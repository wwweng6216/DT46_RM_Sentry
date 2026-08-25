"""
target_pose_server.py — 导航方追击目标点计算节点

功能：
  订阅视觉方发布的 /tracker/enemy_datas，提供 Action 服务。
  调用方发送 Goal（偏移距离），节点计算敌人附近的目标点并返回 PoseStamped。

  敌人坐标为相机坐标系（z前、x右、y下），本节点负责转换到世界坐标系后再计算。

通信链路：
  rm_tracker ── /tracker/enemy_datas (EnemyCenter) ──► 本节点
  rm_serial  ── /imu/rpy (Vector3Stamped) ──► 本节点（用于坐标转换）
  调用方 ── GetTargetPose Action ──► 本节点 ──► 返回 PoseStamped
"""

import math
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time, Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

# Action 相关
from rclpy.action import ActionServer, ActionClient, GoalResponse, CancelResponse
from rm_interfaces.action import GetTargetPose

# 消息类型
from rm_interfaces.msg import EnemyCenter
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Vector3Stamped
from std_msgs.msg import Header

# TF 监听（获取机器人在 map 坐标系下的位姿）
from tf2_ros import TransformListener, Buffer

# QoS 配置（与视觉方 BEST_EFFORT 匹配）
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    欧拉角 → 旋转矩阵（ZYX 顺序：先 Yaw → Pitch → Roll）

    输入：roll, pitch, yaw（弧度）
    输出：3x3 旋转矩阵
    """
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    R = np.array([
        [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [  -sp,            cp*sr,            cp*cr   ]
    ])
    return R


class TargetPoseServer(Node):
    def __init__(self):
        super().__init__('target_pose_server')

        # ==================== 参数声明 ====================
        # 默认偏移距离（米）：当 Goal 中 offset_distance=0 时使用此值
        self.declare_parameter('default_offset_distance', 1.5)
        self.default_offset_distance = self.get_parameter('default_offset_distance').value

        # ==================== TF 监听器 ====================
        # 用于查询机器人在 map 坐标系下的位置，计算偏移方向
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ==================== 订阅 IMU 姿态 ====================
        # 用于将相机坐标系转换到世界坐标系
        imu_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.sub_imu = self.create_subscription(
            Vector3Stamped,
            '/imu/rpy',
            self.imu_callback,
            imu_qos
        )
        self.imu_rpy = None  # [roll, pitch, yaw] 弧度

        # ==================== 相机→世界 旋转矩阵 ====================
        # 相机坐标系：z前、x右、y下
        # 世界坐标系：x前、y左、z上
        # 固定旋转：相机→机体（z前→x前，x右→y左，y下→z上）
        self.R_cam_to_body = np.array([
            [ 0,  0,  1],   # 世界x = 相机z（前→前）
            [-1,  0,  0],   # 世界y = -相机x（右→左）
            [ 0, -1,  0]    # 世界z = -相机y（下→上）
        ], dtype=float)

        # ==================== 订阅敌人数据 ====================
        # QoS 必须为 BEST_EFFORT，与视觉方发布端匹配（否则收不到消息）
        enemy_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.sub_enemy = self.create_subscription(
            EnemyCenter,
            '/tracker/enemy_datas',
            self.enemy_callback,
            enemy_qos
        )

        # ==================== 存储最新数据 ====================
        self.enemy_data = None
        self.enemy_timestamp = None

        # ==================== Action Server ====================
        self.action_callback_group = ReentrantCallbackGroup()
        self.action_server = ActionServer(
            self,
            GetTargetPose,
            '/tracker/get_target_pose',
            self.execute_callback,
            callback_group=self.action_callback_group
        )

        self.get_logger().info('target_pose_server 已启动')
        self.get_logger().info(f'默认偏移距离: {self.default_offset_distance}m')

    def imu_callback(self, msg: Vector3Stamped):
        """存储 IMU 姿态（弧度），用于坐标转换"""
        self.imu_rpy = [msg.vector.x, msg.vector.y, msg.vector.z]

    def enemy_callback(self, msg: EnemyCenter):
        """
        视觉方回调：存储最新的敌人数据（相机坐标系）。
        只做存储，不做计算（轻量级回调，不阻塞订阅线程）。
        """
        self.enemy_data = msg
        self.enemy_timestamp = self.get_clock().now()

    def camera_to_world(self, cam_x, cam_y, cam_z):
        """
        将相机坐标系下的点转换到世界坐标系。

        相机坐标系：z前、x右、y下
        世界坐标系：x前、y左、z上

        变换步骤：
          1. 固定旋转 R_cam_to_body：相机→机体
          2. IMU 旋转 R_imu：机体→世界
          3. P_world = R_imu @ R_cam_to_body @ P_cam
        """
        if self.imu_rpy is None:
            return None

        roll, pitch, yaw = self.imu_rpy
        R_imu = euler_to_rotation_matrix(roll, pitch, yaw)
        R_total = R_imu @ self.R_cam_to_body

        p_cam = np.array([cam_x, cam_y, cam_z])
        p_world = R_total @ p_cam

        return (p_world[0], p_world[1])

    def get_robot_pose(self):
        """
        通过 TF 获取机器人在 map 坐标系下的 (x, y) 坐标。
        返回: (x, y) 或 None（TF 查不到时）
        """
        try:
            transform = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                Time(),
                Duration(seconds=0.1)
            )
            t = transform.transform.translation
            return (t.x, t.y)
        except Exception as e:
            self.get_logger().warn(f'TF 查找失败: {e}')
            return None

    def compute_target_pose(self, offset_distance):
        """
        核心计算：将敌人从相机坐标系转换到世界坐标系，
        再根据机器人位置计算偏移后的导航目标点。

        流程：
          1. 读取敌人相机坐标 (cam_x, cam_y, cam_z)
          2. 转换到世界坐标 (world_x, world_y)
          3. 获取机器人世界坐标 (robot_x, robot_y)
          4. 沿 机器人→敌人 方向，从敌人位置向后偏移
        """
        if self.enemy_data is None:
            return None, False, '无敌人数据'

        if not self.enemy_data.tracked:
            return None, False, '敌人未被跟踪（tracked=false）'

        # 相机坐标系下的敌人位置
        cam_x = self.enemy_data.x
        cam_y = self.enemy_data.y
        cam_z = self.enemy_data.z

        # 旋转：相机坐标系 → 世界坐标系朝向（相对机器人，无平移）
        rel_pos = self.camera_to_world(cam_x, cam_y, cam_z)
        if rel_pos is None:
            return None, False, 'IMU 数据未就绪，无法坐标转换'

        # 获取机器人在 map 坐标系下的绝对位置
        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            # Fallback：TF 不可用时，直接返回敌人原始坐标（无偏移）
            self.get_logger().warn('TF 不可用，返回敌人原始坐标（无偏移）')
            pose = self._make_pose(rel_pos[0], rel_pos[1], 0.0)
            return pose, True, 'TF不可用，返回敌人原始坐标'

        robot_x, robot_y = robot_pose

        # 旋转结果 + 机器人绝对位置 = 敌人在 map 坐标系下的绝对坐标
        enemy_map_x = robot_x + rel_pos[0]
        enemy_map_y = robot_y + rel_pos[1]

        # 计算从机器人指向敌人的方向向量
        dx = enemy_map_x - robot_x
        dy = enemy_map_y - robot_y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 0.01:
            pose = self._make_pose(enemy_map_x, enemy_map_y, 0.0)
            return pose, True, '机器人与敌人距离过近，返回敌人坐标'

        # 单位化方向向量
        ux = dx / dist
        uy = dy / dist

        # 沿方向从敌人位置向后偏移
        target_x = enemy_map_x - ux * offset_distance
        target_y = enemy_map_y - uy * offset_distance

        pose = self._make_pose(target_x, target_y, 0.0)
        return pose, True, f'目标点({target_x:.2f}, {target_y:.2f})，距敌{offset_distance:.1f}m'

    def _make_pose(self, x, y, z):
        """构造 PoseStamped 消息"""
        pose = PoseStamped()
        pose.header = Header()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = 'map'
        pose.pose.position = Point(x=float(x), y=float(y), z=float(z))
        pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        return pose

    def execute_callback(self, goal_handle):
        """
        Action Goal 处理回调。

        流程：
          1. 确定偏移距离（Goal 指定 或 使用默认值）
          2. 循环计算目标点，发布 Feedback
          3. Goal 完成后返回 Result
        """
        offset_distance = goal_handle.request.offset_distance
        if offset_distance <= 0.0:
            offset_distance = self.default_offset_distance

        self.get_logger().info(f'收到 Goal: offset={offset_distance:.2f}m')

        feedback = GetTargetPose.Feedback()
        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal 已被取消')
                return GetTargetPose.Result()

            # 发布 Feedback
            if self.enemy_data is not None and self.enemy_data.tracked:
                rel_pos = self.camera_to_world(
                    self.enemy_data.x, self.enemy_data.y, self.enemy_data.z
                )
                if rel_pos is not None:
                    robot_pose = self.get_robot_pose()
                    if robot_pose is not None:
                        feedback.enemy_x = robot_pose[0] + rel_pos[0]
                        feedback.enemy_y = robot_pose[1] + rel_pos[1]
                    else:
                        feedback.enemy_x = rel_pos[0]
                        feedback.enemy_y = rel_pos[1]
                    feedback.enemy_tracked = True
                else:
                    feedback.enemy_x = 0.0
                    feedback.enemy_y = 0.0
                    feedback.enemy_tracked = True
            else:
                feedback.enemy_x = 0.0
                feedback.enemy_y = 0.0
                feedback.enemy_tracked = False
            goal_handle.publish_feedback(feedback)

            # 计算目标点
            pose, success, message = self.compute_target_pose(offset_distance)

            if success:
                goal_handle.succeed()
                result = GetTargetPose.Result()
                result.target_pose = pose
                result.success = True
                result.message = message
                self.get_logger().info(f'返回结果: {message}')
                return result

            # 未成功，等待后重试
            time.sleep(0.1)

        return GetTargetPose.Result()


def main(args=None):
    rclpy.init(args=args)
    node = TargetPoseServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
