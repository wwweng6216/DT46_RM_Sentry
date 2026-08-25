"""
chase_client.py — 独立追击客户端节点

功能：
  收到触发信号后，调用 target_pose_server 获取敌人附近的导航目标点，
  再通过 Nav2 执行导航，实现追击。

通信链路：
  rm_decision ── /nav/chase_trigger (Bool) ──► 本节点
  本节点 ── GetTargetPose Action ──► target_pose_server
  本节点 ── Nav2 ──► 导航执行
"""

import rclpy
from rclpy.node import Node
from rclpy.time import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

# Action 相关
from rclpy.action import ActionClient
from rm_interfaces.action import GetTargetPose

# 消息类型
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped

# Nav2
from nav2_simple_commander.robot_navigator import BasicNavigator


class ChaseClient(Node):
    def __init__(self):
        super().__init__('chase_client')

        # ==================== 参数声明 ====================
        self.declare_parameter('offset_distance', 1.5)       # 偏移距离（米）
        self.declare_parameter('chase_rate', 0.5)             # 追击刷新频率（Hz）

        self.offset_distance = self.get_parameter('offset_distance').value
        self.chase_rate = self.get_parameter('chase_rate').value

        # ==================== 状态变量 ====================
        self.is_chasing = False          # 当前是否在追击
        self.goal_handle = None          # 当前 Action Goal 的 handle（用于取消）

        # ==================== Nav2 导航器 ====================
        self.navigator = BasicNavigator()

        # ==================== Action Client ====================
        self.action_callback_group = ReentrantCallbackGroup()
        self.action_client = ActionClient(
            self,
            GetTargetPose,
            '/tracker/get_target_pose',
            callback_group=self.action_callback_group
        )

        # ==================== 订阅触发信号 ====================
        self.sub_trigger = self.create_subscription(
            Bool,
            '/nav/chase_trigger',
            self.trigger_callback,
            10
        )

        # ==================== 定时器：周期性检查追击状态 ====================
        self.timer = self.create_timer(1.0 / self.chase_rate, self.chase_loop)

        self.get_logger().info('chase_client 已启动')
        self.get_logger().info(f'偏移距离: {self.offset_distance}m')

    def trigger_callback(self, msg: Bool):
        """
        触发回调：收到 True 开始追击，收到 False 停止追击。
        """
        if msg.data and not self.is_chasing:
            self.get_logger().info('收到追击触发信号，开始追击')
            self.is_chasing = True
            self._send_goal()

        elif not msg.data and self.is_chasing:
            self.get_logger().info('收到停止信号，取消追击')
            self.is_chasing = False
            self._cancel_goal()
            self.navigator.cancelTask()

    def _send_goal(self):
        """向 target_pose_server 发送 Goal"""
        if not self.action_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().warn('target_pose_server 未就绪，无法发送 Goal')
            self.is_chasing = False
            return

        goal = GetTargetPose.Goal()
        goal.offset_distance = self.offset_distance

        self.get_logger().info(f'发送 Goal: offset={self.offset_distance}m')

        future = self.action_client.send_goal_async(
            goal,
            feedback_callback=self._feedback_callback
        )
        future.add_done_callback(self._goal_response_callback)

    def _goal_response_callback(self, future):
        """Goal 被接受后的回调"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal 被拒绝')
            self.is_chasing = False
            return

        self.get_logger().info('Goal 已被接受，等待结果...')
        self.goal_handle = goal_handle

        # 异步等待结果
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_callback)

    def _result_callback(self, future):
        """收到 Result 后的回调：获取目标点并导航"""
        try:
            result = future.result().result
        except Exception as e:
            self.get_logger().error(f'获取 Result 异常: {e}')
            self.is_chasing = False
            return

        if not result.success:
            self.get_logger().warn(f'目标点计算失败: {result.message}')
            # 敌人丢失时，等待后重试
            self.is_chasing = True
            return

        target_pose = result.target_pose
        self.get_logger().info(
            f'收到目标点: ({target_pose.pose.position.x:.2f}, '
            f'{target_pose.pose.position.y:.2f}) - {result.message}'
        )

        # 调用 Nav2 导航
        self.navigator.goToPose(target_pose)

    def _feedback_callback(self, feedback):
        """Feedback 回调：可选，打印敌人实时位置"""
        fb = feedback.feedback
        self.get_logger().debug(
            f'Feedback: 敌人({fb.enemy_x:.2f}, {fb.enemy_y:.2f}) '
            f'跟踪中={fb.enemy_tracked}',
            once=True
        )

    def _cancel_goal(self):
        """取消当前 Goal"""
        if self.goal_handle is not None:
            self.get_logger().info('取消 Goal')
            self.goal_handle.cancel_goal_async()
            self.goal_handle = None

    def chase_loop(self):
        """定时器回调：追击中持续刷新目标点"""
        if not self.is_chasing:
            return

        # 如果 Nav2 正在执行，不重复发送
        if not self.navigator.isTaskComplete():
            return

        # Nav2 完成后，重新发送 Goal 获取新目标点
        self._send_goal()


def main(args=None):
    rclpy.init(args=args)
    node = ChaseClient()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
