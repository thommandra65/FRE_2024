#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav2_simple_commander.robot_navigator import BasicNavigator
from geometry_msgs.msg import PoseStamped, Point
import tf_transformations

class SmoothFollower(Node):
    def __init__(self):
        super().__init__('smooth_follower')
        self.subscription = self.create_subscription(Point, 'mid_point_goal', self.goal_callback, 10)
        self.navigator = BasicNavigator()
        self.current_goal = None
        

    def goal_callback(self, msg):
        self.current_goal = self.create_pose_stamped(msg.x, msg.y, 0.0)
        self.navigator.goToPose(self.current_goal)

    def create_pose_stamped(self, position_x, position_y, rotation_z):
        q_x, q_y, q_z, q_w = tf_transformations.quaternion_from_euler(0.0, 0.0, rotation_z)
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = position_x
        goal_pose.pose.position.y = position_y
        goal_pose.pose.position.z = 0.0
        goal_pose.pose.orientation.x = q_x
        goal_pose.pose.orientation.y = q_y
        goal_pose.pose.orientation.z = q_z
        goal_pose.pose.orientation.w = q_w
        return goal_pose

    def follow_goal(self):

        while rclpy.ok():

            rclpy.spin_once(self)

            if self.current_goal:

                while not self.navigator.isTaskComplete():

                    feedback = self.navigator.getFeedback()

                    self.get_logger().info(f'Feedback: {feedback}')

                    rclpy.spin_once(self)

def main(args=None):
    rclpy.init(args=args)
    follower_node = SmoothFollower()

    # # Uncomment and set the initial pose if needed
    # initial_pose = follower_node.create_pose_stamped(0.0, 0.0, 0.0)
    # follower_node.navigator.setInitialPose(initial_pose)

    # follower_node.navigator.waitUntilNav2Active()

    try:
        follower_node.follow_goal()
    except KeyboardInterrupt:
        pass

    follower_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
