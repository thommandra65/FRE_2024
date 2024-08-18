#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav2_simple_commander.robot_navigator import BasicNavigator
from geometry_msgs.msg import PoseStamped, Quaternion, TransformStamped
from nav_msgs.msg import Odometry
import tf_transformations

class NavigatorWithOdom(Node):
    def __init__(self):
        super().__init__('navigator_with_odom')
        self.nav = BasicNavigator()
        self.odom_subscriber = self.create_subscription(Odometry, 'odometry/filtered', self.odom_callback, 10)
        self.current_pose = PoseStamped()
        self.initial_pose = PoseStamped()
        

    def odom_callback(self, msg):
        self.initial_pose.pose.position.x = 0.0
        self.initial_pose.pose.position.y = 0.0
        self.initial_pose.pose.position.z = 0.0
        self.initial_pose.pose.orientation.x = 0.0
        self.initial_pose.pose.orientation.y = 0.0
        self.initial_pose.pose.orientation.z = 0.0
        self.initial_pose.pose.orientation.w = 1.0
        self.current_pose.header = msg.header
        self.initial_pose.header = msg.header
        self.current_pose.pose = msg.pose.pose

    def create_pose_stamped(self, base_pose, position_x, position_y, rotation_z):
        q_x, q_y, q_z, q_w = tf_transformations.quaternion_from_euler(0.0, 0.0, rotation_z)
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = base_pose.pose.position.x + position_x
        goal_pose.pose.position.y = base_pose.pose.position.y + position_y
        goal_pose.pose.position.z = 0.0

        base_orientation_q = base_pose.pose.orientation
        base_orientation = [base_orientation_q.x, base_orientation_q.y, base_orientation_q.z, base_orientation_q.w]
        base_yaw = tf_transformations.euler_from_quaternion(base_orientation)[2]
        new_yaw = base_yaw + rotation_z

        new_orientation_q = tf_transformations.quaternion_from_euler(0.0, 0.0, new_yaw)
        goal_pose.pose.orientation.x = new_orientation_q[0]
        goal_pose.pose.orientation.y = new_orientation_q[1]
        goal_pose.pose.orientation.z = new_orientation_q[2]
        goal_pose.pose.orientation.w = new_orientation_q[3]
        
        return goal_pose

    def main(self):
        # --- Set initial pose ---
        # initial_pose = self.create_pose_stamped(self.current_pose, 0.0, 0.0, 0.0)
        # self.nav.setInitialPose(initial_pose)

        # --- Wait for Nav2 ---
        # self.nav.waitUntilNav2Active()

        # --- Get current pose from odom ---
        self.current_pose = self.wait_for_pose()

        # --- Create some Nav2 goal poses relative to the current pose ---
        goal_pose0 = self.create_pose_stamped(self.initial_pose, 2.48212, 1.18493, 0)
        goal_pose1 = self.create_pose_stamped(self.initial_pose, 6.68702, 1.27119, -1.62)
        goal_pose2 = self.create_pose_stamped(self.initial_pose, 6.49906, -1.67222, -3.00245)
        goal_pose3 = self.create_pose_stamped(self.initial_pose, 3.86154, -1.64584, 2.18695)
        goal_pose4 = self.create_pose_stamped(self.initial_pose, 1.41322, -0.423331, -2.69012)

        # --- Follow Waypoints ---
        waypoints = [goal_pose0, goal_pose1, goal_pose2, goal_pose3, goal_pose4]
        self.nav.followWaypoints(waypoints)
        while not self.nav.isTaskComplete():
            feedback = self.nav.getFeedback()
            print(feedback)

        # --- Get the result ---
        print(self.nav.getResult())

    def wait_for_pose(self):
        while rclpy.ok() and (self.current_pose.header.stamp.sec == 0 and self.current_pose.header.stamp.nanosec == 0):
            rclpy.spin_once(self)
        return self.current_pose

def main(args=None):
    rclpy.init(args=args)
    navigator = NavigatorWithOdom()
    navigator.main()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
