#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion
import math

class FollowerNode(Node):
    def __init__(self):
        super().__init__('follower_node')
        self.subscription = self.create_subscription(Point, 'mid_point_goal', self.goal_callback, 10)
        self.odom_subscription = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        
        self.goal = None
        self.position = None
        self.orientation = None
        self.timer = self.create_timer(0.1, self.control_loop)

    def goal_callback(self, msg):
        self.goal = msg

    def odom_callback(self, msg):
        self.position = msg.pose.pose.position
        orientation_q = msg.pose.pose.orientation
        _, _, self.orientation = euler_from_quaternion(
            [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        )

    def control_loop(self):
        if self.goal is None or self.position is None or self.orientation is None:
            return
        
        goal_x = self.goal.x
        goal_y = self.goal.y
        position_x = self.position.x
        position_y = self.position.y

        # Calculate the distance and angle to the goal
        distance = math.sqrt((goal_x - position_x) ** 2 + (goal_y - position_y) ** 2)
        angle_to_goal = math.atan2(goal_y - position_y, goal_x - position_x)

        # Calculate the angle difference between the robot's orientation and the goal
        angle_diff = angle_to_goal - self.orientation

        # Normalize the angle difference to the range [-pi, pi]
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

        # Proportional control constants
        kp_linear = 0.5
        kp_angular = 2.0

        # Calculate the control signals
        linear_velocity = kp_linear * distance
        angular_velocity = kp_angular * angle_diff

        # Create and publish the velocity command
        cmd = Twist()
        cmd.linear.x = linear_velocity
        cmd.angular.z = angular_velocity
        self.publisher.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = FollowerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
