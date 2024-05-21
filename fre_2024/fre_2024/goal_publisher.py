# robot_nav_controller/move_robot_nav.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import tf_transformations
import math

class RobotNavController(Node):
    def __init__(self):
        super().__init__('robot_nav_controller')
        self.publisher_ = self.create_publisher(PoseStamped, 'goal_pose', 10)
        self.subscription = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10)
        self.current_position = (0.0, 0.0)
        self.current_orientation = 0.0  # Initial orientation in radians
        self.odom_received = False
        self.move_sequence = [
            (1.5, 0.0),  # Move 1 meter forward
            (0.0, math.pi / 2),  # Move 1 meter at 45 degrees
            # (0.5, -math.pi / 4),  # Move 1 meter at -45 degrees
            #(0.0, math.pi / 2)  # Rotate 90 degrees
        ]
        self.current_move_index = 0

    def odom_callback(self, msg):
        # Update current position and orientation from odometry
        self.current_position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        orientation_q = msg.pose.pose.orientation
        euler = tf_transformations.euler_from_quaternion([
            orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])
        self.current_orientation = euler[2]  # yaw
        self.odom_received = True

    def move_robot(self):
        if not self.odom_received:
            self.get_logger().info("Waiting for odometry data...")
            return

        if self.current_move_index >= len(self.move_sequence):
            self.get_logger().info("All moves completed.")
            return

        distance, angle = self.move_sequence[self.current_move_index]
        x0, y0 = self.current_position
        theta = self.current_orientation + angle  # Update the current angle
        x1 = x0 + distance * math.cos(theta)
        y1 = y0 + distance * math.sin(theta)

        # Create PoseStamped message
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.get_clock().now().to_msg()

        goal.pose.position.x = x1
        goal.pose.position.y = y1

        # Convert yaw angle to quaternion
        quaternion = tf_transformations.quaternion_from_euler(0, 0, theta)
        goal.pose.orientation.x = quaternion[0]
        goal.pose.orientation.y = quaternion[1]
        goal.pose.orientation.z = quaternion[2]
        goal.pose.orientation.w = quaternion[3]

        # Publish PoseStamped message
        self.publisher_.publish(goal)

        self.get_logger().info(f"Moving to position: ({x1}, {y1}) with orientation: {theta}")
        if self.current_position == (x1, y1):
            self.current_move_index += 1

def main(args=None):
    rclpy.init(args=args)
    robot_nav_controller = RobotNavController()
    
    try:
        while rclpy.ok():
            rclpy.spin_once(robot_nav_controller)
            robot_nav_controller.move_robot()
    except KeyboardInterrupt:
        pass
    finally:
        robot_nav_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()