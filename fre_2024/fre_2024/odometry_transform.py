import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
from tf_transformations import euler_from_quaternion, quaternion_from_euler

class OdometryTransformer(Node):
    def __init__(self):
        super().__init__('odometry_transformer')
        self.subscription = self.create_subscription(
            Odometry,
            '/odometry/filtered',
            self.listener_callback,
            10)
        self.publisher = self.create_publisher(Odometry, '/odometry/transformed', 10)

    def listener_callback(self, msg):
        # Transform the position
        transformed_msg = Odometry()
        transformed_msg.header = msg.header
        transformed_msg.child_frame_id = msg.child_frame_id
        
        # Swap x and y
        transformed_msg.pose.pose.position.x = -msg.pose.pose.position.y
        transformed_msg.pose.pose.position.y = msg.pose.pose.position.x
        transformed_msg.pose.pose.position.z = msg.pose.pose.position.z

        # Transform the orientation
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)

        # Adjust yaw to rotate 90 degrees counter-clockwise
        new_yaw = yaw # 90 degrees in radians
        new_orientation_q = quaternion_from_euler(roll, pitch, new_yaw)

        transformed_msg.pose.pose.orientation = Quaternion(
            x=new_orientation_q[0],
            y=new_orientation_q[1],
            z=new_orientation_q[2],
            w=new_orientation_q[3]
        )

        # Copy the rest of the odometry message
        transformed_msg.twist = msg.twist

        self.publisher.publish(transformed_msg)

def main(args=None):
    rclpy.init(args=args)
    node = OdometryTransformer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
