#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import math
import matplotlib
matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PointStamped, PoseStamped, Point
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from tf2_ros import TransformListener, Buffer, LookupException, ConnectivityException, ExtrapolationException
from std_msgs.msg import Int32
import collections
import tf2_geometry_msgs
from nav2_simple_commander.robot_navigator import BasicNavigator
from geometry_msgs.msg import Quaternion

bp = [[-0.5, 0.15], [2, 0.75], [-0.5, -0.75], [2, -0.15]]
width1 = bp[1][0] - bp[0][0]
height1 = bp[1][1] - bp[0][1]
width2 = bp[3][0] - bp[2][0]
height2 = bp[3][1] - bp[2][1]

class LidarRansacNode(Node):
    def __init__(self):
        super().__init__('lidar_ransac_transform')
        
        self.sub_laser_1 = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.sub_laser_2 = self.create_subscription(LaserScan, '/scan', self.range_filter, 10)
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.clbk_odom, 10)
        self.pub_slope_number = self.create_publisher(Int32, 'slope_number', 10)
        
        self.pub_mid_point_goal = self.create_publisher(PoseStamped, 'mid_point_goal', 10)

        self.previous_midpoints = collections.deque(maxlen=5) 

        self.position_ = Point()
        self.yaw_ = 0
        self.num_of_slope_ = 1

        self.right_dist_avg = 0
        self.left_dist_avg = 0

        self.laser_angles = []
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.timer = self.create_timer(0.05, self.timer_callback)

        self.nav = BasicNavigator()
        self.odom_subscriber = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        self.current_pose = PoseStamped()

    def odom_callback(self, msg):
        self.current_pose.header = msg.header
        self.current_pose.pose = msg.pose.pose

    def create_pose_stamped(self, base_pose, position_x, position_y, rotation_z):
        q_x, q_y, q_z, q_w = quaternion_from_euler(0.0, 0.0, rotation_z)
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = base_pose.pose.position.x + position_x
        goal_pose.pose.position.y = base_pose.pose.position.y + position_y
        goal_pose.pose.position.z = 0.0

        base_orientation_q = base_pose.pose.orientation
        base_orientation = [base_orientation_q.x, base_orientation_q.y, base_orientation_q.z, base_orientation_q.w]
        base_yaw = euler_from_quaternion(base_orientation)[2]
        new_yaw = base_yaw + rotation_z

        new_orientation_q = quaternion_from_euler(0.0, 0.0, new_yaw)
        goal_pose.pose.orientation.x = new_orientation_q[0]
        goal_pose.pose.orientation.y = new_orientation_q[1]
        goal_pose.pose.orientation.z = new_orientation_q[2]
        goal_pose.pose.orientation.w = new_orientation_q[3]
        
        return goal_pose

    def moving_average(self, new_point):
        self.previous_midpoints.append(new_point)
        if len(self.previous_midpoints) == 0:
            return new_point
        avg_x = sum(p[0] for p in self.previous_midpoints) / len(self.previous_midpoints)
        avg_y = sum(p[1] for p in self.previous_midpoints) / len(self.previous_midpoints)
        return (avg_x, avg_y)

    def clbk_odom(self, msg):
        self.position_ = msg.pose.pose.position
        quaternion = (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w)
        euler = euler_from_quaternion(quaternion)
        self.yaw_ = euler[2]

    def range_filter(self, msg):
        ranges = [x for x in msg.ranges if not math.isnan(x)]
        custom_range = ranges[270:360] + ranges[0:90]
        right_dist = [x for x in custom_range[0:70] if x < 0.85]
        left_dist = [x for x in custom_range[110:180] if x < 0.85]
        
        if len(right_dist) == 0 and len(left_dist) == 0:
            self.right_dist_avg = 0
            self.left_dist_avg = 0
        elif len(right_dist) == 0:
            self.right_dist_avg = 0
            self.left_dist_avg = sum(left_dist) / len(left_dist)
        elif len(right_dist) > 0 and len(left_dist) == 0:
            self.right_dist_avg = sum(right_dist) / len(right_dist)
            self.left_dist_avg = 0
        else:
            self.right_dist_avg = sum(right_dist) / len(right_dist)
            self.left_dist_avg = sum(left_dist) / len(left_dist)

    def fit_with_least_squares(self, X, y):
        b = np.ones((X.shape[0], 1))
        A = np.hstack((X, b))
        theta = np.linalg.lstsq(A, y, rcond=None)[0]
        return theta

    def evaluate_model(self, X, y, theta, inlier_threshold):
        b = np.ones((X.shape[0], 1))
        y = y.reshape((y.shape[0], 1))
        A = np.hstack((y, X, b))
        theta = np.insert(theta, 0, -1.)

        distances = np.abs(np.sum(A * theta, axis=1)) / np.sqrt(np.sum(np.power(theta[:-1], 2)))
        inliers = distances <= inlier_threshold
        num_inliers = np.count_nonzero(inliers == True)

        return num_inliers

    def ransac(self, X, y, max_iters=100, samples_to_fit=2, inlier_threshold=0.35, min_inliers=7):
        best_model = None
        best_model_performance = 0
        num_samples = X.shape[0]

        for i in range(max_iters):
            sample = np.random.choice(
                num_samples, size=samples_to_fit, replace=False)
            if len(X[sample]) == 0 or len(y[sample]) == 0:
                break
            model_params = self.fit_with_least_squares(X[sample], y[sample])
            model_performance = self.evaluate_model(
                X, y, model_params, inlier_threshold)

            if model_performance < min_inliers:
                continue

            if model_performance > best_model_performance:
                best_model = model_params
                best_model_performance = model_performance

        return best_model

    def polynomial_fit(self, x, y, degree=2):
        if len(x) != len(y):
            raise ValueError("Expected x and y to have the same length.")
        p = np.polyfit(x, y, degree)
        return np.polyval(p, x)

    def scan_callback(self, msg):
        laser_ranges = msg.ranges
        X_all = []
        y_all = []
        
        self.laser_angles = [msg.angle_min + i * msg.angle_increment for i in range(len(laser_ranges))]

        for j in range(0, len(bp), 2):
            X = []
            y = []
            for i in range(len(laser_ranges)):
                pn_x = laser_ranges[i] * math.cos(self.laser_angles[i])
                pn_y = laser_ranges[i] * math.sin(self.laser_angles[i])
                if (not math.isinf(pn_x) and not math.isinf(pn_y) and 
                    bp[j][0] < pn_x < bp[j+1][0] and bp[j][1] < pn_y < bp[j+1][1]):
                    X.append(pn_x)
                    y.append(pn_y)
            if len(X) > 2 and len(X) == len(y):
                X_all.append(np.array(X))
                y_all.append(np.array(y))
            else:
                self.get_logger().info("Skipping a set of points due to insufficient length or mismatch.")
                self.get_logger().debug(f"X length: {len(X)}, y length: {len(y)}")

        if len(X_all) > 0 and len(y_all) > 1:
            if len(X_all) > 1 and len(y_all) > 1:
                left_edge_points = y_all[0]
                right_edge_points = y_all[1]

                if len(left_edge_points) > 0 and len(right_edge_points) > 0:
                    x_points = X_all[0]

                    # Ensure both x_points and y_points have the same length
                    min_length = min(len(x_points), len(left_edge_points), len(right_edge_points))
                    x_points = x_points[:min_length]
                    left_edge_points = left_edge_points[:min_length]
                    right_edge_points = right_edge_points[:min_length]

                    if len(x_points) != len(left_edge_points) or len(x_points) != len(right_edge_points):
                        self.get_logger().error(f"Length mismatch after trimming: x_points({len(x_points)}), left_edge_points({len(left_edge_points)}), right_edge_points({len(right_edge_points)})")
                        return
                    
                    left_edge_poly = self.polynomial_fit(x_points, left_edge_points)
                    right_edge_poly = self.polynomial_fit(x_points, right_edge_points)

                    center_points_poly = (left_edge_poly + right_edge_poly) / 2.0

                    plt.clf()
                    plt.title("Laser Scan and Polynomial Fit on Edges!")
                    rectangle1 = Rectangle(bp[0], width1, height1, linewidth=1, edgecolor='r', facecolor='none')
                    rectangle2 = Rectangle(bp[2], width2, height2, linewidth=1, edgecolor='b', facecolor='none')
                    ax = plt.gca()
                    ax.add_patch(rectangle1)
                    ax.add_patch(rectangle2)
                    
                    plt.plot(x_points, left_edge_points, 'o', label='Left Edge')
                    plt.plot(x_points, right_edge_points, 'o', label='Right Edge')
                    plt.plot(x_points, left_edge_poly, '-', label='Left Edge Smoothed')
                    plt.plot(x_points, right_edge_poly, '-', label='Right Edge Smoothed')
                    plt.plot(x_points, center_points_poly, '-', label='Center Points Smoothed')
                    plt.legend()
                    plt.draw()
                    plt.pause(0.001)

                    mid_point_base = [x_points[len(x_points)//2], center_points_poly[len(x_points)//2]]
                    smoothed_mid_point = self.moving_average(mid_point_base)

                    point = PointStamped()
                    point.header.frame_id = 'base_link'
                    point.point.x = float(smoothed_mid_point[0])
                    point.point.y = float(smoothed_mid_point[1])
                    point.point.z = 0.0
                    try:
                        transform = self.tf_buffer.lookup_transform('odom', 'base_link', rclpy.time.Time())           
                        point_odom = tf2_geometry_msgs.do_transform_point(point, transform)
                        self.get_logger().info(f"mid point in odom frame: x={point_odom.point.x}, y={point_odom.point.y}")
                        
                        pose_msg = PoseStamped()
                        pose_msg.header.frame_id = 'odom'
                        pose_msg.pose.position.x = point_odom.point.x
                        pose_msg.pose.position.y = point_odom.point.y
                        pose_msg.pose.position.z = 0.0
                        pose_msg.pose.orientation.w = 1.0

                        self.pub_mid_point_goal.publish(pose_msg)
                        
                    except (LookupException, ConnectivityException, ExtrapolationException) as e:
                        self.get_logger().error(f"Could not transform point: {e}")

            else:
                self.get_logger().info("Not enough edge points detected for polynomial fitting")
        else:
            self.get_logger().info("Not enough data points for both left and right edges")


    def timer_callback(self):
        # slope_msg = Int32()
        # if self.right_dist_avg > 0 and self.left_dist_avg > 0:
        #     self.num_of_slope_ = 2
        #     self.get_logger().info(f"Number of slopes: {self.num_of_slope_}")
        #     slope_msg.data = self.num_of_slope_
        #     self.pub_slope_number.publish(slope_msg)
        # elif self.right_dist_avg == 0 and self.left_dist_avg > 0:
        #     self.num_of_slope_ = 1
        #     self.get_logger().info(f"Number of slopes: {self.num_of_slope_}")
        #     slope_msg.data = self.num_of_slope_
        #     self.pub_slope_number.publish(slope_msg)
        # elif self.right_dist_avg > 0 and self.left_dist_avg == 0:
        #     self.num_of_slope_ = 1
        #     self.get_logger().info(f"Number of slopes: {self.num_of_slope_}")
        #     slope_msg.data = self.num_of_slope_
        #     self.pub_slope_number.publish(slope_msg)
        # else:
        #     self.num_of_slope_ = 0
        #     self.get_logger().info(f"Number of slopes: {self.num_of_slope_}")
        #     slope_msg.data = self.num_of_slope_
        #     self.pub_slope_number.publish(slope_msg)
        pass

def main(args=None):
    rclpy.init(args=args)
    node = LidarRansacNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

