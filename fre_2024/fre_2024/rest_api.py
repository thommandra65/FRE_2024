#!/usr/bin/env python3
import requests
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json

class PositionPublisher(Node):
    def __init__(self):
        super().__init__('position_publisher')
        self.publisher_ = self.create_publisher(String, 'position_topic', 10)
        self.timer = self.create_timer(3.0, self.publish_position)

    def publish_position(self):
        # Define the URL and headers
        url = 'http://localhost:8000/fre2024/task4/get-positions'
        headers = {
            'accept': 'application/json'
        }

        # Send the GET request
        response = requests.get(url, headers=headers)

        # Check if the request was successful
        if response.status_code == 200:
            # Convert JSON response to string
            response_str = json.dumps(response.json())

            # Publish the JSON response as a string
            msg = String()
            msg.data = response_str
            self.publisher_.publish(msg)
        else:
            # Print an error message if the request failed
            self.get_logger().error(f"Request failed with status code {response.status_code}: {response.text}")

def main(args=None):
    rclpy.init(args=args)
    position_publisher = PositionPublisher()
    rclpy.spin(position_publisher)
    position_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

