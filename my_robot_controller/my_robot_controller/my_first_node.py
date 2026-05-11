#!/usr/bin/env python3
import rclpy

#import node class
from rclpy.node import Node

# Create node class named MyNode
class MyNode(Node):

    # self constuctor
    def __init__(self):

        # initialize node named first_node (for ROS identification)
        # verify all nodes with "ros2 node list" command
        super().__init__("first_node")

        # MyNode counter
        self.counter_ = 0

        # print statement example for ros to terminal
        self.get_logger().info("Hello from ROS")

        # call timer_callback on 1 second intervals
        self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        self.get_logger().info("hello world " + str(self.counter_))
        self.counter_ += 1

# start main function
def main(args=None):
    
    # init ROS communication
    rclpy.init(args=args)

    #init MyNode
    node = MyNode()
    
    # Keep node alive and active
    rclpy.spin(node)

    # kill script and cleanup after exit/kill
    rclpy.shutdown()

# run main() if file run directly
if __name__ == '__main__':
    main()