#!/usr/bin/env python3

import rclpy

from rclpy.node import Node

# from ros package name import message type
from turtlesim.msg import Pose
from geometry_msgs.msg import Twist
# from ros package name import service type
from turtlesim.srv import SetPen


# partial allows multiple arguments for callbacks
from functools import partial

# Create TCN that inherits node
class TurtleControllerNode(Node):

    # init turtle_controller
    def __init__(self):
        super().__init__("turtle_controller")
        self.previous_x_ = 0.0

        # syntax for publisher( message type, topic name, queue size)
        # queue = store newest 10 messages to be handled 

        # publisher only publishes when .publish function called
        self.cmd_vel_publisher_ = self.create_publisher(
            Twist, "/turtle1/cmd_vel" , 10)
        

        # command "ros2 interface show geometry_msgs/msg/Twist" contains linear 
        # and angular fields (both contain x,y,z)

        # "/turtle1/pose" or cmd_vel = turtlesim_node topics

        
        # subscriber always executes/listens  
        # create subscriber( message type, topic name, function call when message received, queue size)
        self.pose_subscriber_ = self.create_subscription(
            Pose, "/turtle1/pose", self.pose_callback, 10 )
        
        # print terminal
        self.get_logger().info("Turtle controller has been started")


    # pose stores "/turtle1/pose", Pose suggests the data type

    # call this function whenever a pose message is received. 

    def pose_callback(self, pose: Pose):

        # create movement command messgae
        cmd = Twist()

        if pose.x > 8.0 or pose.x <3 or pose.y <3 or pose.y >8.0:
            cmd.linear.x = 1.0
            cmd.angular.z = 0.9

        
        else:
            cmd.linear.x = 5.0
            cmd.angular.z = 0.0

        # publish command
        self.cmd_vel_publisher_.publish(cmd)

        if pose.x > 5.5 and self.previous_x_ <= 5.5:

            # update prev
            self.previous_x_ = pose.x


            self.call_set_pen_service(255, 0, 0, 3,0)
            self.get_logger().info("Set color to red")
        elif pose.x <= 5.5 and self.previous_x_ >5.5:

            # update prev
            self.previous_x_ = pose.x
            self.call_set_pen_service(0, 255, 0, 3,0)
            self.get_logger().info("Set color to green")



    # pen service function
    def call_set_pen_service(self, r, g, b, width, off):

        # create service client connection to setpen
        client = self.create_client(SetPen, "turtle1/set_pen")

        # check every second
        while not client.wait_for_service(1.0):
            self.get_logger().warn("waiting for service...")


        # create request object  (make form)
        request = SetPen.Request()
        request.r = r
        request.g = g
        request.b = b
        request.width = width
        request.off = off

        # request service (color change) (send form of new state)
        future = client.call_async(request)

        # run callback_set_pen when dinished
        future.add_done_callback(self.callback_set_pen)

    # call when serive response returns
    def callback_set_pen(self, future):
        try:
            # get serive result
            response = future.result()
        except Exception as e:
            self.get_logger().error("Service call failed: %r" % (e,))


def main(args=None):
    rclpy.init(args=args)
    node = TurtleControllerNode()
    rclpy.spin(node)
    rclpy.shutdown()