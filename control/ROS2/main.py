#!/usr/bin/env python3

import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped

class ServeNode(Node):
    def __init__(self):
        super().__init__('serve_node')
        self.pub = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 10)

    def send(self, linear_x=0.0, linear_y=0.0, linear_z=0.0, angular_z=0.0, duration=1.0):
        end_time = time.time() + duration
        while time.time() < end_time:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "panda_link0"
            msg.twist.linear.x = linear_x
            msg.twist.linear.y = linear_y
            msg.twist.linear.z = linear_z
            msg.twist.angular.z = angular_z
            self.pub.publish(msg)
            time.sleep(0.02)
        self.pub.publish(TwistStamped())

def main():
    rclpy.init()
    node = ServeNode()

    print("x, y, z, duration\n")
    
    try:
        while True:
            cmd = input(">> ")
            x, y, z, dur = map(float, cmd.split(','))

            node.send(linear_x=x, linear_y=y, linear_z=z, duration=dur)

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

