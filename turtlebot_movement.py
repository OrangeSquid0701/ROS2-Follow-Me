#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import numpy as np
import threading

class TurtlebotMovement(Node):
    def __init__(self):
        super().__init__('turtlebot_movement')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/commands/velocity', 10)

        # Subscriptions
        self.create_subscription(Image, '/camera/depth/image_rect_raw', self.depth_image_callback, 10)
        self.create_subscription(Float32MultiArray, '/pose_landmarks', self.pose_callback, 10)
        self.create_subscription(Bool, '/follow_me_stop', self.follow_me_callback, 10)

        # Parameters
        self.image_lock = threading.Lock()
        self.global_image_depth = None
        self.center_x = 0
        self.center_y = 0
        self.emergency_stop = threading.Event()

        # Velocity Smoother
        self.velocity_smoother = VelocitySmoother(alpha=0.1)

        # PID Controller Parameters
        self.integral_error = 0
        self.previous_error = 0
        self.integral_gain = 0.01
        self.proportional_gain = 0.5
        self.derivative_gain = 0.1
        self.dead_zone_threshold = 0.05
        self.damping_factor = 1.0
        self.angular_proportional_gain = 1.0
        self.angular_speed_limit = 0.5
        self.deceleration_base_rate = 0.2

        # Start control thread
        self.control_thread = threading.Thread(target=self.process_and_control_movement)
        self.control_thread.start()

    def depth_image_callback(self, msg):
        try:
            depth_image = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
            with self.image_lock:
                self.global_image_depth = depth_image
        except Exception as e:
            self.get_logger().error(f"Error processing depth image: {e}")

    def calculate_distance(self, depth_image, x, y, min_valid_distance=0.1):
        if x < 0 or x >= depth_image.shape[1] or y < 0 or y >= depth_image.shape[0]:
            return None
        distance = depth_image[y, x] * 0.001  # Convert from mm to meters
        if distance == 0 or np.isnan(distance) or distance < min_valid_distance:
            return None
        return distance

    def process_and_control_movement(self):
        image_width = 640
        image_height = 480

        while rclpy.ok():
            if self.emergency_stop.is_set():
                continue

            with self.image_lock:
                twist = Twist()
                if self.global_image_depth is not None:
                    distance_to_person = self.calculate_distance(self.global_image_depth, self.center_x, self.center_y)
                    if distance_to_person is not None:
                        desired_distance = 1.2
                        error = distance_to_person - desired_distance
                        self.integral_error += error
                        derivative_error = error - self.previous_error
                        self.previous_error = error

                        if abs(error) < self.dead_zone_threshold:
                            twist.linear.x = 0
                        else:
                            twist.linear.x = max(-0.2, min(0.2, self.proportional_gain * error + 
                                                            self.integral_gain * self.integral_error - 
                                                            self.derivative_gain * derivative_error))

                        if distance_to_person < desired_distance + 0.5:
                            twist.linear.x *= 0.5
                        else:
                            twist.linear.x *= self.damping_factor

                        image_center_x = self.global_image_depth.shape[1] // 2
                        if abs(self.center_x - image_center_x) > image_center_x * 0.1:
                            twist.angular.z = self.angular_proportional_gain * (image_center_x - self.center_x) / image_center_x

                        current_linear_x, current_angular_z = self.velocity_smoother.smooth(twist.linear.x, twist.angular.z)
                        twist.linear.x = current_linear_x
                        twist.angular.z = current_angular_z
                        self.cmd_vel_pub.publish(twist)
                else:
                    self.get_logger().warn("No depth image available to calculate distance.")

            self.get_clock().sleep(0.05)

    def follow_me_callback(self, msg):
        if msg.data:
            self.emergency_stop.set()
        else:
            self.emergency_stop.clear()

    def pose_callback(self, msg):
        left_hip_x, left_hip_y, right_hip_x, right_hip_y = msg.data
        self.center_x = int((left_hip_x + right_hip_x) / 2 * 640)
        self.center_y = int((left_hip_y + right_hip_y) / 2 * 480)

class VelocitySmoother:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.linear_x = 0
        self.angular_z = 0

    def smooth(self, linear_x, angular_z):
        self.linear_x = self.alpha * linear_x + (1 - self.alpha) * self.linear_x
        self.angular_z = self.alpha * angular_z + (1 - self.alpha) * self.angular_z
        return self.linear_x, self.angular_z

def main(args=None):
    rclpy.init(args=args)
    node = TurtlebotMovement()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
