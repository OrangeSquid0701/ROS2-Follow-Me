#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
import cv2
import mediapipe as mp
import numpy as np
import threading

class PoseDetectorNode(Node):
    def __init__(self):
        super().__init__('pose_detector')

        # Mediapipe pose detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

        # Global variables for image data
        self.global_image_bgr = None
        self.image_lock = threading.Lock()

        # Subscriber for RGB images
        self.create_subscription(Image, '/camera/rgb/image_raw', self.rgb_image_callback, 10)

        # Publisher for pose landmarks
        self.pose_pub = self.create_publisher(Float32MultiArray, '/pose_landmarks', 10)

        # Thread for processing images and detecting poses
        self.pose_thread = threading.Thread(target=self.detect_pose)
        self.pose_thread.start()

    def rgb_image_callback(self, msg):
        """Callback to process incoming RGB image."""
        try:
            if msg.encoding == 'mono8':
                rgb_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR
            elif msg.encoding == 'rgb8':
                rgb_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
            elif msg.encoding == 'bayer_grbg8':
                bayer_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
                rgb_image = cv2.cvtColor(bayer_image, cv2.COLOR_BAYER_GR2BGR)  # Convert Bayer to BGR
            else:
                self.get_logger().error(f"Unsupported image encoding: {msg.encoding}")
                return

            with self.image_lock:
                self.global_image_bgr = rgb_image
        except Exception as e:
            self.get_logger().error(f"Error processing RGB image: {e}")

    def detect_pose(self):
        """Detect pose and publish hip landmarks."""
        image_width = 640  # Adjust based on your camera resolution
        image_height = 480  # Adjust based on your camera resolution

        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while rclpy.ok():
                with self.image_lock:
                    if self.global_image_bgr is not None:
                        image_rgb = cv2.cvtColor(self.global_image_bgr, cv2.COLOR_BGR2RGB)
                        image_rgb.flags.writeable = False
                        results = pose.process(image_rgb)
                        image_rgb.flags.writeable = True

                        if results.pose_landmarks:
                            # Draw pose landmarks on the image
                            self.mp_drawing.draw_landmarks(
                                self.global_image_bgr, 
                                results.pose_landmarks, 
                                self.mp_pose.POSE_CONNECTIONS,
                                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                                self.mp_drawing.DrawingSpec(color=(144, 238, 144), thickness=2, circle_radius=2)
                            )

                            # Extract left and right hip landmarks
                            left_hip = results.pose_landmarks.landmark[23]
                            right_hip = results.pose_landmarks.landmark[24]

                            # Publish hip landmarks
                            pose_msg = Float32MultiArray()
                            pose_msg.data = [left_hip.x, left_hip.y, right_hip.x, right_hip.y]
                            self.pose_pub.publish(pose_msg)

                        # Display the image with pose landmarks
                        cv2.imshow("RGB Image with Pose", self.global_image_bgr)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            self.get_logger().info("Shutdown initiated by user.")
                            rclpy.shutdown()
                            break

                rclpy.spin_once(self, timeout_sec=0.05)

def main(args=None):
    rclpy.init(args=args)
    node = PoseDetectorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
