#!/usr/bin/env python3
"""
stream_to_ros.py — RTSP-to-ROS 2 bridge node.

Receives an H.264 RTSP stream (e.g. from a Jetson Orin camera), decodes it
with OpenCV, and publishes frames as sensor_msgs/Image on a local ROS 2 topic.

Usage:
    python3 stream_to_ros.py --rtsp-url rtsp://robot-ip:8554/cam
    python3 stream_to_ros.py --rtsp-url rtsp://robot-ip:8554/cam --topic /camera/image_raw --fps 15
"""

import argparse
import sys
import time

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class RTSPBridge(Node):
    """ROS 2 node that reads an RTSP stream and publishes sensor_msgs/Image."""

    def __init__(self, rtsp_url: str, topic: str, fps: float):
        super().__init__("rtsp_bridge")
        self.rtsp_url = rtsp_url
        self.target_fps = fps
        self.frame_interval = 1.0 / fps

        self.publisher = self.create_publisher(Image, topic, 10)
        self.get_logger().info(f"Publishing {topic} at {fps} FPS from {rtsp_url}")

        self.cap: cv2.VideoCapture | None = None
        self._connect()

        # Use a timer to drive the capture loop so rclpy can spin properly
        self.timer = self.create_timer(self.frame_interval, self._capture_and_publish)

    def _connect(self) -> None:
        """Open (or re-open) the RTSP stream."""
        if self.cap is not None:
            self.cap.release()

        self.get_logger().info(f"Connecting to {self.rtsp_url} ...")
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

        if not self.cap.isOpened():
            self.get_logger().warn("Failed to open RTSP stream — will retry next tick")
            self.cap = None
            return

        # Reduce internal buffer to 1 so we always get the latest frame
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.get_logger().info("RTSP stream connected.")

    def _capture_and_publish(self) -> None:
        """Grab a frame and publish it."""
        if self.cap is None or not self.cap.isOpened():
            self._connect()
            return

        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Frame read failed — reconnecting")
            self._connect()
            return

        # Convert OpenCV BGR frame to sensor_msgs/Image
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        msg.height, msg.width = frame.shape[:2]
        msg.encoding = "bgr8"
        msg.is_bigendian = False
        msg.step = msg.width * 3
        msg.data = frame.tobytes()

        self.publisher.publish(msg)

    def destroy_node(self) -> None:
        if self.cap is not None:
            self.cap.release()
        super().destroy_node()


def main() -> None:
    parser = argparse.ArgumentParser(description="RTSP to ROS 2 bridge")
    parser.add_argument(
        "--rtsp-url",
        type=str,
        default="rtsp://robot-ip:8554/cam",
        help="RTSP stream URL (default: rtsp://robot-ip:8554/cam)",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="/camera/image_raw",
        help="ROS 2 topic to publish on (default: /camera/image_raw)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=15.0,
        help="Target publish rate in FPS (default: 15)",
    )
    args = parser.parse_args()

    rclpy.init()
    node = RTSPBridge(args.rtsp_url, args.topic, args.fps)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
