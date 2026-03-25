#!/usr/bin/env python3
"""
vision_agent.py — RAI vision agent for Foxglove + Qwen3-VL.

Subscribes to a ROS 2 camera topic (sensor_msgs/Image) and listens for
HRIMessage prompts on an input topic. When a prompt arrives, grabs the latest
camera frame, sends text + image to a Qwen3-VL model via Ollama, and publishes
the response as an HRIMessage on the output topic.

The model can be switched at runtime from the Foxglove Agent Panel.  The panel
encodes the selected model name as a prefix in the communication_id field:
    model:<model_name>|<unique_id>

Usage:
    python3 vision_agent.py
    python3 vision_agent.py --camera-topic /camera/image_raw \
        --input-topic /from_human --output-topic /to_human \
        --default-model qwen3-vl:8b
"""

from __future__ import annotations

import argparse
import base64
import io
import threading
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

# We interact with rai_interfaces via raw dict publishing over rosbridge, but
# for the ROS 2 side we need the compiled message type.  If rai_interfaces is
# available we import it; otherwise we fall back to a simple string-based shim.
try:
    from rai_interfaces.msg import HRIMessage
    _HAS_RAI_INTERFACES = True
except ImportError:
    _HAS_RAI_INTERFACES = False

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama


# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_CAMERA_TOPIC = "/camera/image_raw"
DEFAULT_INPUT_TOPIC = "/from_human"
DEFAULT_OUTPUT_TOPIC = "/to_human"
DEFAULT_MODEL = "qwen3-vl:8b"
OLLAMA_BASE_URL = "http://localhost:11434"

SYSTEM_PROMPT = (
    "You are a vision-capable robotic assistant. When the user sends a message, "
    "you receive the latest camera frame from the robot alongside their text. "
    "Describe what you see, answer questions about the scene, or follow "
    "instructions as appropriate. Be concise and actionable."
)

MODEL_PREFIX = "model:"
MODEL_SEPARATOR = "|"


# ── Helpers ──────────────────────────────────────────────────────────────────

def parse_model_from_comm_id(communication_id: str, default_model: str) -> tuple[str, str]:
    """Extract model name and clean communication_id from the prefixed string.

    Format: ``model:<model_name>|<original_comm_id>``

    Returns (model_name, original_comm_id).
    """
    if communication_id.startswith(MODEL_PREFIX):
        rest = communication_id[len(MODEL_PREFIX):]
        if MODEL_SEPARATOR in rest:
            model, comm_id = rest.split(MODEL_SEPARATOR, 1)
            return model, comm_id
    return default_model, communication_id


def ros_image_to_base64_jpeg(image_msg: Image) -> str:
    """Convert a sensor_msgs/Image to a base64-encoded JPEG string."""
    # Determine dtype / channels from encoding
    encoding = image_msg.encoding.lower()
    h, w = image_msg.height, image_msg.width

    if encoding in ("bgr8", "rgb8"):
        channels = 3
    elif encoding in ("bgra8", "rgba8"):
        channels = 4
    elif encoding in ("mono8", "8uc1"):
        channels = 1
    else:
        channels = 3  # best-effort fallback

    arr = np.frombuffer(image_msg.data, dtype=np.uint8).reshape((h, w, channels))

    if encoding.startswith("rgb"):
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    elif encoding.startswith("bgra"):
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    elif encoding.startswith("rgba"):
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

    _, jpeg = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(jpeg.tobytes()).decode("utf-8")


# ── Node ─────────────────────────────────────────────────────────────────────

class VisionAgentNode(Node):
    def __init__(
        self,
        camera_topic: str,
        input_topic: str,
        output_topic: str,
        default_model: str,
    ):
        super().__init__("vision_agent")
        self.default_model = default_model
        self.output_topic = output_topic

        # Latest camera frame (guarded by a lock)
        self._latest_image: Optional[Image] = None
        self._image_lock = threading.Lock()

        # LLM instances keyed by model name (lazy-init)
        self._llms: dict[str, ChatOllama] = {}

        # Subscribe to camera topic
        self.create_subscription(Image, camera_topic, self._on_image, 1)
        self.get_logger().info(f"Subscribed to camera: {camera_topic}")

        # Subscribe to HRI input topic
        if _HAS_RAI_INTERFACES:
            self.create_subscription(HRIMessage, input_topic, self._on_hri, 10)
            self._hri_pub = self.create_publisher(HRIMessage, output_topic, 10)
            self.get_logger().info(f"HRI: {input_topic} -> {output_topic} (rai_interfaces)")
        else:
            # Fallback: use std_msgs/String (basic compat)
            from std_msgs.msg import String
            self.create_subscription(String, input_topic, self._on_string, 10)
            self._string_pub = self.create_publisher(String, output_topic, 10)
            self.get_logger().warn(
                "rai_interfaces not found — falling back to std_msgs/String"
            )

    def _get_llm(self, model: str) -> ChatOllama:
        """Return (and cache) a ChatOllama instance for the given model."""
        if model not in self._llms:
            self.get_logger().info(f"Initializing ChatOllama for model: {model}")
            self._llms[model] = ChatOllama(
                model=model,
                base_url=OLLAMA_BASE_URL,
            )
        return self._llms[model]

    # ── Camera callback ──────────────────────────────────────────────────────

    def _on_image(self, msg: Image) -> None:
        with self._image_lock:
            self._latest_image = msg

    # ── HRI callback (rai_interfaces) ────────────────────────────────────────

    def _on_hri(self, msg: HRIMessage) -> None:
        text = msg.text.strip()
        if not text:
            return

        model, comm_id = parse_model_from_comm_id(msg.communication_id, self.default_model)
        self.get_logger().info(f"[{model}] Prompt: {text[:80]}...")

        response_text = self._invoke_vlm(model, text)

        reply = HRIMessage()
        reply.header.stamp = self.get_clock().now().to_msg()
        reply.text = response_text
        reply.communication_id = comm_id
        reply.seq_no = msg.seq_no
        reply.seq_end = True
        self._hri_pub.publish(reply)

    # ── String fallback callback ─────────────────────────────────────────────

    def _on_string(self, msg) -> None:
        text = msg.data.strip()
        if not text:
            return

        self.get_logger().info(f"[{self.default_model}] Prompt: {text[:80]}...")
        response_text = self._invoke_vlm(self.default_model, text)

        from std_msgs.msg import String
        reply = String()
        reply.data = response_text
        self._string_pub.publish(reply)

    # ── VLM invocation ───────────────────────────────────────────────────────

    def _invoke_vlm(self, model: str, text: str) -> str:
        """Send text + latest camera frame to the selected VLM and return the response."""
        # Grab the latest frame
        with self._image_lock:
            image_msg = self._latest_image

        # Build multimodal message
        content: list = []

        if image_msg is not None:
            try:
                b64_jpeg = ros_image_to_base64_jpeg(image_msg)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_jpeg}"},
                    }
                )
            except Exception as e:
                self.get_logger().warn(f"Image encode failed: {e}")
                content.append({"type": "text", "text": "[image unavailable]"})
        else:
            content.append({"type": "text", "text": "[no camera image received yet]"})

        content.append({"type": "text", "text": text})

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=content),
        ]

        try:
            llm = self._get_llm(model)
            response = llm.invoke(messages)
            return response.content
        except Exception as e:
            self.get_logger().error(f"LLM invocation failed: {e}")
            return f"[error] Model invocation failed: {e}"


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="RAI Vision Agent")
    parser.add_argument(
        "--camera-topic",
        type=str,
        default=DEFAULT_CAMERA_TOPIC,
        help=f"Camera image topic (default: {DEFAULT_CAMERA_TOPIC})",
    )
    parser.add_argument(
        "--input-topic",
        type=str,
        default=DEFAULT_INPUT_TOPIC,
        help=f"HRI input topic (default: {DEFAULT_INPUT_TOPIC})",
    )
    parser.add_argument(
        "--output-topic",
        type=str,
        default=DEFAULT_OUTPUT_TOPIC,
        help=f"HRI output topic (default: {DEFAULT_OUTPUT_TOPIC})",
    )
    parser.add_argument(
        "--default-model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Default Ollama model (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    rclpy.init()
    node = VisionAgentNode(
        camera_topic=args.camera_topic,
        input_topic=args.input_topic,
        output_topic=args.output_topic,
        default_model=args.default_model,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
