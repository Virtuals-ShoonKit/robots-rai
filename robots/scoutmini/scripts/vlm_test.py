#!/usr/bin/env python3
"""
vlm_test.py — One-shot or continuous VLM reasoning on ZED camera frames.

Subscribes to a ZED camera image topic and a prompt trigger topic.
When a prompt arrives on /vlm_prompt (std_msgs/String), grabs the latest
camera frame, sends prompt + image to Ollama Qwen VLM, and publishes
the reasoning output on /vlm_response (std_msgs/String).

Modes:
  Default  — runs initial prompt on first frame, then waits for panel triggers.
  --loop   — continuously describes the scene. After each response (and optional
             voice), waits --loop-delay seconds then runs again.

Optional voice output via KokoroTTS (from rai_s2s) — enable with --voice.

Usage:
    cd robots/scoutmini
    python scripts/vlm_test.py
    python scripts/vlm_test.py --voice --loop
    python scripts/vlm_test.py --voice --loop --loop-delay 1.0
"""

from __future__ import annotations

import argparse
import base64
import signal
import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np
import rclpy
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_CAMERA_TOPIC = "/zed/zed_node/rgb/image_rect_color"
DEFAULT_PROMPT_TOPIC = "/vlm_prompt"
DEFAULT_RESPONSE_TOPIC = "/vlm_response"
DEFAULT_STATUS_TOPIC = "/vlm_status"
DEFAULT_REASONING_TOPIC = "/vlm_reasoning"
DEFAULT_MODEL = "qwen2.5-vl"
OLLAMA_BASE_URL = "http://localhost:11434"

DEFAULT_INITIAL_PROMPT = (
    "Describe what you see in this image. "
    "Identify key objects, their spatial relationships, and anything notable."
)

LOOP_PROMPT = "Briefly describe the scene in 1-2 sentences. What do you see right now?"

SYSTEM_PROMPT = (
    "You are a vision analysis system mounted on a Scout Mini robot. "
    "Analyze the camera image and respond to the prompt. "
    "Be concise but thorough. Focus on factual observations."
)

LOOP_SYSTEM_PROMPT = (
    "You are a vision narration system on a mobile robot. "
    "Give a brief, natural spoken description of the scene in 1-2 short sentences. "
    "Focus on what's most prominent. No bullet points or formatting."
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def ros_image_to_base64_jpeg(image_msg: Image) -> str:
    encoding = image_msg.encoding.lower()
    h, w = image_msg.height, image_msg.width

    if encoding in ("bgr8", "rgb8"):
        channels = 3
    elif encoding in ("bgra8", "rgba8"):
        channels = 4
    elif encoding in ("mono8", "8uc1"):
        channels = 1
    else:
        channels = 3

    arr = np.frombuffer(image_msg.data, dtype=np.uint8).reshape((h, w, channels))

    if encoding.startswith("rgb"):
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    elif encoding.startswith("bgra"):
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    elif encoding.startswith("rgba"):
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

    _, jpeg = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(jpeg.tobytes()).decode("utf-8")


# ── TTS wrapper (optional) ──────────────────────────────────────────────────

class VoiceOutput:
    """Wraps KokoroTTS from rai_s2s for optional speech output."""

    def __init__(self, speaker_device: str = "default", voice: str = "af_sarah"):
        from rai_s2s.tts.models import KokoroTTS
        from rai_s2s.sound_device import SoundDeviceConfig

        self._tts = KokoroTTS(voice=voice)
        self._speaker_config = SoundDeviceConfig(
            stream=True,
            channels=1,
            device_name=speaker_device,
            block_size=1280,
            consumer_sampling_rate=24000,
            dtype="int16",
            device_number=None,
            is_input=False,
            is_output=True,
        )
        self._lock = threading.Lock()

    def speak(self, text: str) -> None:
        """Synthesize and play text. Blocks until finished."""
        import sounddevice as sd

        with self._lock:
            audio_seg = self._tts.get_speech(text)
            if audio_seg is not None and len(audio_seg) > 0:
                samples = np.frombuffer(audio_seg.raw_data, dtype=np.int16)
                sd.play(samples, samplerate=audio_seg.frame_rate)
                sd.wait()


# ── ROS 2 Node ──────────────────────────────────────────────────────────────

class VLMTestNode(Node):
    def __init__(
        self,
        camera_topic: str,
        prompt_topic: str,
        response_topic: str,
        status_topic: str,
        model: str,
        initial_prompt: str,
        voice: Optional[VoiceOutput],
        loop: bool = False,
        loop_delay: float = 1.0,
    ):
        super().__init__("vlm_test")
        self._model_name = model
        self._initial_prompt = initial_prompt
        self._voice = voice
        self._initial_prompt_sent = False
        self._loop = loop
        self._loop_delay = loop_delay
        self._shutdown = False

        self._latest_image: Optional[Image] = None
        self._image_lock = threading.Lock()
        self._inference_lock = threading.Lock()

        self._system_prompt = LOOP_SYSTEM_PROMPT if loop else SYSTEM_PROMPT
        self._llm = ChatOllama(model=model, base_url=OLLAMA_BASE_URL, reasoning=True)

        self.create_subscription(Image, camera_topic, self._on_image, 1)
        self.create_subscription(String, prompt_topic, self._on_prompt, 10)
        self._response_pub = self.create_publisher(String, response_topic, 10)
        self._status_pub = self.create_publisher(String, status_topic, 10)
        self._reasoning_pub = self.create_publisher(String, DEFAULT_REASONING_TOPIC, 10)

        mode = "loop" if loop else "on-demand"
        self._publish_status("waiting_for_camera")
        self.get_logger().info(f"VLM Test Node started ({mode})")
        self.get_logger().info(f"  Camera: {camera_topic}")
        self.get_logger().info(f"  Model:  {model} @ {OLLAMA_BASE_URL}")
        self.get_logger().info(f"  Prompt trigger: {prompt_topic}")
        self.get_logger().info(f"  Response: {response_topic}")
        self.get_logger().info(f"  Voice: {'enabled' if voice else 'disabled'}")
        if loop:
            self.get_logger().info(f"  Loop delay: {loop_delay}s after voice/response")

    def _publish_status(self, status: str) -> None:
        msg = String()
        msg.data = status
        self._status_pub.publish(msg)

    def _on_image(self, msg: Image) -> None:
        with self._image_lock:
            first_frame = self._latest_image is None
            self._latest_image = msg

        if first_frame:
            self.get_logger().info("First camera frame received")
            self._publish_status("ready")
            if self._loop and not self._initial_prompt_sent:
                self._initial_prompt_sent = True
                threading.Thread(target=self._loop_runner, daemon=True).start()
            elif self._initial_prompt and not self._initial_prompt_sent:
                self._initial_prompt_sent = True
                threading.Thread(
                    target=self._run_inference,
                    args=(self._initial_prompt,),
                    daemon=True,
                ).start()

    def _loop_runner(self) -> None:
        prompt = self._initial_prompt or LOOP_PROMPT
        self.get_logger().info(f"Continuous loop started (prompt: {prompt[:50]}...)")
        while not self._shutdown:
            self._run_inference(prompt)
            if self._shutdown:
                break
            time.sleep(self._loop_delay)

    def _on_prompt(self, msg: String) -> None:
        text = msg.data.strip()
        if not text:
            return
        self.get_logger().info(f"Prompt received: {text[:80]}...")
        threading.Thread(
            target=self._run_inference, args=(text,), daemon=True
        ).start()

    def _run_inference(self, prompt: str) -> None:
        acquired = self._inference_lock.acquire(blocking=self._loop)
        if not acquired:
            self.get_logger().warn("Inference already in progress, skipping")
            return

        try:
            self._publish_status("processing")
            self.get_logger().info(f"Running inference: {prompt[:60]}...")

            with self._image_lock:
                image_msg = self._latest_image

            content: list = []
            if image_msg is not None:
                try:
                    b64_jpeg = ros_image_to_base64_jpeg(image_msg)
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_jpeg}"},
                    })
                except Exception as e:
                    self.get_logger().warn(f"Image encode failed: {e}")
                    content.append({"type": "text", "text": "[image unavailable]"})
            else:
                content.append({"type": "text", "text": "[no camera frame yet]"})

            content.append({"type": "text", "text": prompt})

            messages = [
                SystemMessage(content=self._system_prompt),
                HumanMessage(content=content),
            ]

            response = self._llm.invoke(messages)
            reasoning = response.additional_kwargs.get("reasoning_content", "")
            if reasoning:
                self.get_logger().info(f"Reasoning: {reasoning[:300]}...")
                r_msg = String()
                r_msg.data = reasoning
                self._reasoning_pub.publish(r_msg)
            response_text = response.content
            self.get_logger().info(f"Response ({len(response_text)} chars): {response_text[:100]}...")

            reply = String()
            reply.data = response_text
            self._response_pub.publish(reply)
            self._publish_status("ready")

            if self._voice:
                self._publish_status("speaking")
                try:
                    self._voice.speak(response_text)
                except Exception as e:
                    self.get_logger().warn(f"TTS failed: {e}")

        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")
            err = String()
            err.data = f"[error] {e}"
            self._response_pub.publish(err)
            self._publish_status("error")
        finally:
            self._inference_lock.release()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="VLM Test — ZED + Ollama Qwen")
    parser.add_argument(
        "--camera-topic", default=DEFAULT_CAMERA_TOPIC,
        help=f"Camera image topic (default: {DEFAULT_CAMERA_TOPIC})",
    )
    parser.add_argument(
        "--prompt-topic", default=DEFAULT_PROMPT_TOPIC,
        help=f"Prompt trigger topic (default: {DEFAULT_PROMPT_TOPIC})",
    )
    parser.add_argument(
        "--response-topic", default=DEFAULT_RESPONSE_TOPIC,
        help=f"Response output topic (default: {DEFAULT_RESPONSE_TOPIC})",
    )
    parser.add_argument(
        "--status-topic", default=DEFAULT_STATUS_TOPIC,
        help=f"Status topic (default: {DEFAULT_STATUS_TOPIC})",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Ollama model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--prompt", default=DEFAULT_INITIAL_PROMPT,
        help="Initial prompt to run on first frame",
    )
    parser.add_argument(
        "--no-initial", action="store_true",
        help="Skip the initial prompt (wait for panel trigger only)",
    )
    parser.add_argument(
        "--loop", action="store_true",
        help="Continuous mode: describe scene, speak, wait, repeat",
    )
    parser.add_argument(
        "--loop-delay", type=float, default=1.0,
        help="Seconds to wait after voice/response before next loop (default: 1.0)",
    )
    parser.add_argument(
        "--voice", action="store_true",
        help="Enable voice output (KokoroTTS)",
    )
    parser.add_argument(
        "--speaker-device", default="default",
        help="Audio output device name (default: default)",
    )
    parser.add_argument(
        "--voice-name", default="af_sarah",
        help="KokoroTTS voice (default: af_sarah)",
    )
    args = parser.parse_args()

    voice: Optional[VoiceOutput] = None
    if args.voice:
        try:
            voice = VoiceOutput(
                speaker_device=args.speaker_device, voice=args.voice_name
            )
            print(f"[vlm_test] Voice output enabled (voice={args.voice_name})")
        except Exception as e:
            print(f"[vlm_test] Voice init failed, continuing without: {e}")

    loop_prompt = args.prompt if args.prompt != DEFAULT_INITIAL_PROMPT else LOOP_PROMPT

    rclpy.init()
    node = VLMTestNode(
        camera_topic=args.camera_topic,
        prompt_topic=args.prompt_topic,
        response_topic=args.response_topic,
        status_topic=args.status_topic,
        model=args.model,
        initial_prompt=loop_prompt if args.loop else ("" if args.no_initial else args.prompt),
        voice=voice,
        loop=args.loop,
        loop_delay=args.loop_delay,
    )

    def shutdown(signum, frame):
        node._shutdown = True
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
