#!/usr/bin/env python3
"""
Dry-test: FastAPI task server + ZED Mini live camera → Ollama VLM → Foxglove.

Starts:
  • http://localhost:8090  — FastAPI task REST API  (same as real robot)

Camera modes:
  default       — pyzed SDK opens the ZED directly (no ROS2 needed).
                  A rosbridge-protocol stub on ws://localhost:9090 streams
                  agent text topics to Foxglove.
  --launch-ros2 — auto-launch zed_wrapper + foxglove_bridge as subprocesses.
                  Camera images and agent topics are real ROS2 topics;
                  foxglove_bridge bridges everything to ws://localhost:8765.
                  All in one terminal; Ctrl-C kills everything.
  --ros2        — same, but you start the ROS2 nodes yourself.
  --no-camera   — text-only fallback (no camera at all).

In Foxglove Studio:
  --ros2 mode:  Open connection → Foxglove WebSocket → ws://localhost:8765
  default mode: Open connection → Rosbridge → ws://localhost:9090

Usage:
    python robots/zedmini/dry_test_api.py
    python robots/zedmini/dry_test_api.py --no-camera

    # ROS2 + foxglove_bridge — single terminal:
    python robots/zedmini/dry_test_api.py --launch-ros2
    python robots/zedmini/dry_test_api.py --launch-ros2 --camera-model zedm

    # ROS2 + foxglove_bridge — separate terminals:
    #   ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zedm
    #   ros2 run foxglove_bridge foxglove_bridge
    python robots/zedmini/dry_test_api.py --ros2
    python robots/zedmini/dry_test_api.py --ros2 --image-topic /zedm/zed_node/rgb/color/rect/image
"""

from __future__ import annotations

import argparse
import atexit
import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
import time
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]  # Python < 3.11
import types
from pathlib import Path

import base64

import cv2
import numpy as np
import requests

# ── sys.path: make robots/common importable ────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT / "robots"))

# ── Load zedmini config.toml for Ollama settings ──────────────────────────────
_config_path = _THIS_DIR / "config.toml"
_cfg: dict = {}
if _config_path.exists():
    with open(_config_path, "rb") as _f:
        _cfg = tomllib.load(_f)

_OLLAMA_BASE_URL: str = _cfg.get("ollama", {}).get("base_url", "http://localhost:11434")
_DEFAULT_MODEL: str = _cfg.get("ollama", {}).get("simple_model", "qwen3:8b")

SYSTEM_PROMPT_VISION = (
    "You are a vision-capable robot assistant (ZED Mini camera). "
    "The user gives you a task. Analyse the attached camera image and reason about "
    "what you see and how you would act. Be concise and factual."
)

SYSTEM_PROMPT_TEXT = (
    "You are a robot assistant. The user gives you a task. "
    "Reason through it and reply with what you would do. Be concise."
)

THINK_START = "<think>"
THINK_END = "</think>"


# ── ZED Mini camera capture (pyzed) ───────────────────────────────────────────

class ZEDCapture:
    """Keeps the ZED open and provides grab-on-demand left-eye JPEG frames."""

    def __init__(self, resolution="HD720", fps: int = 15):
        import pyzed.sl as sl
        self._sl = sl
        self._cam = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = getattr(sl.RESOLUTION, resolution)
        init.camera_fps = fps
        init.depth_mode = sl.DEPTH_MODE.NONE   # no depth needed for VLM

        status = self._cam.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED open failed: {status}")

        self._mat = sl.Mat()
        self._lock = threading.Lock()
        self._latest_b64: str | None = None
        self._running = True

        # Grab frames in background so the latest is always fresh
        threading.Thread(target=self._grab_loop, daemon=True).start()
        print(f"[zed]  opened ({resolution} @ {fps} fps)")

    def _grab_loop(self) -> None:
        sl = self._sl
        rt = sl.RuntimeParameters()
        while self._running:
            if self._cam.grab(rt) == sl.ERROR_CODE.SUCCESS:
                self._cam.retrieve_image(self._mat, sl.VIEW.LEFT)
                arr = self._mat.get_data()           # BGRA uint8
                bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
                _, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                b64 = base64.b64encode(jpeg.tobytes()).decode()
                with self._lock:
                    self._latest_b64 = b64
            else:
                time.sleep(0.05)

    def get_frame_b64(self) -> str | None:
        with self._lock:
            return self._latest_b64

    def close(self) -> None:
        self._running = False
        self._cam.close()


# ── ROS2 native node (rclpy) — camera subscriber + agent topic publishers ────

_DEFAULT_IMAGE_TOPIC = "/zedm/zed_node/rgb/color/rect/image"


class ROS2Node:
    """Minimal rclpy node for --ros2 mode.

    - Subscribes to a ZED camera image topic (raw Image or CompressedImage,
      auto-detected from topic name) and keeps the latest frame as base64 JPEG
      for the Ollama VLM.
    - Publishes agent topics as std_msgs/String so foxglove_bridge exposes them
      to Foxglove at ws://localhost:8765.
    """

    def __init__(self, image_topic: str):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from std_msgs.msg import String
        from rai_interfaces.msg import HRIMessage

        if not rclpy.ok():
            rclpy.init()

        self._String = String
        self._HRIMessage = HRIMessage
        self._node = rclpy.create_node("dry_test_agent")
        self._pubs: dict = {}
        self._lock = threading.Lock()
        self._latest_b64: str | None = None
        self._frame_count = 0

        # Pre-create publishers so DDS discovery completes before the first
        # task arrives (rosbridge won't see messages from a brand-new publisher).
        for t in ("/agent/status", "/agent/reasoning", "/agent/events"):
            self._pubs[t] = self._node.create_publisher(String, t, 10)
        self._pubs["/to_human"] = self._node.create_publisher(HRIMessage, "/to_human", 10)
        self._pubs["/from_human"] = self._node.create_publisher(HRIMessage, "/from_human", 10)

        if image_topic:
            if image_topic.endswith("/compressed"):
                from sensor_msgs.msg import CompressedImage
                self._node.create_subscription(
                    CompressedImage, image_topic, self._on_compressed, 10)
            else:
                from sensor_msgs.msg import Image
                self._node.create_subscription(
                    Image, image_topic, self._on_raw_image, 10)
            self._node.get_logger().info(f"subscribed to {image_topic}")

        executor = SingleThreadedExecutor()
        executor.add_node(self._node)
        threading.Thread(target=executor.spin, daemon=True, name="rclpy-spin").start()

    # ── camera callbacks ──────────────────────────────────────────────────────

    def _on_compressed(self, msg) -> None:
        fmt = getattr(msg, "format", "")
        data = bytes(msg.data)
        if "jpeg" in fmt.lower() or "jpg" in fmt.lower():
            b64 = base64.b64encode(data).decode()
        else:
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return
            _, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64 = base64.b64encode(jpeg.tobytes()).decode()
        with self._lock:
            self._latest_b64 = b64
            self._frame_count += 1
        if self._frame_count == 1:
            print(f"[ros2] first compressed frame ({fmt}, ~{len(b64) // 1024} KB)")

    def _on_raw_image(self, msg) -> None:
        h, w = msg.height, msg.width
        enc = msg.encoding
        raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        if enc in ("bgra8", "8UC4"):
            bgr = cv2.cvtColor(raw.reshape((h, w, 4)), cv2.COLOR_BGRA2BGR)
        elif enc in ("bgr8", "8UC3"):
            bgr = raw.reshape((h, w, 3))
        elif enc == "rgb8":
            bgr = cv2.cvtColor(raw.reshape((h, w, 3)), cv2.COLOR_RGB2BGR)
        elif enc == "mono8":
            bgr = raw.reshape((h, w, 1))
        else:
            bgr = raw.reshape((h, w, -1))
        _, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(jpeg.tobytes()).decode()
        with self._lock:
            self._latest_b64 = b64
            self._frame_count += 1
        if self._frame_count == 1:
            print(f"[ros2] first raw frame ({enc} {w}x{h}, ~{len(b64) // 1024} KB)")

    def get_frame_b64(self) -> str | None:
        with self._lock:
            return self._latest_b64

    # ── agent topic publisher ─────────────────────────────────────────────────

    def publish(self, topic: str, msg_data: dict) -> None:
        """Publish agent data on ROS 2 (foxglove_bridge + rosbridge pick it up)."""
        pub = self._pubs.get(topic)
        if pub is None:
            pub = self._node.create_publisher(self._String, topic, 10)
            self._pubs[topic] = pub

        if topic in ("/to_human", "/from_human"):
            msg = self._HRIMessage()
            msg.text = msg_data.get("text", "")
            msg.communication_id = msg_data.get("communication_id", "")
            msg.seq_no = msg_data.get("seq_no", 0)
            msg.seq_end = msg_data.get("seq_end", False)
        else:
            msg = self._String()
            if "data" in msg_data and isinstance(msg_data["data"], str):
                msg.data = msg_data["data"]
            else:
                msg.data = json.dumps(msg_data)
        pub.publish(msg)

    def close(self) -> None:
        self._node.destroy_node()


# ── Stub ROS2 imports so task_server.py loads without a ROS2 install ──────────

class _FakeHRIMessage:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeHRIConnector:
    """Calls the real Ollama VLM with a live ZED Mini frame per task."""

    def __init__(self, task_manager, ros_publish_fn, model: str, zed: "ZEDCapture | None"):
        self._tm = task_manager
        self._pub = ros_publish_fn      # callable(topic, msg_dict)
        self._model = model
        self._zed = zed

    def send_message(self, msg, topic: str):
        task_text = getattr(msg, "text", "")
        comm_id = getattr(msg, "communication_id", "task:?")
        task_id_str = comm_id.split(":")[-1]
        task_id = int(task_id_str) if task_id_str.isdigit() else None
        print(f"[task] → {topic}  id={task_id}  '{task_text}'")

        now_ms = int(time.time() * 1000)
        self._pub("/from_human", {
            "header": {"stamp": {"sec": now_ms // 1000, "nanosec": (now_ms % 1000) * 1_000_000}, "frame_id": ""},
            "text": task_text,
            "images": [],
            "audios": [],
            "communication_id": comm_id,
            "seq_no": 1,
            "seq_end": True,
        })

        threading.Thread(
            target=self._run_ollama,
            args=(task_text, task_id),
            daemon=True,
        ).start()

    # ── Mock tools ────────────────────────────────────────────────────────────

    _TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "navigate_to_pose",
                "description": (
                    "Navigate the robot to a goal position on the map. "
                    "Use this when the task involves moving to a location."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number", "description": "Goal X in metres (map frame)"},
                        "y": {"type": "number", "description": "Goal Y in metres (map frame)"},
                        "description": {"type": "string", "description": "Human-readable destination label"},
                    },
                    "required": ["x", "y"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_current_pose",
                "description": "Return the robot's current position and orientation in the map frame.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]

    def _execute_tool(self, name: str, args: dict) -> tuple[str, str | None]:
        """Run a mock tool, publish Foxglove events.

        Returns (json_result_str, frame_b64_or_None).
        frame_b64 is a fresh ZED snapshot for tools that move the robot.
        """
        self._pub("/agent/events", {"data": json.dumps({"event": "start", "tool": name, "args": args})})
        self._pub_str("/agent/status", f"executing:{name}")
        frame_b64: str | None = None

        if name == "navigate_to_pose":
            x, y = args.get("x", 0.0), args.get("y", 0.0)
            dest = args.get("description", f"({x}, {y})")
            print(f"[nav]  navigating to {dest}  x={x} y={y}")
            time.sleep(2.0)
            result = {"success": True, "message": f"Arrived at {dest}", "x": x, "y": y}
            # grab a fresh frame to show Ollama what the camera sees on arrival
            if self._zed:
                frame_b64 = self._zed.get_frame_b64()
                if frame_b64:
                    print(f"[nav]  camera snapshot attached ({len(frame_b64)//1024} KB)")

        elif name == "get_current_pose":
            result = {"x": 0.0, "y": 0.0, "yaw_deg": 0.0, "frame": "map"}
            if self._zed:
                frame_b64 = self._zed.get_frame_b64()

        else:
            result = {"error": f"Unknown tool: {name}"}

        elapsed = 2.0 if name == "navigate_to_pose" else 0.05
        self._pub("/agent/events", {"data": json.dumps({"event": "end", "result": result, "elapsed_s": elapsed})})
        self._pub_str("/agent/status", "thinking")
        print(f"[tool] {name} → {result}")
        return json.dumps(result), frame_b64

    def _pub_str(self, topic: str, data: str) -> None:
        self._pub(topic, {"data": data})

    def _run_ollama(self, task_text: str, task_id: int | None) -> None:
        """Agentic loop: grab ZED frame → Ollama → tool calls → final reply."""
        self._pub_str("/agent/status", "thinking")
        self._pub_str("/agent/reasoning", "")

        # ── grab ZED frame ────────────────────────────────────────────────────
        frame_b64 = self._zed.get_frame_b64() if self._zed else None
        if frame_b64:
            user_msg: dict = {"role": "user", "content": task_text, "images": [frame_b64]}
            system_prompt = SYSTEM_PROMPT_VISION
            print(f"[zed]  frame attached ({len(frame_b64)//1024} KB)")
        else:
            user_msg = {"role": "user", "content": task_text}
            system_prompt = SYSTEM_PROMPT_TEXT
            if self._zed:
                print("[zed]  no frame yet — sending text only")

        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            user_msg,
        ]

        reply_text = "(no response)"

        try:
            # ── agentic loop (max 6 turns to avoid runaway) ───────────────────
            for _turn in range(6):
                content_buf = ""
                thinking_buf = ""
                tool_calls: list[dict] = []
                in_think = False

                with requests.post(
                    f"{_OLLAMA_BASE_URL}/api/chat",
                    json={"model": self._model, "messages": messages,
                          "tools": self._TOOLS, "stream": True, "think": True},
                    stream=True,
                    timeout=120,
                ) as resp:
                    resp.raise_for_status()
                    for raw_line in resp.iter_lines():
                        if not raw_line:
                            continue
                        try:
                            chunk = json.loads(raw_line)
                        except Exception:
                            continue

                        msg = chunk.get("message", {})

                        # native thinking field
                        t = msg.get("thinking", "")
                        if t:
                            self._pub_str("/agent/reasoning", t)
                            thinking_buf += t

                        # content tokens (with inline <think> fallback)
                        c = msg.get("content", "")
                        if c:
                            s = c
                            while s:
                                if not in_think:
                                    i = s.find(THINK_START)
                                    if i == -1:
                                        content_buf += s
                                        break
                                    content_buf += s[:i]
                                    s = s[i + len(THINK_START):]
                                    in_think = True
                                else:
                                    j = s.find(THINK_END)
                                    if j == -1:
                                        self._pub_str("/agent/reasoning", s)
                                        break
                                    self._pub_str("/agent/reasoning", s[:j])
                                    s = s[j + len(THINK_END):]
                                    in_think = False

                        if msg.get("tool_calls"):
                            tool_calls = msg["tool_calls"]

                # ── decide next step ─────────────────────────────────────────
                if tool_calls:
                    # add assistant turn to history
                    messages.append({"role": "assistant", "content": content_buf, "tool_calls": tool_calls})
                    # execute each tool and add results
                    for tc in tool_calls:
                        fn = tc.get("function", {})
                        name = fn.get("name", "")
                        args = fn.get("arguments", {})
                        result_str, new_frame_b64 = self._execute_tool(name, args)
                        tool_msg: dict = {"role": "tool", "content": result_str}
                        if new_frame_b64:
                            tool_msg["images"] = [new_frame_b64]
                        messages.append(tool_msg)
                    # continue loop for the model to respond to tool results
                else:
                    # final text reply
                    reply_text = content_buf.strip() or "(no response)"
                    break

        except Exception as e:
            reply_text = f"[ollama error] {e}"
            print(f"[ollama] error: {e}")

        print(f"[ollama] response: {reply_text[:120]}...")
        self._pub_str("/agent/status", "idle")

        now_ms = int(time.time() * 1000)
        self._pub("/to_human", {
            "header": {"stamp": {"sec": now_ms // 1000, "nanosec": (now_ms % 1000) * 1_000_000}, "frame_id": ""},
            "text": reply_text,
            "images": [],
            "audios": [],
            "communication_id": f"task:{task_id}",
            "seq_no": 1,
            "seq_end": True,
        })

        if task_id is not None:
            self._tm.complete_task(task_id, reply_text)
        print(f"[task] completed id={task_id}")


for _name, _mod in [
    ("rai", types.ModuleType("rai")),
    ("rai.communication", types.ModuleType("rai.communication")),
    ("rai.communication.ros2", types.ModuleType("rai.communication.ros2")),
    ("rai.communication.ros2.messages",
     type(sys)("rai.communication.ros2.messages")),
    ("rai.communication.ros2.connectors",
     types.ModuleType("rai.communication.ros2.connectors")),
    ("rai.communication.ros2.connectors.hri_connector",
     types.ModuleType("rai.communication.ros2.connectors.hri_connector")),
]:
    sys.modules[_name] = _mod
_msg_mod = sys.modules["rai.communication.ros2.messages"]
_msg_mod.ROS2HRIMessage = _FakeHRIMessage
_hri_mod = sys.modules["rai.communication.ros2.connectors.hri_connector"]
_hri_mod.ROS2HRIConnector = _FakeHRIConnector

from common.task_server import TaskManager, _build_app  # noqa: E402

import uvicorn  # noqa: E402
import websockets  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

# ── Rosbridge-protocol WebSocket server ───────────────────────────────────────
# Minimal subset: subscribe / unsubscribe / advertise / unadvertise
# Broadcasts {"op":"publish","topic":t,"msg":m} to all subscribers of t.

_ws_loop: asyncio.AbstractEventLoop | None = None
_ws_queue: asyncio.Queue | None = None
_ws_clients: dict[str, set] = {}   # topic → set of open WebSocket connections


async def _ws_handler(websocket):
    """Handle one Foxglove/rosbridge client connection."""
    subscriptions: set[str] = set()
    try:
        async for raw in websocket:
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            op = msg.get("op", "")
            if op == "subscribe":
                topic = msg.get("topic", "")
                subscriptions.add(topic)
                _ws_clients.setdefault(topic, set()).add(websocket)
                print(f"[ws]  subscribe {topic}  (total={len(_ws_clients.get(topic, set()))})")
            elif op in ("unsubscribe",):
                topic = msg.get("topic", "")
                subscriptions.discard(topic)
                _ws_clients.get(topic, set()).discard(websocket)
            elif op in ("advertise", "unadvertise", "publish"):
                pass  # silently accept
    except Exception:
        pass
    finally:
        for topic in subscriptions:
            _ws_clients.get(topic, set()).discard(websocket)
        print(f"[ws]  client disconnected")


async def _drain_queue():
    """Forward items from thread-safe queue → async broadcast."""
    while True:
        topic, msg_data = await _ws_queue.get()
        payload = json.dumps({"op": "publish", "topic": topic, "msg": msg_data})
        clients = list(_ws_clients.get(topic, set()))
        if clients:
            await asyncio.gather(*(c.send(payload) for c in clients), return_exceptions=True)


async def _run_ws_server(host: str, port: int):
    global _ws_queue
    _ws_queue = asyncio.Queue()
    asyncio.create_task(_drain_queue())
    print(f"Rosbridge stub      → ws://{host}:{port}")
    async with websockets.serve(_ws_handler, host, port):
        await asyncio.Future()  # run forever


def _start_ws_thread(host: str, port: int):
    global _ws_loop
    _ws_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_ws_loop)
    _ws_loop.run_until_complete(_run_ws_server(host, port))


def ros_publish(topic: str, msg_data: dict):
    """Thread-safe: enqueue a rosbridge publish from any thread."""
    if _ws_loop and _ws_queue:
        _ws_loop.call_soon_threadsafe(_ws_queue.put_nowait, (topic, msg_data))


# ── ROS2 subprocess launcher ──────────────────────────────────────────────────

_child_procs: list[subprocess.Popen] = []


def _kill_children() -> None:
    for proc in _child_procs:
        if proc.poll() is None:
            print(f"[launch] killing pid {proc.pid}")
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    for proc in _child_procs:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)


def _launch_ros2_nodes(camera_model: str = "zedm") -> None:
    """Start zed_wrapper and foxglove_bridge as child processes."""
    atexit.register(_kill_children)

    # Ensure the local colcon workspace overlay is sourced so that
    # foxglove_bridge (built from source) and zed_wrapper are findable.
    setup_bash = _REPO_ROOT / "install" / "setup.bash"
    shell_prefix = f"source {setup_bash} 2>/dev/null; " if setup_bash.exists() else ""

    cmds = [
        (
            "zed_wrapper",
            f"{shell_prefix}ros2 launch zed_wrapper zed_camera.launch.py"
            f" camera_model:={camera_model}",
        ),
        (
            "foxglove_bridge",
            f"{shell_prefix}ros2 run foxglove_bridge foxglove_bridge",
        ),
        (
            "rosbridge_server",
            f"{shell_prefix}ros2 launch rosbridge_server rosbridge_websocket_launch.xml",
        ),
    ]
    for label, cmd in cmds:
        print(f"[launch] starting {label}")
        proc = subprocess.Popen(
            cmd,
            shell=True,
            executable="/bin/bash",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,
        )
        _child_procs.append(proc)

    print("[launch] waiting for nodes to initialise …")
    time.sleep(5)

    for proc in _child_procs:
        ret = proc.poll()
        if ret is not None:
            stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
            raise RuntimeError(
                f"[launch] process exited early (code {ret}):\n{stderr[-500:]}"
            )
    print("[launch] all ROS2 nodes running")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Dry-test the zedmini FastAPI + Foxglove bridge")
    parser.add_argument("--api-port", type=int, default=8090)
    parser.add_argument("--ws-port", type=int, default=9090,
                        help="Rosbridge stub port (non-ROS2 mode only)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--model", default=_DEFAULT_MODEL,
        help=f"Ollama model (default: {_DEFAULT_MODEL} from config.toml)",
    )

    cam_group = parser.add_mutually_exclusive_group()
    cam_group.add_argument(
        "--no-camera", action="store_true",
        help="Disable ZED capture and run text-only",
    )
    cam_group.add_argument(
        "--ros2", action="store_true",
        help="Use rclpy to subscribe to ZED camera and publish agent topics. "
             "Requires foxglove_bridge running (ws://localhost:8765).",
    )
    cam_group.add_argument(
        "--launch-ros2", action="store_true",
        help="Launch zed_wrapper + foxglove_bridge as subprocesses, then behave like --ros2. "
             "Requires a sourced ROS2 workspace with zed_wrapper & foxglove_bridge installed.",
    )
    parser.add_argument(
        "--camera-model", default="zedm",
        help="ZED camera model for ros2 launch (default: zedm). Only used with --launch-ros2.",
    )
    parser.add_argument(
        "--image-topic",
        default=_DEFAULT_IMAGE_TOPIC,
        help=f"Image topic to subscribe to (default: {_DEFAULT_IMAGE_TOPIC}). "
             "Append /compressed for CompressedImage. "
             "Check with: ros2 topic list | grep image",
    )
    args = parser.parse_args()

    # --launch-ros2 implies --ros2
    if args.launch_ros2:
        args.ros2 = True
        _launch_ros2_nodes(camera_model=args.camera_model)

    # ── camera source + publish function ──────────────────────────────────────
    zed: ZEDCapture | ROS2Node | None = None
    publish_fn = ros_publish  # default: rosbridge stub

    if args.ros2:
        ros2_node = ROS2Node(args.image_topic)
        zed = ros2_node
        publish_fn = ros2_node.publish
    elif not args.no_camera:
        try:
            zed = ZEDCapture()
        except Exception as e:
            print(f"[zed]  could not open camera: {e} — running text-only")

    # In non-ROS2 mode, start the rosbridge stub for Foxglove agent panel
    if not args.ros2:
        ws_thread = threading.Thread(
            target=_start_ws_thread, args=(args.host, args.ws_port), daemon=True,
        )
        ws_thread.start()

    task_manager = TaskManager()
    hri_connector = _FakeHRIConnector(task_manager, publish_fn, model=args.model, zed=zed)

    app = _build_app(task_manager, hri_connector)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if args.ros2:
        cam_label = f"ros2 → {args.image_topic}"
        foxglove_label = "foxglove_bridge → ws://localhost:8765"
    else:
        cam_label = "pyzed SDK" if zed else "disabled (text-only)"
        foxglove_label = f"rosbridge stub → ws://{args.host}:{args.ws_port}"

    print(f"\nOllama model        → {args.model}  ({_OLLAMA_BASE_URL})")
    print(f"ZED Mini camera     → {cam_label}")
    print(f"Dry-test API        → http://{args.host}:{args.api_port}")
    print(f"  GET  /status")
    print(f"  POST /execute_task   body: {{\"task\": \"...\"}}")
    print(f"  GET  /tasks/{{id}}/status")
    print(f"\nFoxglove            → {foxglove_label}")
    if args.ros2:
        print(f"  Open connection → Foxglove WebSocket → ws://localhost:8765")
        print(f"  Agent panel rosbridge → ws://localhost:9090")
    else:
        print(f"  Open connection → Rosbridge → ws://{args.host}:{args.ws_port}")
    print()

    def _shutdown(sig, _frame):
        print(f"\n[main] caught signal {sig}, shutting down …")
        _kill_children()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    uvicorn.run(app, host=args.host, port=args.api_port, log_level="warning")


if __name__ == "__main__":
    main()
