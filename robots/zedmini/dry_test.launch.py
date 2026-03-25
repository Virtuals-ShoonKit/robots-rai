#!/usr/bin/env python3
# Copyright (C) 2025 Eastworld Robotics
#
# Local dry-test launch: ZED Mini + Foxglove bridge + Kokoro TTS voice pipeline
#
# Usage:
#   cd robots/zedmini
#   ros2 launch dry_test.launch.py
#
# Optional args:
#   ros2 launch dry_test.launch.py voice:=false   # skip voice pipeline
#   ros2 launch dry_test.launch.py svo_path:=/path/to/file.svo2  # replay SVO

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import (
    AnyLaunchDescriptionSource,
    PythonLaunchDescriptionSource,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# ── Repo root (two levels up from this file) ───────────────────────────────
_THIS_DIR = Path(__file__).parent.resolve()
_REPO_ROOT = _THIS_DIR.parent.parent

# ── ZED wrapper shared directory ────────────────────────────────────────────
_ZED_WRAPPER_SHARE = get_package_share_directory("zed_wrapper")

def _launch_zed(context, **_):
    svo_path = LaunchConfiguration("svo_path").perform(context)

    zed_params = {
        "camera_model": "zedm",
        "publish_urdf": "true",
        "publish_tf": "true",
        "publish_map_tf": "false",
        # Image transport: publish compressed alongside raw
        "general.pub_resolution": "MEDIUM",
        # Reduce bandwidth — sufficient for Foxglove preview
        "video.quality": "70",
    }

    if svo_path:
        zed_params["svo_file"] = svo_path
        zed_params["svo_loop"] = "true"

    zed_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(_ZED_WRAPPER_SHARE, "launch", "zed_camera.launch.py")
        ),
        launch_arguments={
            "camera_model": "zedm",
            "publish_urdf": "true",
            "publish_tf": "true",
            "publish_map_tf": "false",
            **({} if not svo_path else {"svo_path": svo_path}),
        }.items(),
    )
    return [zed_launch]


def _launch_ollama(context, **_):
    """Start ollama serve if not already running."""
    ollama_enabled = LaunchConfiguration("ollama").perform(context)
    if ollama_enabled.lower() not in ("true", "1", "yes"):
        return []
    return [
        ExecuteProcess(
            cmd=["ollama", "serve"],
            name="ollama",
            output="screen",
        )
    ]


def _launch_vlm(context, **_):
    """Launch vlm_test.py — subscribes to ZED camera, publishes to /vlm_response."""
    if LaunchConfiguration("vlm").perform(context).lower() not in ("true", "1", "yes"):
        return []

    vlm_script = _REPO_ROOT / "robots" / "scoutmini" / "scripts" / "vlm_test.py"
    venv_python = _REPO_ROOT / ".venv" / "bin" / "python"
    python_exe = str(venv_python) if venv_python.exists() else "python3"

    model = LaunchConfiguration("vlm_model").perform(context)
    cmd = [
        python_exe, str(vlm_script),
        "--model", model,
        "--camera-topic", "/zed/zed_node/rgb/color/rect/image",
        "--voice",
    ]
    if LaunchConfiguration("vlm_loop").perform(context).lower() in ("true", "1", "yes"):
        cmd += ["--loop"]

    return [
        ExecuteProcess(
            cmd=cmd,
            cwd=str(_THIS_DIR),
            name="vlm_test",
            output="screen",
        )
    ]


def _launch_voice(context, **_):
    """Launch Kokoro TTS + LocalWhisper voice pipeline as a subprocess."""
    voice_enabled = LaunchConfiguration("voice").perform(context)
    if voice_enabled.lower() not in ("true", "1", "yes"):
        return []

    voice_script = _THIS_DIR / "scripts" / "voice.py"
    if not voice_script.exists():
        # Fall back to scoutmini voice script — same pipeline
        voice_script = _REPO_ROOT / "robots" / "scoutmini" / "scripts" / "voice.py"

    venv_python = _REPO_ROOT / ".venv" / "bin" / "python"
    python_exe = str(venv_python) if venv_python.exists() else "python3"

    return [
        ExecuteProcess(
            cmd=[python_exe, str(voice_script)],
            cwd=str(_THIS_DIR),  # config.toml must be here
            name="voice_pipeline",
            output="screen",
        )
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            # ── Args ─────────────────────────────────────────────────────────
            DeclareLaunchArgument(
                "svo_path",
                default_value="",
                description="Path to an SVO2 file for playback (empty = live camera)",
            ),
            DeclareLaunchArgument(
                "voice",
                default_value="false",
                description="Start the Kokoro TTS + LocalWhisper voice pipeline (mic-based S2S)",
            ),
            DeclareLaunchArgument(
                "ollama",
                default_value="false",
                description="Start ollama serve at launch (disable if already running as service)",
            ),
            DeclareLaunchArgument(
                "foxglove_port",
                default_value="8765",
                description="WebSocket port for Foxglove Studio",
            ),

            # ── Ollama ────────────────────────────────────────────────────────
            OpaqueFunction(function=_launch_ollama),

            # ── ZED Mini camera ───────────────────────────────────────────────
            OpaqueFunction(function=_launch_zed),

            # ── Foxglove bridge ───────────────────────────────────────────────
            # Increased send_buffer_limit to handle compressed video streams.
            # capabilities includes 'assets' so the custom reasoning panel
            # can be served from the Foxglove extension bundle.
            Node(
                package="foxglove_bridge",
                executable="foxglove_bridge",
                name="foxglove_bridge",
                output="screen",
                parameters=[
                    {
                        "port": LaunchConfiguration("foxglove_port"),
                        "address": "0.0.0.0",
                        "send_buffer_limit": 100_000_000,   # 100 MB — handles video
                        "num_threads": 4,
                        "max_qos_depth": 10,
                        "use_compression": True,            # Foxglove msg compression
                        "capabilities": [
                            "clientPublish",
                            "parameters",
                            "parametersSubscribe",
                            "services",
                            "connectionGraph",
                            "assets",
                        ],
                        "topic_whitelist": [".*"],
                        "service_whitelist": [".*"],
                        "ignore_unresponsive_param_nodes": True,
                    }
                ],
            ),

            # ── Rosbridge (port 9090) — required by custom Foxglove panels ──────
            IncludeLaunchDescription(
                AnyLaunchDescriptionSource(
                    "/opt/ros/humble/share/rosbridge_server/launch/"
                    "rosbridge_websocket_launch.xml"
                )
            ),

            # ── VLM test node ─────────────────────────────────────────────────
            DeclareLaunchArgument(
                "vlm",
                default_value="true",
                description="Start vlm_test.py (Ollama Qwen VLM on ZED frames)",
            ),
            DeclareLaunchArgument(
                "vlm_model",
                default_value="qwen3-vl:8b-thinking",
                description="Ollama model for vlm_test.py",
            ),
            DeclareLaunchArgument(
                "vlm_loop",
                default_value="false",
                description="Run VLM in continuous loop mode",
            ),
            OpaqueFunction(function=_launch_vlm),

            # ── Voice pipeline (Kokoro TTS + LocalWhisper) ───────────────────
            OpaqueFunction(function=_launch_voice),
        ]
    )
