# robots-rai

Multi-robot RAI workspace. Each robot runs a ReAct agent (LangChain/LangGraph + ROS 2) that exposes a JSON task API over HTTP and streams telemetry to Foxglove via rosbridge.

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│  Operator / Fleet (no ROS required)                        │
│                                                            │
│  ┌──────────────────────┐  ┌─────────────────────────────┐ │
│  │ External website /   │  │ Foxglove: operator console  │ │
│  │ fleet backend        │  │  - agent chat (/to_human)   │ │
│  │ POST /execute_task   │  │  - streaming reasoning      │ │
│  │ GET  /status, /tasks │  │  - tool activity            │ │
│  └──────────┬───────────┘  │  - LiDAR / camera / map viz │ │
│             │ HTTP :8090   └──────────────┬───────────────┘ │
└────────────┼──────────────────────────────┼────────────────┘
             │                              │ WS :9090 (rosbridge)
             ▼                              ▼
┌────────────────────────────────────────────────────────────┐
│  Robot (Jetson Orin or dev machine)                        │
│  ROS 2: 3D LiDAR · camera → VLM · Nav2 + 2D /map          │
│                                                            │
│  ┌───────────────────┐    ┌──────────────────────────────┐ │
│  │ FastAPI :8090     │    │ ReActAgent (ROS 2 node)      │ │
│  │                   │    │  ├─ NavigateToPoseBlocking   │ │
│  │ GET  /status      │◄──►│  ├─ GetCurrentPose           │ │
│  │ POST /execute_task│    │  ├─ GetOccupancyGrid         │ │
│  │ GET  /tasks/{id}/ │    │  ├─ GetROS2Image (→ VLM)     │ │
│  │       status      │    │  └─ WaitForSeconds           │ │
│  └────────┬──────────┘    │                              │ │
│           │               │ AgentCallbackHandler:        │ │
│  ┌────────▼──────────┐    │  /agent/status               │ │
│  │ TaskManager       │    │  /agent/reasoning            │ │
│  │ (shared state)    │    │  /agent/events               │ │
│  └───────────────────┘    │  /to_human                   │ │
│                           └──────────────────────────────┘ │
│                                        │ HTTP :8000 /v1/    │
└────────────────────────────────────────┼───────────────────┘
                                         ▼
┌────────────────────────────────────────────────────────────┐
│  GPU host (workstation / server)                           │
│  vLLM: Qwen2.5-VL  ·  optional Ollama (embeddings only)   │
│  All LLM/VLM inference lives here — never on the robot.   │
└────────────────────────────────────────────────────────────┘
```

### What runs where

| Location | Services | Notes |
|----------|----------|-------|
| **Robot** (Jetson Orin) | ROS 2 stack (LiDAR, camera, Nav2), `agent.py`, FastAPI `:8090`, rosbridge `:9090` | No model weights. Calls GPU host over HTTP for all inference. |
| **GPU host** | vLLM `:8000` (Qwen2.5-VL), Ollama `:11434` (embeddings, optional) | One host can serve multiple robots. |
| **Operator machine** | Foxglove (desktop + extension), `curl` / fleet website | Connects rosbridge for telemetry; HTTP to `:8090` for task control. |

### Task API (HTTP on robot)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/status` | GET | `{"status": "available"}` or `{"status": "busy"}` |
| `/execute_task` | POST | Submit `{"task": "..."}` → `{"success": bool, "task_id": int}` |
| `/tasks/{id}/status` | GET | `{"status": "in_progress|completed|aborted|rejected", "content": "..."}` |

The robot processes one task at a time. A second submission while busy returns `success: false`.

### Telemetry topics (rosbridge → Foxglove)

| Topic | Type | Content |
|-------|------|---------|
| `/agent/status` | `std_msgs/String` | `"idle"`, `"thinking"`, `"executing:<tool>"`, `"error"` |
| `/agent/reasoning` | `std_msgs/String` | Streaming `<think>` tokens (chain-of-thought) |
| `/agent/events` | `std_msgs/String` | JSON tool start/end/error events |
| `/to_human` | `HRIMessage` | Final agent reply text |

---

## Repo layout

```
robots-rai/
├── robots/
│   ├── common/                   # Shared agent utilities (all robots)
│   │   ├── task_server.py        # TaskManager + FastAPI app + start_task_server()
│   │   └── callback_handler.py   # AgentCallbackHandler (telemetry + task lifecycle)
│   ├── scoutmini/                # Scout Mini delivery robot (active)
│   │   ├── scripts/agent.py      # ReAct agent — imports from common/
│   │   ├── embodiments/          # Robot identity, capabilities, waypoints
│   │   ├── config.toml           # LLM vendor, task server bind, tracing config
│   │   └── launch.sh             # Single-command launcher
│   ├── unitree_g1/               # Planned
│   └── x500_quad/                # Planned
├── rai/                          # RAI framework (git submodule, Robotec.AI)
├── eastworld_robotics_ui/
│   └── foxglove_extension/
│       └── foxglove-agent-panel/ # Foxglove panel: chat, reasoning, tool activity
├── requirements.txt
└── colcon.defaults.yaml
```

---

## Prerequisites

- Ubuntu 22.04 / ROS 2 Humble
- Python 3.10
- [uv](https://docs.astral.sh/uv/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **GPU host**: NVIDIA GPU with CUDA, vLLM installed
- **Robot**: Jetson Orin (or any ROS 2 machine) — no GPU required on robot

---

## Setup

### 1. Clone with submodule

```bash
git clone --recurse-submodules <repo-url>
cd robots-rai
```

### 2. Install rosbridge

```bash
sudo apt update
sudo apt install ros-humble-rosbridge-server
```

### 3. Build rai_interfaces

```bash
source /opt/ros/humble/setup.bash
cd rai
vcs import < ros_deps.repos
colcon build --packages-select rai_interfaces
source install/setup.bash
cd ..
```

### 4. Create venv and install Python packages

```bash
uv venv .venv --python 3.10 --system-site-packages
source .venv/bin/activate
uv pip install -r requirements.txt
```

Verify:

```bash
python -c "import fastapi, uvicorn, rai, rai_whoami; print('OK')"
```

### 5. Set up the GPU host

On the **GPU host** (workstation or server — not the robot):

```bash
# Install vLLM
uv tool install vllm

# Start vLLM (model downloads ~15 GB on first run)
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
    --dtype auto --max-model-len 4096 --host 0.0.0.0 --port 8000

# Optional: Ollama for embeddings
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull nomic-embed-text
```

Verify from the robot:

```bash
curl http://<gpu-host>:8000/v1/models
```

### 6. Configure the robot

Edit `robots/scoutmini/config.toml` — point `[openai]` `base_url` at the GPU host:

```toml
[openai]
base_url = "http://<gpu-host>:8000/v1/"
```

If using Ollama embeddings, point `[ollama]` at the GPU host too (never `localhost` on Jetson):

```toml
[ollama]
base_url = "http://<gpu-host>:11434"
```

---

## Running a Robot Agent

```bash
cd robots/scoutmini

# vLLM on GPU host + Ollama embeddings (default)
bash launch.sh

# Ollama for everything (simpler, slower)
bash launch.sh --ollama
```

The script sources ROS 2, activates the venv, starts rosbridge, and launches the agent. Startup banner shows:

```
=========================================
  Scout Mini Agent — starting
  Task API:  http://localhost:8090
  Foxglove:  ws://localhost:9090
=========================================
```

Ctrl-C shuts everything down cleanly.

---

## Sending Tasks

### curl (no Foxglove needed)

```bash
# Check availability
curl http://<robot-ip>:8090/status
# → {"status":"available"}

# Submit a task
curl -X POST http://<robot-ip>:8090/execute_task \
  -H "Content-Type: application/json" \
  -d '{"task": "navigate to the kitchen and report what you see"}'
# → {"success":true,"content":"The task has been successfully initiated","task_id":1}

# Poll until done
curl http://<robot-ip>:8090/tasks/1/status
# → {"status":"completed","content":"Arrived at the kitchen. I can see ..."}
```

### Foxglove panel

1. Build and install the extension:

```bash
cd eastworld_robotics_ui/foxglove_extension/foxglove-agent-panel
npm install && npm run build && npm run local-install
```

2. Open Foxglove → connect rosbridge at `ws://<robot-ip>:9090`
3. Add the **Agent Panel**
4. In panel settings, set **Task Server URL** to `http://<robot-ip>:8090`

The panel shows:
- **Status dot** — green (idle) / blue pulsing (thinking) / orange (executing tool) / red (error)
- **Robot availability** — available / busy
- **Reasoning** — collapsible, streams chain-of-thought tokens live
- **Tool Activity** — collapsible, shows tool calls with elapsed time
- **Chat** — final agent replies from `/to_human`
- **Execute button** — submits tasks via the HTTP task API

### ROS 2 CLI

```bash
ros2 topic pub /from_human rai_interfaces/msg/HRIMessage \
  "{text: 'go to the pantry'}" --once
ros2 topic echo /to_human
ros2 topic echo /agent/status
```

---

## Adding a New Robot

1. Create `robots/<name>/` with `scripts/agent.py`, `embodiments/<name>.json`, `config.toml`, `launch.sh`
2. In `agent.py`, import from `common`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # robots/

from common.task_server import TaskManager, start_task_server
from common.callback_handler import AgentCallbackHandler
```

3. Wire into `main()` the same way as `scoutmini/scripts/agent.py`
4. Add a `[task_server]` section to `config.toml`
