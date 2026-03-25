#!/usr/bin/env bash
#
# Single-command launcher for the Scout Mini RAI agent.
#
#   ./launch.sh                 — vLLM + Ollama embeddings + Foxglove
#   ./launch.sh --ollama        — Ollama for everything
#
# Prerequisites:
#   - ROS 2 Humble installed at /opt/ros/humble
#   - rai_interfaces built (rai/install/ exists)
#   - .venv created with: uv venv ../../.venv --python 3.10 --system-site-packages
#   - uv pip install -r requirements.txt
#   - ollama and/or vllm installed (uv pip install vllm)
#   - ros-humble-rosbridge-server installed (for Foxglove)
#
# On Jetson: vLLM and Ollama run on the GPU host, not here.
# Point [openai] base_url in config.toml at the GPU host before launching.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$ROBOT_DIR/../.." && pwd)"

VLLM_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
VLLM_PORT=8000
OLLAMA_PORT=11434
ROSBRIDGE_PORT=9090
VLLM_MAX_MODEL_LEN=4096

USE_OLLAMA_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --ollama) USE_OLLAMA_ONLY=true ;;
    esac
done

PIDS=()
CONFIG_BACKUP=""

cleanup() {
    echo ""
    echo "Shutting down..."
    if [[ -n "$CONFIG_BACKUP" && -f "$CONFIG_BACKUP" ]]; then
        mv "$CONFIG_BACKUP" "$ROBOT_DIR/config.toml"
    fi
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
    echo "Done."
}
trap cleanup EXIT INT TERM

wait_for_port() {
    local port=$1 name=$2 timeout=${3:-120}
    echo "Waiting for $name on port $port..."
    for ((i=0; i<timeout; i++)); do
        if curl -sf "http://localhost:$port" >/dev/null 2>&1 || \
           curl -sf "http://localhost:$port/v1/models" >/dev/null 2>&1; then
            echo "$name is ready."
            return 0
        fi
        sleep 1
    done
    echo "ERROR: $name did not start within ${timeout}s"
    exit 1
}

# ── Source ROS 2 + rai_interfaces ──────────────────────────────
source /opt/ros/humble/setup.bash
if [[ -f "$REPO_ROOT/rai/install/setup.bash" ]]; then
    source "$REPO_ROOT/rai/install/setup.bash"
fi

# ── Activate venv ──────────────────────────────────────────────
if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
    source "$REPO_ROOT/.venv/bin/activate"
fi

export OPENAI_API_KEY="${OPENAI_API_KEY:-not-needed}"

# ── Start rosbridge (for Foxglove) ─────────────────────────────
if ! curl -sf "http://localhost:$ROSBRIDGE_PORT" >/dev/null 2>&1; then
    echo "Starting rosbridge_server..."
    ros2 launch rosbridge_server rosbridge_websocket_launch.xml &
    PIDS+=($!)
    sleep 3
    echo "rosbridge_server started (ws://localhost:$ROSBRIDGE_PORT)."
else
    echo "rosbridge_server already running."
fi

# ── Start Ollama (embeddings, or full LLM if --ollama) ─────────
if ! curl -sf "http://localhost:$OLLAMA_PORT" >/dev/null 2>&1; then
    echo "Starting Ollama..."
    ollama serve &
    PIDS+=($!)
    wait_for_port "$OLLAMA_PORT" "Ollama" 30
fi

if [[ "$USE_OLLAMA_ONLY" == true ]]; then
    echo "Mode: Ollama-only"
    echo "Pulling models..."
    ollama pull qwen2.5-vl
    ollama pull nomic-embed-text
else
    echo "Mode: vLLM + Ollama (embeddings)"
    echo "Pulling embeddings model..."
    ollama pull nomic-embed-text

    # ── Start vLLM ─────────────────────────────────────────────
    if ! curl -sf "http://localhost:$VLLM_PORT/v1/models" >/dev/null 2>&1; then
        echo "Starting vLLM ($VLLM_MODEL)..."
        vllm serve "$VLLM_MODEL" \
            --dtype auto \
            --max-model-len "$VLLM_MAX_MODEL_LEN" \
            --port "$VLLM_PORT" &
        PIDS+=($!)
        wait_for_port "$VLLM_PORT" "vLLM" 180
    fi
fi

# ── Prepare config ─────────────────────────────────────────────
if [[ "$USE_OLLAMA_ONLY" == true ]]; then
    CONFIG_BACKUP="$ROBOT_DIR/.config.toml.bak"
    cp "$ROBOT_DIR/config.toml" "$CONFIG_BACKUP"
    sed -i 's/^simple_model = "openai"/simple_model = "ollama"/' "$ROBOT_DIR/config.toml"
    sed -i 's/^complex_model = "openai"/complex_model = "ollama"/' "$ROBOT_DIR/config.toml"
    echo "Config patched for Ollama-only mode."
fi

# ── Launch the agent ───────────────────────────────────────────
echo ""
echo "========================================="
echo "  Scout Mini Agent — starting"
echo "  Task API:  http://localhost:8090"
echo "  Foxglove:  ws://localhost:$ROSBRIDGE_PORT"
echo "========================================="
echo ""

cd "$ROBOT_DIR"
exec python scripts/agent.py
