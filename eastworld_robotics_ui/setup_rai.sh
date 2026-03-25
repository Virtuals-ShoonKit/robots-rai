#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# setup_rai.sh — One-shot installer for RAI + rosbridge + Foxglove extension
#
# Usage:
#   chmod +x setup_rai.sh
#   ./setup_rai.sh
#
# What it does:
#   1. Detects your ROS 2 distro (humble / jazzy)
#   2. Installs rosbridge_server (apt)
#   3. Installs rai_interfaces (apt, with source fallback)
#   4. Installs Ollama and pulls Qwen3-VL models (8b + 32b-thinking)
#   5. Installs uv (if not present) and uses it to create a venv + install rai-core
#   6. Runs rai-config-init to create ~/.config/rai/config.toml
#   7. Builds & locally installs the Foxglove agent-panel extension
#
# Prerequisites:
#   - Ubuntu 22.04+ with ROS 2 (humble or jazzy) sourced
#   - Node.js >= 18 and npm
#   - Python 3.10+
#   - sudo access (for apt packages)
#   - curl (for uv / ollama installers)
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv_rai"
EXTENSION_DIR="${SCRIPT_DIR}/foxglove_extension/foxglove-agent-panel"

# ── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; }

# ── Step 0: Preflight checks ────────────────────────────────────────────────
info "Running preflight checks..."

if [ -z "${ROS_DISTRO:-}" ]; then
    err "ROS_DISTRO is not set. Please source your ROS 2 setup first:"
    echo "    source /opt/ros/<distro>/setup.bash"
    exit 1
fi
ok "ROS 2 distro detected: ${ROS_DISTRO}"

if ! command -v python3 &>/dev/null; then
    err "python3 not found. Please install Python 3.10+."
    exit 1
fi
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
ok "Python ${PYTHON_VERSION}"

if ! command -v node &>/dev/null || ! command -v npm &>/dev/null; then
    err "Node.js / npm not found. Please install Node.js >= 18."
    exit 1
fi
ok "Node $(node --version), npm $(npm --version)"

echo ""

# ── Step 1: Install rosbridge_server ─────────────────────────────────────────
info "Step 1/8: Installing rosbridge_server..."

if ros2 pkg list 2>/dev/null | grep -q rosbridge_server; then
    ok "rosbridge_server already installed."
else
    sudo apt-get update -qq
    sudo apt-get install -y -qq "ros-${ROS_DISTRO}-rosbridge-server"
    ok "rosbridge_server installed."
fi

echo ""

# ── Step 2: Install rai_interfaces ───────────────────────────────────────────
info "Step 2/8: Installing rai_interfaces..."

if ros2 pkg list 2>/dev/null | grep -q rai_interfaces; then
    ok "rai_interfaces already installed."
else
    # Try the apt package first (may lag behind releases)
    if sudo apt-get install -y -qq "ros-${ROS_DISTRO}-rai-interfaces" 2>/dev/null; then
        ok "rai_interfaces installed via apt."
    else
        warn "apt package not available — building from source..."

        RAI_IFACES_WS="${SCRIPT_DIR}/rai_interfaces_ws"
        mkdir -p "${RAI_IFACES_WS}/src"
        if [ ! -d "${RAI_IFACES_WS}/src/rai_interfaces" ]; then
            git clone https://github.com/RobotecAI/rai_interfaces.git \
                "${RAI_IFACES_WS}/src/rai_interfaces"
        fi

        (
            cd "${RAI_IFACES_WS}"
            # Install any rosdep dependencies
            rosdep install --from-paths src --ignore-src -r -y 2>/dev/null || true
            colcon build --symlink-install
        )

        # shellcheck disable=SC1091
        source "${RAI_IFACES_WS}/install/setup.bash"
        ok "rai_interfaces built from source."
        warn "Add this to your .bashrc to persist:"
        echo "    source ${RAI_IFACES_WS}/install/setup.bash"
    fi
fi

echo ""

# ── Step 3: Install Ollama & pull Qwen3-VL models ───────────────────────────
info "Step 3/8: Installing Ollama and pulling Qwen3-VL models..."

if command -v ollama &>/dev/null; then
    ok "Ollama already installed: $(ollama --version 2>/dev/null || echo 'unknown')"
else
    info "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    ok "Ollama installed."
fi

# Ensure Ollama service is running
if ! pgrep -x ollama &>/dev/null; then
    info "Starting Ollama service..."
    ollama serve &>/dev/null &
    sleep 3
fi

info "Pulling qwen3-vl:8b (fast model)..."
ollama pull qwen3-vl:8b

info "Pulling qwen3-vl:32b-thinking (reasoning model)..."
ollama pull qwen3-vl:32b-thinking

ok "Qwen3-VL models ready."

echo ""

# ── Step 4: Install uv, create venv & install rai-core ───────────────────────
info "Step 4/7: Setting up uv, Python venv, and installing rai-core..."

# Install uv if not already present
if command -v uv &>/dev/null; then
    ok "uv already installed: $(uv --version)"
else
    info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add to PATH for the rest of this script
    export PATH="${HOME}/.local/bin:${PATH}"
    ok "uv installed: $(uv --version)"
fi

# Create venv with uv (--system-site-packages so ROS 2 Python pkgs are visible)
if [ ! -d "${VENV_DIR}" ]; then
    uv venv "${VENV_DIR}" --python python3 --system-site-packages
    ok "Created venv at ${VENV_DIR}"
else
    ok "Venv already exists at ${VENV_DIR}"
fi

# Install rai-core into the venv using uv pip
info "Installing rai-core (this should be fast)..."
uv pip install --python "${VENV_DIR}/bin/python" rai-core

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

ok "rai-core installed in venv via uv."

echo ""

# ── Step 5: Install langchain-ollama into venv ──────────────────────────────
info "Step 5/7: Installing langchain-ollama for vision agent..."
uv pip install --python "${VENV_DIR}/bin/python" langchain-ollama
ok "langchain-ollama installed."

echo ""

# ── Step 6: Initialize RAI config ───────────────────────────────────────────
info "Step 6/7: Initializing RAI configuration..."

RAI_CONFIG="${HOME}/.config/rai/config.toml"
if [ -f "${RAI_CONFIG}" ]; then
    ok "RAI config already exists at ${RAI_CONFIG}"
    warn "Run 'rai-config-init' again to reconfigure."
else
    rai-config-init
    ok "RAI config created at ${RAI_CONFIG}"
fi

echo ""

# ── Step 7: Build Foxglove extension ────────────────────────────────────────
info "Step 7/7: Building Foxglove agent-panel extension..."

if [ ! -d "${EXTENSION_DIR}" ]; then
    err "Extension directory not found: ${EXTENSION_DIR}"
    exit 1
fi

(
    cd "${EXTENSION_DIR}"
    npm install --silent
    npm run build
)

ok "Foxglove extension built."
info "To install it locally into Foxglove, run:"
echo "    cd ${EXTENSION_DIR} && npm run local-install"

echo ""

# ── Summary ──────────────────────────────────────────────────────────────────
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Setup complete!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "  To run the full pipeline (3 terminals):"
echo ""
echo "  Terminal 1 — rosbridge:"
echo "       ros2 launch rosbridge_server rosbridge_websocket_launch.xml"
echo ""
echo "  Terminal 2 — RTSP-to-ROS bridge:"
echo "       source ${VENV_DIR}/bin/activate"
echo "       python3 ${SCRIPT_DIR}/stream_to_ros.py --rtsp-url <your-rtsp-url>"
echo ""
echo "  Terminal 3 — Vision agent:"
echo "       source ${VENV_DIR}/bin/activate"
echo "       python3 ${SCRIPT_DIR}/vision_agent.py --camera-topic /camera/image_raw"
echo ""
echo "  Then open Foxglove and add the 'Agent Panel'."
echo "     Panel settings let you change:"
echo "       - Rosbridge URL (default: ws://localhost:9090)"
echo "       - Input/output topics (default: /from_human, /to_human)"
echo "       - Ollama model (qwen3-vl:8b or qwen3-vl:32b-thinking)"
echo ""
echo "  (to add packages later: uv pip install --python ${VENV_DIR}/bin/python <pkg>)"
echo ""
