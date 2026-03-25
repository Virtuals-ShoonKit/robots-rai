"""Scout Mini office delivery agent.

ReAct agent that listens on /from_human and responds on /to_human.
Equipped with Nav2 navigation, ZED camera, and occupancy grid map tools.

Launch from the robots/scoutmini/ directory so RAI picks up config.toml:
    cd robots/scoutmini && python scripts/agent.py
"""

import sys
from pathlib import Path

# Make robots/common/ importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rai.agents import AgentRunner
from rai.agents.langchain.react_agent import ReActAgent
from rai.communication.ros2 import ROS2Connector, ROS2Context
from rai.communication.ros2.connectors.hri_connector import ROS2HRIConnector
from rai.tools.ros2 import GetROS2ImageConfiguredTool
from rai.tools.ros2.navigation.nav2 import GetOccupancyGridTool
from rai.tools.ros2.navigation.nav2_blocking import (
    GetCurrentPoseTool,
    NavigateToPoseBlockingTool,
)
from rai.tools.time import WaitForSecondsTool
from rai_whoami import EmbodimentInfo

from common.callback_handler import AgentCallbackHandler
from common.task_server import TaskManager, start_task_server

EMBODIMENT_PATH = Path(__file__).parent.parent / "embodiments" / "scoutmini.json"

# ZED camera topic — adjust to match your ZED ROS2 wrapper configuration
ZED_IMAGE_TOPIC = "/zed/zed_node/rgb/image_rect_color"

TASK_SERVER_HOST = "0.0.0.0"
TASK_SERVER_PORT = 8090


@ROS2Context()
def main():
    ros2_connector = ROS2Connector()
    hri_connector = ROS2HRIConnector()
    task_manager = TaskManager()

    embodiment_info = EmbodimentInfo.from_file(EMBODIMENT_PATH)

    tools = [
        GetROS2ImageConfiguredTool(
            connector=ros2_connector,
            topic=ZED_IMAGE_TOPIC,
        ),
        NavigateToPoseBlockingTool(
            connector=ros2_connector,
            frame_id="map",
            action_name="navigate_to_pose",
        ),
        GetCurrentPoseTool(
            connector=ros2_connector,
            frame_id="map",
            robot_frame_id="base_link",
        ),
        GetOccupancyGridTool(connector=ros2_connector),
        WaitForSecondsTool(),
    ]

    agent = ReActAgent(
        target_connectors={"/to_human": hri_connector},
        tools=tools,
        system_prompt=embodiment_info.to_langchain(),
    )

    callback = AgentCallbackHandler(ros2_connector, task_manager)
    agent.tracing_callbacks.append(callback)

    agent.subscribe_source("/from_human", hri_connector)

    start_task_server(task_manager, hri_connector, host=TASK_SERVER_HOST, port=TASK_SERVER_PORT, embodiment="scout-mini-rover")

    runner = AgentRunner([agent])
    runner.run_and_wait_for_shutdown()


if __name__ == "__main__":
    main()
