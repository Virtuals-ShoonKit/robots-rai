  To run the full pipeline (3 terminals):

  Terminal 1 — rosbridge:
       ros2 launch rosbridge_server rosbridge_websocket_launch.xml

  Terminal 2 — RTSP-to-ROS bridge:
       source /home/ubuntu/Desktop/VP/eastworld_robotics_ui/.venv_rai/bin/activate
       python3 /home/ubuntu/Desktop/VP/eastworld_robotics_ui/stream_to_ros.py --rtsp-url <your-rtsp-url>

  Terminal 3 — Vision agent:
       source /home/ubuntu/Desktop/VP/eastworld_robotics_ui/.venv_rai/bin/activate
       python3 /home/ubuntu/Desktop/VP/eastworld_robotics_ui/vision_agent.py --camera-topic /camera/image_raw

  Then open Foxglove and add the 'Agent Panel'.
     Panel settings let you change:
       - Rosbridge URL (default: ws://localhost:9090)
       - Input/output topics (default: /from_human, /to_human)
       - Ollama model (qwen3-vl:8b or qwen3-vl:32b-thinking)