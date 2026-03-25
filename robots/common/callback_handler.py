"""LangChain callback handler: /agent/* telemetry and task lifecycle."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import LLMResult
from std_msgs.msg import String

from rai.communication.ros2.connectors.ros2_connector import ROS2Connector

from .task_server import TaskManager

# Qwen-style reasoning delimiters (chain-of-thought); adjust if your model differs.
THINK_START = "<think>"
THINK_END = "</think>"


def _stringify_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content) if content is not None else ""


class AgentCallbackHandler(BaseCallbackHandler):
    """Publishes /agent/status, /agent/reasoning, /agent/events; updates TaskManager."""

    def __init__(self, ros2_connector: ROS2Connector, task_manager: TaskManager) -> None:
        super().__init__()
        self._log = logging.getLogger(__name__)
        self._ros2 = ros2_connector
        self._task_manager = task_manager
        node = ros2_connector.node
        self._pub_status = node.create_publisher(String, "/agent/status", 10)
        self._pub_reasoning = node.create_publisher(String, "/agent/reasoning", 10)
        self._pub_events = node.create_publisher(String, "/agent/events", 10)

        self._think_state: str = "NORMAL"  # NORMAL | IN_THINK
        self._think_carry = ""

        self._tool_starts: dict[UUID, float] = {}

    def _publish_str(self, pub: Any, data: str) -> None:
        pub.publish(String(data=data))

    def _publish_event(self, payload: dict[str, Any]) -> None:
        self._publish_str(self._pub_events, json.dumps(payload, default=str))

    def _reset_think_parser(self) -> None:
        self._think_state = "NORMAL"
        self._think_carry = ""

    def _feed_think_tokens(self, token: str) -> None:
        """Route tokens between THINK_START/THINK_END to /agent/reasoning."""
        s = self._think_carry + token
        self._think_carry = ""

        while s:
            if self._think_state == "NORMAL":
                i = s.find(THINK_START)
                if i == -1:
                    if len(s) > len(THINK_START):
                        self._think_carry = s[-len(THINK_START):]
                        s = s[:-len(THINK_START)]
                    break
                s = s[i + len(THINK_START):]
                self._think_state = "IN_THINK"
                continue

            # IN_THINK
            j = s.find(THINK_END)
            if j == -1:
                if s:
                    self._publish_str(self._pub_reasoning, s)
                break
            chunk = s[:j]
            if chunk:
                self._publish_str(self._pub_reasoning, chunk)
            s = s[j + len(THINK_END):]
            self._think_state = "NORMAL"

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        self._reset_think_parser()
        self._publish_str(self._pub_status, "thinking")

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        self._reset_think_parser()
        self._publish_str(self._pub_status, "thinking")

    def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: Optional[Any] = None,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        if not token:
            return
        self._feed_think_tokens(token)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        if self._think_carry:
            if self._think_state == "IN_THINK":
                self._publish_str(self._pub_reasoning, self._think_carry)
            self._think_carry = ""
        self._think_state = "NORMAL"

        self._publish_str(self._pub_status, "idle")

        try:
            gen = response.generations[0][0]
            msg = gen.message
            if not isinstance(msg, AIMessage):
                return
            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls:
                return
            text = _stringify_message_content(msg.content).strip()
            cur = self._task_manager.get_current_task()
            if cur is not None and cur.status == "in_progress":
                self._task_manager.complete_task(cur.task_id, text)
        except Exception as e:
            self._log.warning("on_llm_end task completion handling: %s", e)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        self._publish_str(self._pub_status, "error")
        self._abort_current_task(str(error))

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        self._publish_str(self._pub_status, "error")
        self._abort_current_task(str(error))

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        inputs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        name = serialized.get("name") or serialized.get("id") or "tool"
        self._tool_starts[run_id] = time.monotonic()
        self._publish_str(self._pub_status, f"executing:{name}")
        args: Any = inputs if inputs is not None else input_str
        self._publish_event(
            {
                "event": "start",
                "tool": name,
                "args": args if isinstance(args, dict) else {"input": args},
            }
        )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        start_t = self._tool_starts.pop(run_id, None)
        elapsed = (time.monotonic() - start_t) if start_t is not None else None
        self._publish_event(
            {
                "event": "end",
                "result": output,
                "elapsed_s": round(elapsed, 3) if elapsed is not None else None,
            }
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        self._tool_starts.pop(run_id, None)
        self._publish_str(self._pub_status, "error")
        self._publish_event({"event": "error", "error": str(error)})
        self._abort_current_task(str(error))

    def _abort_current_task(self, reason: str) -> None:
        cur = self._task_manager.get_current_task()
        if cur is not None and cur.status == "in_progress":
            self._task_manager.abort_task(cur.task_id, reason)
