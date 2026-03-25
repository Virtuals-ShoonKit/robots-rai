"""FastAPI task server and in-memory task state — generic across embodiments."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

if TYPE_CHECKING:
    from rai.communication.ros2.connectors.hri_connector import ROS2HRIConnector


@dataclass
class TaskRecord:
    task_id: int
    status: str  # in_progress | completed | aborted | rejected
    description: str
    content: str = ""
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None


class TaskManager:
    """Thread-safe task store; single in-flight task at a time."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tasks: dict[int, TaskRecord] = {}
        self._next_id = 1
        self._current_in_progress_id: int | None = None

    def create_task(self, description: str) -> TaskRecord:
        with self._lock:
            task_id = self._next_id
            self._next_id += 1
            rec = TaskRecord(
                task_id=task_id,
                status="in_progress",
                description=description,
            )
            self._tasks[task_id] = rec
            self._current_in_progress_id = task_id
            return rec

    def get_task(self, task_id: int) -> TaskRecord | None:
        with self._lock:
            return self._tasks.get(task_id)

    def get_current_task(self) -> TaskRecord | None:
        with self._lock:
            if self._current_in_progress_id is None:
                return None
            return self._tasks.get(self._current_in_progress_id)

    def complete_task(self, task_id: int, content: str) -> None:
        with self._lock:
            rec = self._tasks.get(task_id)
            if rec is None or rec.status != "in_progress":
                return
            rec.status = "completed"
            rec.content = content
            rec.completed_at = time.time()
            if self._current_in_progress_id == task_id:
                self._current_in_progress_id = None

    def abort_task(self, task_id: int, reason: str) -> None:
        with self._lock:
            rec = self._tasks.get(task_id)
            if rec is None or rec.status != "in_progress":
                return
            rec.status = "aborted"
            rec.content = reason
            rec.completed_at = time.time()
            if self._current_in_progress_id == task_id:
                self._current_in_progress_id = None

    @property
    def is_busy(self) -> bool:
        with self._lock:
            return self._current_in_progress_id is not None


class ExecuteTaskRequest(BaseModel):
    task: str


def _build_app(task_manager: TaskManager, hri_connector: "ROS2HRIConnector", embodiment: str = "") -> FastAPI:
    from rai.communication.ros2.messages import ROS2HRIMessage

    app = FastAPI(title="Robot Task Server", version="1.0.0")

    @app.get("/status")
    def status() -> dict[str, str]:
        payload: dict[str, str] = {
            "status": "busy" if task_manager.is_busy else "available",
        }
        if embodiment:
            payload["embodiment"] = embodiment
        return payload

    @app.post("/execute_task")
    def execute_task(body: ExecuteTaskRequest) -> dict[str, Any]:
        if task_manager.is_busy:
            cur = task_manager.get_current_task()
            tid = cur.task_id if cur else "?"
            return {
                "success": False,
                "content": f"Robot is currently busy with task {tid}",
                "task_id": None,
            }
        rec = task_manager.create_task(body.task.strip())
        msg = ROS2HRIMessage(
            text=rec.description,
            message_author="human",
            communication_id=f"task:{rec.task_id}",
            seq_no=0,
            seq_end=True,
        )
        hri_connector.send_message(msg, "/from_human")
        return {
            "success": True,
            "content": "The task has been successfully initiated",
            "task_id": rec.task_id,
        }

    @app.get("/tasks/{task_id}/status")
    def task_status(task_id: int) -> dict[str, str]:
        rec = task_manager.get_task(task_id)
        if rec is None:
            return {
                "status": "unknown",
                "content": f"No task found with id {task_id}",
            }
        return {"status": rec.status, "content": rec.content}

    return app


def start_task_server(
    task_manager: TaskManager,
    hri_connector: "ROS2HRIConnector",
    host: str = "0.0.0.0",
    port: int = 8090,
    embodiment: str = "",
) -> None:
    """Run uvicorn in a daemon thread so the ROS agent keeps the main thread."""
    app = _build_app(task_manager, hri_connector, embodiment=embodiment)
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
    )
    server = uvicorn.Server(config)

    thread = threading.Thread(
        target=lambda: server.run(),
        name="task-server",
        daemon=True,
    )
    thread.start()
