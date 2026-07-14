from typing import Any

__all__ = ["AsyncJointInferenceNode", "WorkerConfig"]


def __getattr__(name: str) -> Any:
    if name == "AsyncJointInferenceNode":
        from .ros_node import AsyncJointInferenceNode

        return AsyncJointInferenceNode
    if name == "WorkerConfig":
        from .config import WorkerConfig

        return WorkerConfig
    raise AttributeError(name)
