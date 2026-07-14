from typing import Any

__all__ = ["JointInferenceNode", "WorkerConfig"]


def __getattr__(name: str) -> Any:
    if name == "JointInferenceNode":
        from .ros_node import JointInferenceNode

        return JointInferenceNode
    if name == "WorkerConfig":
        from .config import WorkerConfig

        return WorkerConfig
    raise AttributeError(name)
