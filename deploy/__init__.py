from typing import Any

__all__ = ["JointInferenceNode"]


def __getattr__(name: str) -> Any:
    if name == "JointInferenceNode":
        from .ros_node import JointInferenceNode

        return JointInferenceNode
    raise AttributeError(name)
