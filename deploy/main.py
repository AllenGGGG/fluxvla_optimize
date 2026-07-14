#!/usr/bin/env python3
"""pistar06 async real-robot inference entry point (joint-angle space).

Main logic is split across the fluxvla_deploy package.
"""

from __future__ import annotations

import rclpy

from .ros_node import AsyncJointInferenceNode


def main(args: list[str] | None = None, *, default_rtc_mode: str = "plain") -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = AsyncJointInferenceNode(default_rtc_mode=default_rtc_mode)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
