#!/usr/bin/env python3
"""pistar06 async real-robot inference entry point (joint-angle space).

Main logic is split across this package's ``ros_node``/``model``/
``exec_engine`` modules.
"""

from __future__ import annotations

import rclpy

from .ros_node import JointInferenceNode


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = JointInferenceNode()
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
