#!/usr/bin/env python3
"""pistar06 async real-robot inference entry point (joint-angle space).

Main logic is split across this package's ``ros_node``/``model``/
``exec_engine`` modules.
"""

from __future__ import annotations

import logging
import pathlib

import rclpy

from .ros_node import JointInferenceNode


def _configure_logging() -> None:
    """Send exec_engine's stdlib logging (scheduler/visualizer) to a file.

    rclpy's own get_logger() and Python's logging module are two unconnected
    systems, and nothing in this package ever called logging.basicConfig --
    so ChunkScheduler's per-chunk latency diagnostics (scheduler.py's
    "Chunk scheduled"/"Splice underestimated inference delay" lines) were
    silently dropped regardless of the debug ROS parameter. Always-on at
    DEBUG level: one log call per chunk (not per 30Hz control tick), so the
    file-write cost is negligible against real-time control.
    """
    log_dir = pathlib.Path("/tmp/rtc_debug")
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_dir / "scheduler.log")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    )
    exec_engine_logger = logging.getLogger("deploy.exec_engine")
    exec_engine_logger.setLevel(logging.DEBUG)
    exec_engine_logger.addHandler(handler)
    exec_engine_logger.propagate = False
    # Belt-and-suspenders: fluxvla.engines.utils.overwatch's dictConfig call
    # (see main()'s comment) sets .disabled on loggers that already existed
    # at that point. Called after JointInferenceNode() now, so this should
    # be a no-op, but explicit is cheaper than a silently-dead log file if
    # some other import ever re-triggers the same footgun.
    exec_engine_logger.disabled = False
    for child_name in ("scheduler", "visualizer"):
        logging.getLogger(f"deploy.exec_engine.{child_name}").disabled = False


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        # Must run after JointInferenceNode() construction: building the
        # inference backend imports fluxvla.engines.utils.overwatch, which
        # calls logging.config.dictConfig(disable_existing_loggers=True) at
        # import time. That silently disables any logger already configured
        # before this point (a one-way flag, not undone by later
        # setLevel/addHandler calls) -- so configuring first meant every
        # scheduler.py log line was a silent no-op regardless of level.
        node = JointInferenceNode()
        _configure_logging()
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
