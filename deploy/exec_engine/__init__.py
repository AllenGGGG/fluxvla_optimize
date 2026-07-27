"""Async inference scheduling: unrelated to RTC guidance math (that's fluxvla's).

Owns only the producer/consumer plumbing between the model's (slow, chunked)
predict calls and the robot's (fast, fixed-rate) control loop: a background
thread that keeps producing action chunks, a thread-safe queue the control
loop drains one step at a time, and delay-aware RTC prefix stitching so
consecutive chunks don't jump at the splice point.
"""

from .base import InferBackend
from .action_queue import ActionQueue
from .config import ChunkSchedulerConfig
from .latency_tracker import LatencyTracker
from .scheduler import ChunkScheduler
from .serial import SerialInferenceBackend

__all__ = [
    "InferBackend",
    "ActionQueue",
    "ChunkSchedulerConfig",
    "LatencyTracker",
    "ChunkScheduler",
    "SerialInferenceBackend",
]
