"""Configuration for ChunkScheduler.

Deliberately excludes RTC guidance parameters (schedule/max_guidance_weight) --
those belong to fluxvla's own ``rtc_config`` dict (see ``deploy/model.py``'s
``_build_rtc_kwargs``), not to this scheduler. This dataclass only carries
what ``ChunkScheduler`` itself consumes: how much of the queue's tail to
treat as RTC prefix context, and debug output.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChunkSchedulerConfig:
    # Latency alignment is only meaningful for RTC prefix/guidance execution.
    rtc_enabled: bool = False

    # How many unconsumed queue steps to hand to predict_fn as RTC prefix
    # context (clamped to whatever is actually left in the queue).
    execution_horizon: int = 10
    # Minimum number of actions to keep before starting the next RTC
    # prediction.  0 lets the scheduler derive it from delay + 2*horizon
    # (handoff window + a trailing horizon of real data for the decay taper).
    replan_remaining: int = 0

    # How many predicted actions to enqueue from each chunk.
    # 0 means enqueue the full model chunk.
    publish_horizon: int = 0
    # RTC-specific override. 0 means publish the full model chunk.
    rtc_publish_horizon: int = 0

    # Debug collection (zero overhead when disabled).
    debug: bool = False
    debug_dir: str = "/tmp/rtc_debug"

    def __post_init__(self) -> None:
        if self.execution_horizon <= 0:
            raise ValueError(
                f"execution_horizon must be positive, got {self.execution_horizon}"
            )
        if self.replan_remaining < 0:
            raise ValueError(
                f"replan_remaining must be non-negative, got {self.replan_remaining}"
            )
        if self.publish_horizon < 0:
            raise ValueError(
                f"publish_horizon must be non-negative, got {self.publish_horizon}"
            )
        if self.rtc_publish_horizon < 0:
            raise ValueError(
                "rtc_publish_horizon must be non-negative, "
                f"got {self.rtc_publish_horizon}"
            )
