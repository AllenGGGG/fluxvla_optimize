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

    # Debug collection (zero overhead when disabled).
    debug: bool = False
    debug_dir: str = "/tmp/rtc_debug"

    def __post_init__(self) -> None:
        if self.execution_horizon <= 0:
            raise ValueError(
                f"execution_horizon must be positive, got {self.execution_horizon}"
            )
