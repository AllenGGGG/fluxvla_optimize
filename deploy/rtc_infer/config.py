"""Configuration dataclasses for all inference backends."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AttentionSchedule(Enum):
    """Prefix attention weight schedule for RTC guidance."""

    ZEROS = "zeros"
    ONES = "ones"
    LINEAR = "linear"
    EXP = "exp"


@dataclass
class RTCConfig:
    """Configuration for Real-Time Chunking inference.

    RTC steers each denoising step toward the unexecuted tail of the previous
    chunk (the "prefix"), reducing discontinuities at chunk boundaries.
    """

    # Guidance schedule over the overlap window.
    prefix_attention_schedule: AttentionSchedule = AttentionSchedule.LINEAR
    # Hard cap on the scalar guidance weight.
    max_guidance_weight: float = 10.0
    # How many prefix steps to use as the RTC anchor.
    execution_horizon: int = 10

    # Debug collection (zero overhead when disabled).
    debug: bool = False
    debug_maxlen: int = 100
    # Directory for chunk boundary debug plots.
    debug_dir: str = "/tmp/rtc_debug"

    def __post_init__(self) -> None:
        if self.max_guidance_weight <= 0:
            raise ValueError(f"max_guidance_weight must be positive, got {self.max_guidance_weight}")
        if self.execution_horizon <= 0:
            raise ValueError(f"execution_horizon must be positive, got {self.execution_horizon}")
        if self.debug_maxlen <= 0:
            raise ValueError(f"debug_maxlen must be positive, got {self.debug_maxlen}")


@dataclass
class PlainConfig:
    execution_fraction: float = 0.5
    debug: bool = False
    debug_dir: str = "/tmp/plain_debug"


@dataclass
class OverlapConfig:
    overlap_steps: int = 15
    action_blend_steps: int = 10
    state_fusion_alpha: float = 0.5
    discard_latency_steps: bool = True
    min_queue_keep_steps: int = 1
    debug: bool = False
    debug_dir: str = "/tmp/overlap_debug"
