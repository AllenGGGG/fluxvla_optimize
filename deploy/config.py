from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


# Set PISTAR_MODEL_ID to the fine-tuned checkpoint file
# (e.g. .../checkpoints/step-XXXXXX.safetensors). dataset_statistics.json is
# read from its grandparent directory, matching training's checkpoint layout.
DEFAULT_MODEL_ID: str = os.environ.get("PISTAR_MODEL_ID", "")

# The eager no-RTC inference config lives in this repo now, so no
# separate training-repo root discovery (ACCVLA_ROOT/accvla_root) is needed.
_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INFERENCE_CONFIG = os.environ.get(
    "PISTAR_INFERENCE_CONFIG",
    str(_REPO_ROOT /
        "configs/pi05/pi05_parcel_sort_none_pytorch_inference.py"),
)
DEFAULT_TASK = (
    "Pick up the parcel with the left hand, then move it onto the conveyor "
    "belt with the right hand."
)
VALID_NORM_TYPES = {"quantile", "min_max"}
# fluxvla's own RTC (PI0FlowMatching.predict_action's rtc_config['method']):
# 'none' skips it, 'prefix' hard-pins the shared prefix, and 'guidance'
# softly aligns it
# (fluxvla/engines/utils/rtc_guidance.py). FluxVLAPolicy
# (model.py)'s predict_chunk feeds it the exec engine's real
# unconsumed queue tail as prev_actions (see deploy/exec_engine). Only
# takes effect when the loaded inference_model's predict_action actually
# implements RTC: the eager PI0/PI05FlowMatching classes and the fused
# PI05FlowMatchingGuidanceInference do; plain PI05FlowMatchingInference does not.
VALID_RTC_METHODS = {"none", "prefix", "guidance"}


def default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return "mps"
    return "cpu"


@dataclass
class WorkerConfig:
    model_id: str
    inference_config: str
    device: str
    dtype: str
    num_inference_steps_override: int = 0
    # Explicit dataset_statistics.json path. If unset, BaseInferenceRunner's
    # default (ckpt_path's grandparent directory) is used instead.
    norm_stats_path: str | None = None
    tokenizer_path: str | None = None

    # fluxvla RTC guidance; see VALID_RTC_METHODS above. Default 'none'.
    rtc_method: str = "none"
    rtc_execution_horizon: int = 10
    # 0 = derive the trigger from measured inference delay + 2 * RTC horizon
    # (handoff window + a trailing horizon of real data for the decay taper).
    rtc_replan_remaining: int = 0
    chunk_publish_horizon: int = 0  # 0 = publish the full predicted chunk
    rtc_publish_horizon: int = 0  # 0 = publish the full model chunk in RTC mode
    rtc_max_guidance_weight: float = 10.0
    # 'exp' matches the PI Kinetix reference implementation's recommended
    # schedule (fluxvla/engines/utils/rtc_guidance.py's compute_prefix_weights
    # docstring). Real-robot A/B data (2026-07-26, guidance+triton,
    # execution_horizon=10): with 'linear', 50-65% of large per-tick command
    # jumps landed within 150ms of a chunk-switch marker (vs a ~35% random
    # baseline) across two independent runs -- delay estimation was ruled
    # out separately (measured vs applied delay matched within ~0.1 step in
    # steady state). linear holds guidance weight >=0.5 for roughly half the
    # decay window then drops sharply near the end, concentrating the
    # old/new-trajectory discrepancy release right at the window's tail
    # (near the next switch); exp releases it gradually from the start of
    # the window instead. Not yet re-validated with 'exp' on the real robot
    # -- if a follow-up run doesn't show the jump concentration dropping
    # below the random baseline, reconsider.
    rtc_schedule: str = "exp"  # 'linear', 'exp', 'ones', or 'zeros'
