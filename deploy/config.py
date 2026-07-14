from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


# Set PISTAR_MODEL_ID to the fine-tuned checkpoint file
# (e.g. .../checkpoints/step-XXXXXX.safetensors). dataset_statistics.json is
# read from its grandparent directory, matching training's checkpoint layout.
DEFAULT_MODEL_ID: str = os.environ.get("PISTAR_MODEL_ID", "")

# configs/pi05/pi05_parcel_sort_inference.py lives in this repo now, so no
# separate training-repo root discovery (ACCVLA_ROOT/accvla_root) is needed.
_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INFERENCE_CONFIG = os.environ.get(
    "PISTAR_INFERENCE_CONFIG",
    str(_REPO_ROOT / "configs/pi05/pi05_parcel_sort_inference.py"),
)
DEFAULT_TASK = (
    "Pick up the parcel with the left hand, then move it onto the conveyor "
    "belt with the right hand."
)
VALID_NORM_TYPES = {"quantile", "min_max"}
# fluxvla's own RTC (PI0FlowMatching.predict_action's rtc_config['method']):
# 'none' skips it, 'prefix' hard-pins the shared prefix, 'guidance' soft-
# blends it (fluxvla/engines/utils/rtc_guidance.py). FluxVLAPolicy
# (model.py)'s predict_chunk feeds it the exec engine's real
# unconsumed queue tail as prev_actions (see deploy/exec_engine). Only
# takes effect when the loaded inference_model's predict_action actually
# implements RTC: the eager PI0/PI05FlowMatching classes and the Triton
# PI05FlowMatchingRTCInference (prefix only) do; plain
# PI05FlowMatchingInference does not.
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

    # fluxvla RTC guidance; see VALID_RTC_METHODS above. Default 'none'.
    rtc_method: str = "none"
    rtc_execution_horizon: int = 10
    rtc_max_guidance_weight: float = 10.0
    rtc_schedule: str = "linear"  # 'linear', 'exp', 'ones', or 'zeros'

    # CFG / advantage-conditioning (see model.py's apply_cfg_blend docstring).
    # Inert against the currently deployed checkpoint (no advantage
    # training yet); infrastructure for collecting the advantage signal.
    advantage_enabled: bool = False
    cfg_enabled: bool = False
    cfg_scale: float = 2.0
    cfg_scale_joint: float = -1.0  # -1 = use cfg_scale
    cfg_scale_gripper: float = -1.0  # -1 = use cfg_scale
    cfg_cond_advantage_tag: str | None = "positive"
    cfg_uncond_advantage_tag: str | None = None
    arm_dof: int = 7
