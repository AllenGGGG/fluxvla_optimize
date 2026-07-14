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
VALID_RTC_MODES = {"plain", "overlap", "rtc_infer"}


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
    num_inference_steps_override: int
    max_normalized_action_abs: float = 1.25

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
