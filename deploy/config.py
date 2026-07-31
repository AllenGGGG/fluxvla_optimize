from __future__ import annotations

import os
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


def load_inference_options(inference_config_path: str) -> dict:
    """Read the ``inference_options`` dict from the selected mmengine config.

    This is the single source of truth for RTC/algorithm knobs (rtc_method,
    rtc_execution_horizon, rtc_replan_remaining, chunk_publish_horizon,
    rtc_publish_horizon, rtc_max_guidance_weight, rtc_schedule,
    num_inference_steps_override) -- see VALID_RTC_METHODS above and
    configs/pi05/pi05_parcel_sort_none_pytorch_inference.py's
    inference_options, which every other pi05_parcel_sort_*_inference.py
    inherits via _base_.
    """
    from mmengine import Config

    path = Path(inference_config_path).expanduser().resolve()
    return Config.fromfile(str(path)).get('inference_options', {})
