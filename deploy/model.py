"""FluxVLA PI0.5 loading and CFG/advantage-aware RTC inference.

All preprocessing/postprocessing (image resize+normalize, state
normalization, prompt building, action denormalization) go through
``PIStar06InferenceRunner``'s ``dataset``/``denormalize_action`` objects,
which are built via ``fluxvla.engines``' ``build_dataset_from_cfg``/
``build_transform_from_cfg`` reading ``configs/pi05/pi05_parcel_sort_inference.py``
- the exact same ``fluxvla.transforms`` classes used by training. There is no
hand-rolled normalize/denormalize/prompt-building code in this file.
"""

from __future__ import annotations

import copy
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .config import WorkerConfig
from .pistar06_inference_runner import MODEL_JOINT_DIM, MODEL_TENSOR_DIM


@dataclass
class PredictBundle:
    """Callable policy wrapper with action-horizon metadata."""

    fn: Callable
    n_action_steps: int

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# CFG / advantage-conditioning support.
#
# Ported from fluxvla_deploy main branch's inference.config module. main's
# advantage-mode switch worked by scanning a LeRobot preprocessor for a
# TokenizerProcessorStep and toggling its `advantage_mode` attribute - a
# mechanism specific to LeRobot's processor pipeline that fluxvla.transforms
# has no equivalent of (pi05_parcel_sort.py has no "advantage" concept at
# all yet). Re-expressed here against fluxvla's own prompt convention: the
# cond/uncond distinction is carried entirely in the task-description text
# fed into PreparePromptWithState/ProcessPrompts (an "Advantage:positive" /
# "Advantage:negative" suffix, matching main's DEFAULT_TASK tagging
# convention), and the two forward passes are two independent calls into the
# *same* dataset transform pipeline used for the untagged case - no
# hand-rolled preprocessing either way.
#
# Since the currently deployed checkpoint was never trained on
# advantage-tagged prompts, this is infrastructure for *collecting* the
# advantage signal during rollouts; the blend is inert (or meaningless)
# until a future checkpoint is trained on advantage-tagged data.
# ---------------------------------------------------------------------------

_ADVANTAGE_TAG_RE = re.compile(
    r"\s*Advantage\s*:\s*(positive|negative)\s*", re.IGNORECASE)


def _strip_advantage_text(task: str) -> str:
    """Remove any existing 'Advantage:positive/negative' tag from a task string."""
    if not isinstance(task, str):
        return task
    task = _ADVANTAGE_TAG_RE.sub(" ", task)
    return re.sub(r"\s+", " ", task).strip()


def _with_advantage_tag(task: str, tag: str | None) -> str:
    """Replace any existing advantage tag on ``task`` with ``tag`` (or none)."""
    base = _strip_advantage_text(task)
    if not tag:
        return base
    return f"{base}Advantage:{tag}"


def apply_cfg_blend(
    cond: np.ndarray,
    uncond: np.ndarray,
    *,
    scale: float,
    scale_joint: float,
    scale_gripper: float,
    arm_dof: int,
) -> np.ndarray:
    """Classifier-free-guidance blend of cond/uncond denormalized actions.

    Ported verbatim (pure array math, no LeRobot dependency) from
    fluxvla_deploy main branch's ``inference.config.apply_cfg_blend``.

    pistar06's 28-D body+dual-arm+dual-hand action layout does not match
    this function's built-in left-arm/left-gripper/right-arm/right-gripper
    index split (that assumption fit main's single-arm+gripper robot). With
    the default ``scale_joint=scale_gripper=-1`` ("use scale"), the
    per-segment split reduces algebraically to a uniform
    ``uncond + scale * (cond - uncond)`` regardless of ``arm_dof``, so it is
    safe to keep the function unmodified until a real per-limb scale is
    defined for this robot's joint layout.
    """
    if cond.shape != uncond.shape:
        return uncond + float(scale) * (cond - uncond)

    s_j = float(scale_joint) if scale_joint >= 0.0 else float(scale)
    s_g = float(scale_gripper) if scale_gripper >= 0.0 else float(scale)
    delta = cond - uncond
    out = uncond.copy()

    action_dim = 2 * arm_dof + 2
    if out.shape[-1] < action_dim:
        return uncond + float(scale) * delta

    left_grip = arm_dof
    right_grip = 2 * arm_dof + 1

    out[..., :arm_dof] = uncond[..., :arm_dof] + s_j * delta[..., :arm_dof]
    out[..., left_grip] = uncond[..., left_grip] + s_g * delta[..., left_grip]
    out[..., arm_dof + 1:right_grip] = (
        uncond[..., arm_dof + 1:right_grip]
        + s_j * delta[..., arm_dof + 1:right_grip])
    out[..., right_grip] = uncond[..., right_grip] + s_g * delta[..., right_grip]
    if out.shape[-1] > action_dim:
        out[..., action_dim:] = (
            uncond[..., action_dim:] + float(scale) * delta[..., action_dim:])
    return out


class FluxVLAPolicyWorker:
    """Loads the pi05_parcel_sort checkpoint via PIStar06InferenceRunner and
    runs cond/uncond/CFG-blended inference using fluxvla's training-native
    transform pipeline."""

    def __init__(
        self,
        worker_cfg: WorkerConfig,
        log_info: Callable[[str], None] | None = None,
    ) -> None:
        # Imported here (not module level) to keep ROS protocol inspection
        # importable without mmengine/torch installed, matching
        # BaseInferenceRunner's own lazy-import convention.
        from mmengine import Config

        from .pistar06_inference_runner import PIStar06InferenceRunner

        startup_started = time.perf_counter()
        stage_started = startup_started
        emit = log_info or (lambda _message: None)

        def log_success(message: str) -> None:
            nonlocal stage_started
            now = time.perf_counter()
            emit(
                f"[startup] {message} "
                f"(stage={now - stage_started:.2f}s total={now - startup_started:.2f}s)"
            )
            stage_started = now

        self.cfg = worker_cfg
        if not worker_cfg.model_id:
            raise ValueError("model_id must point to a fine-tuned checkpoint")
        if (
            not np.isfinite(worker_cfg.max_normalized_action_abs)
            or worker_cfg.max_normalized_action_abs <= 0.0
        ):
            raise ValueError("max_normalized_action_abs must be finite and positive")

        checkpoint_path = Path(worker_cfg.model_id).expanduser().resolve()
        inference_config_path = Path(worker_cfg.inference_config).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        if not inference_config_path.is_file():
            raise FileNotFoundError(
                f"Inference config not found: {inference_config_path}")
        log_success(
            "paths validated: "
            f"checkpoint={checkpoint_path} inference_config={inference_config_path}"
        )

        inference_cfg = Config.fromfile(str(inference_config_path))
        if worker_cfg.num_inference_steps_override > 0:
            inference_cfg.inference_model['num_steps'] = (
                worker_cfg.num_inference_steps_override)
        log_success(
            f"inference config loaded: type={inference_cfg.inference_model['type']}"
        )

        self.runner = PIStar06InferenceRunner(
            cfg=inference_cfg,
            ckpt_path=str(checkpoint_path),
            dataset=copy.deepcopy(inference_cfg.dataset),
            denormalize_action=copy.deepcopy(inference_cfg.denormalize_action),
            operator=dict(type='NullOperator'),
            mixed_precision_dtype=worker_cfg.dtype,
            enable_mixed_precision=worker_cfg.device == 'cuda',
        )
        log_success(f"model structure built: type={type(self.runner.vla).__name__}")

        parameter_count = sum(
            parameter.numel() for parameter in self.runner.vla.parameters())
        log_success(f"checkpoint loaded strictly: parameters={parameter_count:,}")

        # BaseInferenceRunner.run_setup() moves the model to CUDA and sets
        # the global seed; it does not depend on ROS being available.
        self.runner.run_setup()
        self.n_action_steps = int(getattr(self.runner.vla, 'n_action_steps', 50))
        log_success(
            "model ready: "
            f"dtype={worker_cfg.dtype} eval=true action_horizon={self.n_action_steps}"
        )

        # Reused (not reimplemented) for RTC prev_actions normalization: the
        # first dataset transform is the same NormalizeStatesAndActions
        # instance, built from the same checkpoint statistics, used for
        # every real preprocessing call below.
        self._normalize_transform = self.runner.dataset.transforms[0]
        self._norm_stats = self.runner.dataset.norm_stats['private']

        emit(
            f"[startup] policy worker ready: "
            f"total={time.perf_counter() - startup_started:.2f}s"
        )

    def _check_action_envelope(self, raw_action: torch.Tensor) -> np.ndarray:
        array = raw_action.detach().float().cpu().numpy()
        if array.ndim == 3:
            if array.shape[0] != 1:
                raise ValueError(f"Expected batch size 1, got action shape {array.shape}")
            array = array[0]
        if array.ndim != 2 or array.shape[1] < MODEL_JOINT_DIM:
            raise ValueError(f"Unexpected action shape: {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError("Model returned NaN or Inf")
        trained_values = array[:, :MODEL_JOINT_DIM]
        max_abs = float(np.max(np.abs(trained_values)))
        if max_abs > self.cfg.max_normalized_action_abs:
            raise ValueError(
                "Model action is outside the normalized training envelope: "
                f"max_abs={max_abs:.3f} > {self.cfg.max_normalized_action_abs:.3f}"
            )
        return array

    def _normalize_prev_actions(self, prev_actions: np.ndarray) -> np.ndarray:
        """Normalize (T, 28) robot-space actions via the training transform."""
        previous = np.asarray(prev_actions, dtype=np.float32)
        if previous.ndim != 2 or previous.shape[1] != MODEL_JOINT_DIM:
            raise ValueError(
                f"prev_actions must have shape (T, {MODEL_JOINT_DIM}), "
                f"got {previous.shape}")
        padded = np.zeros((previous.shape[0], MODEL_TENSOR_DIM), dtype=np.float32)
        padded[:, :MODEL_JOINT_DIM] = previous
        result = self._normalize_transform({
            'states': padded[0],
            'actions': padded,
            'stats': self._norm_stats,
        })
        return result['actions']

    def _forward(self, obs: dict[str, Any], **rtc_kwargs: Any) -> torch.Tensor:
        inputs = self.runner.dataset(obs)
        return self.runner.vla.predict_action(**inputs, **rtc_kwargs)

    def _denormalize(self, raw_action: torch.Tensor) -> np.ndarray:
        self._check_action_envelope(raw_action)
        return self.runner.denormalize_action(
            dict(action=raw_action.detach().float().cpu().numpy()))

    def _predict_denormalized(
        self,
        obs: dict[str, Any],
        **rtc_kwargs: Any,
    ) -> np.ndarray:
        cfg = self.cfg
        if not cfg.advantage_enabled:
            return self._denormalize(self._forward(obs, **rtc_kwargs))

        if not cfg.cfg_enabled:
            cond_obs = dict(obs)
            cond_obs['task_description'] = _with_advantage_tag(
                obs['task_description'], cfg.cfg_cond_advantage_tag)
            return self._denormalize(self._forward(cond_obs, **rtc_kwargs))

        cond_obs = dict(obs)
        cond_obs['task_description'] = _with_advantage_tag(
            obs['task_description'], cfg.cfg_cond_advantage_tag)
        uncond_obs = dict(obs)
        uncond_obs['task_description'] = _with_advantage_tag(
            obs['task_description'], cfg.cfg_uncond_advantage_tag)

        cond = self._denormalize(self._forward(cond_obs, **rtc_kwargs))
        uncond = self._denormalize(self._forward(uncond_obs, **rtc_kwargs))
        return apply_cfg_blend(
            cond,
            uncond,
            scale=cfg.cfg_scale,
            scale_joint=cfg.cfg_scale_joint,
            scale_gripper=cfg.cfg_scale_gripper,
            arm_dof=cfg.arm_dof,
        )

    def predict(self, obs: dict[str, Any]) -> np.ndarray:
        return self._predict_denormalized(obs)

    def predict_rtc(
        self,
        obs: dict[str, Any],
        delay: int,
        prev_actions: np.ndarray | None,
        rtc_cfg: Any,
    ) -> dict[str, np.ndarray]:
        kwargs: dict[str, Any] = {}
        if prev_actions is not None and len(prev_actions):
            previous = self._normalize_prev_actions(prev_actions)
            prefix_len = min(max(delay, 1), len(previous))
            kwargs = {
                "prev_actions": torch.from_numpy(previous).unsqueeze(0).cuda(),
                "prefix_len": prefix_len,
                "rtc_config": {
                    "method": "guidance",
                    "decay_end": min(
                        max(delay, 0) + rtc_cfg.execution_horizon, len(previous)),
                    "schedule": rtc_cfg.prefix_attention_schedule.value,
                    "max_guidance_weight": rtc_cfg.max_guidance_weight,
                    "use_vjp": False,
                },
            }
        return {"chunk": self._predict_denormalized(obs, **kwargs)}


def build_predict_fn(
    worker_cfg: WorkerConfig,
    log_info: Callable[[str], None] | None = None,
) -> PredictBundle:
    worker = FluxVLAPolicyWorker(worker_cfg, log_info=log_info)
    return PredictBundle(fn=worker.predict, n_action_steps=worker.n_action_steps)


def build_rtc_predict_fn(
    worker_cfg: WorkerConfig,
    rtc_cfg: Any,
    log_info: Callable[[str], None] | None = None,
) -> PredictBundle:
    worker = FluxVLAPolicyWorker(worker_cfg, log_info=log_info)

    def predict_fn(
        obs: dict[str, Any],
        delay: int,
        prev_actions: np.ndarray | None,
    ) -> dict[str, np.ndarray]:
        return worker.predict_rtc(obs, delay, prev_actions, rtc_cfg)

    return PredictBundle(fn=predict_fn, n_action_steps=worker.n_action_steps)
