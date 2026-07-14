"""FluxVLA PI0.5 loading and CFG/advantage/RTC-aware inference.

All preprocessing/postprocessing (image resize+normalize, state
normalization, prompt building, action denormalization) go through
``self.dataset``/``self.denormalize_action`` (built by
``BaseInferenceRunner.__init__`` via ``fluxvla.engines``'
``build_dataset_from_cfg``/``build_transform_from_cfg`` reading
``configs/pi05/pi05_parcel_sort_inference.py``) - the exact same
``fluxvla.transforms`` classes used by training. There is no hand-rolled
normalize/denormalize/prompt-building code in this file.
"""

from __future__ import annotations

import copy
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from fluxvla.engines.runners.base_inference_runner import BaseInferenceRunner
from fluxvla.engines.utils.root import OPERATORS

from .config import WorkerConfig
from .utils import MODEL_JOINT_DIM, MODEL_TENSOR_DIM


# ---------------------------------------------------------------------------
# BaseInferenceRunner unconditionally builds a ROS operator
# (fluxvla/engines/runners/base_inference_runner.py:154); pistar06's ROS2
# node (ros_node.JointInferenceNode) owns all ROS I/O itself, so
# FluxVLAPolicy needs an operator that does nothing.
# ---------------------------------------------------------------------------


@OPERATORS.register_module()
class NullOperator:
    """No-op stand-in for BaseInferenceRunner's ROS1 operator abstraction.

    pistar06 is ROS2/rclpy; ``ros_node.JointInferenceNode`` owns all
    subscriptions/publishers directly rather than through this abstraction.
    """


# inference_model.type names whose predict_action actually consumes
# prev_actions/prefix_len/rtc_config for each rtc_method. Anything else
# (e.g. plain PI05FlowMatchingInference) silently drops those kwargs into
# **kwargs and produces un-guided actions -- see FluxVLAPolicy.__init__'s
# startup check below.
_RTC_METHOD_SUPPORTED_TYPES = {
    "prefix": {"PI05FlowMatchingRTCInference", "PI05FlowMatching", "PI0FlowMatching"},
    "guidance": {"PI05FlowMatching", "PI0FlowMatching"},
}


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


class FluxVLAPolicy(BaseInferenceRunner):
    """Loads the pi05_parcel_sort checkpoint and runs cond/uncond/CFG-blended
    inference using fluxvla's training-native transform pipeline.

    Subclasses BaseInferenceRunner directly (rather than composing an
    instance of it) so ``self.dataset``/``self.vla``/``self.denormalize_action``
    -- built by BaseInferenceRunner.__init__ from pi05_parcel_sort_inference.py
    -- are used as-is, with no wrapper class or extra attribute indirection
    in between. Its ROS1-shaped run()/get_ros_observation()/etc. are never
    called (ros_node.JointInferenceNode owns real ROS I/O).
    """

    def __init__(
        self,
        worker_cfg: WorkerConfig,
        log_info: Callable[[str], None] | None = None,
    ) -> None:
        # Imported here (not module level) to keep ROS protocol inspection
        # importable without mmengine/torch installed, matching
        # BaseInferenceRunner's own lazy-import convention.
        from mmengine import Config

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
        self.device = worker_cfg.device
        checkpoint_path = Path(worker_cfg.model_id).expanduser().resolve()
        inference_config_path = Path(worker_cfg.inference_config).expanduser().resolve()
        log_success(
            "paths validated: "
            f"checkpoint={checkpoint_path} inference_config={inference_config_path}"
        )

        inference_cfg = Config.fromfile(str(inference_config_path))
        inference_options = inference_cfg.get('inference_options', {})
        for name, value in inference_options.items():
            if hasattr(worker_cfg, name):
                setattr(worker_cfg, name, value)
        if worker_cfg.num_inference_steps_override > 0:
            inference_cfg.inference_model['num_steps'] = (
                worker_cfg.num_inference_steps_override)
        norm_type = inference_cfg.dataset['transforms'][0]['norm_type']
        if inference_cfg.denormalize_action['norm_type'] != norm_type:
            raise ValueError(
                "dataset and action denormalization norm_type must match"
            )
        log_success(
            f"inference config loaded: type={inference_cfg.inference_model['type']} "
            f"norm_type={norm_type} rtc_method={worker_cfg.rtc_method}"
        )

        super().__init__(
            cfg=inference_cfg,
            ckpt_path=str(checkpoint_path),
            dataset=copy.deepcopy(inference_cfg.dataset),
            denormalize_action=copy.deepcopy(inference_cfg.denormalize_action),
            operator=dict(type='NullOperator'),
            mixed_precision_dtype=worker_cfg.dtype,
            enable_mixed_precision=worker_cfg.device == 'cuda',
        )
        vla_type = type(self.vla).__name__
        log_success(f"model structure built: type={vla_type}")

        if worker_cfg.rtc_method != "none":
            supported = _RTC_METHOD_SUPPORTED_TYPES.get(worker_cfg.rtc_method, set())
            if vla_type not in supported:
                raise ValueError(
                    f"rtc_method={worker_cfg.rtc_method!r} requires an "
                    f"inference_config whose inference_model.type implements "
                    f"RTC {worker_cfg.rtc_method!r} (one of {sorted(supported)}), "
                    f"but got type={vla_type!r}, which silently ignores "
                    f"prev_actions/prefix_len/rtc_config."
                )

        parameter_count = sum(
            parameter.numel() for parameter in self.vla.parameters())
        log_success(f"checkpoint loaded strictly: parameters={parameter_count:,}")

        # BaseInferenceRunner.__init__ always derives dataset_statistics.json
        # from ckpt_path's grandparent directory; override with an explicit
        # path here if the caller gave one (e.g. stats live somewhere else).
        if worker_cfg.norm_stats_path:
            norm_stats_path = Path(worker_cfg.norm_stats_path).expanduser().resolve()
            with open(norm_stats_path, 'r', encoding='utf-8') as f:
                norm_stats = json.load(f)
            self.dataset.norm_stats = norm_stats
            self.denormalize_action.norm_stats = norm_stats
            log_success(f"norm_stats overridden: path={norm_stats_path}")
        # log_success(f"norm_stats loaded: {self.dataset.norm_stats}")

        # run_setup() moves the model to CUDA and sets the global seed; it
        # does not depend on ROS being available.
        self.run_setup()
        self.n_action_steps = int(getattr(self.vla, 'n_action_steps', 50))
        log_success(
            "model ready: "
            f"dtype={worker_cfg.dtype} eval=true action_horizon={self.n_action_steps}"
        )

        # Reused (not reimplemented) for RTC prev_actions normalization: the
        # first dataset transform is the same NormalizeStatesAndActions
        # instance, built from the same checkpoint statistics, used for
        # every real preprocessing call below.
        self._normalize_transform = self.dataset.transforms[0]
        self._norm_stats = self.dataset.norm_stats['private']

        emit(
            f"[startup] policy worker ready: "
            f"total={time.perf_counter() - startup_started:.2f}s"
        )

    def _normalize_prev_actions(self, prev_actions: np.ndarray) -> np.ndarray:
        """Normalize (T, 28) robot-space actions via the training transform."""
        previous = np.asarray(prev_actions, dtype=np.float32)
        padded = np.zeros((previous.shape[0], MODEL_TENSOR_DIM), dtype=np.float32)
        padded[:, :MODEL_JOINT_DIM] = previous
        result = self._normalize_transform({
            'states': padded[0],
            'actions': padded,
            'stats': self._norm_stats,
        })
        return result['actions']

    def _build_rtc_kwargs(
        self, guidance_prev: np.ndarray | None
    ) -> dict[str, Any]:
        """predict_action's RTC kwargs from the caller-supplied prefix
        context (empty = no RTC: method 'none', or no prefix context yet).

        ``guidance_prev`` is the ground truth for what the new chunk's
        prefix must match -- the exec engine's real unconsumed queue tail,
        not a self-tracked guess (see ``predict_chunk``).
        """
        cfg = self.cfg
        if cfg.rtc_method == "none" or guidance_prev is None or not len(
                guidance_prev):
            return {}
        previous = self._normalize_prev_actions(guidance_prev)
        prefix_len = min(cfg.rtc_execution_horizon, len(previous))
        if prefix_len <= 0:
            return {}
        return {
            "prev_actions": torch.from_numpy(previous).unsqueeze(0).to(self.device),
            "prefix_len": prefix_len,
            "rtc_config": {
                "method": cfg.rtc_method,
                "decay_end": min(2 * cfg.rtc_execution_horizon, len(previous)),
                "schedule": cfg.rtc_schedule,
                "max_guidance_weight": cfg.rtc_max_guidance_weight,
                "use_vjp": False,
            },
        }

    def _preprocess(self, obs: dict[str, Any]) -> dict[str, Any]:
        image_keys = tuple(self.dataset.img_keys)
        for key in image_keys:
            if key not in obs or obs[key] is None:
                raise ValueError(f"missing inference image field: {key}")
            image = np.asarray(obs[key])
            if image.size == 0:
                raise ValueError(f"empty inference image field: {key}")
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(
                    f"inference image {key} must be HWC RGB, got {image.shape}"
                )

        if 'qpos' not in obs or obs['qpos'] is None:
            raise ValueError("missing inference state field: qpos")
        qpos = np.asarray(obs['qpos'])
        if qpos.shape != (MODEL_TENSOR_DIM,):
            raise ValueError(
                f"inference qpos must have shape ({MODEL_TENSOR_DIM},), "
                f"got {qpos.shape}"
            )
        if not np.isfinite(qpos).all():
            raise ValueError("inference qpos contains NaN or Inf")

        task = obs.get('task_description')
        if not isinstance(task, str) or not task.strip():
            raise ValueError("missing or empty inference field: task_description")

        inputs = self.dataset(obs)
        for key in ('images', 'img_masks', 'lang_tokens', 'lang_masks', 'states'):
            value = inputs.get(key)
            if value is None:
                raise ValueError(f"preprocessed model field is missing: {key}")
            if not isinstance(value, torch.Tensor) or value.numel() == 0:
                raise ValueError(f"preprocessed model field is empty: {key}")
        return inputs

    def _forward(
        self, inputs: dict[str, Any], **rtc_kwargs: Any
    ) -> torch.Tensor:
        # The shared PI0 sampler creates float32 noise by default. The eager
        # deployment model is converted to bf16, so provide a matching noise
        # tensor explicitly rather than relying on that training-time default.
        action_proj = getattr(self.vla, 'action_in_proj', None)
        action_param = next(action_proj.parameters(), None) if action_proj else None
        action_dtype = action_param.dtype if action_param is not None else None
        if action_dtype is not None:
            states = inputs['states']
            noise = torch.randn(
                states.shape[0],
                int(self.vla.n_action_steps),
                int(self.vla.max_action_dim),
                device=states.device,
                dtype=action_dtype,
            )
            rtc_kwargs = dict(rtc_kwargs)
            rtc_kwargs['noise'] = noise
            with torch.inference_mode():
                if states.device.type == 'cuda':
                    with torch.autocast(
                            device_type='cuda', dtype=action_dtype):
                        return self.vla.predict_action(**inputs, **rtc_kwargs)
                return self.vla.predict_action(**inputs, **rtc_kwargs)
        with torch.inference_mode():
            return self.vla.predict_action(**inputs, **rtc_kwargs)

    def _postprocess(self, raw_action: torch.Tensor) -> np.ndarray:
        actions = np.asarray(self.denormalize_action(
            dict(action=raw_action.detach().float().cpu().numpy())))
        # PI0.5 predicts the 32-D training tensor, while Pistar06 exposes only
        # 28 controllable joints. The final four tensor dimensions are padding
        # (their normalization mask is false) and must never enter the ROS
        # command queue or RTC prefix history.
        if actions.shape[-1] == MODEL_TENSOR_DIM:
            actions = actions[..., :MODEL_JOINT_DIM]
        if actions.shape[-1] != MODEL_JOINT_DIM:
            raise ValueError(
                "inference action has unexpected dimension: "
                f"expected {MODEL_JOINT_DIM} or {MODEL_TENSOR_DIM}, "
                f"got shape {actions.shape}"
            )
        return actions

    def _predict_once(self, obs: dict[str, Any], **rtc_kwargs: Any) -> np.ndarray:
        inputs = self._preprocess(obs)
        raw_action = self._forward(inputs, **rtc_kwargs)
        return self._postprocess(raw_action)

    def predict_chunk(
        self,
        obs: dict[str, Any],
        delay: int,
        guidance_prev: np.ndarray | None,
    ) -> dict[str, Any]:
        """``ChunkScheduler``'s predict_fn protocol: predict_fn(obs,
        delay, prev_chunk) -> {"chunk": ndarray}.

        ``guidance_prev`` is the exec engine's real unconsumed queue tail
        (already delay-aligned by ``aligned_guidance_window``, length up to
        ``delay + rtc_execution_horizon``). Its first ``delay`` rows are
        already stale -- they will have been executed by the time this call
        returns -- so only the remainder anchors the RTC prefix.
        """
        cfg = self.cfg
        anchor = (
            guidance_prev[delay:]
            if guidance_prev is not None and len(guidance_prev) > delay
            else None
        )
        rtc_kwargs = self._build_rtc_kwargs(anchor)

        if not cfg.advantage_enabled:
            result = self._predict_once(obs, **rtc_kwargs)
        else:
            cond_obs = dict(obs)
            cond_obs['task_description'] = _with_advantage_tag(
                obs['task_description'], cfg.cfg_cond_advantage_tag)
            if not cfg.cfg_enabled:
                result = self._predict_once(cond_obs, **rtc_kwargs)
            else:
                uncond_obs = dict(obs)
                uncond_obs['task_description'] = _with_advantage_tag(
                    obs['task_description'], cfg.cfg_uncond_advantage_tag)

                cond = self._predict_once(cond_obs, **rtc_kwargs)
                uncond = self._predict_once(uncond_obs, **rtc_kwargs)
                result = apply_cfg_blend(
                    cond,
                    uncond,
                    scale=cfg.cfg_scale,
                    scale_joint=cfg.cfg_scale_joint,
                    scale_gripper=cfg.cfg_scale_gripper,
                    arm_dof=cfg.arm_dof,
                )

        return {"chunk": result}


def build_predict_fn(
    worker_cfg: WorkerConfig,
    log_info: Callable[[str], None] | None = None,
) -> PredictBundle:
    worker = FluxVLAPolicy(worker_cfg, log_info=log_info)
    return PredictBundle(
        fn=worker.predict_chunk, n_action_steps=worker.n_action_steps
    )
