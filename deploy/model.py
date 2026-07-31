"""FluxVLA PI0.5 loading and RTC-aware inference.

All preprocessing/postprocessing (image resize+normalize, state
normalization, prompt building, action denormalization) go through
``self.dataset``/``self.denormalize_action`` (built by
``BaseInferenceRunner.__init__`` via ``fluxvla.engines``'
``build_dataset_from_cfg``/``build_transform_from_cfg`` reading
``configs/pi05/pi05_parcel_sort_none_pytorch_inference.py``) - the exact same
``fluxvla.transforms`` classes used by training. There is no hand-rolled
normalize/denormalize/prompt-building code in this file.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from fluxvla.engines.runners.base_inference_runner import BaseInferenceRunner
from fluxvla.engines.utils.root import OPERATORS

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
    "prefix": {
        "PI05FlowMatchingRTCInference",
        "PI05FlowMatching",
        "PI0FlowMatching",
    },
    "guidance": {
        "PI05FlowMatchingGuidanceInference",
        "PI05FlowMatching",
        "PI0FlowMatching",
    },
}


@dataclass
class PredictBundle:
    """Callable policy wrapper with action-horizon metadata."""

    fn: Callable
    n_action_steps: int

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)


class FluxVLAPolicy(BaseInferenceRunner):
    """Loads the pi05_parcel_sort checkpoint and runs single-pass inference
    using fluxvla's training-native transform pipeline.

    Advantage-conditioning, if any, is a static prompt condition set by the
    inference config's EpisodeMetadataPrompter (dataset.transforms), not a
    runtime CFG toggle -- there used to be a dual cond/uncond forward pass
    blended via apply_cfg_blend here; it has no equivalent now.

    Subclasses BaseInferenceRunner directly (rather than composing an
    instance of it) so ``self.dataset``/``self.vla``/``self.denormalize_action``
    -- built by BaseInferenceRunner.__init__ from the selected inference config
    -- are used as-is, with no wrapper class or extra attribute indirection
    in between. Its ROS1-shaped run()/get_ros_observation()/etc. are never
    called (ros_node.JointInferenceNode owns real ROS I/O).
    """

    def __init__(
        self,
        *,
        model_id: str,
        inference_config: str,
        device: str,
        dtype: str,
        num_inference_steps_override: int = 0,
        norm_stats_path: str | None = None,
        tokenizer_path: str | None = None,
        rtc_method: str = "none",
        rtc_execution_horizon: int = 10,
        rtc_replan_remaining: int = 0,
        chunk_publish_horizon: int = 0,
        rtc_publish_horizon: int = 0,
        rtc_max_guidance_weight: float = 10.0,
        rtc_schedule: str = "exp",
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

        self.device = device
        self.rtc_method = rtc_method
        self.rtc_execution_horizon = rtc_execution_horizon
        self.rtc_replan_remaining = rtc_replan_remaining
        self.chunk_publish_horizon = chunk_publish_horizon
        self.rtc_publish_horizon = rtc_publish_horizon
        self.rtc_max_guidance_weight = rtc_max_guidance_weight
        self.rtc_schedule = rtc_schedule

        checkpoint_path = Path(model_id).expanduser().resolve()
        inference_config_path = Path(inference_config).expanduser().resolve()
        log_success(
            "paths validated: "
            f"checkpoint={checkpoint_path} inference_config={inference_config_path}"
        )

        inference_cfg = Config.fromfile(str(inference_config_path))
        if num_inference_steps_override > 0:
            inference_cfg.inference_model['num_steps'] = (
                num_inference_steps_override)
        if tokenizer_path:
            tokenizer_path_resolved = Path(tokenizer_path).expanduser().resolve()
            if not tokenizer_path_resolved.exists():
                raise FileNotFoundError(
                    f"tokenizer path does not exist: {tokenizer_path_resolved}")
            if not any(
                    transform.get('type') == 'ProcessPrompts'
                    for transform in inference_cfg.dataset['transforms']):
                raise ValueError(
                    "inference config has no ProcessPrompts tokenizer transform"
                )
            # PrivateInferenceDataset gives this explicit path precedence over
            # its normal <checkpoint-root>/tokenizer convention.
            inference_cfg.dataset['tokenizer_path'] = str(tokenizer_path_resolved)
        norm_type = inference_cfg.dataset['transforms'][0]['norm_type']
        if inference_cfg.denormalize_action['norm_type'] != norm_type:
            raise ValueError(
                "dataset and action denormalization norm_type must match"
            )
        log_success(
            f"inference config loaded: type={inference_cfg.inference_model['type']} "
            f"norm_type={norm_type} rtc_method={rtc_method}"
        )

        super().__init__(
            cfg=inference_cfg,
            ckpt_path=str(checkpoint_path),
            dataset=copy.deepcopy(inference_cfg.dataset),
            denormalize_action=copy.deepcopy(inference_cfg.denormalize_action),
            operator=dict(type='NullOperator'),
            mixed_precision_dtype=dtype,
            enable_mixed_precision=device == 'cuda',
        )
        vla_type = type(self.vla).__name__
        log_success(f"model structure built: type={vla_type}")

        if rtc_method != "none":
            supported = _RTC_METHOD_SUPPORTED_TYPES.get(rtc_method, set())
            if vla_type not in supported:
                raise ValueError(
                    f"rtc_method={rtc_method!r} requires an "
                    f"inference_config whose inference_model.type implements "
                    f"RTC {rtc_method!r} (one of {sorted(supported)}), "
                    f"but got type={vla_type!r}, which silently ignores "
                    f"prev_actions/prefix_len/rtc_config."
                )

        parameter_count = sum(
            parameter.numel() for parameter in self.vla.parameters())
        log_success(f"checkpoint loaded strictly: parameters={parameter_count:,}")

        # BaseInferenceRunner.__init__ always derives dataset_statistics.json
        # from ckpt_path's grandparent directory; override with an explicit
        # path here if the caller gave one (e.g. stats live somewhere else).
        if norm_stats_path:
            norm_stats_path_resolved = Path(norm_stats_path).expanduser().resolve()
            with open(norm_stats_path_resolved, 'r', encoding='utf-8') as f:
                norm_stats = json.load(f)
            self.dataset.norm_stats = norm_stats
            self.denormalize_action.norm_stats = norm_stats
            log_success(f"norm_stats overridden: path={norm_stats_path_resolved}")
        # log_success(f"norm_stats loaded: {self.dataset.norm_stats}")

        # run_setup() moves the model to CUDA and sets the global seed; it
        # does not depend on ROS being available.
        self.run_setup()
        self.n_action_steps = int(getattr(self.vla, 'n_action_steps', 50))
        log_success(
            "model ready: "
            f"dtype={dtype} eval=true action_horizon={self.n_action_steps}"
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
        self,
        guidance_prev: np.ndarray | None,
        prefix_len: int | None = None,
    ) -> dict[str, Any]:
        """predict_action's RTC kwargs from the caller-supplied prefix
        context (empty = no RTC: method 'none', or no prefix context yet).

        ``guidance_prev`` is the ground truth for what the new chunk's
        prefix must match -- the exec engine's real unconsumed queue tail,
        not a self-tracked guess (see ``predict_chunk``).
        """
        if self.rtc_method == "none" or guidance_prev is None or not len(
                guidance_prev):
            return {}
        previous = self._normalize_prev_actions(guidance_prev)
        if prefix_len is None:
            prefix_len = self.rtc_execution_horizon
        prefix_len = min(prefix_len, len(previous))
        if prefix_len <= 0:
            return {}
        return {
            "prev_actions": torch.from_numpy(previous).unsqueeze(0).to(self.device),
            "prefix_len": prefix_len,
            "rtc_config": {
                "method": self.rtc_method,
                "decay_end": min(
                    prefix_len + self.rtc_execution_horizon,
                    len(previous),
                ),
                "schedule": self.rtc_schedule,
                "max_guidance_weight": self.rtc_max_guidance_weight,
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
        # Sample the initial noise in fp32, not in the model's compute dtype
        # (bf16). predict_action's denoise loop accumulates the ODE state
        # with in-place ops (x_t += dt * v_t), which preserve x_t's own
        # dtype regardless of v_t's dtype -- confirmed: even under
        # autocast(bfloat16), where v_t is forced to bf16, an fp32 x_t stays
        # fp32 through all 10 steps. Seeding with bf16 noise instead would
        # make x_t itself bf16 for the whole loop, re-quantizing the ODE
        # state every step -- the same class of error the fp32
        # diffusion_noise buffer fix (triton path) exists to avoid, and it
        # would apply here to the eager path too. autocast below still
        # controls the actual matmul/attention compute dtype; only the
        # persistent integration state is kept fp32.
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
                dtype=torch.float32,
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

        ``guidance_prev`` is the old queue segment beginning at the
        observation time and extending ``delay + 2 * rtc_execution_horizon``
        steps deep: the first ``delay`` rows are stale by the time this
        call returns, the next ``rtc_execution_horizon`` rows are the locked
        handoff window, and the final ``rtc_execution_horizon`` rows exist
        only to give the RTC guidance decay region real data to taper
        against (see ``_build_rtc_kwargs``'s ``decay_end``) instead of
        collapsing decay_end onto prefix_len and dropping guidance weight
        straight from 1.0 to 0 with no taper.
        """
        prefix_len = (
            delay + self.rtc_execution_horizon
            if guidance_prev is not None
            else None
        )
        rtc_kwargs = self._build_rtc_kwargs(guidance_prev, prefix_len=prefix_len)

        # Advantage-conditioning is no longer a runtime CFG toggle here: it is
        # a static prompt condition declared in the inference config's
        # dataset.transforms (EpisodeMetadataPrompter(training=False,
        # desired_advantage=...)), applied by self.dataset itself before this
        # method ever sees obs -- see fluxvla/transforms/prompters.py. There
        # is exactly one forward pass; the old dual cond/uncond pass +
        # apply_cfg_blend has no equivalent, matching training's own prompt
        # convention exactly instead of a deploy-only regex-built tag.
        result = self._predict_once(obs, **rtc_kwargs)

        return {"chunk": result}


def build_predict_fn(
    *,
    log_info: Callable[[str], None] | None = None,
    **kwargs: Any,
) -> PredictBundle:
    worker = FluxVLAPolicy(log_info=log_info, **kwargs)
    return PredictBundle(
        fn=worker.predict_chunk, n_action_steps=worker.n_action_steps
    )
