# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import json
import math
import os
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
import tqdm
from libero.libero import benchmark
from safetensors.torch import load_file
from scipy.fft import dct

from fluxvla.engines.utils import initialize_overwatch
from fluxvla.engines.utils.name_map import str_to_dtype
from fluxvla.engines.utils.torch_utils import set_seed_everywhere
from ..utils.root import RUNNERS

overwatch = initialize_overwatch(__name__)


@RUNNERS.register_module()
class LiberoEvalRunner:
    """Runner for evaluating models using Hugging Face Transformers.
    This class sets up the evaluation environment, loads the model,
    and runs the evaluation process.
    Args:
        cfg (Dict): Configuration dictionary containing model and
            evaluation settings.
        seed (int): Random seed for reproducibility.
        ckpt_path (str): Path to the model checkpoint.
        model_family (str): Model family for evaluation.
        task_suite_name (str): Name of the task suite for evaluation.
        dataset (Dict): Configuration for the dataset to be used in evaluation.
        denormalize_action (Dict): Configuration for denormalizing actions.
        eval_chunk_size (int): Size of the chunks for evaluation.
            Default is 1.
        resize_size (int): Size to which images will be resized.
            Default is 224.
        num_trials_per_task (int): Number of trials per task in the evaluation.
            Default is 50.
        num_steps_wait (int): Number of steps to wait before
            starting evaluation.
            Default is 10.
        mixed_precision_dtype (str): Data type for mixed precision training.
            Default is 'bf16'.
        enable_mixed_precision_training (bool): Whether to enable mixed
            precision training.
            Default is True.
    """

    def __init__(self,
                 cfg: Dict,
                 seed: int,
                 ckpt_path: str,
                 model_family: str,
                 task_suite_name: str,
                 dataset: Dict,
                 denormalize_action: Dict,
                 norm_stats_key: str = None,
                 eval_chunk_size: int = 1,
                 resize_size: int = 224,
                 num_trials_per_task: int = 50,
                 num_steps_wait: int = 10,
                 mixed_precision_dtype: str = 'bf16',
                 eval_task_ids=None,
                 measure_predict_latency: bool = False,
                 decoupled_alternating: bool = False,
                 decoupled_exec_chunk_size: int = None,
                 action_postprocess: Dict = None,
                 action_plot_dir: str = None,
                 action_plot_max_chunks: int = 0,
                 enable_mixed_precision_training: bool = True):
        from fluxvla.engines import (build_dataset_from_cfg,
                                     build_transform_from_cfg,
                                     build_vla_from_cfg)
        self.device_id = overwatch.local_rank()
        if hasattr(cfg, 'inference_model'):
            self.vla = build_vla_from_cfg(cfg.inference_model).eval()
        else:
            self.vla = build_vla_from_cfg(cfg.model).eval()
        # Load checkpoint weights if ckpt_path is provided
        if ckpt_path is not None:
            assert Path.exists(Path(ckpt_path)), \
                f'Checkpoint path {ckpt_path} does not exist!'

            if ckpt_path.endswith('.safetensors'):
                state_dict = load_file(ckpt_path, device='cpu')
            else:
                # A sibling .safetensors is preferred when available because
                # the .pt file also contains the optimizer/scheduler state
                # which is unnecessary for inference and quickly exhausts
                # CPU RAM when loaded on every rank (SIGKILL / exit -9).
                sf_candidate = (
                    ckpt_path[:-len('.pt')] +
                    '.safetensors' if ckpt_path.endswith('.pt') else None)
                if sf_candidate is not None and os.path.exists(sf_candidate):
                    state_dict = load_file(sf_candidate, device='cpu')
                else:
                    # mmap=True avoids copying the whole checkpoint into RAM
                    # on every rank.
                    try:
                        checkpoint = torch.load(
                            ckpt_path, map_location='cpu', mmap=True)
                    except TypeError:
                        checkpoint = torch.load(ckpt_path, map_location='cpu')
                    if isinstance(checkpoint, dict) and 'model' in checkpoint:
                        state_dict = checkpoint['model']
                        # Drop optimizer/scheduler state ASAP to reclaim RAM.
                        checkpoint.pop('optimizer_state_dict', None)
                        checkpoint.pop('scheduler_state_dict', None)
                        checkpoint.pop('optimizer_state_index_to_name', None)
                    else:
                        state_dict = checkpoint
                    del checkpoint
                    gc.collect()
            self.vla.load_state_dict(state_dict, strict=True)
            del state_dict
            gc.collect()
        self.cfg = cfg
        self.seed = seed
        self.ckpt_path = ckpt_path
        self.eval_task_ids = eval_task_ids
        self.measure_predict_latency = measure_predict_latency
        self.decoupled_alternating = decoupled_alternating
        self.decoupled_exec_chunk_size = decoupled_exec_chunk_size
        self.action_postprocess = action_postprocess
        self.action_plot_dir = action_plot_dir
        self.action_plot_max_chunks = int(action_plot_max_chunks or 0)
        self._action_plot_count = 0
        self._action_postprocess_log_count = 0
        self._last_action_postprocess_info = None
        self._action_postprocess_prev_action = None
        data_stat_path = os.path.join(
            Path(self.ckpt_path).resolve().parent.parent,
            'dataset_statistics.json')  # noqa: E501
        assert os.path.exists(data_stat_path), \
            f'Dataset statistics file not found at {data_stat_path}!'
        # Load dataset and denormalization action
        denormalize_action['norm_stats'] = data_stat_path
        self.norm_stats_key = norm_stats_key or f'{task_suite_name}_no_noops'
        dataset['task_suite_name'] = task_suite_name
        dataset['norm_stats_key'] = self.norm_stats_key
        dataset['norm_stats'] = data_stat_path
        self.dataset = build_dataset_from_cfg(dataset)
        self.denormalize_action = build_transform_from_cfg(denormalize_action)
        self.eval_chunk_size = eval_chunk_size
        self.model_family = model_family
        self.task_suite_name = task_suite_name
        self.resize_size = resize_size
        self.num_trials_per_task = num_trials_per_task
        self.num_steps_wait = num_steps_wait
        self.mixed_precision_dtype = str_to_dtype(mixed_precision_dtype)
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.distributed_state = overwatch.distributed_state

        if os.path.isfile(data_stat_path):
            with open(data_stat_path, 'r') as f:
                norm_stats = json.load(f)
            self.vla.norm_stats = norm_stats
        else:
            overwatch.warning(
                'WARNING: No local dataset_statistics.json file found for current checkpoint.\n'  # noqa: E501
                'You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint.'  # noqa: E501
                'Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`.'  # noqa: E501
            )

    @staticmethod
    def _dct_resample(signal_2d, keep, output_steps):
        input_steps = len(signal_2d)
        coeffs = dct(signal_2d.astype(np.float64), axis=0, norm='ortho')
        ks = np.arange(keep, dtype=np.float64)
        n_new = np.linspace(0, input_steps - 1, output_steps)
        basis = np.cos(
            np.pi * (2 * n_new[:, None] + 1) * ks[None, :] /
            (2 * input_steps))
        norms = np.where(ks == 0, 1.0 / np.sqrt(input_steps),
                         np.sqrt(2.0 / input_steps))
        return basis @ (coeffs[:keep] * norms[:, None])

    @staticmethod
    def _haar_dwt(signal_2d):
        coeffs = signal_2d.astype(np.float64).copy()
        length = coeffs.shape[0]
        sqrt2 = np.sqrt(2.0)
        while length > 1:
            half = length // 2
            even = coeffs[:length:2].copy()
            odd = coeffs[1:length:2].copy()
            coeffs[:half] = (even + odd) / sqrt2
            coeffs[half:length] = (even - odd) / sqrt2
            length = half
        return coeffs

    @staticmethod
    def _haar_idwt(coeffs):
        signal = coeffs.astype(np.float64).copy()
        length = 1
        sqrt2 = np.sqrt(2.0)
        while length < signal.shape[0]:
            approx = signal[:length].copy()
            detail = signal[length:2 * length].copy()
            signal[:2 * length:2] = (approx + detail) / sqrt2
            signal[1:2 * length:2] = (approx - detail) / sqrt2
            length *= 2
        return signal

    @classmethod
    def _dwt_resample(cls,
                      signal_2d,
                      keep,
                      output_steps,
                      coefficient_mode='largest'):
        input_steps = len(signal_2d)
        padded_steps = 1 << (input_steps - 1).bit_length()
        padded = np.empty((padded_steps, signal_2d.shape[1]), dtype=np.float64)
        padded[:input_steps] = signal_2d.astype(np.float64)
        if padded_steps > input_steps:
            padded[input_steps:] = signal_2d[-1]

        coeffs = cls._haar_dwt(padded)
        keep = min(max(1, int(keep)), padded_steps)
        filtered = np.zeros_like(coeffs)
        if coefficient_mode == 'lowpass':
            filtered[:keep] = coeffs[:keep]
        elif coefficient_mode == 'largest':
            filtered[0] = coeffs[0]
            if keep > 1:
                count = keep - 1
                if count >= padded_steps - 1:
                    filtered[1:] = coeffs[1:]
                else:
                    top = np.argpartition(
                        np.abs(coeffs[1:]), -count, axis=0)[-count:]
                    dim_idx = np.arange(coeffs.shape[1])[None, :]
                    filtered[top + 1, dim_idx] = coeffs[top + 1, dim_idx]
        else:
            raise ValueError(
                'Unsupported DWT coefficient_mode: '
                f'{coefficient_mode!r}. Expected "largest" or "lowpass".')

        recon = cls._haar_idwt(filtered)[:input_steps]
        old_idx = np.arange(input_steps, dtype=np.float64)
        new_idx = np.linspace(0, input_steps - 1, output_steps)
        out = np.empty((output_steps, signal_2d.shape[1]), dtype=np.float64)
        for dim in range(signal_2d.shape[1]):
            out[:, dim] = np.interp(new_idx, old_idx, recon[:, dim])
        return out

    @staticmethod
    def _interp_path(path, sample_pos):
        old_pos = np.arange(len(path), dtype=np.float64)
        out = np.empty((len(sample_pos), path.shape[1]), dtype=np.float64)
        for dim in range(path.shape[1]):
            out[:, dim] = np.interp(sample_pos, old_pos, path[:, dim])
        return out

    @staticmethod
    def _normalize_nonnegative(values):
        values = np.asarray(values, dtype=np.float64)
        values = np.maximum(values, 0.0)
        scale = float(values.max())
        if scale <= 1e-12:
            return np.zeros_like(values)
        return values / scale

    @classmethod
    def _smooth_path(cls, path, cfg):
        smooth_type = cfg.get('smooth_type', 'none')
        if smooth_type in (None, 'none'):
            return path
        if smooth_type != 'dct':
            raise ValueError(
                f'Unsupported arc_retime smooth_type: {smooth_type!r}')

        keep_ratio = float(cfg.get('smooth_keep_ratio',
                                   cfg.get('keep_ratio', 0.6)))
        if not 0 < keep_ratio <= 1:
            raise ValueError(
                'action_postprocess.smooth_keep_ratio must be in (0, 1], '
                f'got {keep_ratio}.')
        keep = min(max(2, int(len(path) * keep_ratio)), len(path))
        smoothed = cls._dct_resample(path, keep, len(path))

        # Keep the path endpoints fixed so retiming preserves the VLA's
        # predicted total displacement exactly.
        start_offset = path[0] - smoothed[0]
        smoothed = smoothed + start_offset
        end_drift = path[-1] - smoothed[-1]
        alpha = np.linspace(0, 1, len(path))[:, None]
        smoothed = smoothed + end_drift * alpha
        smoothed[0] = path[0]
        smoothed[-1] = path[-1]
        return smoothed

    @classmethod
    def _importance_sample_positions(cls, score_cont, gripper, output_steps,
                                     cfg):
        input_steps = len(score_cont)
        segment_len = np.linalg.norm(score_cont, axis=1)

        accel = np.zeros(input_steps, dtype=np.float64)
        if input_steps > 1:
            accel_norm = np.linalg.norm(np.diff(score_cont, axis=0), axis=1)
            accel[:-1] = np.maximum(accel[:-1], accel_norm)
            accel[1:] = np.maximum(accel[1:], accel_norm)

        turn = np.zeros(input_steps, dtype=np.float64)
        if input_steps > 1:
            norm = np.linalg.norm(score_cont, axis=1)
            valid = (norm[:-1] > 1e-12) & (norm[1:] > 1e-12)
            dots = np.zeros(input_steps - 1, dtype=np.float64)
            dots[valid] = (
                np.sum(
                    score_cont[:-1][valid] * score_cont[1:][valid],
                    axis=1) /
                (norm[:-1][valid] * norm[1:][valid]))
            angles = np.arccos(np.clip(dots, -1.0, 1.0)) / np.pi
            turn[:-1] = np.maximum(turn[:-1], angles)
            turn[1:] = np.maximum(turn[1:], angles)

        gripper_event = np.zeros(input_steps, dtype=np.float64)
        if gripper.size and input_steps > 1:
            grip_delta = np.max(np.abs(np.diff(gripper, axis=0)), axis=1)
            gripper_event[:-1] = np.maximum(gripper_event[:-1], grip_delta)
            gripper_event[1:] = np.maximum(gripper_event[1:], grip_delta)

        score = (
            float(cfg.get('length_weight', 1.0)) *
            cls._normalize_nonnegative(segment_len) +
            float(cfg.get('accel_weight', 0.35)) *
            cls._normalize_nonnegative(accel) +
            float(cfg.get('turn_weight', 0.35)) *
            cls._normalize_nonnegative(turn) +
            float(cfg.get('gripper_weight', 0.5)) *
            cls._normalize_nonnegative(gripper_event))
        score = np.power(
            np.maximum(score, 0.0),
            float(cfg.get('importance_power', 1.0)))
        score += float(cfg.get('min_segment_weight', 0.05))

        cumulative = np.concatenate([[0.0], np.cumsum(score)])
        if cumulative[-1] <= 1e-12:
            return np.linspace(0, input_steps, output_steps + 1)
        target = np.linspace(0, cumulative[-1], output_steps + 1)
        sample_pos = np.interp(
            target, cumulative, np.arange(input_steps + 1))
        sample_pos[0] = 0.0
        sample_pos[-1] = float(input_steps)
        return sample_pos

    @classmethod
    def _arc_retime_once(cls, actions, cont_idx, gripper_idx, output_steps,
                         cfg):
        anchor_steps = int(cfg.get('anchor_first_steps', 1))
        if (anchor_steps > 0 and output_steps > anchor_steps
                and len(actions) > anchor_steps):
            tail_cfg = dict(cfg)
            tail_cfg['anchor_first_steps'] = 0
            tail = cls._arc_retime_once(
                actions[anchor_steps:],
                cont_idx,
                gripper_idx,
                output_steps - anchor_steps,
                tail_cfg)
            return np.concatenate([actions[:anchor_steps], tail], axis=0)

        action_dim = actions.shape[1]
        cont = actions[:, cont_idx].astype(np.float64)
        gripper = (
            actions[:, gripper_idx].astype(np.float64)
            if gripper_idx else np.empty((len(actions), 0), dtype=np.float64))
        metric_idx = cfg.get('metric_idx', [0, 1, 2])
        metric_idx = [idx for idx in metric_idx if idx in cont_idx]
        if metric_idx:
            metric_local_idx = [cont_idx.index(idx) for idx in metric_idx]
            score_cont = cont[:, metric_local_idx]
        else:
            score_cont = cont

        path = np.concatenate(
            [np.zeros((1, cont.shape[1]), dtype=np.float64),
             np.cumsum(cont, axis=0)],
            axis=0)
        path = cls._smooth_path(path, cfg)
        sample_pos = cls._importance_sample_positions(
            score_cont, gripper, output_steps, cfg)
        sampled_path = cls._interp_path(path, sample_pos)
        sampled_path[0] = path[0]
        sampled_path[-1] = path[-1]
        retimed_cont = np.diff(sampled_path, axis=0)

        out = np.zeros((output_steps, action_dim), dtype=actions.dtype)
        for i, dim in enumerate(cont_idx):
            out[:, dim] = retimed_cont[:, i].astype(actions.dtype)

        if gripper_idx:
            gripper_sample = cfg.get('gripper_sample', 'end')
            if gripper_sample == 'mid':
                pick_pos = 0.5 * (sample_pos[:-1] + sample_pos[1:])
                source_idx = np.floor(pick_pos).astype(int)
            elif gripper_sample == 'start':
                source_idx = np.floor(sample_pos[:-1]).astype(int)
            elif gripper_sample == 'end':
                source_idx = np.ceil(sample_pos[1:]).astype(int) - 1
            else:
                raise ValueError(
                    'Unsupported arc_retime gripper_sample: '
                    f'{gripper_sample!r}. Expected "start", "mid", or "end".')
            source_idx = source_idx.clip(0, len(actions) - 1)
            for dim in gripper_idx:
                out[:, dim] = actions[source_idx, dim]
        return out

    @classmethod
    def _arc_retime_delta(cls, actions, cont_idx, gripper_idx, output_steps,
                          cfg):
        return cls._guarded_delta_retime(
            actions,
            cont_idx,
            gripper_idx,
            output_steps,
            cfg,
            cls._arc_retime_once)

    @staticmethod
    def _delta_cumulative_path(actions, cont_idx):
        cont = actions[:, cont_idx].astype(np.float64)
        return np.concatenate(
            [np.zeros((1, cont.shape[1]), dtype=np.float64),
             np.cumsum(cont, axis=0)],
            axis=0)

    @classmethod
    def _resample_cumulative_path(cls, actions, cont_idx, target_len):
        path = cls._delta_cumulative_path(actions, cont_idx)
        sample_pos = np.linspace(0, len(path) - 1, target_len)
        out = cls._interp_path(path, sample_pos)
        out[0] = path[0]
        out[-1] = path[-1]
        return out

    @classmethod
    def _path_fidelity_ok(cls, raw_actions, candidate, cont_idx, cfg):
        if not bool(cfg.get('path_fidelity_guard', True)):
            return True, 0.0, 0.0

        raw_path = cls._delta_cumulative_path(raw_actions, cont_idx)
        candidate_path = cls._resample_cumulative_path(
            candidate, cont_idx, len(raw_path))
        err = np.linalg.norm(raw_path - candidate_path, axis=1)
        raw_step = raw_actions[:, cont_idx].astype(np.float64)
        scale = max(
            float(np.linalg.norm(raw_path[-1] - raw_path[0])),
            float(np.sum(np.linalg.norm(raw_step, axis=1))),
            float(np.max(np.linalg.norm(raw_step, axis=1), initial=0.0)),
            1e-8)
        mean_ratio = float(np.mean(err) / scale)
        max_ratio = float(np.max(err, initial=0.0) / scale)
        mean_limit = float(cfg.get('path_error_mean_ratio', 0.10))
        max_limit = float(cfg.get('path_error_max_ratio', 0.35))
        return (
            mean_ratio <= mean_limit and max_ratio <= max_limit,
            mean_ratio,
            max_ratio)

    @classmethod
    def _first_step_ok(cls, raw_actions, candidate, cont_idx, cfg):
        if not bool(cfg.get('first_step_guard', True)):
            return True, 0.0
        if not len(raw_actions) or not len(candidate):
            return True, 0.0
        raw_first = raw_actions[0, cont_idx].astype(np.float64)
        cand_first = candidate[0, cont_idx].astype(np.float64)
        raw_scale = max(float(np.linalg.norm(raw_first)), 1e-8)
        err_ratio = float(np.linalg.norm(cand_first - raw_first) / raw_scale)
        return (
            err_ratio <= float(cfg.get('first_step_error_ratio', 1.0)),
            err_ratio)

    @classmethod
    def _chunk_continuity_ok(cls, previous_action, raw_actions, candidate,
                             cont_idx, cfg):
        if not bool(cfg.get('chunk_continuity_guard', False)):
            return True, 0.0
        if previous_action is None or not len(candidate):
            return True, 0.0

        previous = previous_action[cont_idx].astype(np.float64)
        cand_first = candidate[0, cont_idx].astype(np.float64)
        raw_cont = raw_actions[:, cont_idx].astype(np.float64)
        raw_norms = np.linalg.norm(raw_cont, axis=1)
        scale = max(
            float(np.linalg.norm(previous)),
            float(np.median(raw_norms)) if len(raw_norms) else 0.0,
            0.25 * float(np.max(raw_norms, initial=0.0)),
            float(cfg.get('continuity_abs_floor', 1e-8)))
        ratio = float(np.linalg.norm(cand_first - previous) / scale)
        return (
            ratio <= float(cfg.get('continuity_error_ratio', 1.75)),
            ratio)

    @staticmethod
    def _candidate_reachability_metrics(raw_actions, candidate, cont_idx,
                                        cfg):
        raw_cont = raw_actions[:, cont_idx].astype(np.float64)
        cand_cont = candidate[:, cont_idx].astype(np.float64)
        abs_floor = float(cfg.get('guard_abs_floor', 1e-8))
        raw_delta = np.linalg.norm(raw_cont, axis=1)
        cand_delta = np.linalg.norm(cand_cont, axis=1)
        delta_base = max(float(np.max(raw_delta, initial=0.0)), abs_floor)
        delta_ratio = float(
            np.max(cand_delta, initial=0.0) / delta_base)

        if len(raw_cont) > 1:
            raw_accel = np.linalg.norm(np.diff(raw_cont, axis=0), axis=1)
        else:
            raw_accel = np.zeros(0, dtype=np.float64)
        if len(cand_cont) > 1:
            cand_accel = np.linalg.norm(np.diff(cand_cont, axis=0), axis=1)
        else:
            cand_accel = np.zeros(0, dtype=np.float64)
        accel_base = max(
            float(np.max(raw_accel, initial=0.0)),
            float(cfg.get('accel_floor_delta_ratio', 0.75)) * delta_base,
            abs_floor)
        accel_ratio = float(
            np.max(cand_accel, initial=0.0) / accel_base)
        return delta_ratio, accel_ratio

    @classmethod
    def _score_race_candidate(cls, raw_actions, candidate, cont_idx,
                              previous_action, path_mean, path_max,
                              first_ratio, continuity_ratio, cfg):
        delta_ratio, accel_ratio = cls._candidate_reachability_metrics(
            raw_actions, candidate, cont_idx, cfg)
        compression = len(raw_actions) / max(len(candidate), 1)
        score = (
            float(cfg.get('race_compression_weight', 0.45)) * compression -
            float(cfg.get('race_path_mean_weight', 8.0)) * path_mean -
            float(cfg.get('race_path_max_weight', 3.0)) * path_max -
            float(cfg.get('race_first_step_weight', 0.5)) * first_ratio -
            float(cfg.get('race_delta_weight', 0.35)) *
            max(0.0, delta_ratio - 1.0) -
            float(cfg.get('race_accel_weight', 0.25)) *
            max(0.0, accel_ratio - 1.0))
        if previous_action is not None:
            score -= (
                float(cfg.get('race_continuity_weight', 0.6)) *
                continuity_ratio)
        return float(score), delta_ratio, accel_ratio

    @classmethod
    def _scheduled_min_output_steps(cls, actions, cont_idx, gripper_idx,
                                    min_steps, input_steps, cfg):
        if not bool(cfg.get('adaptive_min_scheduler', False)):
            return min_steps, {}

        cont = actions[:, cont_idx].astype(np.float64)
        metric_idx = cfg.get('metric_idx', [0, 1, 2])
        metric_idx = [idx for idx in metric_idx if idx in cont_idx]
        if metric_idx:
            metric_local_idx = [cont_idx.index(idx) for idx in metric_idx]
            score_cont = cont[:, metric_local_idx]
        else:
            score_cont = cont

        eps = 1e-8
        step_norm = np.linalg.norm(score_cont, axis=1)
        path_len = float(np.sum(step_norm))
        net_len = float(np.linalg.norm(np.sum(score_cont, axis=0)))
        max_step = float(np.max(step_norm, initial=0.0))

        turn_risk = 0.0
        accel_risk = 0.0
        if len(score_cont) > 1:
            prev = score_cont[:-1]
            nxt = score_cont[1:]
            prev_norm = np.linalg.norm(prev, axis=1)
            nxt_norm = np.linalg.norm(nxt, axis=1)
            valid = (prev_norm > eps) & (nxt_norm > eps)
            if np.any(valid):
                cos = (
                    np.sum(prev[valid] * nxt[valid], axis=1) /
                    (prev_norm[valid] * nxt_norm[valid]))
                angles = np.arccos(np.clip(cos, -1.0, 1.0)) / np.pi
                turn_risk = float(np.max(angles, initial=0.0))
            accel_risk = float(
                np.max(np.linalg.norm(np.diff(score_cont, axis=0), axis=1),
                       initial=0.0) / max(max_step, eps))
            accel_risk = min(accel_risk, 1.0)

        tortuosity_risk = 0.0
        concentration_risk = 0.0
        if path_len > eps:
            tortuosity_risk = float(
                np.clip((path_len - net_len) / path_len, 0.0, 1.0))
            max_frac = max_step / path_len
            concentration_risk = float(
                np.clip((max_frac - 0.35) / 0.40, 0.0, 1.0))

        gripper_risk = 0.0
        if gripper_idx and len(actions) > 1:
            gripper = actions[:, gripper_idx].astype(np.float64)
            gripper_span = float(np.max(gripper) - np.min(gripper))
            gripper_step = float(np.max(np.abs(np.diff(gripper, axis=0))))
            trigger = max(
                float(cfg.get('protect_gripper_change_threshold', 0.02)),
                eps)
            gripper_risk = min(max(gripper_span, gripper_step) / trigger,
                               1.0)

        risk_score = max(
            gripper_risk,
            0.85 * turn_risk,
            0.70 * accel_risk,
            0.75 * tortuosity_risk,
            0.55 * concentration_risk,
        )
        risk_score = float(np.clip(risk_score, 0.0, 1.0))

        thresholds = list(
            cfg.get('scheduler_risk_thresholds',
                    [0.15, 0.35, 0.55, 0.75]))
        steps = list(cfg.get('scheduler_min_steps', [6, 7, 8, 9, 10]))
        if len(steps) != len(thresholds) + 1:
            raise ValueError(
                'adaptive_min_scheduler expects len(scheduler_min_steps) == '
                'len(scheduler_risk_thresholds) + 1, got '
                f'{len(steps)} and {len(thresholds)}.')

        scheduled_min = int(steps[-1])
        for threshold, step in zip(thresholds, steps):
            if risk_score <= float(threshold):
                scheduled_min = int(step)
                break
        scheduled_min = min(max(min_steps, scheduled_min), input_steps)
        return scheduled_min, dict(
            scheduler_risk_score=risk_score,
            scheduler_gripper_risk=float(gripper_risk),
            scheduler_turn_risk=float(turn_risk),
            scheduler_accel_risk=float(accel_risk),
            scheduler_tortuosity_risk=float(tortuosity_risk),
            scheduler_concentration_risk=float(concentration_risk),
            scheduler_min_output_steps=scheduled_min,
        )

    @classmethod
    def _arc_adaptive_delta(cls, actions, cont_idx, gripper_idx, output_steps,
                            cfg):
        input_steps = len(actions)
        min_steps = int(cfg.get('min_output_steps', output_steps))
        min_steps = min(max(1, min_steps), input_steps)
        max_steps = int(cfg.get('max_output_steps', input_steps))
        max_steps = min(max(min_steps, max_steps), input_steps)
        if not bool(cfg.get('fallback_until_feasible', True)):
            max_steps = min_steps

        scheduler_info = {}
        min_steps, scheduler_info = cls._scheduled_min_output_steps(
            actions, cont_idx, gripper_idx, min_steps, input_steps, cfg)
        max_steps = min(max(min_steps, max_steps), input_steps)

        if (bool(cfg.get('protect_gripper_change', False)) and gripper_idx
                and input_steps > 1):
            gripper = actions[:, gripper_idx].astype(np.float64)
            gripper_span = float(np.max(gripper) - np.min(gripper))
            gripper_step = float(np.max(np.abs(np.diff(gripper, axis=0))))
            trigger = max(
                float(cfg.get('protect_gripper_change_threshold', 0.02)),
                1e-8)
            if gripper_span >= trigger or gripper_step >= trigger:
                protected_min = int(
                    cfg.get('protected_min_output_steps', input_steps))
                min_steps = min(max(min_steps, protected_min), input_steps)
                max_steps = min(max(min_steps, max_steps), input_steps)
                scheduler_info['protected_gripper_change'] = True

        base_anchor = int(cfg.get('anchor_first_steps', 0))
        anchor_when_k_ge = int(cfg.get('anchor_first_steps_when_k_ge', 5))
        previous_action = cfg.get('_previous_action', None)
        use_race_search = bool(cfg.get('race_candidate_search', False))
        search_objective = cfg.get('race_search_objective', 'score')
        if (use_race_search and bool(cfg.get('race_exclude_uncompressed',
                                             False))
                and input_steps > 1):
            max_steps = min(max_steps, input_steps - 1)
            min_steps = min(min_steps, max_steps)
        best = None
        best_info = None
        best_score = -np.inf
        fallback = None
        fallback_info = None
        fallback_score = -np.inf
        last_candidate = None
        last_info = None
        for steps in range(min_steps, max_steps + 1):
            step_cfg = dict(cfg)
            step_cfg['fallback_until_feasible'] = False
            step_cfg['allow_zero_motion_passthrough'] = False
            if base_anchor > 0 and steps >= anchor_when_k_ge:
                step_cfg['anchor_first_steps'] = base_anchor
            else:
                step_cfg['anchor_first_steps'] = 0

            if steps >= input_steps:
                candidate = actions.copy()
            else:
                candidate = cls._guarded_delta_retime(
                    actions,
                    cont_idx,
                    gripper_idx,
                    steps,
                    step_cfg,
                    cls._arc_retime_once)
            path_ok, path_mean, path_max = cls._path_fidelity_ok(
                actions, candidate, cont_idx, step_cfg)
            first_ok, first_ratio = cls._first_step_ok(
                actions, candidate, cont_idx, step_cfg)
            continuity_ok, continuity_ratio = cls._chunk_continuity_ok(
                previous_action, actions, candidate, cont_idx, step_cfg)
            race_score, delta_ratio, accel_ratio = cls._score_race_candidate(
                actions, candidate, cont_idx, previous_action, path_mean,
                path_max, first_ratio, continuity_ratio, step_cfg)
            candidate_info = dict(
                selected_steps=len(candidate),
                target_steps=steps,
                path_error_mean_ratio=path_mean,
                path_error_max_ratio=path_max,
                first_step_error_ratio=first_ratio,
                continuity_error_ratio=continuity_ratio,
                race_candidate_score=race_score,
                delta_norm_ratio=delta_ratio,
                accel_norm_ratio=accel_ratio,
                path_fidelity_ok=path_ok,
                first_step_ok=first_ok,
                continuity_ok=continuity_ok,
            )
            candidate_info.update(scheduler_info)
            last_candidate = candidate
            last_info = dict(candidate_info)
            if use_race_search:
                if (search_objective == 'fastest_feasible' and path_ok
                        and first_ok and continuity_ok):
                    candidate_info['race_selected_by_search'] = True
                    candidate_info['race_search_objective'] = (
                        search_objective)
                    cls._pending_action_postprocess_info = candidate_info
                    return candidate
                if search_objective not in ('score', 'fastest_feasible'):
                    raise ValueError(
                        'Unsupported race_search_objective: '
                        f'{search_objective!r}. Expected "score" or '
                        '"fastest_feasible".')
                if race_score > fallback_score:
                    fallback = candidate
                    fallback_info = dict(candidate_info)
                    fallback_score = race_score
                if (path_ok and first_ok and continuity_ok
                        and race_score > best_score):
                    best = candidate
                    best_info = dict(candidate_info)
                    best_score = race_score
                continue
            if path_ok and first_ok and continuity_ok:
                cls._pending_action_postprocess_info = candidate_info
                return candidate

        if use_race_search:
            if best is not None:
                best_info['race_selected_by_search'] = True
                cls._pending_action_postprocess_info = best_info
                return best
            if fallback_info is not None:
                fallback_info['race_fallback_no_feasible'] = True
            cls._pending_action_postprocess_info = fallback_info
            return fallback

        cls._pending_action_postprocess_info = last_info
        return last_candidate

    @staticmethod
    def _rdp_point_line_dist(points, start, end):
        segment = end - start
        seg_len2 = float(np.dot(segment, segment))
        if seg_len2 <= 1e-12:
            return np.linalg.norm(points - start, axis=1)
        t = np.clip(((points - start) @ segment) / seg_len2, 0.0, 1.0)
        projection = start + t[:, None] * segment
        return np.linalg.norm(points - projection, axis=1)

    @classmethod
    def _rdp_recurse(cls, points, epsilon, keep_mask, start, end):
        if end <= start + 1:
            return
        dists = cls._rdp_point_line_dist(
            points[start + 1:end], points[start], points[end])
        if len(dists) == 0:
            return
        local_idx = int(np.argmax(dists))
        if float(dists[local_idx]) > epsilon:
            split = start + 1 + local_idx
            keep_mask[split] = True
            cls._rdp_recurse(points, epsilon, keep_mask, start, split)
            cls._rdp_recurse(points, epsilon, keep_mask, split, end)

    @classmethod
    def _rdp_mask(cls, points, epsilon):
        keep_mask = np.zeros(len(points), dtype=bool)
        keep_mask[0] = True
        keep_mask[-1] = True
        cls._rdp_recurse(points, epsilon, keep_mask, 0, len(points) - 1)
        return keep_mask

    @classmethod
    def _rdp_indices_for_count(cls, points, target_count, max_iter=60):
        del max_iter
        target_count = min(max(2, int(target_count)), len(points))
        if target_count >= len(points):
            return np.arange(len(points), dtype=np.int64)
        if target_count == 2:
            return np.array([0, len(points) - 1], dtype=np.int64)

        scores = np.zeros(len(points), dtype=np.float64)
        scores[0] = np.inf
        scores[-1] = np.inf

        def assign_scores(start, end):
            if end <= start + 1:
                return
            dists = cls._rdp_point_line_dist(
                points[start + 1:end], points[start], points[end])
            if len(dists) == 0:
                return
            local_idx = int(np.argmax(dists))
            score = float(dists[local_idx])
            if score <= 1e-12:
                return
            split = start + 1 + local_idx
            scores[split] = max(scores[split], score)
            assign_scores(start, split)
            assign_scores(split, end)

        assign_scores(0, len(points) - 1)

        interior = np.arange(1, len(points) - 1, dtype=np.int64)
        positive = interior[scores[interior] > 1e-12]
        keep_interior = target_count - 2
        if len(positive) >= keep_interior:
            order = np.argsort(-scores[positive], kind='stable')
            idx = np.sort(
                np.concatenate([
                    np.array([0, len(points) - 1], dtype=np.int64),
                    positive[order[:keep_interior]],
                ]))
            return idx.astype(np.int64)

        idx = np.sort(
            np.concatenate([
                np.array([0, len(points) - 1], dtype=np.int64),
                positive,
            ])).astype(np.int64)
        if len(idx) < target_count:
            missing = target_count - len(idx)
            all_positions = np.arange(len(points), dtype=np.int64)
            candidate_mask = np.ones(len(points), dtype=bool)
            candidate_mask[idx] = False
            candidate_mask[0] = False
            candidate_mask[-1] = False
            candidates = all_positions[candidate_mask]
            if len(candidates) > 0:
                chosen = np.linspace(
                    0, len(candidates) - 1, missing).round().astype(np.int64)
                idx = np.sort(
                    np.concatenate([idx, candidates[chosen]])).astype(np.int64)
            if len(idx) > target_count:
                ranks = np.linspace(
                    0, len(idx) - 1, target_count).round().astype(np.int64)
                idx = idx[ranks]
                idx[0] = 0
                idx[-1] = len(points) - 1

        if len(idx) != target_count:
            idx = np.linspace(
                0, len(points) - 1, target_count).round().astype(np.int64)
            idx[0] = 0
            idx[-1] = len(points) - 1
        return idx

    @classmethod
    def _rdp_delta_once(cls, actions, cont_idx, gripper_idx, output_steps, cfg):
        anchor_steps = int(cfg.get('anchor_first_steps', 1))
        if (anchor_steps > 0 and output_steps > anchor_steps
                and len(actions) > anchor_steps):
            tail_cfg = dict(cfg)
            tail_cfg['anchor_first_steps'] = 0
            tail = cls._rdp_delta_once(
                actions[anchor_steps:],
                cont_idx,
                gripper_idx,
                output_steps - anchor_steps,
                tail_cfg)
            return np.concatenate([actions[:anchor_steps], tail], axis=0)

        action_dim = actions.shape[1]
        cont = actions[:, cont_idx].astype(np.float64)
        path = np.concatenate(
            [np.zeros((1, cont.shape[1]), dtype=np.float64),
             np.cumsum(cont, axis=0)],
            axis=0)
        metric_idx = cfg.get('metric_idx', [0, 1, 2])
        metric_idx = [idx for idx in metric_idx if idx in cont_idx]
        if metric_idx:
            metric_local_idx = [cont_idx.index(idx) for idx in metric_idx]
            metric_path = path[:, metric_local_idx]
        else:
            metric_path = path

        time_weight = float(cfg.get('time_weight', 0.05))
        t_axis = np.linspace(0.0, 1.0, len(metric_path))[:, None]
        scale = np.ptp(metric_path, axis=0)
        scale[scale < 1e-8] = 1.0
        metric_norm = (metric_path - metric_path.min(axis=0)) / scale
        rdp_features = [time_weight * t_axis, metric_norm]
        if gripper_idx and bool(cfg.get('include_gripper_in_metric', True)):
            gripper_path = np.concatenate(
                [actions[:1, gripper_idx],
                 actions[:, gripper_idx]],
                axis=0).astype(np.float64)
            grip_scale = np.ptp(gripper_path, axis=0)
            grip_scale[grip_scale < 1e-8] = 1.0
            grip_norm = (
                gripper_path - gripper_path.min(axis=0)) / grip_scale
            grip_weight = float(cfg.get('gripper_metric_weight', 1.0))
            rdp_features.append(grip_weight * grip_norm)
        rdp_points = np.concatenate(rdp_features, axis=1)

        path_indices = cls._rdp_indices_for_count(
            rdp_points,
            output_steps + 1,
            max_iter=int(cfg.get('rdp_max_iter', 60)))
        if len(path_indices) != output_steps + 1:
            ranks = np.linspace(
                0, len(path_indices) - 1,
                output_steps + 1).round().astype(np.int64)
            path_indices = path_indices[ranks]
            path_indices[0] = 0
            path_indices[-1] = len(path) - 1

        sampled_path = path[path_indices]
        retimed_cont = np.diff(sampled_path, axis=0)
        out = np.zeros((len(retimed_cont), action_dim), dtype=actions.dtype)
        for i, dim in enumerate(cont_idx):
            out[:, dim] = retimed_cont[:, i].astype(actions.dtype)

        if gripper_idx:
            gripper_sample = cfg.get('gripper_sample', 'end')
            if gripper_sample == 'mid':
                path_pos = 0.5 * (path_indices[:-1] + path_indices[1:])
                action_indices = np.floor(path_pos).astype(np.int64)
            elif gripper_sample == 'start':
                action_indices = path_indices[:-1]
            elif gripper_sample == 'end':
                action_indices = path_indices[1:] - 1
            else:
                raise ValueError(
                    'Unsupported rdp_delta gripper_sample: '
                    f'{gripper_sample!r}. Expected "start", "mid", or "end".')
            action_indices = np.clip(action_indices, 0, len(actions) - 1)
            for dim in gripper_idx:
                out[:, dim] = actions[action_indices, dim]
        return out

    @classmethod
    def _guarded_delta_retime(cls, actions, cont_idx, gripper_idx,
                              output_steps, cfg, once_fn):
        input_steps = len(actions)
        cont = actions[:, cont_idx].astype(np.float64)
        if (bool(cfg.get('allow_zero_motion_passthrough', True))
                and float(np.max(np.abs(cont))) <=
                float(cfg.get('zero_motion_eps', 1e-10))):
            return actions
        max_delta_ratio = float(cfg.get('max_delta_ratio', 1.75))
        max_accel_ratio = float(cfg.get('max_accel_ratio', 2.5))
        abs_floor = float(cfg.get('guard_abs_floor', 1e-8))
        accel_floor_delta_ratio = float(
            cfg.get('accel_floor_delta_ratio', 0.75))
        guard_mode = cfg.get('guard_mode', 'per_dim')
        if guard_mode == 'per_dim':
            delta_base = np.maximum(np.max(np.abs(cont), axis=0), abs_floor)
            delta_limit = max_delta_ratio * delta_base
            orig_accel = (
                np.diff(cont, axis=0)
                if input_steps > 1 else np.zeros((0, cont.shape[1])))
            accel_base = (
                np.max(np.abs(orig_accel), axis=0)
                if len(orig_accel) else np.zeros(cont.shape[1]))
            accel_limit = max_accel_ratio * np.maximum(
                accel_base,
                accel_floor_delta_ratio * delta_base)
        elif guard_mode == 'norm':
            orig_delta = np.linalg.norm(cont, axis=1)
            delta_base = max(float(orig_delta.max(initial=0.0)), abs_floor)
            orig_accel = (
                np.linalg.norm(np.diff(cont, axis=0), axis=1)
                if input_steps > 1 else np.zeros(0, dtype=np.float64))
            delta_limit = max_delta_ratio * delta_base
            accel_limit = (
                max_accel_ratio *
                max(float(orig_accel.max(initial=0.0)),
                    accel_floor_delta_ratio * delta_base))
        else:
            raise ValueError(
                f'Unsupported action_postprocess guard_mode: {guard_mode!r}')
        fallback = bool(cfg.get('fallback_until_feasible', True))

        last = None
        max_steps = input_steps if fallback else output_steps
        for steps in range(output_steps, max_steps + 1):
            if steps >= input_steps:
                candidate = actions.copy()
            else:
                candidate = once_fn(actions, cont_idx, gripper_idx, steps, cfg)
            last = candidate
            cand_cont = candidate[:, cont_idx].astype(np.float64)
            if guard_mode == 'per_dim':
                cand_delta = np.max(np.abs(cand_cont), axis=0)
                delta_ok = bool(np.all(cand_delta <= delta_limit))
                if len(cand_cont) > 1:
                    cand_accel = np.max(
                        np.abs(np.diff(cand_cont, axis=0)), axis=0)
                    accel_ok = bool(np.all(cand_accel <= accel_limit))
                else:
                    accel_ok = True
            else:
                cand_delta = np.linalg.norm(cand_cont, axis=1)
                delta_ok = float(cand_delta.max(initial=0.0)) <= delta_limit
                if len(cand_cont) > 1:
                    cand_accel = np.linalg.norm(
                        np.diff(cand_cont, axis=0), axis=1)
                    accel_ok = (
                        float(cand_accel.max(initial=0.0)) <= accel_limit)
                else:
                    accel_ok = True
            if delta_ok and accel_ok:
                return candidate
        return last

    @classmethod
    def _rdp_delta(cls, actions, cont_idx, gripper_idx, output_steps, cfg):
        return cls._guarded_delta_retime(
            actions,
            cont_idx,
            gripper_idx,
            output_steps,
            cfg,
            cls._rdp_delta_once)

    @staticmethod
    def _plot_action_chunk(raw_actions, processed_actions, out_png, title):
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt  # noqa: WPS433

        raw_cont = np.concatenate(
            [np.zeros((1, 6), dtype=np.float64),
             np.cumsum(raw_actions[:, :6], axis=0)],
            axis=0)
        proc_cont = np.concatenate(
            [np.zeros((1, 6), dtype=np.float64),
             np.cumsum(processed_actions[:, :6], axis=0)],
            axis=0)
        raw_xyz = raw_cont[:, :3]
        proc_xyz = proc_cont[:, :3]

        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(2, 4)
        ax3d = fig.add_subplot(gs[:, 0], projection='3d')
        ax3d.plot(
            raw_xyz[:, 0],
            raw_xyz[:, 1],
            raw_xyz[:, 2],
            'o-',
            label=(f'raw {len(raw_actions)} actions '
                   f'({len(raw_xyz)} path pts)'),
            linewidth=1.8)
        ax3d.plot(
            proc_xyz[:, 0],
            proc_xyz[:, 1],
            proc_xyz[:, 2],
            's-',
            label=(f'processed {len(processed_actions)} actions '
                   f'({len(proc_xyz)} path pts)'),
            linewidth=1.8)
        ax3d.scatter(
            raw_xyz[0, 0], raw_xyz[0, 1], raw_xyz[0, 2],
            marker='^', s=80, c='tab:blue')
        ax3d.scatter(
            raw_xyz[-1, 0], raw_xyz[-1, 1], raw_xyz[-1, 2],
            marker='x', s=80, c='tab:blue')
        ax3d.scatter(
            proc_xyz[0, 0], proc_xyz[0, 1], proc_xyz[0, 2],
            marker='^', s=80, c='tab:orange')
        ax3d.scatter(
            proc_xyz[-1, 0], proc_xyz[-1, 1], proc_xyz[-1, 2],
            marker='x', s=80, c='tab:orange')
        ax3d.set_xlabel('cum dx')
        ax3d.set_ylabel('cum dy')
        ax3d.set_zlabel('cum dz')
        ax3d.set_title('XYZ cumulative path')
        ax3d.legend(loc='best')

        labels = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
        for dim in range(6):
            ax = fig.add_subplot(gs[dim // 3, dim % 3 + 1])
            ax.plot(
                np.arange(len(raw_cont)),
                raw_cont[:, dim],
                'o-',
                label='raw',
                linewidth=1.5)
            ax.plot(
                np.linspace(0, len(raw_actions), len(proc_cont)),
                proc_cont[:, dim],
                's-',
                label='processed',
                linewidth=1.5)
            ax.set_title(f'cumulative {labels[dim]}')
            ax.grid(True, alpha=0.25)
            if dim == 0:
                ax.legend(loc='best')

        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(out_png, dpi=180)
        plt.close(fig)

    def _maybe_save_action_plot(self, raw_actions, processed_actions, task_id,
                                trial_id, step_id, seed=None):
        if not self.action_plot_dir or self.action_plot_max_chunks <= 0:
            return
        if self._action_plot_count >= self.action_plot_max_chunks:
            return
        out_dir = Path(self.action_plot_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        seed_tag = f'_seed{seed}' if seed is not None else ''
        stem = (
            f'rank{overwatch.rank()}{seed_tag}_task{task_id}_trial{trial_id}_'
            f'step{step_id}_chunk{self._action_plot_count:03d}')
        out_png = out_dir / f'{stem}.png'
        self._plot_action_chunk(
            raw_actions,
            processed_actions,
            out_png,
            f'{stem}: raw {len(raw_actions)} vs processed '
            f'{len(processed_actions)}')
        np.savez_compressed(
            out_dir / f'{stem}.npz',
            raw_actions=raw_actions,
            processed_actions=processed_actions,
            raw_cumulative_cont=np.concatenate(
                [np.zeros((1, 6), dtype=np.float64),
                 np.cumsum(raw_actions[:, :6], axis=0)],
                axis=0),
            processed_cumulative_cont=np.concatenate(
                [np.zeros((1, 6), dtype=np.float64),
                 np.cumsum(processed_actions[:, :6], axis=0)],
                axis=0),
            action_postprocess=self.action_postprocess,
            action_postprocess_info=self._last_action_postprocess_info,
            raw_steps=len(raw_actions),
            processed_steps=len(processed_actions),
            endpoint_error=np.max(
                np.abs(
                    np.sum(raw_actions[:, :6], axis=0) -
                    np.sum(processed_actions[:, :6], axis=0))),
            first_step_error=(
                np.max(np.abs(raw_actions[0, :6] - processed_actions[0, :6]))
                if len(raw_actions) and len(processed_actions) else np.nan))
        self._action_plot_count += 1

    def _postprocess_action_chunk(self, actions):
        self._last_action_postprocess_info = None
        if not self.action_postprocess:
            return actions

        cfg = dict(self.action_postprocess)
        method = cfg.get('type', 'dct_delta')
        if method not in ('dct_delta', 'dct', 'action_subsample',
                          'delta_subsample', 'dwt_delta',
                          'arc_retime_delta', 'arc_adaptive_delta',
                          'race_lite_delta', 'rdp_delta'):
            raise ValueError(f'Unsupported action_postprocess type: {method}')
        action_space = cfg.get('action_space', 'delta')
        if action_space != 'delta':
            raise ValueError(
                'LiberoEvalRunner action_postprocess currently supports only '
                f"action_space='delta', got {action_space!r}.")

        input_steps, action_dim = actions.shape
        speed = float(cfg.get('speed', 1.5))
        if speed <= 0:
            raise ValueError(
                f'action_postprocess.speed must be positive, got {speed}.')
        output_steps = cfg.get('output_steps', None)
        if output_steps is None:
            output_steps = max(2, int(input_steps / speed))
        else:
            output_steps = int(output_steps)
        output_steps = min(max(1, output_steps), input_steps)
        if output_steps == input_steps and method != 'arc_adaptive_delta':
            return actions

        cont_idx = cfg.get('cont_idx', list(range(action_dim - 1)))
        gripper_idx = cfg.get('gripper_idx', [action_dim - 1])

        source_idx = np.round(
            np.linspace(0, input_steps - 1,
                        output_steps)).astype(int).clip(0, input_steps - 1)
        if method in ('action_subsample', 'delta_subsample'):
            out = actions[source_idx]
            self._last_action_postprocess_info = dict(
                selected_steps=len(out), target_steps=output_steps)
            return out
        if method == 'arc_retime_delta':
            out = self._arc_retime_delta(
                actions, cont_idx, gripper_idx, output_steps, cfg)
            self._last_action_postprocess_info = dict(
                selected_steps=len(out), target_steps=output_steps)
            return out
        if method in ('arc_adaptive_delta', 'race_lite_delta'):
            type(self)._pending_action_postprocess_info = None
            if method == 'race_lite_delta':
                cfg.setdefault('adaptive_min_scheduler', True)
                cfg.setdefault('chunk_continuity_guard', True)
                cfg['_previous_action'] = self._action_postprocess_prev_action
            out = self._arc_adaptive_delta(
                actions, cont_idx, gripper_idx, output_steps, cfg)
            info = getattr(type(self), '_pending_action_postprocess_info',
                           None)
            self._last_action_postprocess_info = info or dict(
                selected_steps=len(out), target_steps=output_steps)
            if len(out):
                self._action_postprocess_prev_action = out[-1].copy()
            return out
        if method == 'rdp_delta':
            out = self._rdp_delta(
                actions, cont_idx, gripper_idx, output_steps, cfg)
            self._last_action_postprocess_info = dict(
                selected_steps=len(out), target_steps=output_steps)
            return out

        out = np.zeros((output_steps, action_dim), dtype=actions.dtype)
        cont = actions[:, cont_idx].astype(np.float64)

        cumulative = np.cumsum(cont, axis=0)

        keep_ratio = float(cfg.get('keep_ratio', 0.4))
        if not 0 < keep_ratio <= 1:
            raise ValueError(
                'action_postprocess.keep_ratio must be in (0, 1], '
                f'got {keep_ratio}.')
        keep = max(2, int(input_steps * keep_ratio))
        keep = min(keep, input_steps)
        if method == 'dwt_delta':
            compressed = self._dwt_resample(
                cumulative,
                keep,
                output_steps,
                coefficient_mode=cfg.get('coefficient_mode', 'largest'))
        else:
            compressed = self._dct_resample(cumulative, keep, output_steps)
        drift = cumulative[-1] - compressed[-1]
        alpha = np.linspace(0, 1, output_steps)[:, None]
        corrected = compressed + drift * alpha

        recon = np.empty_like(corrected)
        recon[0] = corrected[0]
        recon[1:] = np.diff(corrected, axis=0)
        for i, dim in enumerate(cont_idx):
            out[:, dim] = recon[:, i].astype(actions.dtype)

        for dim in gripper_idx:
            out[:, dim] = actions[source_idx, dim]
        self._last_action_postprocess_info = dict(
            selected_steps=len(out), target_steps=output_steps)
        return out

    def run_setup(self):
        """Set up the evaluation environment and model."""
        set_seed_everywhere(self.seed)
        torch.cuda.set_device(device_id := self.device_id)  # noqa: F841
        self.vla.eval()
        self.vla.freeze_vision_backbone = True
        self.vla.freeze_llm_backbone = True
        self.vla.freeze_projector = True
        self.vla.freeze_vlm_backbone = True
        if self.enable_mixed_precision_training:
            self.vla.to(
                device=self.device_id, dtype=self.mixed_precision_dtype)
        else:
            self.vla.cuda(self.device_id)

    def run(self):
        """Run the evaluation process."""
        from fluxvla.engines.utils.eval_utils import (
            get_libero_dummy_action, get_libero_env, save_rollout_video)

        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[self.task_suite_name]()
        num_tasks_in_suite = task_suite.n_tasks

        if self.eval_task_ids is None:
            eval_task_ids = list(range(num_tasks_in_suite))
        else:
            eval_task_ids = list(self.eval_task_ids)
            invalid_task_ids = [
                task_id for task_id in eval_task_ids
                if task_id < 0 or task_id >= num_tasks_in_suite
            ]
            if invalid_task_ids:
                raise ValueError(
                    f'eval_task_ids contains invalid task ids '
                    f'{invalid_task_ids}; valid range is '
                    f'0..{num_tasks_in_suite - 1}')

        init_state_counts = {
            task_id: len(task_suite.get_task_init_states(task_id))
            for task_id in eval_task_ids
        }
        insufficient_init_states = {
            task_id: count
            for task_id, count in init_state_counts.items()
            if self.num_trials_per_task > count
        }
        if insufficient_init_states:
            raise ValueError(
                f'num_trials_per_task={self.num_trials_per_task} exceeds '
                f'available LIBERO initial states per task: '
                f'{insufficient_init_states}. Use at most the listed count '
                'per task, or run multiple seeds instead of requesting more '
                'trials than LIBERO provides.')

        global_episodes = [
            task_id * self.num_trials_per_task + trial_id
            for task_id in eval_task_ids
            for trial_id in range(self.num_trials_per_task)
        ]
        overwatch.info(f'Task suite: {self.task_suite_name}')
        overwatch.info(f'Running evaluation on {len(eval_task_ids)} tasks '
                       f'with {self.num_trials_per_task} trials each.')
        if len(eval_task_ids) != num_tasks_in_suite:
            overwatch.info(f'Evaluating task ids: {eval_task_ids}')
        overwatch.info(f'Using model family: {self.model_family}')
        overwatch.info(f'Using resize size: {self.resize_size}')
        overwatch.info(f'Using evaluation chunk size: {self.eval_chunk_size}')
        if self.action_postprocess:
            overwatch.info(
                f'Using action postprocess: {self.action_postprocess}')
        if self.decoupled_alternating:
            if not hasattr(self.vla, 'predict_action_decoder_only_prefix'):
                raise ValueError(
                    'decoupled_alternating=True requires a model with '
                    'predict_action_decoder_only_prefix(), but '
                    f'{type(self.vla).__name__} does not provide it.')
            exec_chunk_size = (
                self.decoupled_exec_chunk_size
                if self.decoupled_exec_chunk_size is not None else
                getattr(self.vla, 'exec_chunk_size', self.eval_chunk_size))
            overwatch.info(
                'Using decoupled prefix/suffix inference: '
                f'full executes [0:{exec_chunk_size}], decoder-only clamps '
                f'prefix and executes [{exec_chunk_size}:'
                f'{2 * exec_chunk_size}]')
        else:
            exec_chunk_size = self.eval_chunk_size
        overwatch.info(
            f'Using mixed precision dtype: {self.mixed_precision_dtype}')
        rank = overwatch.rank()
        world_size = overwatch.world_size()
        local_episodes = global_episodes[rank::world_size]
        num_local_episodes = math.ceil(len(global_episodes) / world_size)
        data_time = time.strftime('%Y_%m_%d-%H_%M_%S')
        run_id = f'EVAL-{self.task_suite_name}-{self.model_family}-{data_time}'  # noqa: E501
        local_log_filepath = os.path.join(
            Path(self.ckpt_path).resolve().parent.parent, run_id + '.txt')
        log_file = open(local_log_filepath, 'w')
        total_episodes, total_successes = torch.zeros(
            1, device=torch.cuda.current_device()), torch.zeros(
                1, device=torch.cuda.current_device())
        latency_count = torch.zeros(1, device=torch.cuda.current_device())
        latency_sum_ms = torch.zeros(1, device=torch.cuda.current_device())
        latency_action_steps = torch.zeros(
            1, device=torch.cuda.current_device())
        success_action_summaries = []
        unnorm_key = self.task_suite_name
        if rank == 0:
            pbar = tqdm.tqdm(
                total=len(global_episodes),
                desc='Evaluation',
                dynamic_ncols=True)
        else:
            pbar = None
        if self.model_family == 'openvla':
            # In some cases, the key must be manually modified (e.g. after
            # training on a modified version of the dataset
            # with the suffix "_no_noops" in the dataset name)
            candidate_unnorm_key = f'{unnorm_key}_no_noops'
            if (unnorm_key not in self.vla.norm_stats
                    and candidate_unnorm_key in self.vla.norm_stats):
                unnorm_key = candidate_unnorm_key
            assert unnorm_key in self.vla.norm_stats, (
                f'Action un-norm key {unnorm_key} '
                'not found in VLA norm_stats!')
        for id in range(num_local_episodes):
            if id >= len(local_episodes):
                step_tensor = torch.zeros(
                    1, device=torch.cuda.current_device())
            else:
                local_id = local_episodes[id]
                # Get task ID from local episode index
                task_id = local_id // self.num_trials_per_task
                # Get trial ID within the task
                trial_id = local_id % self.num_trials_per_task

                # Log the current task and trial
                overwatch.info(f'Evaluating Task {task_id}, Trial {trial_id}')
                log_file.write(
                    f'Evaluating Task {task_id}, Trial {trial_id}\n')

                # Initialize the task suite and environment
                # Get task
                task = task_suite.get_task(task_id)

                # Get default LIBERO initial states
                initial_states = task_suite.get_task_init_states(task_id)

                # Initialize LIBERO environment and task description
                env, task_description = get_libero_env(task, resolution=256)
                overwatch.info(f'\nTask: {task_description}')
                log_file.write(f'\nTask: {task_description}\n')

                # Reset environment
                env.reset()

                # Set initial states
                obs = env.set_init_state(initial_states[trial_id])
                is_new_episode = True

                # Setup
                t = 0
                replay_images = []
                next_batch = None
                use_decoder_only = False
                previous_full_actions = None
                episode_raw_chunk_steps = 0
                episode_selected_steps = 0
                episode_executed_steps = 0
                episode_selected_sequence = []
                episode_raw_sequence = []
                episode_target_sequence = []
                episode_scheduled_min_sequence = []
                episode_path_error_max = []
                self._action_postprocess_prev_action = None
                if self.task_suite_name == 'libero_spatial':
                    max_steps = 220  # longest training demo has 193 steps
                elif self.task_suite_name == 'libero_object':
                    max_steps = 280  # longest training demo has 254 steps
                elif self.task_suite_name == 'libero_goal':
                    max_steps = 300  # longest training demo has 270 steps
                elif self.task_suite_name == 'libero_10':
                    max_steps = 520  # longest training demo has 505 steps
                elif self.task_suite_name == 'libero_90':
                    max_steps = 400  # longest training demo has 373 steps

                overwatch.info(f'Starting episode {trial_id+1}...')

                log_file.write(f'Starting episode {trial_id+1}...\n')
                while t < max_steps + self.num_steps_wait:
                    # IMPORTANT: Do nothing for the first
                    # few timesteps
                    # because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < self.num_steps_wait:
                        obs, reward, done, info = env.step(
                            get_libero_dummy_action())
                        t += 1
                        continue
                    if next_batch is None:
                        obs['task_description'] = task_description
                        obs['is_new_episode'] = is_new_episode
                        batch, replay_img = self.dataset(obs)
                        if len(replay_images) == 0:
                            replay_images.append(replay_img)
                    else:
                        batch = next_batch
                        next_batch = None
                    is_new_episode = False
                    batch['unnorm_key'] = unnorm_key
                    if self.measure_predict_latency:
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)
                        start_event.record()
                    with torch.autocast(
                            'cuda',
                            dtype=self.mixed_precision_dtype,
                            enabled=self.enable_mixed_precision_training):
                        with torch.no_grad():
                            if (self.decoupled_alternating
                                    and use_decoder_only):
                                if previous_full_actions is None:
                                    raise RuntimeError(
                                        'Decoupled prefix/suffix mode requires '
                                        'a previous full action chunk')
                                actions = (
                                    self.vla
                                    .predict_action_decoder_only_prefix(
                                        previous_full_actions,
                                        exec_chunk_size))
                            else:
                                actions = self.vla.predict_action(**batch)
                    if self.measure_predict_latency:
                        end_event.record()
                        torch.cuda.synchronize()
                        latency_sum_ms += start_event.elapsed_time(end_event)
                        latency_count += 1
                    if len(actions.shape) == 3:
                        actions_tensor = actions.detach()
                        if self.decoupled_alternating and use_decoder_only:
                            action_start = exec_chunk_size
                        else:
                            action_start = 0
                        action_end = min(action_start + exec_chunk_size,
                                         actions.shape[1])
                        if action_end <= action_start:
                            raise ValueError(
                                'Cannot slice actions with '
                                f'action_start={action_start}, '
                                f'exec_chunk_size={exec_chunk_size}, '
                                f'action_shape={tuple(actions.shape)}')
                        action_steps_this_call = action_end - action_start
                        actions = actions[
                            0, action_start:action_end, :].float().cpu().numpy()
                        if (self.decoupled_alternating
                                and not use_decoder_only):
                            previous_full_actions = actions_tensor
                    else:
                        assert len(actions.shape) == 2, \
                            f'Unexpected action shape: {actions.shape}'
                        action_steps_this_call = 1
                        actions = actions[0, None, :].float().cpu().numpy()
                    denormed_actions = []
                    for action in actions:
                        inputs = dict(
                            action=action,
                            task_suite_name=self.task_suite_name,
                            norm_stats_key=self.norm_stats_key,
                        )
                        denormed_actions.append(
                            self.denormalize_action(inputs))
                    denormed_actions = np.asarray(
                        denormed_actions, dtype=np.float32)
                    raw_denormed_actions = denormed_actions.copy()
                    target_post_steps = len(denormed_actions)
                    denormed_actions = self._postprocess_action_chunk(
                        denormed_actions)
                    post_info = self._last_action_postprocess_info or {}
                    selected_steps = len(denormed_actions)
                    episode_raw_chunk_steps += target_post_steps
                    episode_selected_steps += selected_steps
                    episode_raw_sequence.append(target_post_steps)
                    episode_selected_sequence.append(selected_steps)
                    episode_target_sequence.append(
                        int(post_info.get('target_steps', selected_steps)))
                    if 'scheduler_min_output_steps' in post_info:
                        episode_scheduled_min_sequence.append(
                            int(post_info['scheduler_min_output_steps']))
                    if 'path_error_max_ratio' in post_info:
                        episode_path_error_max.append(
                            float(post_info['path_error_max_ratio']))
                    if (self.action_postprocess
                            and self._action_postprocess_log_count < 8):
                        overwatch.info(
                            'action_postprocess steps: '
                            f'{target_post_steps}->{selected_steps}')
                        self._action_postprocess_log_count += 1
                    self._maybe_save_action_plot(
                        raw_denormed_actions,
                        denormed_actions,
                        task_id,
                        trial_id,
                        t,
                        seed=self.seed)
                    action_steps_this_call = len(denormed_actions)
                    if self.measure_predict_latency:
                        latency_action_steps += action_steps_this_call
                    chunk_executed_steps = 0
                    for action_denormed in denormed_actions:
                        obs, reward, done, info = env.step(
                            action_denormed.tolist())
                        chunk_executed_steps += 1
                        obs['task_description'] = task_description
                        batch, replay_img = self.dataset(obs)
                        replay_images.append(replay_img)
                        if done:
                            total_successes += 1
                            next_batch = None
                            break
                        next_batch = batch
                        t += 1
                    episode_executed_steps += chunk_executed_steps
                    if self.decoupled_alternating:
                        use_decoder_only = not use_decoder_only
                    if done:
                        break
                total_episodes += 1
                step_tensor = torch.ones(1, device=torch.cuda.current_device())
                episode_success = (
                    bool(done.item()) if isinstance(done, torch.Tensor)
                    else bool(done))
                compression_speedup = (
                    episode_raw_chunk_steps / episode_selected_steps
                    if episode_selected_steps > 0 else 1.0)
                executed_speedup = (
                    episode_raw_chunk_steps / episode_executed_steps
                    if episode_executed_steps > 0 else 1.0)
                step_line = (
                    '# action_postprocess episode steps: '
                    f'task={task_id} trial={trial_id} '
                    f'success={episode_success} '
                    'raw_chunk_steps_if_uncompressed='
                    f'{episode_raw_chunk_steps} '
                    f'postprocess_selected_steps={episode_selected_steps} '
                    f'env_control_steps_executed={episode_executed_steps} '
                    f'chunk_compression_speedup={compression_speedup:.4f} '
                    f'executed_speedup={executed_speedup:.4f} '
                    'raw_step_sequence='
                    f'{",".join(map(str, episode_raw_sequence))} '
                    'selected_step_sequence='
                    f'{",".join(map(str, episode_selected_sequence))} '
                    'target_step_sequence='
                    f'{",".join(map(str, episode_target_sequence))}')
                if episode_scheduled_min_sequence:
                    step_line += (
                        ' scheduled_min_step_sequence=' +
                        ','.join(map(str, episode_scheduled_min_sequence)))
                if episode_path_error_max:
                    step_line += (
                        ' path_error_max_ratio_sequence=' +
                        ','.join(f'{value:.4f}'
                                 for value in episode_path_error_max))
                log_file.write(step_line + '\n')
                if episode_success:
                    success_action_summaries.append(
                        dict(
                            raw_steps=episode_raw_chunk_steps,
                            selected_steps=episode_selected_steps,
                            executed_steps=episode_executed_steps,
                            compression_speedup=compression_speedup,
                            executed_speedup=executed_speedup,
                            selected_sequence=list(episode_selected_sequence),
                            scheduled_min_sequence=list(
                                episode_scheduled_min_sequence)))
                # Save a replay video of the episode
                save_rollout_video(
                    replay_images,
                    local_id,
                    success=done,
                    task_description=task_description,
                    work_dir=Path(self.ckpt_path).resolve().parent.parent,
                    log_file=log_file)
                env.close()

                # except Exception as e:
                #     print(f'Error during action prediction: {e}')
                #     log_file.write(f'Caught exception: {e}\n')
                #     action = get_libero_dummy_action()
            dist.barrier()
            dist.all_reduce(step_tensor, op=dist.ReduceOp.SUM)
            if rank == 0 and pbar is not None:
                pbar.update(int(step_tensor.item()))

            global_episodes = total_episodes.clone()
            global_successes = total_successes.clone()
            global_latency_count = latency_count.clone()
            global_latency_sum_ms = latency_sum_ms.clone()
            global_latency_action_steps = latency_action_steps.clone()
            dist.all_reduce(global_episodes, op=dist.ReduceOp.SUM)
            dist.all_reduce(global_successes, op=dist.ReduceOp.SUM)
            dist.all_reduce(global_latency_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(global_latency_sum_ms, op=dist.ReduceOp.SUM)
            dist.all_reduce(
                global_latency_action_steps, op=dist.ReduceOp.SUM)
            done = done.item() if isinstance(done, torch.Tensor) else done
            if rank == 0:
                # Log current results
                overwatch.info(
                    f'# episodes completed so far: {int(global_episodes[0])}')
                success_rate = (global_successes[0] / global_episodes[0] * 100)
                success_text = (f'# successes: {int(global_successes[0])} '
                                f'({success_rate:.1f}%)')  # noqa: E231
                overwatch.info(success_text)
                log_file.write(f'Success: {done}\n')
                log_file.write(
                    f'# episodes completed so far: {global_episodes[0]}\n')
                success_log = (f'# successes: {global_successes[0]} '
                               f'({success_rate:.1f}%)\n')  # noqa: E231
                log_file.write(success_log)
                if (self.measure_predict_latency
                        and global_latency_count.item() > 0):
                    mean_ms = (
                        global_latency_sum_ms / global_latency_count).item()
                    call_hz = 1000.0 / mean_ms
                    action_step_hz = (
                        global_latency_action_steps /
                        (global_latency_sum_ms / 1000.0)).item()
                    latency_lines = [
                        f'# predict_action calls: '
                        f'{int(global_latency_count.item())}',
                        f'# predict_action mean ms: {mean_ms:.3f}',
                        f'# predict_action call Hz: {call_hz:.3f}',
                        f'# predict_action action-step Hz: '
                        f'{action_step_hz:.3f}',
                    ]
                    for line in latency_lines:
                        overwatch.info(line)
                        log_file.write(line + '\n')
                log_file.flush()
        if success_action_summaries:
            raw_values = np.asarray(
                [row['raw_steps'] for row in success_action_summaries],
                dtype=np.float64)
            selected_values = np.asarray(
                [row['selected_steps'] for row in success_action_summaries],
                dtype=np.float64)
            executed_values = np.asarray(
                [row['executed_steps'] for row in success_action_summaries],
                dtype=np.float64)
            compression_values = np.asarray(
                [row['compression_speedup']
                 for row in success_action_summaries],
                dtype=np.float64)
            executed_speedup_values = np.asarray(
                [row['executed_speedup'] for row in success_action_summaries],
                dtype=np.float64)
            hist = {}
            scheduled_min_hist = {}
            for row in success_action_summaries:
                for selected in row['selected_sequence']:
                    hist[selected] = hist.get(selected, 0) + 1
                for scheduled_min in row.get('scheduled_min_sequence', []):
                    scheduled_min_hist[scheduled_min] = (
                        scheduled_min_hist.get(scheduled_min, 0) + 1)
            hist_text = ','.join(
                f'{steps}:{hist[steps]}' for steps in sorted(hist))
            scheduled_min_hist_text = ','.join(
                f'{steps}:{scheduled_min_hist[steps]}'
                for steps in sorted(scheduled_min_hist))
            summary_lines = [
                f'# action_postprocess success episodes: '
                f'{len(success_action_summaries)}',
                '# action_postprocess success raw_chunk_steps_if_uncompressed '
                f'mean: {raw_values.mean():.3f}',
                '# action_postprocess success postprocess_selected_steps '
                f'mean: {selected_values.mean():.3f}',
                '# action_postprocess success env_control_steps_executed '
                f'mean: {executed_values.mean():.3f}',
                '# action_postprocess success chunk_compression_speedup '
                f'mean: {compression_values.mean():.4f}',
                '# action_postprocess success executed_speedup '
                f'mean: {executed_speedup_values.mean():.4f}',
                f'# action_postprocess success selected_step_hist: '
                f'{hist_text}',
            ]
            if scheduled_min_hist:
                summary_lines.append(
                    '# action_postprocess success scheduled_min_step_hist: '
                    f'{scheduled_min_hist_text}')
            for line in summary_lines:
                overwatch.info(line)
                log_file.write(line + '\n')
            log_file.flush()
        dist.barrier()
        exit(0)
