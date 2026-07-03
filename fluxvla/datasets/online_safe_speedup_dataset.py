# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Online phase-aware action speedup dataset.

This is a deployment-neutral training augmentation for delta-action LIBERO
data. It does not add TempoVLA speed tokens or modify the model. For each
episode it builds a protected, non-uniformly downsampled action sequence:
gripper-change windows and the final placement segment stay dense while
low-risk segments are compressed, with a hard max-gap cap to avoid jumps.
"""

from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from fluxvla.engines import DATASETS
from .parquet_dataset import ParquetDataset


class _SafeStep(NamedTuple):
    source_idx: Optional[int]
    action: np.ndarray
    valid_observation: bool


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = np.maximum(values, 0.0)
    scale = float(np.max(values, initial=0.0))
    if scale <= 1e-12:
        return np.zeros_like(values)
    return values / scale


def _stats(values: np.ndarray) -> Dict[str, List[float]]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    return {
        'min': np.min(values, axis=0).tolist(),
        'max': np.max(values, axis=0).tolist(),
        'mean': np.mean(values, axis=0).tolist(),
        'std': np.std(values, axis=0).tolist(),
        'count': [int(len(values))],
    }


@DATASETS.register_module()
class OnlineSafeSpeedupParquetDataset(ParquetDataset):
    """Online safe speedup dataset for pixshuffle+MLP baseline training."""

    def __init__(
        self,
        *args,
        target_speed: float = 2.0,
        min_keep: int = 12,
        final_protect_ratio: float = 0.22,
        gripper_window: int = 8,
        gripper_threshold: float = 1e-4,
        max_gap: int = 4,
        length_weight: float = 1.0,
        accel_weight: float = 0.5,
        turn_weight: float = 0.5,
        gripper_weight: float = 2.0,
        floor: float = 0.04,
        recompute_action_stats: bool = True,
        **kwargs,
    ) -> None:
        if 'use_delta' in kwargs:
            kwargs.pop('use_delta')
        kwargs['use_delta'] = False
        super().__init__(*args, **kwargs)

        if target_speed <= 0:
            raise ValueError('target_speed must be positive')
        if max_gap < 1:
            raise ValueError('max_gap must be >= 1')

        self.target_speed = float(target_speed)
        self.min_keep = int(min_keep)
        self.final_protect_ratio = float(final_protect_ratio)
        self.gripper_window = int(gripper_window)
        self.gripper_threshold = float(gripper_threshold)
        self.max_gap = int(max_gap)
        self.length_weight = float(length_weight)
        self.accel_weight = float(accel_weight)
        self.turn_weight = float(turn_weight)
        self.gripper_weight = float(gripper_weight)
        self.floor = float(floor)

        self.episode_ranges: Dict[Tuple[int, int], List[int]] = {}
        self.episode_observation_indices: Dict[Tuple[int, int], List[int]] = {}
        self.index_to_observation_rank: Dict[int, int] = {}
        self.episode_retimed: Dict[Tuple[int, int], List[_SafeStep]] = {}
        self.episode_valid_positions: Dict[Tuple[int, int], List[int]] = {}
        self._build_episode_cache()
        if recompute_action_stats:
            self._replace_action_stats()

    def _task_is_valid(self, dataset_idx: int, index: int) -> bool:
        task = self.tasks[dataset_idx][self.dataset[index]['task_index']]['task']
        return task not in ('empty', 'static')

    def _is_train_observation_index(self, index: int) -> bool:
        if index < 0 or index >= len(self.dataset) - 1:
            return False
        dataset_idx = self._get_dataset_index(index)
        next_idx = index + 1
        return (
            self._get_dataset_index(next_idx) == dataset_idx
            and self.dataset[index]['episode_index']
            == self.dataset[next_idx]['episode_index']
            and self._task_is_valid(dataset_idx, next_idx))

    def _resolve_train_observation_index(self, index: int) -> int:
        attempts = 0
        while not self._is_train_observation_index(index):
            index = self._rand_another()
            attempts += 1
            if attempts > 1000:
                raise RuntimeError(
                    'Unable to sample a valid training observation index.')
        return index

    def _episode_actions(self, episode_indices: List[int]) -> np.ndarray:
        return np.stack([
            np.asarray(self.dataset[idx][self.action_key], dtype=np.float32)
            for idx in episode_indices
        ],
                        axis=0)

    def _importance(self, actions: np.ndarray) -> np.ndarray:
        cont = actions[:, :-1].astype(np.float64)
        gripper = actions[:, -1].astype(np.float64)
        n = len(actions)

        step_len = np.linalg.norm(cont, axis=1)

        accel = np.zeros(n, dtype=np.float64)
        if n > 1:
            accel_step = np.linalg.norm(np.diff(cont, axis=0), axis=1)
            accel[:-1] = np.maximum(accel[:-1], accel_step)
            accel[1:] = np.maximum(accel[1:], accel_step)

        turn = np.zeros(n, dtype=np.float64)
        if n > 1:
            prev = cont[:-1]
            nxt = cont[1:]
            prev_norm = np.linalg.norm(prev, axis=1)
            nxt_norm = np.linalg.norm(nxt, axis=1)
            valid = (prev_norm > 1e-8) & (nxt_norm > 1e-8)
            if np.any(valid):
                cos = np.zeros(n - 1, dtype=np.float64)
                cos[valid] = (
                    np.sum(prev[valid] * nxt[valid], axis=1) /
                    (prev_norm[valid] * nxt_norm[valid]))
                angle = np.arccos(np.clip(cos, -1.0, 1.0)) / np.pi
                turn[:-1] = np.maximum(turn[:-1], angle)
                turn[1:] = np.maximum(turn[1:], angle)

        grip = np.zeros(n, dtype=np.float64)
        if n > 1:
            grip_step = np.abs(np.diff(gripper))
            grip[:-1] = np.maximum(grip[:-1], grip_step)
            grip[1:] = np.maximum(grip[1:], grip_step)

        score = (
            self.length_weight * _normalize(step_len) +
            self.accel_weight * _normalize(accel) +
            self.turn_weight * _normalize(turn) +
            self.gripper_weight * _normalize(grip))
        score += self.floor
        score[0] = max(score[0], 1.0)
        score[-1] = max(score[-1], 1.0)
        return score

    def _protected_mask(self, actions: np.ndarray) -> np.ndarray:
        n = len(actions)
        protected = np.zeros(n, dtype=bool)
        protected[0] = True
        protected[-1] = True

        final_start = int(np.floor(n * (1.0 - self.final_protect_ratio)))
        final_start = int(np.clip(final_start, 0, n - 1))
        protected[final_start:] = True

        gripper = actions[:, -1].astype(np.float64)
        if n > 1:
            changes = np.where(
                np.abs(np.diff(gripper)) > self.gripper_threshold)[0]
            for idx in changes:
                lo = max(0, int(idx) - self.gripper_window)
                hi = min(n, int(idx) + self.gripper_window + 2)
                protected[lo:hi] = True
        return protected

    def _sample_keep(self, score: np.ndarray,
                     protected: np.ndarray) -> np.ndarray:
        n = len(score)
        target_count = max(self.min_keep, int(round(n / self.target_speed)))
        target_count = int(np.clip(target_count, 2, n))
        protected = protected.copy()
        protected[0] = True
        protected[-1] = True

        keep = set(int(i) for i in np.where(protected)[0])
        target_count = max(target_count, len(keep))
        if target_count >= n:
            return np.arange(n, dtype=np.int64)

        remaining = target_count - len(keep)
        candidates = np.where(~protected)[0]
        if remaining > 0 and len(candidates) > 0:
            weights = np.maximum(score[candidates], 1e-8)
            cumulative = np.concatenate([[0.0], np.cumsum(weights)])
            targets = np.linspace(0.0, cumulative[-1], remaining + 2)[1:-1]
            picked_pos = np.searchsorted(
                cumulative, targets, side='right') - 1
            picked_pos = np.clip(picked_pos, 0, len(candidates) - 1)
            keep.update(int(i) for i in candidates[picked_pos])

        if len(keep) < target_count and len(candidates) > 0:
            missing = target_count - len(keep)
            unused = np.array([i for i in candidates if int(i) not in keep],
                              dtype=np.int64)
            if len(unused):
                ranked = unused[np.argsort(-score[unused], kind='stable')]
                keep.update(int(i) for i in ranked[:missing])

        return np.array(sorted(keep), dtype=np.int64)

    def _enforce_max_gap(self, keep: np.ndarray, n: int) -> np.ndarray:
        if self.max_gap <= 1:
            return np.arange(n, dtype=np.int64)
        keep = np.asarray(sorted(set(int(i) for i in keep)), dtype=np.int64)
        expanded = []
        for start, end in zip(keep[:-1], keep[1:]):
            expanded.append(int(start))
            if int(end - start) > self.max_gap:
                extra = np.arange(
                    int(start) + self.max_gap, int(end), self.max_gap,
                    dtype=np.int64)
                expanded.extend(int(i) for i in extra)
        expanded.append(int(keep[-1]))
        return np.array(sorted(set(expanded)), dtype=np.int64)

    def _retime_episode(self, episode_indices: List[int]) -> List[_SafeStep]:
        actions = self._episode_actions(episode_indices)
        score = self._importance(actions)
        protected = self._protected_mask(actions)
        keep = self._sample_keep(score, protected)
        keep = self._enforce_max_gap(keep, len(actions))

        steps = []
        for out_i, src_pos in enumerate(keep):
            if out_i < len(keep) - 1:
                next_pos = int(keep[out_i + 1])
                block = actions[int(src_pos):next_pos]
                action = np.zeros_like(actions[0])
                action[:-1] = np.sum(block[:, :-1], axis=0)
                action[-1] = block[-1, -1]
            else:
                action = np.zeros_like(actions[0])
                action[-1] = actions[int(src_pos), -1]
            steps.append(
                _SafeStep(episode_indices[int(src_pos)],
                          action.astype(np.float32), True))
        return steps

    def _build_episode_cache(self) -> None:
        for i in range(len(self.dataset)):
            dataset_idx = self._get_dataset_index(i)
            episode_idx = self.dataset[i]['episode_index']
            key = (dataset_idx, episode_idx)
            self.episode_ranges.setdefault(key, []).append(i)

        for key, episode_indices in self.episode_ranges.items():
            observations = [
                idx for idx in episode_indices
                if self._is_train_observation_index(idx)
            ]
            self.episode_observation_indices[key] = observations
            for rank, idx in enumerate(observations):
                self.index_to_observation_rank[idx] = rank

            retimed = self._retime_episode(episode_indices)
            self.episode_retimed[key] = retimed
            valid_positions = [
                pos for pos, step in enumerate(retimed)
                if step.valid_observation and step.source_idx is not None
                and self._is_train_observation_index(step.source_idx)
            ]
            self.episode_valid_positions[key] = valid_positions

    def _replace_action_stats(self) -> None:
        grouped_actions: Dict[int, List[np.ndarray]] = {}
        for key, steps in self.episode_retimed.items():
            dataset_idx, episode_idx = key
            if not steps:
                continue
            first_src = steps[0].source_idx
            if first_src is None:
                continue
            task_idx = int(self.dataset[first_src]['task_index'])
            grouped_actions.setdefault(task_idx, [])
            grouped_actions[task_idx].extend([step.action for step in steps])

        action_stats = {
            task_idx: _stats(np.stack(actions, axis=0))
            for task_idx, actions in grouped_actions.items()
            if actions
        }
        for stat in self.stats:
            task_idx = int(stat['stats']['task_index']['min'][0])
            if task_idx in action_stats:
                stat['stats']['action'] = action_stats[task_idx]

    def _select_retimed_observation(self, index: int,
                                    key: Tuple[int, int]) -> int:
        valid_positions = self.episode_valid_positions[key]
        if not valid_positions:
            raise RuntimeError(f'Retimed episode {key} has no observations.')

        base_observations = self.episode_observation_indices[key]
        rank = self.index_to_observation_rank.get(index)
        if rank is None or not base_observations:
            return valid_positions[np.random.randint(0, len(valid_positions))]

        target_rank = int(rank * len(valid_positions) / len(base_observations))
        target_rank = min(target_rank, len(valid_positions) - 1)
        return valid_positions[target_rank]

    def __getitem__(self, index, dataset_statistics):
        index = self._resolve_train_observation_index(index)
        data = self.dataset[index]
        dataset_idx = self._get_dataset_index(index)
        key = (dataset_idx, data['episode_index'])

        retimed = self.episode_retimed[key]
        best_idx = self._select_retimed_observation(index, key)
        chosen_step = retimed[best_idx]
        assert chosen_step.source_idx is not None
        chosen_src_idx = chosen_step.source_idx

        actions = []
        action_masks = []
        for i in range(self.action_window_size):
            action_idx = best_idx + self.window_start_idx + i
            if action_idx < len(retimed):
                actions.append(retimed[action_idx].action)
                action_masks.append(1.0)
            else:
                actions.append(actions[-1] if actions else np.zeros_like(
                    retimed[0].action))
                action_masks.append(0.0)

        data = dict(self.dataset[chosen_src_idx])
        data['actions'] = np.stack(actions, axis=0).astype(np.float32)
        data['action_masks'] = np.array(action_masks, dtype=np.float32)

        if self.frame_window_size > 1:
            frame_timestamps = [data['timestamp']]
            frame_masks = [1]
            for fi in range(1, self.frame_window_size):
                future_idx = chosen_src_idx + fi
                if (future_idx < len(self.dataset)
                        and self.dataset[future_idx]['episode_index']
                        == data['episode_index']
                        and self._get_dataset_index(future_idx)
                        == dataset_idx):
                    frame_timestamps.append(
                        self.dataset[future_idx]['timestamp'])
                    frame_masks.append(1)
                else:
                    frame_timestamps.append(frame_timestamps[-1])
                    frame_masks.append(0)
            data['frame_timestamps'] = frame_timestamps
            data['frame_masks'] = np.array(frame_masks, dtype=np.float32)

        visual_delay = self._sample_visual_delay()
        if visual_delay > 0:
            frame_count = len(data.get('frame_timestamps',
                                       [data['timestamp']]))
            visual_timestamps, visual_delay = self._visual_timestamps_for_delay(
                chosen_src_idx, visual_delay, dataset_idx, frame_count)
            data['visual_frame_timestamps'] = visual_timestamps
            data['visual_delay_steps'] = np.array(visual_delay, dtype=np.int64)
        elif self.visual_delay_max_steps > 0:
            data['visual_delay_steps'] = np.array(0, dtype=np.int64)

        data['info'] = self.info[dataset_idx]
        data['stats'] = dataset_statistics[self.statistic_name]
        if self.expose_index:
            data['index'] = np.array(chosen_src_idx, dtype=np.int64)
        data['task_description'] = self.tasks[dataset_idx][
            data['task_index']]['task']
        data['data_root'] = self.data_root_path[dataset_idx]

        for transform in self.transforms:
            data = transform(data)
        return data
