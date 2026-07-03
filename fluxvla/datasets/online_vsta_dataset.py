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

"""Online Variable-Speed Trajectory Augmentation (VSTA) for TempoVLA.

Implements VSTA algorithm from TempoVLA paper (Section 3.2, Algorithm 1).
VSTA operates on linearly-composable action representations. For LIBERO, the
stored actions are already in delta/velocity form and can be directly accumulated.
"""

import math
from fractions import Fraction
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from fluxvla.engines import DATASETS
from .parquet_dataset import ParquetDataset


class _RetimedStep(NamedTuple):
    """One action step in the retimed trajectory.

    source_idx is only meaningful when valid_observation is True. Synthetic
    split steps do not have a matching image frame and cannot start a sample.
    """

    source_idx: Optional[int]
    action: np.ndarray
    valid_observation: bool


@DATASETS.register_module()
class OnlineVSTATempoParquetDataset(ParquetDataset):
    """Online TempoVLA/VSTA dataset implementing Algorithm 1.

    IMPORTANT: This dataset assumes actions are in linearly-composable form
    (e.g., delta positions, velocities). For LIBERO, actions are stored as
    deltas and can be directly accumulated. The parent dataset should have
    use_delta=False to provide the original stored actions.

    Args:
        tempo_speeds: Candidate speeds to sample from
        tempo_speed_key: Key to store sampled speed in data dict
        keep_last_action_dim_nearest: Keep last action dim (gripper) as nearest
        random_phase: Enable random chunk-start offset sampling per segment
        motion_mode_threshold: Threshold for translation/rotation magnitude
        direction_reversal_threshold_deg: Cosine threshold for direction reversal
        gripper_threshold: Threshold for gripper state change detection
    """

    def __init__(
        self,
        *args,
        tempo_speeds: Sequence[float] = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0),
        tempo_speed_key: str = 'tempo_speed',
        keep_last_action_dim_nearest: bool = True,
        random_phase: bool = True,
        motion_mode_threshold: float = 1e-4,
        direction_reversal_threshold_deg: float = 90.0,
        gripper_threshold: float = 0.5,
        **kwargs
    ) -> None:
        # Parent dataset should provide original actions (not deltas of deltas)
        if 'use_delta' in kwargs:
            kwargs.pop('use_delta')
        kwargs['use_delta'] = False

        super().__init__(*args, **kwargs)

        if not tempo_speeds:
            raise ValueError('tempo_speeds must contain at least one value')
        self.tempo_speeds = tuple(float(s) for s in tempo_speeds)
        if any(s <= 0 for s in self.tempo_speeds):
            raise ValueError(f'All tempo_speeds must be positive: {tempo_speeds}')

        self.tempo_speed_key = tempo_speed_key
        self.keep_last_action_dim_nearest = keep_last_action_dim_nearest
        self.random_phase = random_phase
        self.motion_mode_threshold = motion_mode_threshold
        self.direction_reversal_cos_threshold = math.cos(
            math.radians(direction_reversal_threshold_deg))
        self.gripper_threshold = gripper_threshold

        # Build episode index cache
        self._build_episode_cache()

    def _build_episode_cache(self) -> None:
        """Build episode index ranges and segmentation cache."""
        self.episode_ranges: Dict[Tuple[int, int], List[int]] = {}
        self.episode_observation_indices: Dict[Tuple[int, int], List[int]] = {}
        self.episode_segments: Dict[Tuple[int, int], List[List[int]]] = {}
        self.index_to_observation_rank: Dict[int, int] = {}

        for i in range(len(self.dataset)):
            dataset_idx = self._get_dataset_index(i)
            episode_idx = self.dataset[i]['episode_index']
            key = (dataset_idx, episode_idx)
            if key not in self.episode_ranges:
                self.episode_ranges[key] = []
            self.episode_ranges[key].append(i)

        # Pre-compute segmentation for all episodes
        for key, episode_indices in self.episode_ranges.items():
            observation_indices = [
                idx for idx in episode_indices
                if self._is_train_observation_index(idx)
            ]
            self.episode_observation_indices[key] = observation_indices
            for rank, idx in enumerate(observation_indices):
                self.index_to_observation_rank[idx] = rank
            self.episode_segments[key] = self._segment_episode(episode_indices)

    def _task_is_valid(self, dataset_idx: int, index: int) -> bool:
        task = self.tasks[dataset_idx][self.dataset[index]['task_index']]['task']
        return task not in ('empty', 'static')

    def _is_train_observation_index(self, index: int) -> bool:
        """Match ParquetDataset's train-observation validity rule.

        The baseline avoids the final frame of an episode because it cannot
        provide the same forward-looking supervision context. Keep that
        behavior so VSTA does not silently change the training distribution.
        """
        if index < 0 or index >= len(self.dataset) - 1:
            return False

        dataset_idx = self._get_dataset_index(index)
        next_idx = index + 1
        if self._get_dataset_index(next_idx) != dataset_idx:
            return False
        if self.dataset[index]['episode_index'] != self.dataset[next_idx][
                'episode_index']:
            return False
        if not self._task_is_valid(dataset_idx, next_idx):
            return False
        return True

    def _resolve_train_observation_index(self, index: int) -> int:
        """Resample until the raw index can start a baseline-style sample."""
        attempts = 0
        while not self._is_train_observation_index(index):
            index = self._rand_another()
            attempts += 1
            if attempts > 1000:
                raise RuntimeError(
                    'Unable to sample a valid training observation index.')
        return index

    def _sample_speed(self) -> float:
        """Randomly sample a target speed."""
        return float(np.random.choice(self.tempo_speeds))

    def _speed_to_coprime(self, speed: float) -> Tuple[int, int]:
        """Convert speed to coprime integers q, p where speed = q/p."""
        frac = Fraction(speed).limit_denominator(100)
        return frac.numerator, frac.denominator

    def _get_motion_mode(self, action: np.ndarray) -> str:
        """Classify motion mode from action."""
        translation_mag = np.linalg.norm(action[:3])
        rotation_mag = np.linalg.norm(action[3:6])

        trans_moving = translation_mag > self.motion_mode_threshold
        rot_moving = rotation_mag > self.motion_mode_threshold

        if trans_moving and rot_moving:
            return 'translate-and-rotate'
        elif trans_moving:
            return 'translate'
        elif rot_moving:
            return 'rotate'
        else:
            return 'still'

    def _segment_episode(
        self,
        episode_indices: List[int]
    ) -> List[List[int]]:
        """Motion-consistent segmentation (Algorithm 1, line 2)."""
        if len(episode_indices) <= 1:
            return [episode_indices]

        segments = []
        current_segment = [episode_indices[0]]

        prev_row = self.dataset[episode_indices[0]]
        prev_action = np.array(prev_row[self.action_key], dtype=np.float32)
        prev_gripper = prev_action[-1]
        prev_mode = self._get_motion_mode(prev_action)

        for idx in episode_indices[1:]:
            row = self.dataset[idx]
            action = np.array(row[self.action_key], dtype=np.float32)
            gripper = action[-1]

            should_split = False

            # 1. Gripper event (hard boundary)
            if abs(gripper - prev_gripper) > self.gripper_threshold:
                should_split = True

            # Always compute current mode for comparison and update
            current_mode = self._get_motion_mode(action)

            # 2. Motion mode change
            if not should_split:
                if current_mode != prev_mode:
                    should_split = True

            # 3. Direction reversal
            if not should_split:
                trans = action[:3]
                prev_trans = prev_action[:3]
                trans_mag = np.linalg.norm(trans)
                prev_trans_mag = np.linalg.norm(prev_trans)

                if trans_mag > self.motion_mode_threshold and prev_trans_mag > self.motion_mode_threshold:
                    cos_sim = np.dot(trans, prev_trans) / (trans_mag * prev_trans_mag)
                    if cos_sim < self.direction_reversal_cos_threshold:
                        should_split = True

                if not should_split:
                    rot = action[3:6]
                    prev_rot = prev_action[3:6]
                    rot_mag = np.linalg.norm(rot)
                    prev_rot_mag = np.linalg.norm(prev_rot)

                    if rot_mag > self.motion_mode_threshold and prev_rot_mag > self.motion_mode_threshold:
                        cos_sim = np.dot(rot, prev_rot) / (rot_mag * prev_rot_mag)
                        if cos_sim < self.direction_reversal_cos_threshold:
                            should_split = True

            if should_split:
                segments.append(current_segment)
                current_segment = [idx]
            else:
                current_segment.append(idx)

            # Always update prev state
            prev_action = action
            prev_gripper = gripper
            prev_mode = current_mode

        if current_segment:
            segments.append(current_segment)

        return segments

    def _retime_segment(
        self,
        segment_indices: List[int],
        q: int,
        p: int
    ) -> List[_RetimedStep]:
        """Chunk-level speed transform (Algorithm 1, lines 7-11).

        Returns a retimed action sequence. Only real source frames are marked as
        valid observations; synthetic split steps are action-only.
        """
        if len(segment_indices) == 0:
            return []

        # Sample chunk-start offset (Algorithm 1, line 5)
        r = np.random.randint(0, q) if (self.random_phase and q > 1) else 0

        result = []

        # Leading passthrough: frames [0, r) (Algorithm 1, line 6)
        for i in range(min(r, len(segment_indices))):
            idx = segment_indices[i]
            action = np.array(self.dataset[idx][self.action_key], dtype=np.float32)
            result.append(_RetimedStep(idx, action, True))

        # Process chunks starting at r (Algorithm 1, lines 7-11)
        chunk_start = r
        while chunk_start + q <= len(segment_indices):
            chunk_indices = segment_indices[chunk_start:chunk_start + q]

            # Accumulate total motion over q frames (Algorithm 1, line 8)
            # Actions are already in delta form, so we directly sum them
            total_motion = np.zeros_like(
                np.array(self.dataset[chunk_indices[0]][self.action_key], dtype=np.float32))

            for idx in chunk_indices:
                action = np.array(self.dataset[idx][self.action_key], dtype=np.float32)
                # Accumulate all dims except gripper
                total_motion[:-1] += action[:-1]

            # Get gripper from chunk last frame
            gripper_value = np.array(
                self.dataset[chunk_indices[-1]][self.action_key], dtype=np.float32)[-1]

            # Re-split into p equal steps (Algorithm 1, line 9)
            step_motion = total_motion[:-1] / p if p > 0 else total_motion[:-1]

            for step_idx in range(p):
                new_action = np.zeros_like(total_motion)
                new_action[:-1] = step_motion

                # Gripper: use nearest or copy from chunk last
                if self.keep_last_action_dim_nearest:
                    new_action[-1] = gripper_value
                else:
                    # Linear interpolation of gripper
                    gripper_first = np.array(
                        self.dataset[chunk_indices[0]][self.action_key], dtype=np.float32)[-1]
                    gripper_last = gripper_value
                    alpha = (step_idx + 0.5) / p
                    new_action[-1] = (1 - alpha) * gripper_first + alpha * gripper_last

                # Only the first split step has a real observation. Later split
                # steps are supervision targets inside the action window.
                source_idx = chunk_indices[0] if step_idx == 0 else None
                result.append(
                    _RetimedStep(source_idx, new_action, step_idx == 0))

            chunk_start += q

        # Trailing passthrough: leftover < q frames (Algorithm 1, line 12)
        for i in range(chunk_start, len(segment_indices)):
            idx = segment_indices[i]
            action = np.array(self.dataset[idx][self.action_key], dtype=np.float32)
            result.append(_RetimedStep(idx, action, True))

        return result

    def _select_retimed_observation(
        self,
        index: int,
        key: Tuple[int, int],
        all_retimed: List[_RetimedStep],
    ) -> int:
        """Map a baseline-valid raw sample index to a valid retimed sample.

        DistributedRepeatingDataset still samples over the original dataset
        length. Mapping by rank keeps valid retimed observation starts roughly
        uniformly sampled and avoids the nearest-neighbor bias caused by
        duplicated chunk-start source indices.
        """
        valid_positions = [
            pos for pos, step in enumerate(all_retimed)
            if step.valid_observation and step.source_idx is not None
            and self._is_train_observation_index(step.source_idx)
        ]
        if not valid_positions:
            raise RuntimeError('Retimed episode has no valid observations.')

        base_observations = self.episode_observation_indices[key]
        rank = self.index_to_observation_rank.get(index)
        if rank is None or not base_observations:
            return valid_positions[np.random.randint(0, len(valid_positions))]

        target_rank = int(rank * len(valid_positions) / len(base_observations))
        target_rank = min(target_rank, len(valid_positions) - 1)
        return valid_positions[target_rank]

    def __getitem__(self, index, dataset_statistics):
        """Apply VSTA online during training (Algorithm 1)."""
        index = self._resolve_train_observation_index(index)

        # Sample speed
        speed = self._sample_speed()
        q, p = self._speed_to_coprime(speed)

        # Get episode indices and cached segments
        data = self.dataset[index]
        dataset_idx = self._get_dataset_index(index)
        episode_index = data['episode_index']
        key = (dataset_idx, episode_index)

        if key not in self.episode_segments:
            # This should not happen in normal operation - segmentation is
            # pre-computed for all episodes in _build_episode_cache
            raise RuntimeError(
                f"Episode key {key} not found in cached segments. "
                f"This indicates a bug in episode cache construction.")

        # Use cached segmentation
        segments = self.episode_segments[key]

        # Retime all segments
        all_retimed = []
        for seg in segments:
            retimed = self._retime_segment(seg, q, p)
            all_retimed.extend(retimed)

        if not all_retimed:
            raise RuntimeError(
                f"Episode {key} produced no retimed steps after segmentation. "
                f"Original episode length: {len(self.episode_ranges[key])}, "
                f"Number of segments: {len(segments)}")

        # Pick a retimed position that has a real image/proprio observation.
        best_idx = self._select_retimed_observation(index, key, all_retimed)
        chosen_step = all_retimed[best_idx]
        assert chosen_step.source_idx is not None
        chosen_src_idx = chosen_step.source_idx

        # Build action window starting from this observation
        actions = []
        action_masks = []

        for i in range(self.action_window_size):
            action_idx = best_idx + self.window_start_idx + i
            if action_idx < len(all_retimed):
                actions.append(all_retimed[action_idx].action)
                action_masks.append(1.0)
            else:
                # Pad with last valid action
                if actions:
                    actions.append(actions[-1])
                else:
                    # Fallback: zero action
                    actions.append(np.zeros_like(all_retimed[0].action))
                action_masks.append(0.0)

        # Build output data
        data = dict(self.dataset[chosen_src_idx])
        data['actions'] = np.stack(actions, axis=0).astype(np.float32)
        data['action_masks'] = np.array(action_masks, dtype=np.float32)
        if self.frame_window_size > 1:
            frame_timestamps = [data['timestamp']]
            frame_masks = [1]
            dataset_idx = self._get_dataset_index(chosen_src_idx)
            for fi in range(1, self.frame_window_size):
                future_idx = chosen_src_idx + fi
                if (future_idx < len(self.dataset)
                        and self.dataset[future_idx]['episode_index']
                        == data['episode_index'] and
                        self._get_dataset_index(future_idx) == dataset_idx):
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

        data[self.tempo_speed_key] = float(speed)
        data['info'] = self.info[dataset_idx]
        data['stats'] = dataset_statistics[self.statistic_name]
        if self.expose_index:
            data['index'] = np.array(chosen_src_idx, dtype=np.int64)
        data['task_description'] = self.tasks[dataset_idx][
            data['task_index']]['task']
        data['data_root'] = self.data_root_path[dataset_idx]

        # Apply transforms
        for transform in self.transforms:
            data = transform(data)

        return data
