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

"""
VSTAParquetDataset — window-level online VSTA augmentation.

At each __getitem__, a random speed s = q/p is drawn from speed_set.
Instead of augmenting the full episode, only ceil(window_size × s) original
frames are fetched and passed to VSTADelta, which produces action_window_size
output steps.  The observation frame is always the current index (unchanged).

Memory overhead vs ParquetDataset: zero (no init-time data loading).
Per-getitem overhead: ceil(window_size × max_speed) extra row reads at most
  (e.g. window=10, max_speed=2 → at most 20 rows instead of 10).

Correctness vs episode-level (paper Algorithm 1):
  - Equivalent when no gripper transition falls inside the fetched window.
  - When a gripper transition is present, VSTADelta's segment_gripper clips
    the segment at the transition, preserving the delta-axis-angle validity.
  - Observation is the current frame's original state (no episode mapping).

Action format: 7D [Δx, Δy, Δz, Δax, Δay, Δaz, gripper_abs]
State  format: 8D [x, y, z, ax, ay, az, g1, g2]  (absolute, axis-angle)
"""

import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from fluxvla.engines import DATASETS
from .parquet_dataset import ParquetDataset

# ---------------------------------------------------------------------------
# Locate and import vsta_delta.py
# ---------------------------------------------------------------------------
_VSTA_ROOT = os.environ.get(
    'VSTA_ROOT',
    str(Path(__file__).parent.parent.parent.parent / 'vsta'),
)
if _VSTA_ROOT not in sys.path:
    sys.path.insert(0, _VSTA_ROOT)

from vsta_delta import VSTADelta, VSTADeltaConfig  # noqa: E402

_DEFAULT_VSTA_CONFIG = dict(
    delta_indices=[0, 1, 2, 3, 4, 5],
    gripper_indices=[6],
    speed_set=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
)


@DATASETS.register_module()
class VSTAParquetDataset(ParquetDataset):
    """ParquetDataset with online window-level VSTA speed augmentation.

    At each __getitem__ a speed s is sampled from speed_set.  The action
    window is built by fetching ceil(action_window_size × s) original frames
    then resampling to action_window_size steps via VSTADelta.  Zero extra
    memory beyond what ParquetDataset already uses.

    Args:
        vsta_config (dict | None): Forwarded to VSTADeltaConfig.
        speed_conditioning (bool): Prepend speed text to task_description.
        seed (int): RNG seed.  Override per DataLoader worker via
            worker_init_fn to avoid identical random sequences.
    """

    def __init__(
        self,
        *args,
        vsta_config: Optional[Dict] = None,
        speed_conditioning: bool = False,
        seed: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        cfg = vsta_config if vsta_config is not None else _DEFAULT_VSTA_CONFIG
        self.vsta = VSTADelta(VSTADeltaConfig(**cfg))
        self.speed_conditioning = speed_conditioning
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Getitem
    # ------------------------------------------------------------------

    def __getitem__(self, index, dataset_statistics):
        index = self._resolve_index(index)
        for _ in range(16):
            result = self._try_getitem(index, dataset_statistics)
            if result is not None:
                return result
            index = self._rand_another()
        return super().__getitem__(index, dataset_statistics)

    def _try_getitem(self, index: int, dataset_statistics):
        data        = self.dataset[index]
        dataset_idx = self._get_dataset_index(index)

        if self._invalid_start_index(index, dataset_idx, data):
            return None

        # ── random speed ───────────────────────────────────────────────────
        speed_set = self.vsta.cfg.speed_set
        speed     = float(speed_set[int(self._rng.integers(0, len(speed_set)))])

        # ── fetch raw actions ──────────────────────────────────────────────
        # speed = q/p  →  need ceil(window × q/p) = ceil(window × speed)
        # original frames to produce action_window_size augmented steps.
        fetch_len   = math.ceil(self.action_window_size * speed)
        raw_actions: List = []

        window_idx = self.window_start_idx
        while len(raw_actions) < fetch_len:
            ai = index + window_idx
            if not self._same_episode_and_dataset(ai, dataset_idx, data):
                break
            task = self._get_task_name(dataset_idx, ai)
            if task == 'empty':
                break
            if task == 'static':
                window_idx += 1
                continue
            raw_actions.append(self.dataset[ai][self.action_key])
            window_idx += 1

        if not raw_actions:
            return None

        # ── apply VSTADelta to the raw window ─────────────────────────────
        raw_np   = np.array(raw_actions, dtype=np.float32)
        aug_seed = int(self._rng.integers(0, 2**31))
        aug      = self.vsta(
            {'action': torch.from_numpy(raw_np)},
            speed=speed,
            seed=aug_seed,
        )
        aug_actions = aug['action'].numpy()  # [T_aug, action_dim]

        # ── clip / pad to action_window_size ──────────────────────────────
        actions: List      = []
        action_masks: List = []
        for i in range(self.action_window_size):
            if i < len(aug_actions):
                actions.append(aug_actions[i])
                action_masks.append(1)
            else:
                # episode ended before filling the window
                actions.append(actions[-1] if actions else raw_np[-1])
                action_masks.append(0)

        # ── build output dict ──────────────────────────────────────────────
        obs_data = dict(data)
        obs_data['actions']          = np.array(actions, dtype=np.float32)
        obs_data['action_masks']     = np.array(action_masks, dtype=np.float32)
        obs_data['info']             = self.info[dataset_idx]
        obs_data['stats']            = dataset_statistics[self.statistic_name]
        obs_data['task_description'] = self._get_task_name(dataset_idx, index)
        obs_data['data_root']        = self.data_root_path[dataset_idx]

        if self.speed_conditioning:
            obs_data['task_description'] = (
                aug['speed_text'] + obs_data['task_description'])

        # ── multi-frame video window (mirrors parent logic) ────────────────
        if self.frame_window_size > 1:
            frame_timestamps = [data['timestamp']]
            frame_masks      = [1]
            for fi in range(1, self.frame_window_size):
                future_idx = index + fi
                if self._same_episode_and_dataset(future_idx, dataset_idx, data):
                    frame_timestamps.append(
                        self.dataset[future_idx]['timestamp'])
                    frame_masks.append(1)
                else:
                    frame_timestamps.append(frame_timestamps[-1])
                    frame_masks.append(0)
            obs_data['frame_timestamps'] = frame_timestamps
            obs_data['frame_masks']      = np.array(frame_masks, dtype=np.float32)

        if self.expose_index:
            obs_data['index'] = np.array(index, dtype=np.int64)

        # ── transform chain ────────────────────────────────────────────────
        for transform in self.transforms:
            obs_data = transform(obs_data)

        return obs_data
