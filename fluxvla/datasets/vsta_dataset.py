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
VSTADataset — window-level online VSTA speed augmentation for absolute
(non-delta) action spaces, e.g. continuous arm/hand joint control.

Dataset-integration structure restored from git history (commit aa2824a,
path vsta/FluxVLA/fluxvla/datasets/vsta_parquet_dataset.py — a historical,
fully parallel codebase snapshot with no version relationship to this repo,
later deleted in 56c2c0b in favor of fluxvla/datasets/online_vsta_dataset.py).
That file fetches only ceil(action_window_size x speed) raw frames per
sample instead of retiming the whole episode, which is the efficiency
property we want; its resampling core, however, is superseded here.

The resampling core is corrected against /data/wqz/fluxvla_optimize/vsta/
vsta.py (a separate, actively maintained repo implementing the same
TempoVLA VSTA algorithm, used there for inference-side benchmarking, not
dataset construction). That repo's `DimSpec(delta_idx, abs_idx, gripper_idx)`
abstraction is the reference for what "correct" abs-joint handling looks
like:
  - `vsta_delta.py`/`VSTADelta` (the module the historical
    vsta_parquet_dataset.py imports, no longer present anywhere) assumed
    every non-gripper dim was a delta/velocity and used
    "accumulate-then-split" (sum deltas, interpolate the cumulative sum).
    Summing consecutive *absolute* values has no physical meaning, so this
    is wrong for this project's data.
  - `vsta/vsta.py`'s `abs_idx` dims are instead resampled via direct
    interpolation of the absolute trajectory — that is the approach used
    below (`_resample_chunk`).
  - This project's end effector is a continuous, multi-DOF hand, not a
    binary gripper, so there is no default "gripper channel" to assume;
    `discrete_indices` defaults to empty (every action dim continuous, i.e.
    every dim is treated like `vsta.py`'s `abs_idx`). Pass explicit indices
    only if a given deployment's action vector does contain a channel that
    must never be blended across a segment boundary (nearest-frame lookup
    instead of interpolation, mirroring `vsta.py`'s `gripper_idx` handling),
    with segment boundaries placed on value-jumps (not sign changes, unlike
    the historical file) so it isn't tied to a signed-delta convention.
  - `vsta.py` additionally supports delta dims and three non-uniform
    resampling backends (rdp/dct/idf); this dataset only needs the
    uniform, abs+discrete case for now and does not vendor those.

Efficiency trade-off vs. online_vsta_dataset.py (kept, unmodified, and no
longer used by new configs):
  - online_vsta_dataset.py segments and retimes the *entire* episode on
    every __getitem__ call, even though only one small window
    (action_window_size frames) is ever consumed from the result — O(episode
    length) work per sample.
  - VSTADataset (this file) fetches only ceil(action_window_size x speed)
    raw frames and resamples just that window — O(window size) work per
    sample, zero init-time episode/segment cache.
  - Correctness caveat (present in the historical delta-space
    implementation too): segmentation only sees the fetched window, not the
    whole episode. If a discrete-channel transition falls right at the edge
    of the window, the segment touching that edge is truncated to the
    window instead of its true full extent. With `discrete_indices` empty
    (this project's default), there is nothing to segment on and this
    caveat does not apply.
"""

import math
import os
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from fluxvla.engines import DATASETS
from .parquet_dataset import ParquetDataset

# ---------------------------------------------------------------------------
# VSTAResampler — resampling core for absolute action spaces
# ---------------------------------------------------------------------------


class VSTAResampler:
    """VSTA speed transform for absolute (non-delta) action spaces.

    Resamples a (T, action_dim) trajectory to a different execution speed by
    directly interpolating the absolute values (as opposed to accumulating
    deltas and interpolating the cumulative sum, which only makes sense for
    delta/velocity actions — see module docstring).

    Args:
        discrete_indices: Action dims that must never be blended across a
            segment boundary (resampled via nearest-frame lookup instead of
            linear interpolation), e.g. a binary gripper. Defaults to none:
            every dim is treated as continuous.
        discrete_threshold: Segment boundary is placed wherever any
            `discrete_indices` dim jumps by more than this amount between
            consecutive frames. Unused if `discrete_indices` is empty.
        speed_set: Candidate speeds to sample from when `speed` is not
            passed explicitly to `__call__`.
        fixed_speed: If set, always use this speed instead of sampling from
            `speed_set`.
        seed: RNG seed.

    Input episode keys:
      "action":      Tensor / ndarray  (T, action_dim)   required
      "observation": Tensor / ndarray  (T, ...)           optional

    Output adds:
      "valid_mask":  ndarray (T',) bool
      "speed":       float

    Note: this class does not produce any prompt text. Injecting the sampled
    speed into the language prompt (with dropout, bucketing, etc.) is the
    sole responsibility of `EpisodeMetadataPrompter`
    (fluxvla/transforms/prompters.py), which reads the `tempo_speed` value
    this dataset attaches to each sample.
    """

    def __init__(
        self,
        discrete_indices: Optional[Sequence[int]] = None,
        discrete_threshold: float = 0.5,
        speed_set: Sequence[float] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
        fixed_speed: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.discrete_indices = list(discrete_indices or [])
        self.discrete_threshold = discrete_threshold
        self.speed_set = list(speed_set)
        self.fixed_speed = fixed_speed
        self._rng = np.random.default_rng(seed)

    @staticmethod
    def _speed_to_qp(s: float, max_denom: int = 20) -> Tuple[int, int]:
        """s = q/p with coprime positive integers q, p."""
        frac = Fraction(s).limit_denominator(max_denom)
        q, p = frac.numerator, frac.denominator
        assert q > 0 and p > 0, f'invalid speed s={s}'
        return q, p

    def _segment(self, actions: np.ndarray) -> List[List[int]]:
        """Split a trajectory into segments at discrete-channel boundaries.

        A boundary is added wherever any `discrete_indices` dim changes by
        more than `discrete_threshold` between consecutive frames. With no
        discrete_indices configured (this project's default), the whole
        trajectory is a single segment.
        """
        T = len(actions)
        if T == 0:
            return []
        if not self.discrete_indices:
            return [list(range(T))]

        boundaries = {0}
        for t in range(1, T):
            prev = actions[t - 1, self.discrete_indices]
            curr = actions[t, self.discrete_indices]
            if np.any(np.abs(curr - prev) > self.discrete_threshold):
                boundaries.add(t)

        sorted_b = sorted(boundaries) + [T]
        return [
            list(range(sorted_b[i], sorted_b[i + 1]))
            for i in range(len(sorted_b) - 1) if sorted_b[i + 1] > sorted_b[i]
        ]

    def _resample_chunk(self, chunk: np.ndarray, p: int) -> np.ndarray:
        """Map q source frames -> p output frames for one chunk.

        Continuous dims: linearly interpolate the absolute trajectory at p
        evenly-spaced sample times spanning the chunk's original time span
        [0, q-1] (mirrors `vsta.py`'s `abs_idx` handling).

        Discrete dims: resample via nearest-frame lookup (round the query
        time to the nearest source frame index) so the output never blends
        two distinct discrete states.
        """
        q = len(chunk)
        D = chunk.shape[1]
        out = np.zeros((p, D), dtype=np.float64)
        src_t = np.arange(q, dtype=np.float64)
        out_t = (np.linspace(0, q - 1, p) if p > 1 else
                 np.array([(q - 1) / 2.0]))

        discrete = set(self.discrete_indices)
        continuous_indices = [vi for vi in range(D) if vi not in discrete]

        for vi in continuous_indices:
            col = chunk[:, vi].astype(np.float64)
            out[:, vi] = np.interp(out_t, src_t, col)

        if discrete:
            nearest_idx = np.clip(np.round(out_t).astype(int), 0, q - 1)
            for gi in self.discrete_indices:
                out[:, gi] = chunk[nearest_idx, gi]

        return out.astype(chunk.dtype)

    def _resample_segment(
        self,
        actions_seg: np.ndarray,  # (L, D)
        obs_seg: Optional[np.ndarray],  # (L, ...) or None
        q: int,
        p: int,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Speed-transform one segment via chunk-level absolute resampling.

        Chunk-start offset r ~ U{0, ..., q-1} per segment (online sampling of
        the chunk phase).

        Valid-mask:
          - Leading passthrough [0, r): all valid
          - Each chunk: only j=0 (chunk-start observation) is valid
          - Trailing remainder: all valid
        """
        L = len(actions_seg)
        # r is sampled from [0, q), but a short fetched window (L < q -- the
        # window-level fetch ran out of frames near the end of an episode,
        # or the episode itself is shorter than one chunk at this speed) can
        # make r land past the end of actions_seg. Clamp to L: with no room
        # for even one full chunk, the whole segment is just leading
        # passthrough (the while loop below never enters, chunk_start == L).
        r = min(int(rng.integers(0, q)) if q > 1 else 0, L)

        new_actions: List[np.ndarray] = []
        valid: List[bool] = []
        new_obs: Optional[List] = [] if obs_seg is not None else None

        def _append(a, v, o=None):
            new_actions.append(a)
            valid.append(v)
            if new_obs is not None:
                new_obs.append(o)

        # Leading passthrough [0, r) — verbatim, all valid
        for t in range(r):
            _append(actions_seg[t], True,
                    obs_seg[t] if obs_seg is not None else None)

        # Non-overlapping chunks of size q
        chunk_start = r
        while chunk_start + q <= L:
            chunk = actions_seg[chunk_start:chunk_start + q]
            new_steps = self._resample_chunk(chunk, p)
            obs_chunk = obs_seg[chunk_start] if obs_seg is not None else None
            for j in range(p):
                _append(new_steps[j], j == 0, obs_chunk)
            chunk_start += q

        # Trailing remainder — verbatim, all valid
        for t in range(chunk_start, L):
            _append(actions_seg[t], True,
                    obs_seg[t] if obs_seg is not None else None)

        out_actions = np.stack(new_actions, axis=0)
        out_valid = np.array(valid, dtype=bool)
        out_obs = np.stack(new_obs, axis=0) if new_obs is not None else None

        return out_actions, out_valid, out_obs

    def __call__(
        self,
        episode: Dict,
        speed: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Dict:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        actions = episode['action']
        if isinstance(actions, torch.Tensor):
            actions = actions.numpy()
        actions = np.asarray(actions, dtype=np.float32)

        obs = episode.get('observation', None)
        if obs is not None and isinstance(obs, torch.Tensor):
            obs = obs.numpy()

        if speed is None:
            speed = self.fixed_speed
        if speed is None:
            speed = float(self._rng.choice(self.speed_set))

        q, p = self._speed_to_qp(speed)
        segments = self._segment(actions)

        all_actions: List[np.ndarray] = []
        all_valid: List[np.ndarray] = []
        all_obs: Optional[List] = [] if obs is not None else None

        for seg_idx in segments:
            seg_act = actions[seg_idx]
            seg_obs = obs[seg_idx] if obs is not None else None
            new_act, valid_mask, new_obs = self._resample_segment(
                seg_act, seg_obs, q, p, self._rng)
            all_actions.append(new_act)
            all_valid.append(valid_mask)
            if all_obs is not None:
                all_obs.append(new_obs)

        new_actions = np.concatenate(all_actions, axis=0)
        valid_mask = np.concatenate(all_valid, axis=0)

        out = dict(episode)
        out['action'] = torch.from_numpy(new_actions)
        out['valid_mask'] = valid_mask
        out['speed'] = speed

        if all_obs is not None:
            out['observation'] = torch.from_numpy(
                np.concatenate(all_obs, axis=0))

        return out


# ---------------------------------------------------------------------------
# VSTADataset — window-level dataset integration
# ---------------------------------------------------------------------------


@DATASETS.register_module()
class VSTADataset(ParquetDataset):
    """ParquetDataset with online window-level VSTA speed augmentation.

    At each __getitem__ a speed s is sampled from speed_set.  The action
    window is built by fetching ceil(action_window_size x s) original frames
    then resampling to action_window_size steps via VSTAResampler.  Zero extra
    memory beyond what ParquetDataset already uses.

    Args:
        vsta_kwargs (dict | None): Forwarded to VSTAResampler
            (discrete_indices, discrete_threshold, speed_set, fixed_speed).
            Defaults to VSTAResampler's own defaults — no discrete channels,
            i.e. every action dim resampled via continuous interpolation.
        seed (int): RNG seed.  Override per DataLoader worker via
            worker_init_fn to avoid identical random sequences.
    """

    def __init__(
        self,
        *args,
        vsta_kwargs: Optional[Dict] = None,
        seed: int = 42,
        **kwargs,
    ) -> None:
        # Actions must be the raw absolute values stored in the parquet
        # (not the base class's frame-to-frame delta), since VSTAResampler
        # expects an absolute-value trajectory to interpolate.
        if 'use_delta' in kwargs:
            kwargs.pop('use_delta')
        kwargs['use_delta'] = False

        super().__init__(*args, **kwargs)
        self.vsta = VSTAResampler(**(vsta_kwargs or {}))
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Getitem
    # ------------------------------------------------------------------

    def __getitem__(self, index, dataset_statistics):
        # Mirrors ParquetDataset.__getitem__'s own retry loop exactly (same
        # _invalid_start_index gate, same _rand_another resampling): keep
        # redrawing index until it points at a frame whose action window can
        # actually start there. Unbounded, like the parent — guaranteed to
        # terminate as long as the dataset has at least one valid index.
        index = self._resolve_index(index)
        data = self.dataset[index]
        dataset_idx = self._get_dataset_index(index)
        while self._invalid_start_index(index, dataset_idx, data):
            index = self._rand_another()
            data = self.dataset[index]
            dataset_idx = self._get_dataset_index(index)
        return self._build_sample(index, dataset_idx, data,
                                  dataset_statistics)

    def _build_sample(self, index: int, dataset_idx: int, data,
                      dataset_statistics):
        # ── random speed ────────────────────────────────────────────────
        speed = float(self.vsta.speed_set[int(
            self._rng.integers(0, len(self.vsta.speed_set)))])

        # ── fetch raw actions ───────────────────────────────────────────
        # speed = q/p  ->  need ceil(window x q/p) = ceil(window x speed)
        # original frames to produce action_window_size augmented steps.
        fetch_len = math.ceil(self.action_window_size * speed)
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

        # __getitem__ only calls _build_sample once _invalid_start_index has
        # confirmed index + window_start_idx is a valid frame, so the loop
        # above always appends at least that one frame before it can break.
        assert raw_actions, (
            'raw_actions unexpectedly empty; _invalid_start_index should '
            'have already excluded this index.')

        # ── apply VSTAResampler to the raw window ───────────────────────
        raw_np = np.array(raw_actions, dtype=np.float32)
        aug_seed = int(self._rng.integers(0, 2**31))
        aug = self.vsta(
            {'action': torch.from_numpy(raw_np)},
            speed=speed,
            seed=aug_seed,
        )
        aug_actions = aug['action'].numpy()  # [T_aug, action_dim]

        # ── clip / pad to action_window_size ────────────────────────────
        actions: List = []
        action_masks: List = []
        for i in range(self.action_window_size):
            if i < len(aug_actions):
                actions.append(aug_actions[i])
                action_masks.append(1)
            else:
                # episode ended before filling the window
                actions.append(actions[-1] if actions else raw_np[-1])
                action_masks.append(0)

        # ── build output dict ───────────────────────────────────────────
        obs_data = dict(data)
        obs_data['actions'] = np.array(actions, dtype=np.float32)
        obs_data['action_masks'] = np.array(action_masks, dtype=np.float32)
        obs_data['info'] = self.info[dataset_idx]
        obs_data['stats'] = dataset_statistics[self.statistic_name]
        obs_data['task_description'] = self._get_task_name(dataset_idx, index)
        obs_data['data_root'] = self.data_root_path[dataset_idx]
        # advantage/intervention/length, if migrated into episodes.jsonl by
        # scripts/migrate_episode_advantage.py; {} otherwise.
        obs_data['episode_metadata'] = self.episode_metadata_by_key.get(
            (dataset_idx, data['episode_index']), {})
        # Consumed downstream by EpisodeMetadataPrompter to derive the
        # "Speed: ..." prompt text (tempo_speed + original episode length).
        obs_data['tempo_speed'] = speed

        # ── multi-frame video window (mirrors parent logic) ─────────────
        if self.frame_window_size > 1:
            frame_timestamps = [data['timestamp']]
            frame_masks = [1]
            for fi in range(1, self.frame_window_size):
                future_idx = index + fi * self.frame_sample_stride
                if self._same_episode_and_dataset(future_idx, dataset_idx,
                                                  data):
                    frame_timestamps.append(
                        self.dataset[future_idx]['timestamp'])
                    frame_masks.append(1)
                else:
                    frame_timestamps.append(frame_timestamps[-1])
                    frame_masks.append(0)
            obs_data['frame_timestamps'] = frame_timestamps
            obs_data['frame_masks'] = np.array(frame_masks, dtype=np.float32)

        if self.expose_index:
            obs_data['index'] = np.array(index, dtype=np.int64)

        # ── transform chain ─────────────────────────────────────────────
        for transform in self.transforms:
            obs_data = transform(obs_data)

        return obs_data


# ---------------------------------------------------------------------------
# VSTAPackedParquetDatasetV3 — window-level dataset integration for LeRobot
# v3.0 packed-video layouts (e.g. pi05_parcel_sort's production data)
# ---------------------------------------------------------------------------

from .packed_parquet_dataset_v3 import PackedParquetDatasetV3  # noqa: E402


@DATASETS.register_module()
class VSTAPackedParquetDatasetV3(PackedParquetDatasetV3):
    """PackedParquetDatasetV3 with window-level VSTA speed augmentation.

    Same window-fetch-then-resample approach as VSTADataset (this module),
    adapted to LeRobot v3.0's packed-video layout: PackedParquetDatasetV3's
    ``__getitem__`` is a two-stage pipeline (row/action assembly via
    ParquetDatasetV3.__getitem__, then video decoding on top using
    ``episode_records_by_index``'s chunk/file/from_timestamp fields — see
    that module's docstring). This class does not chain through
    ``super().__getitem__()`` for the first stage, because that stage is
    exactly the part being replaced (its plain future-frame lookahead action
    window, swapped for the VSTA-resampled one); instead it re-implements
    that stage's index-validity resampling loop (identical semantics to
    ParquetDatasetV3.__getitem__) and calls the inherited video-decoding
    helpers (``_build_video_path``, ``_video_from_timestamp``,
    ``_decode_video_frames``) directly, exactly as
    PackedParquetDatasetV3.__getitem__ does.

    episode_metadata (advantage/intervention/length) comes straight from
    ``self.episodes_meta[dataset_idx][episode_index]`` -- the full
    episodes.jsonl record for that episode, already loaded by
    ParquetDatasetV3.__init__ -- no separate lookup table needed (unlike the
    v2.1 VSTADataset, which has to build one itself; see
    ParquetDataset.episode_metadata_by_key).
    """

    def __init__(
        self,
        *args,
        vsta_kwargs: Optional[Dict] = None,
        seed: int = 42,
        **kwargs,
    ) -> None:
        if 'use_delta' in kwargs:
            kwargs.pop('use_delta')
        kwargs['use_delta'] = False

        super().__init__(*args, **kwargs)
        self.vsta = VSTAResampler(**(vsta_kwargs or {}))
        self._rng = np.random.default_rng(seed)

    def __getitem__(self, index, dataset_statistics):
        # ── resolve to a valid starting row ──────────────────────────────
        # Identical semantics to ParquetDatasetV3.__getitem__'s own loop:
        # an index is invalid if it is the dataset's last row, or the next
        # row belongs to a different episode/dataset, or the next row's task
        # is 'empty'/'static'. Duplicated rather than called via super()
        # because this whole method replaces ParquetDatasetV3.__getitem__'s
        # row-assembly stage; there is no clean way to reuse only its first
        # half through the standard super() chain.
        data = self.dataset[index]
        dataset_idx = self._get_dataset_index(index)
        while True:
            if index == len(self.dataset) - 1:
                needs_resample = True
            else:
                next_data = self.dataset[index + 1]
                next_task = self._resolve_task_description(
                    dataset_idx, next_data)
                needs_resample = (
                    data['episode_index'] != next_data['episode_index']
                    or self._get_dataset_index(index + 1) != dataset_idx
                    or next_task in ('empty', 'static'))
            if not needs_resample:
                break
            index = self._rand_another()
            data = self.dataset[index]
            dataset_idx = self._get_dataset_index(index)

        episode_index = int(np.asarray(data['episode_index']).item())

        # ── random speed + windowed raw-action fetch ─────────────────────
        speed = float(self.vsta.speed_set[int(
            self._rng.integers(0, len(self.vsta.speed_set)))])
        fetch_len = math.ceil(self.action_window_size * speed)
        raw_actions: List = []

        window_idx = self.window_start_idx
        while len(raw_actions) < fetch_len:
            ai = index + window_idx
            if ai >= len(self.dataset):
                break
            future_data = self.dataset[ai]
            if (future_data['episode_index'] != data['episode_index']
                    or self._get_dataset_index(ai) != dataset_idx):
                break
            future_task = self._resolve_task_description(
                dataset_idx, future_data)
            if future_task == 'empty':
                break
            if future_task == 'static':
                window_idx += 1
                continue
            raw_actions.append(future_data[self.action_key])
            window_idx += 1

        # See VSTADataset._build_sample: guaranteed non-empty because the
        # index-validity loop above already confirmed index + window_start_idx
        # is a valid frame.
        assert raw_actions, (
            'raw_actions unexpectedly empty; the index-validity resampling '
            'loop above should have already excluded this index.')

        raw_np = np.array(raw_actions, dtype=np.float32)
        aug_seed = int(self._rng.integers(0, 2**31))
        aug = self.vsta(
            {'action': torch.from_numpy(raw_np)},
            speed=speed,
            seed=aug_seed,
        )
        aug_actions = aug['action'].numpy()

        actions: List = []
        action_masks: List = []
        for i in range(self.action_window_size):
            if i < len(aug_actions):
                actions.append(aug_actions[i])
                action_masks.append(1)
            else:
                actions.append(actions[-1] if actions else raw_np[-1])
                action_masks.append(0)

        # ── assemble the row-level sample, mirroring
        # ParquetDatasetV3.__getitem__'s tail exactly (frame_window_size
        # block, info/stats/task_description/data_root), plus the fields
        # EpisodeMetadataPrompter reads ──────────────────────────────────
        if self.frame_window_size > 1:
            frame_timestamps = [data['timestamp']]
            frame_masks = [1]
            for fi in range(1, self.frame_window_size):
                future_idx = index + fi * self.frame_sample_stride
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

        data['info'] = self.info[dataset_idx]
        data['stats'] = dataset_statistics[self.statistic_name]
        data['actions'] = np.array(actions, dtype=np.float32)
        data['action_masks'] = np.array(action_masks, dtype=np.float32)
        if self.expose_index:
            data['index'] = np.array(index, dtype=np.int64)
        data['task_description'] = self._resolve_task_description(
            dataset_idx, data)
        data['data_root'] = self.data_root_path[dataset_idx]
        # advantage/intervention/length: the full episodes.jsonl record for
        # this episode, already loaded into episodes_meta by
        # ParquetDatasetV3.__init__ -- no separate lookup table needed here.
        data['episode_metadata'] = self.episodes_meta[dataset_idx].get(
            episode_index, {})
        # Consumed downstream by EpisodeMetadataPrompter for "Speed: ...".
        data['tempo_speed'] = speed

        # ── video decoding (PackedParquetDatasetV3's contribution) ───────
        current_ts = float(np.asarray(data['timestamp']).reshape(-1)[0])
        images = []
        img_masks = []
        for video_key in self.video_keys:
            video_path = self._build_video_path(dataset_idx, episode_index,
                                                video_key)
            assert os.path.exists(video_path), \
                f'Video file not found: {video_path}'
            from_ts = self._video_from_timestamp(dataset_idx, episode_index,
                                                 video_key)
            abs_ts = from_ts + current_ts
            frame = self._decode_video_frames(video_path, [abs_ts])[0]
            images.append(frame.numpy())
            img_masks.append(True)

        inputs = {
            'states': np.asarray(data[self.state_key], dtype=np.float32),
            'actions': data['actions'],
            'action_masks': data['action_masks'],
            'images': images,
            'img_masks': np.array(img_masks),
            'task_description': data.get('task_description', ''),
            'stats': data['stats'],
            'episode_metadata': data['episode_metadata'],
            'tempo_speed': data['tempo_speed'],
            # meta/info.json's features.*.names -- consumed by
            # SelectJointDims (fluxvla/transforms/transform_states.py) to
            # look up which dims to zero by joint name rather than a fixed
            # index, so the same config stays correct even if a different
            # dataset's recorder config orders its joints differently.
            'info': data['info'],
        }

        for transform in self._post_transforms:
            inputs = transform(inputs)
        return inputs
