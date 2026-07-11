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
"""Dataset for LeRobot v3.0 layouts that pack many episodes per video file.

The standard :class:`ParquetDataset`/:class:`ParquetDatasetV3` + the
``ProcessParquetInputs`` transform assume a v2-style layout where every
episode maps to a single video file and per-frame ``timestamp`` values index
directly into that file. Genuine LeRobot v3.0 datasets (``codebase_version``
``v3.0``) instead concatenate multiple episodes into a shared
``videos/<key>/chunk-XXX/file-YYY.mp4`` file, and reset each episode's
``timestamp`` to zero. Correct frame retrieval therefore requires, per camera:

* resolving the ``chunk_index``/``file_index`` of the file that stores the
  episode (cameras for the same episode may live in different files);
* offsetting the query timestamp by that camera's per-episode
  ``from_timestamp``.

This dataset reuses :class:`ParquetDatasetV3` for metadata loading and the
future-action window construction, then performs v3-correct video decoding
inline (replacing ``ProcessParquetInputs``). Downstream transforms
(normalization, prompt building, resize, image normalization) run unchanged.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torchvision

from fluxvla.datasets.parquet_dataset_v3 import ParquetDatasetV3
from fluxvla.engines import DATASETS


@DATASETS.register_module()
class PackedParquetDatasetV3(ParquetDatasetV3):
    """ParquetDatasetV3 variant with v3-correct packed-video decoding.

    Args:
        data_root_path (Union[str, List[str]]): One dataset root, or a list of
            dataset roots, each containing ``meta``, ``data`` and ``videos``.
        video_keys (List[str]): Camera video keys to decode (must match the
            ``observation.images.*`` keys present in ``meta/info.json`` and the
            per-episode metadata).
        transforms (List[Dict]): Transforms applied after the sample (with
            decoded ``images`` and mapped ``states``/``actions``) is assembled.
            Do NOT include ``ProcessParquetInputs`` here; this dataset performs
            the equivalent work internally.
        action_window_size (int): Number of future actions returned per sample.
        action_key (str): Dataset column used as the action/state source.
        state_key (str): Dataset column used as the current-frame state.
        statistic_name (str): Statistics entry read from ``dataset_statistics``.
        window_start_idx (int): First offset used when building the action
            window.
        use_delta (bool): Whether actions are converted to frame-to-frame
            deltas.
        video_tolerance_s (float): Max allowed distance (seconds) between the
            requested timestamp and the decoded frame.
        video_backend (str): torchvision video backend. Defaults to ``pyav``.
    """

    def __init__(self,
                 data_root_path: Union[str, List[str]],
                 video_keys: List[str],
                 transforms: List[Dict],
                 action_window_size: int = 50,
                 action_key: str = 'action',
                 state_key: str = 'observation.state',
                 statistic_name: str = 'private',
                 window_start_idx: int = 1,
                 use_delta: bool = False,
                 video_tolerance_s: float = 0.1,
                 video_backend: str = 'pyav') -> None:
        super().__init__(
            data_root_path=data_root_path,
            transforms=transforms,
            action_window_size=action_window_size,
            action_key=action_key,
            use_delta=use_delta,
            statistic_name=statistic_name,
            window_start_idx=window_start_idx,
        )
        self.video_keys = video_keys
        self.state_key = state_key
        self.video_tolerance_s = video_tolerance_s
        self.video_backend = video_backend

        # The parent already applied the transform configs into
        # ``self.transforms`` and runs them at the end of its ``__getitem__``.
        # We move them to a post-processing list so the parent assembles the
        # raw sample only, and we apply transforms ourselves after decoding
        # videos and mapping state/action keys.
        self._post_transforms = self.transforms
        self.transforms = []

        # episode_index -> episode metadata record, per dataset root.
        self.episodes_meta = self.episode_records_by_index

    def _first_available(self, meta: Dict, keys: List[str]) -> Optional[float]:
        for key in keys:
            if key in meta and meta[key] is not None:
                return meta[key]
        return None

    def _build_video_path(self, dataset_idx: int, episode_index: int,
                          video_key: str) -> str:
        info = self.info[dataset_idx]
        episode_meta = self.episodes_meta[dataset_idx][episode_index]
        video_root_path = info['video_path']
        format_kwargs = {
            'video_key': video_key,
            'episode_index': episode_index,
        }
        chunk_index = self._first_available(episode_meta, [
            f'videos/{video_key}/chunk_index',
            'data/chunk_index',
            'chunk_index',
        ])
        file_index = self._first_available(episode_meta, [
            f'videos/{video_key}/file_index',
            'data/file_index',
            'file_index',
        ])
        if chunk_index is not None:
            format_kwargs['chunk_index'] = int(chunk_index)
        if file_index is not None:
            format_kwargs['file_index'] = int(file_index)
        if 'chunks_size' in info:
            format_kwargs['episode_chunk'] = episode_index // int(
                info['chunks_size'])
        return os.path.join(self.data_root_path[dataset_idx],
                            video_root_path.format(**format_kwargs))

    def _video_from_timestamp(self, dataset_idx: int, episode_index: int,
                              video_key: str) -> float:
        episode_meta = self.episodes_meta[dataset_idx][episode_index]
        from_ts = self._first_available(
            episode_meta, [f'videos/{video_key}/from_timestamp'])
        return float(from_ts) if from_ts is not None else 0.0

    def _decode_video_frames(self,
                             video_path: Union[Path, str],
                             timestamps: List[float]) -> torch.Tensor:
        """Decode the frames closest to ``timestamps`` from a packed video.

        Args:
            video_path (Union[Path, str]): Path to the mp4 file.
            timestamps (List[float]): Absolute timestamps (already offset by
                the episode ``from_timestamp``) to retrieve.

        Returns:
            torch.Tensor: Stacked uint8 CHW frames, one per requested
            timestamp.
        """
        video_path = str(video_path)
        first_ts = min(timestamps)
        last_ts = max(timestamps)
        keyframes_only = self.video_backend == 'pyav'

        def _load(seek_ts):
            torchvision.set_video_backend(self.video_backend)
            reader = torchvision.io.VideoReader(video_path, 'video')
            reader.seek(max(0.0, seek_ts), keyframes_only=keyframes_only)
            frames, loaded_ts = [], []
            try:
                for frame in reader:
                    current_ts = float(frame['pts'])
                    frames.append(frame['data'])
                    loaded_ts.append(current_ts)
                    if current_ts >= last_ts:
                        break
            finally:
                if self.video_backend == 'pyav':
                    reader.container.close()
            return frames, loaded_ts

        def _match(frames, loaded_ts):
            if not loaded_ts:
                return None
            query = torch.tensor(timestamps, dtype=torch.float32)
            loaded = torch.tensor(loaded_ts, dtype=torch.float32)
            dist = torch.cdist(query[:, None], loaded[:, None], p=1)
            min_dist, argmin = dist.min(1)
            if not (min_dist <= self.video_tolerance_s).all():
                return None
            return torch.stack([frames[idx] for idx in argmin])

        # Seek slightly before the first requested timestamp so the closest
        # preceding key frame is available, then fall back to a full scan.
        frames, loaded_ts = _load(first_ts - 1.0)
        matched = _match(frames, loaded_ts)
        if matched is None:
            frames, loaded_ts = _load(0.0)
            matched = _match(frames, loaded_ts)
        if matched is None:
            raise ValueError(
                'Failed to find frames within tolerance '
                f'({self.video_tolerance_s}s) for {video_path} '
                f'at timestamps {timestamps}')
        return matched

    def __getitem__(self, index, dataset_statistics):
        """Assemble one sample with v3-correct packed-video decoding.

        Args:
            index (int): Dataset row index.
            dataset_statistics (Dict): Statistics mapping supplied by the
                dataset wrapper.

        Returns:
            Dict: Transformed sample with decoded ``images``, mapped
            ``states``/``actions`` and prompt tensors.
        """
        # Parent assembles actions window, stats, info, task_description,
        # data_root and (crucially) may resample ``index`` to a valid frame.
        # ``self.transforms`` is empty here, so no transform runs yet.
        data = super().__getitem__(index, dataset_statistics)

        # ``index`` may have been resampled inside the parent; recover the true
        # dataset and episode via the returned row rather than the incoming
        # index.
        dataset_idx = self.data_root_path.index(data['data_root'])
        episode_index = int(np.asarray(data['episode_index']).item())
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

        # Build a fresh, minimal sample dict (mirroring ProcessParquetInputs)
        # so raw parquet columns do not leak into the collator, which rejects
        # any key not declared as stackable or meta.
        inputs = {
            'states': np.asarray(data[self.state_key], dtype=np.float32),
            'actions': np.asarray(data['actions'], dtype=np.float32),
            'action_masks': np.asarray(data['action_masks'],
                                       dtype=np.float32),
            'images': images,
            'img_masks': np.array(img_masks),
            'task_description': data.get('task_description', ''),
            'stats': data['stats'],
        }

        for transform in self._post_transforms:
            inputs = transform(inputs)
        return inputs
