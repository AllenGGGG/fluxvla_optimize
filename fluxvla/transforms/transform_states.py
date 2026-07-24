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

from typing import Dict, List, Sequence, Union

import numpy as np

from fluxvla.engines import TRANSFORMS


@TRANSFORMS.register_module()
class AddStateGaussianNoise:
    """Add Gaussian noise to `data['states']`.

    Meant to run after `NormalizeStatesAndActions` in the transform
    pipeline, so `noise_std` is in normalized units (e.g. a fraction of the
    [-1, 1] range when the upstream norm_type is 'quantile' or 'min_max'),
    not raw joint-angle units.

    Args:
        noise_std (float | List[float]): Standard deviation of the Gaussian
            noise. A scalar applies the same noise level to every state
            dimension; a list must match the last dimension of
            `data['states']` for per-joint noise levels. Default: 0.01.
        prob (float): Probability of applying noise on a given call.
            Default: 1.0.
    """

    def __init__(self,
                 noise_std: Union[float, List[float]] = 0.01,
                 prob: float = 1.0,
                 *args,
                 **kwargs):
        self.noise_std = noise_std
        self.prob = prob

    def __call__(self, data: dict) -> dict:
        assert 'states' in data, "Input data must contain 'states' key"
        if np.random.random() > self.prob:
            return data

        states = np.asarray(data['states'], dtype=np.float32)
        noise = np.random.normal(
            loc=0.0, scale=self.noise_std, size=states.shape).astype(
                np.float32)
        data['states'] = states + noise
        return data


@TRANSFORMS.register_module()
class SelectJointDims:
    """Drop specific joints from `data['states']`/`data['actions']` by name,
    compact the remaining dims together, then zero-pad at the end back up to
    `pad_to` -- not an in-place zero.

    Why by name, not index: recorder.default.yaml's `joint_names` list (and
    therefore the column order baked into a given dataset's parquet rows) is
    a per-deployment config choice. A fixed index list (e.g. "drop dims
    0-5") silently drops the wrong joints the moment a different dataset
    orders its dims differently -- exactly the failure mode this transform
    exists to rule out. Instead, it looks up each name in
    `data['info']['features'][<feature>]['names']` (populated by
    ros2_data_collection's writer from the same `joint_names` config, see
    lerobot_v21_writer.py/lerobot_v3_writer.py's `_write_meta_files`) at
    runtime, per sample, so the same `exclude_joint_names` config is correct
    for any dataset that declares those names, regardless of position.

    Why drop-and-repad instead of zeroing in place: the resulting layout is
    "N real joints (in their original relative order, excluded ones
    removed) followed by zero padding" -- the exact convention
    NormalizeStatesAndActions' state_norm_mask/action_norm_mask already
    assumes (e.g. pi05_parcel_sort.py's
    `[True]*_JOINT_DIM + [False]*(_MODEL_ACTION_DIM-_JOINT_DIM)`). Zeroing
    in place instead would leave gaps in the middle of the vector at the
    excluded joints' original positions, which does not match that
    contiguous-prefix mask convention.

    Must run BEFORE NormalizeStatesAndActions in the transform pipeline, and
    `pad_to` must equal that transform's `state_dim`/`action_dim` (typically
    the model's `_MODEL_ACTION_DIM`), so the padded zeros land exactly where
    state_norm_mask/action_norm_mask already expect padding and skip
    normalizing them.

    Args:
        exclude_joint_names (Sequence[str]): Joint names to drop, e.g.
            ["body_joint1", "body_joint2", "body_joint3", "body_joint4",
            "head_joint1", "head_joint2"] to drop the two non-arm/hand
            joint groups in ros2_data_collection/config/recorder.default.yaml.
        pad_to (int | None): Target last-dim size after dropping and
            compacting. Must be >= (original dim count - len(
            exclude_joint_names)). If None, no padding is applied and the
            output is simply shorter by len(exclude_joint_names).
        state_key (str): Key in `data` holding the current-frame state
            array. Default: 'states'.
        action_key (str): Key in `data` holding the action window array.
            Default: 'actions'.
        state_feature_name (str): Key into
            `data['info']['features']` for the state names list.
            Default: 'observation.state'.
        action_feature_name (str): Key into
            `data['info']['features']` for the action names list.
            Default: 'action'.
    """

    def __init__(self,
                 exclude_joint_names: Sequence[str],
                 pad_to: Union[int, None] = None,
                 state_key: str = 'states',
                 action_key: str = 'actions',
                 state_feature_name: str = 'observation.state',
                 action_feature_name: str = 'action',
                 *args,
                 **kwargs) -> None:
        self.exclude_joint_names = list(exclude_joint_names)
        self.pad_to = pad_to
        self.state_key = state_key
        self.action_key = action_key
        self.state_feature_name = state_feature_name
        self.action_feature_name = action_feature_name
        # (feature_name, names tuple) -> resolved keep-indices. A dataset's
        # joint order is fixed for its whole lifetime, so this is safe to
        # cache across samples/episodes rather than re-resolving every call,
        # while still being recomputed correctly if two datasets with
        # different orders are ever mixed in one process (different cache
        # key).
        self._index_cache: Dict = {}

    def _resolve_keep_indices(self, info: Dict, feature_name: str) -> List[int]:
        features = info.get('features') if info else None
        names = (features or {}).get(feature_name, {}).get('names')
        assert names, (
            f"SelectJointDims needs data['info']['features']['{feature_name}']"
            "['names'] to look up joints by name, but it is missing/empty "
            'for this sample. Was this dataset recorded with joint_names '
            'set in its recorder config?')

        cache_key = (feature_name, tuple(names))
        cached = self._index_cache.get(cache_key)
        if cached is not None:
            return cached

        name_set = set(names)
        missing = [
            name for name in self.exclude_joint_names if name not in name_set
        ]
        assert not missing, (
            f"exclude_joint_names {missing} not found in this dataset's "
            f"declared '{feature_name}' names {names} -- check "
            'exclude_joint_names against this dataset\'s actual recorder '
            'config (e.g. recorder.default.yaml\'s joint_names) instead of '
            'assuming a fixed position.')

        exclude_set = set(self.exclude_joint_names)
        keep_indices = [
            i for i, name in enumerate(names) if name not in exclude_set
        ]
        self._index_cache[cache_key] = keep_indices
        return keep_indices

    def _select_and_pad(self, values, keep_indices: List[int]) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        selected = values[..., keep_indices]
        if self.pad_to is None:
            return selected

        n_keep = selected.shape[-1]
        pad_amount = self.pad_to - n_keep
        assert pad_amount >= 0, (
            f'pad_to={self.pad_to} is smaller than the {n_keep} dims left '
            f'after dropping {len(self.exclude_joint_names)} joint(s) -- '
            'raise pad_to or drop fewer joints.')
        if pad_amount == 0:
            return selected
        pad_width = [(0, 0)] * (selected.ndim - 1) + [(0, pad_amount)]
        return np.pad(selected, pad_width, mode='constant', constant_values=0.0)

    def __call__(self, data: Dict) -> Dict:
        assert 'info' in data, (
            "SelectJointDims needs data['info'] (populated by "
            'ParquetDataset/ParquetDatasetV3-family datasets) to look up '
            'joint names; it was not found in this sample.')
        info = data['info']

        if self.state_key in data:
            idx = self._resolve_keep_indices(info, self.state_feature_name)
            data[self.state_key] = self._select_and_pad(
                data[self.state_key], idx)

        if self.action_key in data:
            idx = self._resolve_keep_indices(info, self.action_feature_name)
            data[self.action_key] = self._select_and_pad(
                data[self.action_key], idx)

        return data
