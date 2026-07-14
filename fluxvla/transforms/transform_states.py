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

from typing import List, Union

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
