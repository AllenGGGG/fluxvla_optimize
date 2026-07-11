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

import unittest

import numpy as np

from fluxvla.transforms.normalize import NormalizeStatesAndActions


class TestNormalizeStatesAndActions(unittest.TestCase):

    def test_masks_are_matched_before_padding(self):
        transform = NormalizeStatesAndActions(
            state_key='proprio',
            action_key='action',
            state_dim=4,
            action_dim=4,
            norm_type='min_max',
            state_norm_mask=[True, True, False, False],
            action_norm_mask=[True, True, False, False])
        data = {
            'states': np.array([5.0, 10.0], dtype=np.float32),
            'actions': np.array([[2.5, 15.0], [7.5, 5.0]],
                                dtype=np.float32),
            'stats': {
                'proprio': {
                    'min': [0.0, 0.0],
                    'max': [10.0, 20.0],
                },
                'action': {
                    'min': [0.0, 0.0],
                    'max': [10.0, 20.0],
                },
            },
        }

        output = transform(data)

        np.testing.assert_allclose(
            output['states'], [0.0, 0.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(
            output['actions'],
            [[-0.5, 0.5, 0.0, 0.0], [0.5, -0.5, 0.0, 0.0]],
            atol=1e-6)


if __name__ == '__main__':
    unittest.main()
