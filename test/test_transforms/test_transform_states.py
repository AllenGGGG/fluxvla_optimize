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

from fluxvla.transforms.transform_states import AddStateGaussianNoise


class TestAddStateGaussianNoise(unittest.TestCase):

    def test_zero_std_leaves_states_unchanged(self):
        transform = AddStateGaussianNoise(noise_std=0.0, prob=1.0)
        states = np.array([0.1, -0.2, 0.5], dtype=np.float32)
        out = transform({'states': states.copy()})
        np.testing.assert_allclose(out['states'], states)

    def test_zero_prob_leaves_states_unchanged(self):
        transform = AddStateGaussianNoise(noise_std=0.5, prob=0.0)
        states = np.array([0.1, -0.2, 0.5], dtype=np.float32)
        out = transform({'states': states.copy()})
        np.testing.assert_allclose(out['states'], states)

    def test_noise_statistics_match_configured_std(self):
        np.random.seed(0)
        transform = AddStateGaussianNoise(noise_std=0.05, prob=1.0)
        states = np.zeros((5000, 4), dtype=np.float32)
        out = transform({'states': states})
        noise = out['states']
        self.assertAlmostEqual(float(noise.mean()), 0.0, delta=0.01)
        self.assertAlmostEqual(float(noise.std()), 0.05, delta=0.01)

    def test_output_shape_matches_input(self):
        transform = AddStateGaussianNoise(noise_std=0.01, prob=1.0)
        states = np.random.uniform(-1, 1, size=(8,)).astype(np.float32)
        out = transform({'states': states})
        self.assertEqual(out['states'].shape, states.shape)

    def test_per_dimension_std_preserves_padding(self):
        np.random.seed(0)
        transform = AddStateGaussianNoise(
            noise_std=[0.01] * 28 + [0.0] * 4, prob=1.0)
        states = np.zeros(32, dtype=np.float32)

        out = transform({'states': states})

        self.assertTrue(np.any(out['states'][:28] != 0.0))
        np.testing.assert_array_equal(out['states'][28:], 0.0)


if __name__ == '__main__':
    unittest.main()
