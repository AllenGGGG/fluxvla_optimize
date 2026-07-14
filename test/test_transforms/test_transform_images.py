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

from fluxvla.transforms.transform_images import RandomMaskImages


class TestRandomMaskImages(unittest.TestCase):

    def _make_images(self, n=3, h=64, w=64):
        # Matches ResizeImages/SimpleNormalizeImages's convention: N cameras'
        # RGB channels concatenated along axis 0 into a single (N*3, H, W)
        # array, not a list of separate (3, H, W) images.
        rng = np.random.default_rng(0)
        return rng.uniform(0, 255, size=(n * 3, h, w)).astype(np.float32)

    def test_zero_prob_leaves_images_unchanged(self):
        transform = RandomMaskImages(
            num_masks_range=(1, 3), mask_size_range=(0.1, 0.3), prob=0.0)
        images = self._make_images()
        out = transform({'images': images.copy()})
        np.testing.assert_array_equal(images, out['images'])

    def test_output_shape_matches_input(self):
        transform = RandomMaskImages(
            num_masks_range=(1, 3), mask_size_range=(0.1, 0.3), prob=1.0)
        images = self._make_images()
        out = transform({'images': images.copy()})
        self.assertEqual(out['images'].shape, (9, 64, 64))

    def test_masked_region_equals_image_mean(self):
        np.random.seed(0)
        transform = RandomMaskImages(
            num_masks_range=(1, 1), mask_size_range=(0.5, 0.5), prob=1.0)
        image = np.zeros((3, 32, 32), dtype=np.float32)
        image[0] = 10.0
        image[1] = 20.0
        image[2] = 30.0
        expected_mean = image.mean(axis=(1, 2))

        out = transform({'images': image.copy()})
        masked = out['images']
        for ch in range(3):
            self.assertTrue(
                np.any(np.isclose(masked[ch], expected_mean[ch])))

    def test_masking_changes_some_pixels(self):
        images = self._make_images(n=1)
        transform = RandomMaskImages(
            num_masks_range=(2, 2), mask_size_range=(0.05, 0.05), prob=1.0)
        out = transform({'images': images.copy()})
        num_changed = np.sum(~np.isclose(out['images'], images))
        self.assertGreater(num_changed, 0)


if __name__ == '__main__':
    unittest.main()
