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

from fluxvla.datasets.parquet_dataset_v3 import ParquetDatasetV3


class TestParquetDatasetV3(unittest.TestCase):

    def test_empty_action_window_is_padded_from_current_action(self):
        dataset = ParquetDatasetV3.__new__(ParquetDatasetV3)
        dataset.dataset = [
            {
                'episode_index': 0,
                'task': 'parcel sort',
                'action': [1.0, 2.0],
            },
            {
                'episode_index': 0,
                'task': 'parcel sort',
                'action': [3.0, 4.0],
            },
        ]
        dataset.dataset_cumulative_sizes = np.array([0, 2])
        dataset.action_window_size = 3
        dataset.action_key = 'action'
        dataset.use_delta = False
        dataset.statistic_name = 'private'
        dataset.window_start_idx = 2
        dataset.frame_window_size = 1
        dataset.expose_index = False
        dataset.info = [{}]
        dataset.tasks = [{}]
        dataset.episodes_by_dataset = [[]]
        dataset.episode_records_by_index = [{}]
        dataset.data_root_path = ['/tmp/dataset']
        dataset.transforms = []

        output = dataset.__getitem__(0, {'private': {}})

        np.testing.assert_array_equal(output['actions'],
                                      [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        np.testing.assert_array_equal(output['action_masks'], [0.0, 0.0, 0.0])


if __name__ == '__main__':
    unittest.main()
