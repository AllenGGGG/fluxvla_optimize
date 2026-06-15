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

"""TempoVLA Speed-Modulated training on LIBERO-10 full dataset.

This config trains on ALL 10 LIBERO-10 tasks with speed modulation.

Training:
    python train.py --config configs/pi05/pi05_libero10_full_tempovla_speed_modulated.py

Evaluation:
    python scripts/compare_pi05_task0_with_l_pixshuffle.py \\
        --tag libero10_full_tempovla \\
        --variant speed_modulated \\
        configs/pi05/pi05_libero10_task0_tempovla_speed_modulated_inference.py \\
        work_dirs/libero10_full_tempovla/checkpoints/latest-checkpoint.safetensors \\
        --base-weights work_dirs/libero10_full_tempovla/checkpoints/latest-checkpoint.safetensors \\
        --eval-speeds 0.5,0.75,1.0,1.25,1.5,1.75,2.0

Difference from task0-only:
    - task_indices: [0] → [0,1,2,3,4,5,6,7,8,9]
    - 10x more training data
    - Model needs to generalize across all LIBERO-10 tasks
"""

_base_ = './pi05_libero10_task0_tempovla_speed_modulated.py'

# Override to use all 10 tasks
train_dataloader = dict(
    dataset=dict(
        datasets=dict(
            task_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # All 10 LIBERO-10 tasks
        )))

# Note: All other settings (tempo_speeds, speed_mlp, etc.) are inherited
# from the base config pi05_libero10_task0_tempovla_speed_modulated.py
