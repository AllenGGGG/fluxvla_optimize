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

"""Ultra-V2 inference with DCT delta action-chunk postprocessing.

The DCT step runs after action denormalization, on the real LIBERO 7D action
space. The first six dimensions are treated as delta EEF control commands and
the gripper dimension is sampled by nearest neighbor.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/eval.py \
        --config configs/pi05/pi05_libero10_task0_tempovla_ultra_v2_dct_delta_inference.py \
        --ckpt-path work_dirs/libero10_full_tempovla_speed_modulated/checkpoints/latest-checkpoint.safetensors \
        --cfg-options eval.action_postprocess.speed=1.5
"""

_base_ = './pi05_libero10_task0_tempovla_ultra_v2_inference.py'

eval = dict(
    measure_predict_latency=True,
    action_postprocess=dict(
        type='dct_delta',
        action_space='delta',
        speed=1.5,
        keep_ratio=0.4,
        cont_idx=[0, 1, 2, 3, 4, 5],
        gripper_idx=[6],
    ),
)
