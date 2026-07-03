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

"""Normal-speed v1 checkpoint eval with DCT delta postprocessing speed=1.5.

The VLA still predicts a normal 10-step chunk. The postprocess stage derives
the executed chunk length from speed: int(10 / 1.5) = 6 steps.
"""

_base_ = './pi05_libero10_task0_with_l_pixshuffle_mlp_inference.py'

eval = dict(
    action_postprocess=dict(
        type='dct_delta',
        action_space='delta',
        speed=1.5,
        keep_ratio=0.4,
        cont_idx=[0, 1, 2, 3, 4, 5],
        gripper_idx=[6],
    ),
)
