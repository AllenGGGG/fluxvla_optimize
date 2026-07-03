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

"""Inference config for Ultra-V2: autotuned ffn_gate kernel.

Uses PI05FlowMatchingUltraV2Inference which replaces only the ffn_gate kernel
with an autotuned version (1.46x faster at M=10 in microbench). Everything
else (attn_o, ffn_down, VLM, encoder, CUDA Graph, speed caching) is identical
to the standard speed-modulated inference path.

Expected improvement: ~1-3ms over standard path (端到端, not microbench extrapolation).

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/eval.py \
        --config configs/pi05/pi05_libero10_task0_tempovla_ultra_v2_inference.py \
        --ckpt-path work_dirs/libero10_full_tempovla_speed_modulated/checkpoints/latest-checkpoint.safetensors
"""

_base_ = './pi05_libero10_task0_with_l_pixshuffle_mlp_inference.py'

inference_model = dict(
    type='PI05FlowMatchingUltraV2Inference',
    speed_mlp_hidden_dim=256,
    default_tempo_speed=1.0,
    use_ultra_fusion=False,  # v2 does not use old ultra fusion path
)
