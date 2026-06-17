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

"""Standard inference config WITHOUT RealTimeVLA ultra optimizations.

This config uses the standard Triton/CUDA Graph path without aggressive fusion.
Use this as a baseline to compare against the ultra-optimized version.

Expected latency: ~40-50ms on A100

Usage:
    python scripts/eval.py --config configs/pi05/pi05_libero10_task0_tempovla_standard_inference.py \
                           --checkpoint checkpoints/speed_modulated/latest-checkpoint.safetensors

Compare with ultra version:
    configs/pi05/pi05_libero10_task0_tempovla_speed_modulated_inference.py

Training config: pi05_libero10_task0_tempovla_speed_modulated.py
"""

_base_ = './pi05_libero10_task0_tempovla_speed_modulated.py'

# Override the accelerated inference model - STANDARD version (no ultra fusion)
inference_model = dict(
    type='PI05FlowMatchingSpeedModulatedInference',
    speed_mlp_hidden_dim=256,
    default_tempo_speed=1.0,
    use_ultra_fusion=False,  # DISABLED for baseline comparison
)
