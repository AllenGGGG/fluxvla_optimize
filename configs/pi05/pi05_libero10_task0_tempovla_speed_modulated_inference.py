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

"""Inference config for TempoVLA Speed-Modulated RMSNorm with Triton acceleration.

This config is for deployment/demo that requires low-latency inference (< 50ms).
It uses PI05FlowMatchingSpeedModulatedInference with Triton/CUDA Graph acceleration.

Usage:
    # Load checkpoint from training
    python inference.py --config configs/pi05/pi05_libero10_task0_tempovla_speed_modulated_inference.py \
                        --checkpoint checkpoints/speed_modulated/latest-checkpoint.pt \
                        --tempo_speed 1.5

Training config: pi05_libero10_task0_tempovla_speed_modulated.py
"""

# Base on the official Triton inference config so all submodules are inference
# variants with prepare_triton() implementations.
_base_ = './pi05_libero10_task0_with_l_pixshuffle_mlp_inference.py'

# Override the accelerated inference model while inheriting the inference
# backbone/projector/action-time module definitions from the base config.
inference_model = dict(
    type='PI05FlowMatchingSpeedModulatedInference',
    speed_mlp_hidden_dim=256,
    default_tempo_speed=1.0,  # Default speed for inference
)

# Note: This config assumes the checkpoint was trained with
# pi05_libero10_task0_tempovla_speed_modulated.py
# The speed_mlp weights will be loaded from the checkpoint.
#
# The model structure (backbone, projector, etc.) is inherited from the
# inference base config. These are the variants that provide prepare_triton().
