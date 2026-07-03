# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Ultra-fusion inference config for TempoVLA Speed-Modulated.

This is an A/B benchmark config. It keeps the same model/checkpoint structure as
the standard speed-modulated inference config, but enables the optional
RealTimeVLA-style decoder fusion path.
"""

_base_ = './pi05_libero10_task0_tempovla_speed_modulated_inference.py'

inference_model = dict(
    use_ultra_fusion=True,
)
