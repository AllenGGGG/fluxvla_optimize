# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Ultra-V2 inference for normal pixshuffle+MLP PI0.5 checkpoints.

This does not add TempoVLA speed conditioning. It only swaps the inference
decoder kernel to the Ultra-V2 ffn_gate path, so it is compatible with normal
PI05FlowMatching checkpoints such as origin_speedup16.
"""

_base_ = './pi05_libero10_task0_with_l_pixshuffle_mlp_inference.py'

inference_model = dict(
    type='PI05FlowMatchingPlainUltraV2Inference',
)
