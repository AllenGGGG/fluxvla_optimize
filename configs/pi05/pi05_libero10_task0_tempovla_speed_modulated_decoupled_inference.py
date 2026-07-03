# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Speed-modulated 5+5 decoupled evaluation.

Full call refreshes VLM/encoder features and executes actions [0:5].
Decoder-only call reuses the previous visual/encoder cache, clamps [0:5],
generates the suffix, and executes actions [5:10].
"""

_base_ = './pi05_libero10_task0_tempovla_speed_modulated_inference.py'

inference_model = dict(
    type='PI05FlowMatchingSpeedModulatedDecoupledInference',
    exec_chunk_size=5,
)

eval = dict(
    decoupled_alternating=True,
    decoupled_exec_chunk_size=5,
    eval_chunk_size=10,
)
