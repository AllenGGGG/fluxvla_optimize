# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Normal-speed v1 checkpoint eval with DCT delta postprocessing speed=1.75.

The VLA still predicts a normal 10-step chunk. The postprocess stage derives
the executed chunk length from speed: int(10 / 1.75) = 5 steps.
"""

_base_ = './pi05_libero10_task0_with_l_pixshuffle_mlp_inference.py'

eval = dict(
    action_postprocess=dict(
        type='dct_delta',
        action_space='delta',
        speed=1.75,
        keep_ratio=0.4,
        cont_idx=[0, 1, 2, 3, 4, 5],
        gripper_idx=[6],
    ),
)
