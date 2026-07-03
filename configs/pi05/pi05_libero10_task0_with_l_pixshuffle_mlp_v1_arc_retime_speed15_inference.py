# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Normal-speed v1 checkpoint eval with adaptive arc-length retiming speed=1.5.

The VLA still predicts a normal 10-step chunk. The postprocess stage retimes
the denormalized delta-action path toward 6 executed steps, while preserving
important motion/gripper regions and falling back to more steps if the retimed
chunk becomes too aggressive.
"""

_base_ = './pi05_libero10_task0_with_l_pixshuffle_mlp_inference.py'

eval = dict(
    action_postprocess=dict(
        type='arc_retime_delta',
        action_space='delta',
        speed=1.5,
        smooth_type='none',
        length_weight=1.0,
        accel_weight=0.35,
        turn_weight=0.35,
        gripper_weight=0.5,
        min_segment_weight=0.05,
        importance_power=1.0,
        max_delta_ratio=1.75,
        max_accel_ratio=2.5,
        accel_floor_delta_ratio=0.75,
        anchor_first_steps=1,
        metric_idx=[0, 1, 2],
        guard_mode='per_dim',
        gripper_sample='end',
        allow_zero_motion_passthrough=True,
        fallback_until_feasible=True,
        cont_idx=[0, 1, 2, 3, 4, 5],
        gripper_idx=[6],
    ),
)
