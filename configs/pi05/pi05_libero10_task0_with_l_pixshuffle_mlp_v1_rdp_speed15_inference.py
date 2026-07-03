# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Normal-speed v1 checkpoint eval with RDP path downsampling speed=1.5."""

_base_ = './pi05_libero10_task0_with_l_pixshuffle_mlp_inference.py'

eval = dict(
    action_postprocess=dict(
        type='rdp_delta',
        action_space='delta',
        speed=1.5,
        time_weight=0.05,
        rdp_max_iter=60,
        max_delta_ratio=1.75,
        max_accel_ratio=2.5,
        accel_floor_delta_ratio=0.75,
        anchor_first_steps=1,
        metric_idx=[0, 1, 2],
        include_gripper_in_metric=True,
        gripper_metric_weight=1.0,
        guard_mode='per_dim',
        gripper_sample='end',
        allow_zero_motion_passthrough=True,
        fallback_until_feasible=False,
        cont_idx=[0, 1, 2, 3, 4, 5],
        gripper_idx=[6],
    ),
)
