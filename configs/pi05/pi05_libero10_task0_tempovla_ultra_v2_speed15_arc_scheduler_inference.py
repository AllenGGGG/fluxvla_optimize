# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Ultra-V2 tempo speed=1.5 + risk-scheduled adaptive retiming."""

_base_ = './pi05_libero10_task0_tempovla_ultra_v2_inference.py'

inference_model = dict(
    default_tempo_speed=1.5,
)

eval = dict(
    measure_predict_latency=True,
    action_postprocess=dict(
        type='arc_adaptive_delta',
        action_space='delta',
        speed=1.67,
        min_output_steps=6,
        max_output_steps=10,
        adaptive_min_scheduler=True,
        scheduler_risk_thresholds=[0.12, 0.30, 0.50, 0.72],
        scheduler_min_steps=[6, 7, 8, 9, 10],
        smooth_type='none',
        length_weight=1.0,
        accel_weight=0.35,
        turn_weight=0.35,
        gripper_weight=0.5,
        min_segment_weight=0.05,
        importance_power=1.0,
        max_delta_ratio=1.75,
        max_accel_ratio=2.0,
        accel_floor_delta_ratio=0.75,
        anchor_first_steps=1,
        anchor_first_steps_when_k_ge=6,
        first_step_guard=True,
        first_step_error_ratio=0.5,
        path_fidelity_guard=True,
        path_error_mean_ratio=0.035,
        path_error_max_ratio=0.10,
        metric_idx=[0, 1, 2],
        guard_mode='per_dim',
        gripper_sample='end',
        protect_gripper_change=True,
        protect_gripper_change_threshold=0.02,
        protected_min_output_steps=9,
        allow_zero_motion_passthrough=True,
        fallback_until_feasible=True,
        cont_idx=[0, 1, 2, 3, 4, 5],
        gripper_idx=[6],
    ),
)
