# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""RACE-lite fastest-feasible search from 5 steps, Ultra-V2 tempo=1.0."""

_base_ = './pi05_libero10_task0_race_lite_inference.py'

inference_model = dict(
    default_tempo_speed=1.0,
)

eval = dict(
    action_postprocess=dict(
        min_output_steps=5,
        scheduler_min_steps=[5, 6, 7, 8, 10],
    ),
)
