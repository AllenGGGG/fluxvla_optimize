# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Speed-modulated TempoVLA training with delayed visual frames.

This keeps the same speed-modulated, pixshuffle, OnlineVSTA setup used by
full_task_finetune_v2, but decodes images from t-d while actions/states stay
anchored at t.
"""

_base_ = './pi05_libero10_task0_tempovla_speed_modulated.py'

train_dataloader = dict(
    dataset=dict(
        datasets=dict(
            task_indices=[0],
            visual_delay_max_steps=5,
            visual_delay_min_steps=1,
            visual_delay_distribution='uniform',
            visual_delay_probability=1.0,
        )),
)

runner = dict(
    collator=dict(
        meta_keys=[
            'task_description', 'prompt', 'info', 'stats',
            'visual_delay_steps'
        ]),
)
