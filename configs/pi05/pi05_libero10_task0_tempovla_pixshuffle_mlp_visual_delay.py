# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""TempoVLA training with delayed visual frames.

The action/state sample stays anchored at time t, while images are decoded
from t-d. This matches asynchronous VLM/AE serving where visual features may
be stale relative to the action decoder target.
"""

_base_ = './pi05_libero10_task0_tempovla_pixshuffle_mlp.py'

train_dataloader = dict(
    dataset=dict(
        datasets=dict(
            visual_delay_max_steps=5,
            visual_delay_min_steps=1,
            visual_delay_distribution='uniform',
            visual_delay_probability=1.0,
        )),
)

runner = dict(
    collator=dict(
        meta_keys=[
            'task_description', 'prompt', 'info', 'stats', 'tempo_speed',
            'visual_delay_steps'
        ]),
)
