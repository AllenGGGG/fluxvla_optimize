# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Task0 pixshuffle+MLP training with online safe action speedup.

This keeps the normal PI0.5 pixshuffle+MLP model and prompt path. The only
change from the baseline config is the training dataset: action windows are
retimed online with protected gripper/placement phases and a max-gap cap.
"""

_base_ = './pi05_libero10_task0_with_l_pixshuffle_mlp.py'

train_dataloader = dict(
    per_device_batch_size=8,
    per_device_num_workers=4,
    dataset=dict(
        datasets=dict(
            type='OnlineSafeSpeedupParquetDataset',
            target_speed=2.0,
            max_gap=4,
            final_protect_ratio=0.22,
            gripper_window=8,
            gripper_threshold=1e-4,
            min_keep=12,
            length_weight=1.0,
            accel_weight=0.5,
            turn_weight=0.5,
            gripper_weight=2.0,
            floor=0.04,
            recompute_action_stats=True,
            task_indices=[0],
        ),
    ),
)
