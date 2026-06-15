# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

_base_ = './pi05_libero10_task0_with_l_pixshuffle_mlp.py'

# ========================================================================
# TempoVLA Method 1: Textual Prefix (SPEED IN PROMPT)
# ========================================================================
# 1. Variable-Speed Trajectory Augmentation (VSTA): online accumulate-then-split
#    of linearly-composable actions (LIBERO uses delta commands that sum correctly).
# 2. Speed conditioning: append the sampled speed to the language instruction
#    as a textual prefix: "put the blue block in the drawer at 1.5x speed"
#
# Reference: TempoVLA paper Section 3.3, Method 1
#
# For Method 2 (speed-modulated RMSNorm without prompt), use:
#   pi05_libero10_task0_tempovla_speed_modulated.py
# ========================================================================

tempo_speeds = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
tempo_prompt_template = '{task_description} at {speed:g}x speed'

train_dataloader = dict(
    dataset=dict(
        datasets=dict(
            type='OnlineVSTATempoParquetDataset',
            data_root_path='./datasets/libero_10_no_noops_lerobotv2.1',
            task_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # All 10 LIBERO-10 tasks
            tempo_speeds=tempo_speeds,
            tempo_speed_key='tempo_speed',
            keep_last_action_dim_nearest=True,
            random_phase=True,  # Random phase to avoid alignment bias
            transforms=[
                dict(
                    type='ProcessParquetInputs',
                    parquet_keys=[
                        'observation.state', 'timestamp', 'actions', 'info',
                        'stats', 'action_masks', 'tempo_speed'
                    ],
                    video_keys=[
                        'observation.images.image',
                        'observation.images.wrist_image',
                    ],
                    name_mappings={
                        'observation.state': ['states'],
                        'actions': ['actions'],
                    }),
                dict(
                    type='ParquetPrompter',
                    use_conversation=False,
                    speed_key='tempo_speed',
                    speed_prompt_template=tempo_prompt_template),
                dict(
                    type='ProcessPrompts',
                    tokenizer=dict(type='PaligemmaTokenizer')),
                dict(type='ResizeImages', height=224, width=224),
                dict(
                    type='NormalizeImages',
                    means=[[123.515625, 116.04492188, 103.59375],
                           [123.515625, 116.04492188, 103.59375]],
                    stds=[[58.27148438, 57.02636719, 57.27539062],
                          [58.27148438, 57.02636719, 57.27539062]],
                ),
                dict(
                    type='NormalizeStatesAndActions',
                    action_dim=32,
                    state_dim=32,
                    state_key='proprio',
                    action_key='action',
                    norm_type='mean_std')
            ],
            action_window_size=10,
            action_key='action',
            statistic_name='libero_10_no_noops',
            window_start_idx=0)))

runner = dict(
    collator=dict(
        meta_keys=['task_description', 'prompt', 'info', 'stats',
                   'tempo_speed']),
    metric=dict(active_trackers=('jsonl', )))

eval = dict(
    dataset=dict(
        transforms=[
            dict(
                type='ProcessLiberoEvalInputs',
                img_keys=['agentview_image', 'robot0_eye_in_hand_image'],
            ),
            dict(
                type='TransformImage',
                image_resize_strategy='resize-naive',
                input_sizes=[[3, 224, 224], [3, 224, 224]],
                means=[[123.515625, 116.04492188, 103.59375],
                       [123.515625, 116.04492188, 103.59375]],
                stds=[[58.27148438, 57.02636719, 57.27539062],
                      [58.27148438, 57.02636719, 57.27539062]],
            ),
            dict(
                type='LiberoPromptFromInputs',
                use_conversation=False,
                speed=1.0,
                speed_prompt_template=tempo_prompt_template,
                tokenizer=dict(type='PaligemmaTokenizer')),
            dict(
                type='LiberoProprioFromInputs',
                norm_type='mean_std',
                pos_key='robot0_eef_pos',
                quat_key='robot0_eef_quat',
                gripper_key='robot0_gripper_qpos',
                state_dim=32,
                out_key='states'),
        ]))
