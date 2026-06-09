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

retimed_data_roots = [
    './datasets/libero_10_task0_retimed/speed_0_5',
    './datasets/libero_10_task0_retimed/speed_0_75',
    './datasets/libero_10_task0_retimed/speed_1_0',
    './datasets/libero_10_task0_retimed/speed_1_25',
    './datasets/libero_10_task0_retimed/speed_1_5',
    './datasets/libero_10_task0_retimed/speed_1_75',
    './datasets/libero_10_task0_retimed/speed_2_0',
]

train_dataloader = dict(
    dataset=dict(
        datasets=dict(
            data_root_path=retimed_data_roots,
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
                    speed_prompt_template=(
                        '{task_description} at {speed:g}x speed')),
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
            ])))

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
                speed_prompt_template=(
                    '{task_description} at {speed:g}x speed'),
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
