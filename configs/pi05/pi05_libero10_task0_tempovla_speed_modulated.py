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
# TempoVLA Method 2: Speed-Modulated RMSNorm (NO PROMPT)
# ========================================================================
# Speed is embedded via a small MLP and added to the flow-matching timestep
# embedding, which then modulates the action expert's RMSNorm layers.
# Speed is NOT included in the text prompt.
#
# Reference: TempoVLA paper Section 3.3, Method 2
#
# For Method 1 (textual prefix with speed in prompt), use:
#   pi05_libero10_task0_tempovla_online_vsta_pixshuffle_mlp.py
# ========================================================================

tempo_speeds = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

# Change model to speed-modulated version
model = dict(
    type='PI05FlowMatchingSpeedModulated',
    speed_mlp_hidden_dim=256  # Hidden dim for speed MLP
)

train_dataloader = dict(
    dataset=dict(
        datasets=dict(
            type='OnlineVSTATempoParquetDataset',
            data_root_path='./datasets/libero_10_no_noops_lerobotv2.1',
            task_indices=[0],
            tempo_speeds=tempo_speeds,
            tempo_speed_key='tempo_speed',
            keep_last_action_dim_nearest=True,
            random_phase=True,
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
                    # No speed in prompt - speed goes through MLP modulation
                    ),
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
                    state_dim=32,  # No change to state_dim
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
        keys=[
            'states', 'observation.eepose', 'timestamp', 'images', 'img_masks',
            'lang_tokens', 'lang_masks', 'actions', 'action_masks', 'tempo_speed'
        ],
        meta_keys=['task_description', 'prompt', 'info', 'stats']),
    metric=dict(active_trackers=('jsonl', )))

eval = dict(
    # Note: Current eval runner does not pass tempo_speed to predict_action.
    # The model defaults to 1.0x speed during evaluation, which matches the
    # paper's evaluation protocol.
    # To evaluate at different speeds, modify libero_eval_runner.py to add
    # batch['tempo_speed'] = desired_speed before predict_action(**batch).
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
                # No speed in prompt
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
