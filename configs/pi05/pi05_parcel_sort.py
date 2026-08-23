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
# PI0.5 parcel sorting real-robot fine-tuning.
# LeRobot v3 data with three RGB cameras and 30 Hz joint-space control.
# State/action use 32 values: 28 robot joints followed by 4 zero pads.
# For state[t], the 50-step target starts at action[t + 1].

_JOINT_DIM = 28
_MODEL_ACTION_DIM = 32
_ACTION_HORIZON = 50

model = dict(
    type='PI05FlowMatching',
    llm_backbone=dict(
        type='ConditionGemmaModel',
        adarms_cond_dim=None,
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=2,
        eos_token_id=1,
        head_dim=256,
        hidden_act='gelu_pytorch_tanh',
        hidden_activation='gelu_pytorch_tanh',
        hidden_size=2048,
        initializer_range=0.02,
        intermediate_size=16384,
        max_position_embeddings=8192,
        model_type='gemma',
        num_attention_heads=8,
        num_hidden_layers=18,
        num_key_value_heads=1,
        rms_norm_eps=1e-06,
        rope_theta=10000.0,
        torch_dtype='float32',
        use_cache=True,
        vocab_size=257152,
    ),
    vision_backbone=dict(
        type='SigLIPViTBackbone',
        vision_backbone_id='siglip_224',
        vision_config=dict(
            attention_dropout=0.0,
            hidden_act='gelu_pytorch_tanh',
            hidden_size=1152,
            image_size=224,
            intermediate_size=4304,
            layer_norm_eps=1e-06,
            model_type='siglip_vision_model',
            num_attention_heads=16,
            num_channels=3,
            num_hidden_layers=27,
            patch_size=14,
            projection_dim=2048,
            projector_hidden_act='gelu_fast',
            torch_dtype='float32',
            vision_use_head=False,
        ),
    ),
    projector=dict(
        type='LinearProjector',
        in_dim=1152,
        out_dim=2048,
    ),
    proj_width=1024,
    n_action_steps=_ACTION_HORIZON,
    action_in_proj=dict(
        type='LinearProjector', in_dim=_MODEL_ACTION_DIM, out_dim=1024),
    action_out_proj=dict(
        type='LinearProjector', in_dim=1024, out_dim=_MODEL_ACTION_DIM),
    time_mlp_in=dict(type='LinearProjector', in_dim=1024, out_dim=1024),
    time_mlp_out=dict(type='LinearProjector', in_dim=1024, out_dim=1024),
    max_action_dim=_MODEL_ACTION_DIM,
    llm_expert=dict(
        type='ConditionGemmaModel',
        attention_bias=False,
        adarms_cond_dim=1024,
        attention_dropout=0.0,
        bos_token_id=2,
        eos_token_id=1,
        head_dim=256,
        hidden_act='gelu_pytorch_tanh',
        hidden_activation='gelu_pytorch_tanh',
        hidden_size=1024,
        initializer_range=0.02,
        intermediate_size=4096,
        max_position_embeddings=8192,
        model_type='gemma',
        num_attention_heads=8,
        num_hidden_layers=18,
        num_key_value_heads=1,
        pad_token_id=0,
        rms_norm_eps=1e-06,
        rope_theta=10000.0,
        torch_dtype='float32',
        transformers_version='4.48.1',
        use_adarms=True,
        use_cache=True,
        vocab_size=257152),
    freeze_llm_backbone=False,
    freeze_vision_backbone=False,
    pretrained_name_or_path=  # noqa: E251
    '/home/guohao/fluxvla_optimize/checkpoints/pi05_base/model.safetensors',  # noqa: E501
    name_mapping={
        'llm_backbone': 'paligemma_with_expert.paligemma.model.language_model',
        'vision_backbone.vision':
        'paligemma_with_expert.paligemma.model.vision_tower',
        'projector.projector':
        'paligemma_with_expert.paligemma.model.multi_modal_projector.linear',
        'llm_expert': 'paligemma_with_expert.gemma_expert.model',
        'time_mlp_in.projector': 'time_mlp_in',
        'time_mlp_out.projector': 'time_mlp_out',
        'action_in_proj.projector': 'action_in_proj',
        'action_out_proj.projector': 'action_out_proj',
        'llm_backbone.embed_tokens': 'paligemma_with_expert.paligemma.lm_head',
    },
    params_to_change_dtype=[
        'llm_expert.llm.model.layers',
        'vlm_backbone.vlm.model.language_model.layers',
        'vlm_backbone.vlm.model.vision_tower',
        'vlm_backbone.vlm.model.multi_modal_projector',
    ],
    ori_action_dim=_JOINT_DIM,
)

train_dataloader = dict(
    per_device_batch_size=32,
    per_device_num_workers=8,
    dataset=dict(
        type='DistributedRepeatingDataset',
        statistic_keys=['observation.state', 'action'],
        name_mappings={
            'observation.state': 'proprio',
            'action': 'action',
        },
        datasets=[
            dict(
                type='PackedParquetDatasetV3',
                data_root_path=  # noqa: E251
                [
                    '/home/guohao/fluxvla_optimize/workdirs/dataset/2026_08_17',
                    '/home/guohao/fluxvla_optimize/workdirs/dataset/2026_08_13',
                    '/home/guohao/fluxvla_optimize/workdirs/dataset/2026_08_22',
                    '/home/guohao/fluxvla_optimize/workdirs/dataset/2026_08_21',
                ],
                video_keys=[
                    'observation.images.base_0_rgb',
                    'observation.images.left_wrist_0_rgb',
                    'observation.images.right_wrist_0_rgb',
                ],
                state_key='observation.state',
                action_key='action',
                statistic_name='private',
                window_start_idx=1,
                # Train absolute actions, matching the A100 training branch.
                use_delta=False,
                # This parcel-sort parquet data has no `advantage` column;
                # use_advantage's True default would raise at load time.
                use_advantage=False,
                # Dataset's own task text ("Pick up the parcel with the left
                # hand, then move it onto the conveyor belt with the right
                # hand.") is used as-is; no override needed.
                transforms=[
                    # Match the 2026_07_17 checkpoint: raw state padding is
                    # min-max normalized from 0 to -1 before prompt encoding.
                    dict(
                        type='NormalizeStatesAndActions',
                        action_dim=_MODEL_ACTION_DIM,
                        state_dim=_MODEL_ACTION_DIM,
                        state_key='proprio',
                        action_key='action',
                        norm_type='min_max',
                        action_norm_mask=[True] * _JOINT_DIM + [False] *
                        (_MODEL_ACTION_DIM - _JOINT_DIM)),
                    dict(type='AddStateGaussianNoise', noise_std=0.01),
                    dict(type='PreparePromptWithState'),
                    dict(
                        type='ProcessPrompts',
                        max_len=200,
                        tokenizer=dict(
                            type='PretrainedTokenizer',
                            model_path=  # noqa: E251
                            '/home/guohao/fluxvla_optimize/checkpoints/pi05_base',  # noqa: E501
                        )),
                    dict(type='ResizeImages', height=224, width=224),
                    dict(
                        type='RandomMaskImages',
                        num_masks_range=(0, 3),
                        mask_size_range=(0.05, 0.2),
                        prob=0.5),
                    dict(type='SimpleNormalizeImages'),
                ],
                action_window_size=_ACTION_HORIZON)
        ]))

runner = dict(
    type='FSDPTrainRunner',
    max_epochs=30,
    optimizer=dict(lr=5e-5, type='AdamW', weight_decay=0.01),
    max_grad_norm=1.0,
    sharding_strategy='full-shard',
    save_iter_interval=5000,
    collator=dict(
        type='DictCollator',
        keys=[
            'states', 'images', 'img_masks', 'lang_tokens', 'lang_masks',
            'actions', 'action_masks'
        ],
        meta_keys=['task_description', 'prompt', 'stats']),
    sampler=None,
    tokenizer=dict(
        type='PretrainedTokenizer',
        model_path=  # noqa: E251
        '/home/guohao/fluxvla_optimize/checkpoints/pi05_base',  # noqa: E501
    ),
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        grad_accumulation_steps=1,
        window_size=1),
    lr_scheduler=dict(
        type='linear-warmup+cosine-decay',
        warmup_ratio=0.03,
    ),
    enable_gradient_checkpointing=True,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    change_key_name=False)
