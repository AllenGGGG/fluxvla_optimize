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
# Complete plain PyTorch eager inference config for the parcel-sort checkpoint.

_JOINT_DIM = 28
_MODEL_ACTION_DIM = 32
_ACTION_HORIZON = 50
_NORM_TYPE = 'min_max'

inference_model = dict(
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
    './checkpoints/pi05_base/model.safetensors',  # noqa: E501
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

# Same transform pipeline as pi05_parcel_sort_recap.py's training dataset
# (norm_type, EpisodeMetadataPrompter's advantage-conditioning text), reused
# verbatim via PrivateInferenceDataset (fluxvla/datasets/
# parquet_dataset.py:531) so a training-side transform change propagates
# automatically instead of drifting. desired_advantage=True is a static
# eval-time prompt condition, not a runtime toggle -- see deploy/model.py's
# predict_chunk, which used to switch this via a CFG dual-pass; that
# mechanism is gone, replaced by this transform matching training's own
# "Advantage: true/false." text exactly (see
# fluxvla/transforms/prompters.py's EpisodeMetadataPrompter).
dataset = dict(
    type='PrivateInferenceDataset',
    img_keys=[
        'observation.images.base_0_rgb',
        'observation.images.left_wrist_0_rgb',
        'observation.images.right_wrist_0_rgb',
    ],
    transforms=[
        dict(
            type='NormalizeStatesAndActions',
            action_dim=_MODEL_ACTION_DIM,
            state_dim=_MODEL_ACTION_DIM,
            state_key='proprio',
            action_key='action',
            norm_type=_NORM_TYPE,
            action_norm_mask=[True] * _JOINT_DIM + [False] *
            (_MODEL_ACTION_DIM - _JOINT_DIM)),
        dict(
            type='EpisodeMetadataPrompter',
            training=False,
            control_mode='joint',
            desired_advantage=True),
        dict(type='PreparePromptWithState'),
        dict(
            type='ProcessPrompts',
            max_len=200,
            tokenizer=dict(
                type='PretrainedTokenizer',
                model_path='checkpoints/pi05_base',
            )),
        dict(type='ResizeImages', height=224, width=224),
        dict(type='SimpleNormalizeImages'),
    ],
)

denormalize_action = dict(
    type='DenormalizePrivateAction',
    norm_type=_NORM_TYPE,
    action_dim=_MODEL_ACTION_DIM,
    action_norm_mask=[True] * _JOINT_DIM + [False] *
    (_MODEL_ACTION_DIM - _JOINT_DIM),
)

inference_options = dict(
    num_inference_steps_override=10,
    rtc_method='none', # none guidance
    rtc_execution_horizon=10,
    # 0 = automatically keep inference-delay + 2 * RTC handoff steps.
    rtc_replan_remaining=0,
    chunk_publish_horizon=50,
    # Keep the RTC guidance window at 10. RTC keeps the full 50-step model
    # chunk so latency alignment does not consume the guidance prefix.
    rtc_publish_horizon=0,
    rtc_max_guidance_weight=5.0,
    rtc_schedule='linear',
)
