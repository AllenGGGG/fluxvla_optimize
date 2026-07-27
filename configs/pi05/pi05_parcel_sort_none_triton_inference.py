# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Triton + CUDA Graph parcel-sort inference without RTC.

The training architecture and checkpoint contract stay unchanged: three
camera views, a normal linear vision projector, 32-D model state/action
tensors, and a 50-step action horizon.  Only inference implementations are
replaced by their fused Triton variants. TempoVLA, pixel shuffle, and the
slower-at-horizon-50 Ultra-V2 ffn-gate path are not used.

The guidance/prefix Triton configs inherit this shared accelerated model
definition and only replace the RTC-capable model class/method.
"""

_base_ = ['./pi05_parcel_sort_none_pytorch_inference.py']

_JOINT_DIM = 28
_MODEL_ACTION_DIM = 32
_ACTION_HORIZON = 50
_NUM_VIEWS = 3
_PROMPT_MAX_LEN = 200

# Keep preprocessing and the static CUDA Graph on the same fixed prompt size
# used by training. The prompt text and valid token count may vary, while both
# paths retain the same 200-row padded attention/KV layout.
#
# This redefines dataset.transforms wholesale (mmengine replaces lists, not
# deep-merges them), so EpisodeMetadataPrompter must be repeated here too --
# see the base pytorch config's comment for why desired_advantage=True.
dataset = dict(
    transforms=[
        dict(
            type='NormalizeStatesAndActions',
            action_dim=_MODEL_ACTION_DIM,
            state_dim=_MODEL_ACTION_DIM,
            state_key='proprio',
            action_key='action',
            norm_type='min_max',
            action_norm_mask=[True] * _JOINT_DIM + [False] *
            (_MODEL_ACTION_DIM - _JOINT_DIM),
        ),
        dict(
            type='EpisodeMetadataPrompter',
            training=False,
            control_mode='joint',
            desired_advantage=True),
        dict(type='PreparePromptWithState'),
        dict(
            type='ProcessPrompts',
            max_len=_PROMPT_MAX_LEN,
            tokenizer=dict(
                type='PretrainedTokenizer',
                model_path='checkpoints/pi05_base',
            ),
        ),
        dict(type='ResizeImages', height=224, width=224),
        dict(type='SimpleNormalizeImages'),
    ],
)

inference_model = dict(
    _delete_=True,
    type='PI05FlowMatchingInference',
    num_views=_NUM_VIEWS,
    # Match ProcessPrompts so no state values or trailing Action marker are
    # truncated before entering the static graph.
    triton_max_prompt_len=_PROMPT_MAX_LEN,
    llm_backbone=dict(
        type='ConditionGemmaInferenceModel',
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
        type='SigLIPViTBackboneInference',
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
        type='LinearProjectorInference',
        in_dim=1152,
        out_dim=2048,
    ),
    proj_width=1024,
    n_action_steps=_ACTION_HORIZON,
    action_in_proj=dict(
        type='LinearProjectorInference',
        in_dim=_MODEL_ACTION_DIM,
        out_dim=1024,
    ),
    action_out_proj=dict(
        type='LinearProjectorInference',
        in_dim=1024,
        out_dim=_MODEL_ACTION_DIM,
    ),
    time_mlp_in=dict(
        type='LinearProjectorInference', in_dim=1024, out_dim=1024),
    time_mlp_out=dict(
        type='LinearProjectorInference', in_dim=1024, out_dim=1024),
    max_action_dim=_MODEL_ACTION_DIM,
    llm_expert=dict(
        type='ConditionGemmaInferenceModel',
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
        vocab_size=257152,
    ),
    freeze_llm_backbone=False,
    freeze_vision_backbone=False,
    pretrained_name_or_path='./checkpoints/pi05_base/model.safetensors',
    name_mapping={
        'llm_backbone':
        'paligemma_with_expert.paligemma.model.language_model',
        'vision_backbone.vision':
        'paligemma_with_expert.paligemma.model.vision_tower',
        'projector.projector':
        'paligemma_with_expert.paligemma.model.multi_modal_projector.linear',
        'llm_expert': 'paligemma_with_expert.gemma_expert.model',
        'time_mlp_in.projector': 'time_mlp_in',
        'time_mlp_out.projector': 'time_mlp_out',
        'action_in_proj.projector': 'action_in_proj',
        'action_out_proj.projector': 'action_out_proj',
        'llm_backbone.embed_tokens':
        'paligemma_with_expert.paligemma.lm_head',
    },
    ori_action_dim=_JOINT_DIM,
)

inference_options = dict(rtc_method='none')
