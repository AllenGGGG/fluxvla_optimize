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
# Real-robot inference config for the pi05_parcel_sort checkpoint, for
# deploy/'s rtc_method="prefix" (see deploy/config.py's VALID_RTC_METHODS).
#
# Identical to pi05_parcel_sort_inference.py except inference_model.type:
# PI05FlowMatchingRTCInference (fluxvla/models/vlas/
# pi05_flowmatching_inference_rtc.py) is a Triton/CUDA-graph-accelerated
# variant that implements real RTC prefix guidance (prev_actions/prefix_len
# are baked into the same captured graph as data, not Python branching);
# plain PI05FlowMatchingInference silently ignores those kwargs. It only
# supports rtc_config['method']='prefix' (raises NotImplementedError for
# 'guidance') -- see pi05_parcel_sort_inference_guidance_rtc.py for that case.

_JOINT_DIM = 28
_MODEL_ACTION_DIM = 32
_ACTION_HORIZON = 50
_NUM_VIEWS = 3

inference_model = dict(
    type='PI05FlowMatchingRTCInference',
    num_views=_NUM_VIEWS,
    # Must be >= dataset.transforms' ProcessPrompts.max_len (200 below):
    # the Triton CUDA-graph prompt buffer overflows if a real prompt (task
    # text + 32-value discretized state) exceeds this bound. The class
    # default of 48 is far too small for PreparePromptWithState's prompts.
    triton_max_prompt_len=200,
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
        type='LinearProjectorInference', in_dim=_MODEL_ACTION_DIM,
        out_dim=1024),
    action_out_proj=dict(
        type='LinearProjectorInference', in_dim=1024,
        out_dim=_MODEL_ACTION_DIM),
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
    ori_action_dim=_JOINT_DIM,
)

# Same 5-transform pipeline as pi05_parcel_sort.py's training dataset,
# reused verbatim via PrivateInferenceDataset (fluxvla/datasets/
# parquet_dataset.py:531) so a training-side transform change (e.g.
# norm_type) propagates automatically instead of drifting.
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
            norm_type='quantile',
            state_norm_mask=[True] * _JOINT_DIM + [False] *
            (_MODEL_ACTION_DIM - _JOINT_DIM),
            action_norm_mask=[True] * _JOINT_DIM + [False] *
            (_MODEL_ACTION_DIM - _JOINT_DIM)),
        dict(type='PreparePromptWithState'),
        dict(
            type='ProcessPrompts',
            max_len=200,
            tokenizer=dict(
                type='PretrainedTokenizer',
                model_path=  # noqa: E251
                'checkpoints/pi05_base',  # noqa: E501
            )),
        dict(type='ResizeImages', height=224, width=224),
        dict(type='SimpleNormalizeImages'),
    ],
)

# norm_type must match whatever the deployed checkpoint was actually
# trained with (currently 'quantile', see train_dataloader.dataset.datasets
# in pi05_parcel_sort.py). Keep in sync by hand if that ever changes.
denormalize_action = dict(
    type='DenormalizePrivateAction',
    norm_type='quantile',
    action_dim=_MODEL_ACTION_DIM,
    action_norm_mask=[True] * _JOINT_DIM + [False] *
    (_MODEL_ACTION_DIM - _JOINT_DIM),
)
