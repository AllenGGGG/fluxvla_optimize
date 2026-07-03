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

"""RTC-like decoupled prefix/suffix evaluation for textual TempoVLA/VSTA.

Uses PI05FlowMatchingDecoupledInference which maintains two CUDA Graphs:
- Full graph (VLM + Encoder + Decoder)
- Decoder-only prefix graph (reuses encoder K/V from last full call)

Execution pattern:
- Full call predicts A0:A10 and executes A0:A5.
- Decoder-only call clamps A0:A5 during denoising and executes the generated
  suffix A5:A10.

Run the actual 5+5 alternating evaluation with:
    CUDA_VISIBLE_DEVICES=1 python scripts/eval.py \
        --config configs/pi05/pi05_libero10_task0_tempovla_decoupled_inference.py \
        --ckpt-path work_dirs/mlp_tempovla/checkpoints/latest-checkpoint.safetensors

For speed/correctness verification only:
    CUDA_VISIBLE_DEVICES=1 python scripts/verify_decoupled.py
"""

_base_ = './pi05_libero10_task0_with_l_pixshuffle_mlp_inference.py'

tempo_prompt_template = '{task_description} at {speed:g}x speed'

inference_model = dict(
    type='PI05FlowMatchingDecoupledInference',
    exec_chunk_size=5,
)

eval = dict(
    decoupled_alternating=True,
    decoupled_exec_chunk_size=5,
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
        ]),
)
