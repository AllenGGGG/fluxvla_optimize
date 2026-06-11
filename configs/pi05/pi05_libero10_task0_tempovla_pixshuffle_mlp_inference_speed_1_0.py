_base_ = './pi05_libero10_task0_with_l_pixshuffle_mlp_inference.py'

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
