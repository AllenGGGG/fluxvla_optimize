# RealTimeVLA + TempoVLA Integration

## Current Status

The realtime-named TempoVLA entry point is safe to use with the existing
FluxVLA evaluation pipeline, but raw RealTimeVLA kernels are not enabled for
FluxVLA checkpoints yet.

`PI05RealTimeSpeedModulatedInference` currently inherits the stable
`PI05FlowMatchingSpeedModulatedInference` implementation. This preserves:

- FluxVLA checkpoint `load_state_dict(strict=True)` compatibility.
- Existing LIBERO `predict_action(**batch)` compatibility.
- Speed conditioning through `inference_model.default_tempo_speed`.
- No speed text in the prompt.

## Why Raw RealTimeVLA Is Disabled

The copied RealTimeVLA code expects a flattened checkpoint dictionary with keys
such as:

```text
decoder_time_embeds
decoder_time_mlp_in_w
decoder_action_in_proj_w
language_embeds
embedding_weight
```

FluxVLA training checkpoints use normal module state-dict keys such as:

```text
time_mlp_in.projector.weight
action_in_proj.projector.weight
llm_backbone.layers.*.self_attn.*
speed_mlp.0.weight
```

A complete and verified converter is required before those raw kernels can be
used safely. Until then, `enable_realtime_kernels=False` is the default.

## Config

Use:

```text
configs/pi05/pi05_libero10_task0_tempovla_realtime_inference.py
```

This config inherits:

```text
configs/pi05/pi05_libero10_task0_tempovla_speed_modulated_inference.py
```

and only changes the model type to:

```python
PI05RealTimeSpeedModulatedInference
```

## Evaluation

Use the same comparison command shape as standard speed-modulated inference:

```bash
setsid /home/guohao/miniconda3/envs/fluxvla/bin/python scripts/compare_pi05_task0_with_l_pixshuffle.py \
--tag realtime_tempovla_task4 \
--variant realtime \
configs/pi05/pi05_libero10_task0_tempovla_realtime_inference.py \
work_dirs/libero10_full_tempovla_speed_modulated/checkpoints/latest-checkpoint.safetensors \
--eval-speeds 0.5,0.75,1.0,1.25,1.5,1.75,2.0 \
--task-id 4 \
--skip-speed \
--success-trials-per-task 50 \
--success-seeds 7,8 \
--success-gpus 2,3,4,5,6,7 \
--success-nproc-per-node 6 \
> logs/realtime_tempovla_task4.log 2>&1 < /dev/null &
```

Do not pass the fine-tuned checkpoint as `--base-weights`; the positional
checkpoint argument is the fine-tuned checkpoint.

## Next Work To Enable Raw RealTimeVLA

1. Implement a FluxVLA-to-RealTimeVLA checkpoint converter.
2. Recompute `decoder_style_attn`, `decoder_style_ffn`, and
   `decoder_style_final` after adding speed conditioning.
3. Build a `predict_action(**batch)` adapter for the raw RealTimeVLA path.
4. Benchmark against the stable FluxVLA Triton path.

