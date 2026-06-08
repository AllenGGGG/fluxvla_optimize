# PI0.5 Inference Speed Comparison

This benchmark measures model-side `predict_action` latency only.

## Benchmark Settings

- Warmup iterations: `10`
- Measured iterations per worker: `100`
- Prompt length: `32`
- Camera views: `2`
- Action chunk size: `10` action steps per `predict_action` call

Only measured iterations are used for Mean / P50 / P90 / P99 / Hz.

## Summary

| Variant | Mean ms | P50 ms | P90 ms | P99 ms | Hz | Action-step Hz | Cold start ms | Peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| accelerated | 30.338 | 30.313 | 30.391 | 30.635 | 32.962 | 329.625 | 4854.892 | 12.036 |

## Run Details

| Variant | Config | Checkpoint | CUDA_VISIBLE_DEVICES | Dtype | Prompt len | Iters |
| --- | --- | --- | --- | --- | ---: | ---: |
| accelerated | `configs/pi05/pi05_libero10_task0_with_l_pixshuffle_inference.py` | `work_dirs/pi05_libero10_task0_with_l_pixshuffle/checkpoints/step-008460-epoch-60-loss=0.0918.safetensors` | `3` | bfloat16 | 32 | 100 |
