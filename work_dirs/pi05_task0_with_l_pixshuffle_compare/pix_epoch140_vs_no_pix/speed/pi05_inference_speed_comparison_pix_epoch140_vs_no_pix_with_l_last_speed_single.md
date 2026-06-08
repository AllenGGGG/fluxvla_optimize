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
| accelerated | 38.831 | 38.786 | 38.869 | 39.966 | 25.753 | 257.526 | 1362.404 | 12.075 |

## Run Details

| Variant | Config | Checkpoint | CUDA_VISIBLE_DEVICES | Dtype | Prompt len | Iters |
| --- | --- | --- | --- | --- | ---: | ---: |
| accelerated | `work_dirs/pi05_task0_with_l_pixshuffle_compare/pix_epoch140_vs_no_pix/configs/pi05_task0_with_l_last_accelerated.py` | `work_dirs/pi05_libero10_task0_with_l/checkpoints/latest-checkpoint.safetensors` | `7` | bfloat16 | 32 | 100 |
