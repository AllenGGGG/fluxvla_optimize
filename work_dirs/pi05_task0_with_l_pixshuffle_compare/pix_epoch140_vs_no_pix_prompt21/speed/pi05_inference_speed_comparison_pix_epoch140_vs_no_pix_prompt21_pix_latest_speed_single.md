# PI0.5 Inference Speed Comparison

This benchmark measures model-side `predict_action` latency only.

## Benchmark Settings

- Warmup iterations: `10`
- Measured iterations per worker: `100`
- Prompt length: `21`
- Camera views: `2`
- Action chunk size: `10` action steps per `predict_action` call

Only measured iterations are used for Mean / P50 / P90 / P99 / Hz.

## Summary

| Variant | Mean ms | P50 ms | P90 ms | P99 ms | Hz | Action-step Hz | Cold start ms | Peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| accelerated | 31.678 | 30.767 | 31.436 | 48.348 | 31.568 | 315.677 | 12008.719 | 12.033 |

## Run Details

| Variant | Config | Checkpoint | CUDA_VISIBLE_DEVICES | Dtype | Prompt len | Iters |
| --- | --- | --- | --- | --- | ---: | ---: |
| accelerated | `work_dirs/pi05_task0_with_l_pixshuffle_compare/pix_epoch140_vs_no_pix_prompt21/configs/pi05_task0_pix_latest_accelerated_prompt21.py` | `work_dirs/pi05_libero10_task0_with_l_pixshuffle/checkpoints/latest-checkpoint.safetensors` | `7` | bfloat16 | 21 | 100 |
