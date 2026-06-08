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
| accelerated | 39.185 | 38.752 | 38.869 | 49.654 | 127.626 | 1276.264 | 2638.298 | 60.373 |

## Multi-GPU Throughput

| Variant | Workers | Total Hz | Per-worker Hz | Total Action-step Hz | Per-worker Mean ms | Total Peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| accelerated | 5 | 127.626 | 25.525 | 1276.264 | 39.185 | 60.373 |

### Per-Worker Details

#### accelerated

| Worker | Device | Mean ms | P90 ms | Hz | Peak GiB |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | cuda:0 | 40.247 | 38.972 | 24.847 | 12.075 |
| 1 | cuda:1 | 39.060 | 38.847 | 25.602 | 12.075 |
| 2 | cuda:2 | 38.701 | 38.807 | 25.839 | 12.075 |
| 3 | cuda:3 | 38.793 | 38.834 | 25.778 | 12.075 |
| 4 | cuda:4 | 39.123 | 38.882 | 25.561 | 12.075 |


## Run Details

| Variant | Config | Checkpoint | CUDA_VISIBLE_DEVICES | Dtype | Prompt len | Iters |
| --- | --- | --- | --- | --- | ---: | ---: |
| accelerated | `work_dirs/pi05_task0_with_l_pixshuffle_compare/pix_epoch140_vs_no_pix/configs/pi05_task0_with_l_last_accelerated.py` | `work_dirs/pi05_libero10_task0_with_l/checkpoints/latest-checkpoint.safetensors` | `1,4,5,6,7` | bfloat16 | 32 | 100 |
