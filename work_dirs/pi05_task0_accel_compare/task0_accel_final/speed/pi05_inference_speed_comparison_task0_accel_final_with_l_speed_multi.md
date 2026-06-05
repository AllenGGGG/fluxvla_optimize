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
| accelerated | 39.085 | 38.664 | 39.002 | 44.935 | 127.930 | 1279.299 | 2511.346 | 60.448 |

## Multi-GPU Throughput

| Variant | Workers | Total Hz | Per-worker Hz | Total Action-step Hz | Per-worker Mean ms | Total Peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| accelerated | 5 | 127.930 | 25.586 | 1279.299 | 39.085 | 60.448 |

### Per-Worker Details

#### accelerated

| Worker | Device | Mean ms | P90 ms | Hz | Peak GiB |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | cuda:0 | 39.100 | 39.025 | 25.575 | 12.090 |
| 1 | cuda:1 | 39.343 | 39.403 | 25.417 | 12.090 |
| 2 | cuda:2 | 38.637 | 38.781 | 25.882 | 12.090 |
| 3 | cuda:3 | 39.060 | 38.945 | 25.602 | 12.090 |
| 4 | cuda:4 | 39.288 | 38.856 | 25.453 | 12.090 |


## Run Details

| Variant | Config | Checkpoint | CUDA_VISIBLE_DEVICES | Dtype | Prompt len | Iters |
| --- | --- | --- | --- | --- | ---: | ---: |
| accelerated | `work_dirs/pi05_task0_accel_compare/task0_accel_final/configs/pi05_task0_with_l_accelerated.py` | `work_dirs/pi05_libero10_task0_with_l/checkpoints/latest-checkpoint.safetensors` | `3,4,5,6,7` | bfloat16 | 32 | 100 |
