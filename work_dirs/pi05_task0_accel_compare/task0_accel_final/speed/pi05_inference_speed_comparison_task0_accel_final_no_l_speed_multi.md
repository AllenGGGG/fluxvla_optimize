# PI0.5 Inference Speed Comparison

This benchmark measures model-side `predict_action` latency only.

## Benchmark Settings

- Warmup iterations: `10`
- Measured iterations per worker: `100`
- Prompt length: `0`
- Camera views: `2`
- Action chunk size: `10` action steps per `predict_action` call

Only measured iterations are used for Mean / P50 / P90 / P99 / Hz.

## Summary

| Variant | Mean ms | P50 ms | P90 ms | P99 ms | Hz | Action-step Hz | Cold start ms | Peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| accelerated | 38.901 | 38.541 | 38.779 | 44.593 | 128.533 | 1285.329 | 2382.464 | 60.447 |

## Multi-GPU Throughput

| Variant | Workers | Total Hz | Per-worker Hz | Total Action-step Hz | Per-worker Mean ms | Total Peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| accelerated | 5 | 128.533 | 25.707 | 1285.329 | 38.901 | 60.447 |

### Per-Worker Details

#### accelerated

| Worker | Device | Mean ms | P90 ms | Hz | Peak GiB |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | cuda:0 | 39.020 | 38.973 | 25.628 | 12.089 |
| 1 | cuda:1 | 38.646 | 38.616 | 25.876 | 12.089 |
| 2 | cuda:2 | 38.896 | 38.767 | 25.709 | 12.089 |
| 3 | cuda:3 | 38.932 | 38.752 | 25.686 | 12.089 |
| 4 | cuda:4 | 39.011 | 38.788 | 25.634 | 12.089 |


## Run Details

| Variant | Config | Checkpoint | CUDA_VISIBLE_DEVICES | Dtype | Prompt len | Iters |
| --- | --- | --- | --- | --- | ---: | ---: |
| accelerated | `work_dirs/pi05_task0_accel_compare/task0_accel_final/configs/pi05_task0_no_l_accelerated.py` | `work_dirs/pi05_libero10_task0_no_l/checkpoints/latest-checkpoint.safetensors` | `3,4,5,6,7` | bfloat16 | 0 | 100 |
