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
| accelerated | 39.015 | 38.784 | 38.896 | 43.937 | 153.790 | 1537.898 | 2650.758 | 72.454 |

## Multi-GPU Throughput

| Variant | Workers | Total Hz | Per-worker Hz | Total Action-step Hz | Per-worker Mean ms | Total Peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| accelerated | 6 | 153.790 | 25.632 | 1537.898 | 39.015 | 72.454 |

### Per-Worker Details

#### accelerated

| Worker | Device | Mean ms | P90 ms | Hz | Peak GiB |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | cuda:0 | 39.101 | 38.954 | 25.575 | 12.076 |
| 1 | cuda:1 | 38.774 | 38.794 | 25.790 | 12.076 |
| 2 | cuda:2 | 39.123 | 38.963 | 25.560 | 12.076 |
| 3 | cuda:3 | 38.726 | 38.773 | 25.822 | 12.076 |
| 4 | cuda:4 | 39.236 | 39.070 | 25.487 | 12.076 |
| 5 | cuda:5 | 39.131 | 38.823 | 25.555 | 12.076 |


## Run Details

| Variant | Config | Checkpoint | CUDA_VISIBLE_DEVICES | Dtype | Prompt len | Iters |
| --- | --- | --- | --- | --- | ---: | ---: |
| accelerated | `work_dirs/pi05_task0_with_l_pixshuffle_compare/mlp_vs_base/configs/pi05_task0_with_l_last_accelerated.py` | `work_dirs/pi05_libero10_task0_with_l/checkpoints/latest-checkpoint.safetensors` | `2,3,4,5,6,7` | bfloat16 | 32 | 100 |
