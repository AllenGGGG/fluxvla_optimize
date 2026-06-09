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
| accelerated | 30.419 | 30.320 | 30.374 | 32.273 | 197.251 | 1972.511 | 2001.019 | 72.220 |

## Multi-GPU Throughput

| Variant | Workers | Total Hz | Per-worker Hz | Total Action-step Hz | Per-worker Mean ms | Total Peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| accelerated | 6 | 197.251 | 32.875 | 1972.511 | 30.419 | 72.220 |

### Per-Worker Details

#### accelerated

| Worker | Device | Mean ms | P90 ms | Hz | Peak GiB |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | cuda:0 | 30.316 | 30.338 | 32.986 | 12.037 |
| 1 | cuda:1 | 30.344 | 30.353 | 32.956 | 12.037 |
| 2 | cuda:2 | 30.364 | 30.425 | 32.934 | 12.037 |
| 3 | cuda:3 | 30.596 | 30.309 | 32.684 | 12.037 |
| 4 | cuda:4 | 30.316 | 30.357 | 32.986 | 12.037 |
| 5 | cuda:5 | 30.575 | 30.464 | 32.707 | 12.037 |


## Run Details

| Variant | Config | Checkpoint | CUDA_VISIBLE_DEVICES | Dtype | Prompt len | Iters |
| --- | --- | --- | --- | --- | ---: | ---: |
| accelerated | `work_dirs/pi05_task0_with_l_pixshuffle_compare/mlp_450_vs_mean/configs/pi05_task0_pix_lastest_accelerated.py` | `work_dirs/pi05_libero10_task0_with_l_pixshuffle/checkpoints/latest-checkpoint.safetensors` | `2,3,4,5,6,7` | bfloat16 | 32 | 100 |
