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
| accelerated | 30.484 | 30.324 | 30.396 | 33.116 | 196.826 | 1968.257 | 2678.395 | 72.220 |

## Multi-GPU Throughput

| Variant | Workers | Total Hz | Per-worker Hz | Total Action-step Hz | Per-worker Mean ms | Total Peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| accelerated | 6 | 196.826 | 32.804 | 1968.257 | 30.484 | 72.220 |

### Per-Worker Details

#### accelerated

| Worker | Device | Mean ms | P90 ms | Hz | Peak GiB |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | cuda:0 | 30.544 | 30.439 | 32.739 | 12.037 |
| 1 | cuda:1 | 30.350 | 30.367 | 32.948 | 12.037 |
| 2 | cuda:2 | 30.406 | 30.455 | 32.888 | 12.037 |
| 3 | cuda:3 | 30.662 | 30.337 | 32.613 | 12.037 |
| 4 | cuda:4 | 30.306 | 30.346 | 32.997 | 12.037 |
| 5 | cuda:5 | 30.637 | 30.430 | 32.640 | 12.037 |


## Run Details

| Variant | Config | Checkpoint | CUDA_VISIBLE_DEVICES | Dtype | Prompt len | Iters |
| --- | --- | --- | --- | --- | ---: | ---: |
| accelerated | `work_dirs/pi05_task0_with_l_pixshuffle_compare/mlp_vs_mean/configs/pi05_task0_pix_lastest_accelerated.py` | `work_dirs/pi05_libero10_task0_with_l_pixshuffle/checkpoints/latest-checkpoint.safetensors` | `2,3,4,5,6,7` | bfloat16 | 32 | 100 |
