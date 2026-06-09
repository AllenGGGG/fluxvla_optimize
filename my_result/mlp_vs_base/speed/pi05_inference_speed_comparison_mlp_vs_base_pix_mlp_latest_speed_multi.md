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
| accelerated | 30.498 | 30.288 | 30.431 | 34.078 | 196.736 | 1967.361 | 2803.474 | 72.368 |

## Multi-GPU Throughput

| Variant | Workers | Total Hz | Per-worker Hz | Total Action-step Hz | Per-worker Mean ms | Total Peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| accelerated | 6 | 196.736 | 32.789 | 1967.361 | 30.498 | 72.368 |

### Per-Worker Details

#### accelerated

| Worker | Device | Mean ms | P90 ms | Hz | Peak GiB |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | cuda:0 | 30.317 | 30.351 | 32.985 | 12.061 |
| 1 | cuda:1 | 30.315 | 30.351 | 32.987 | 12.061 |
| 2 | cuda:2 | 30.664 | 30.644 | 32.612 | 12.061 |
| 3 | cuda:3 | 30.460 | 30.337 | 32.830 | 12.061 |
| 4 | cuda:4 | 30.656 | 30.426 | 32.620 | 12.061 |
| 5 | cuda:5 | 30.579 | 30.479 | 32.702 | 12.061 |


## Run Details

| Variant | Config | Checkpoint | CUDA_VISIBLE_DEVICES | Dtype | Prompt len | Iters |
| --- | --- | --- | --- | --- | ---: | ---: |
| accelerated | `work_dirs/pi05_task0_with_l_pixshuffle_compare/mlp_vs_base/configs/pi05_task0_pix_mlp_latest_accelerated.py` | `work_dirs/pi05_libero10_task0_with_l_pixshuffle_mlp/checkpoints/latest-checkpoint.safetensors` | `2,3,4,5,6,7` | bfloat16 | 32 | 100 |
