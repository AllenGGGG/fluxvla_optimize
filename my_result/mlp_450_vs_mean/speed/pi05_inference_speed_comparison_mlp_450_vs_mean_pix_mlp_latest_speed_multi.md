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
| accelerated | 30.551 | 30.287 | 30.408 | 34.230 | 196.402 | 1964.016 | 2296.543 | 72.368 |

## Multi-GPU Throughput

| Variant | Workers | Total Hz | Per-worker Hz | Total Action-step Hz | Per-worker Mean ms | Total Peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| accelerated | 6 | 196.402 | 32.734 | 1964.016 | 30.551 | 72.368 |

### Per-Worker Details

#### accelerated

| Worker | Device | Mean ms | P90 ms | Hz | Peak GiB |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | cuda:0 | 30.671 | 30.382 | 32.604 | 12.061 |
| 1 | cuda:1 | 30.664 | 30.365 | 32.612 | 12.061 |
| 2 | cuda:2 | 30.325 | 30.347 | 32.976 | 12.061 |
| 3 | cuda:3 | 30.520 | 30.424 | 32.765 | 12.061 |
| 4 | cuda:4 | 30.260 | 30.283 | 33.047 | 12.061 |
| 5 | cuda:5 | 30.865 | 30.648 | 32.399 | 12.061 |


## Run Details

| Variant | Config | Checkpoint | CUDA_VISIBLE_DEVICES | Dtype | Prompt len | Iters |
| --- | --- | --- | --- | --- | ---: | ---: |
| accelerated | `work_dirs/pi05_task0_with_l_pixshuffle_compare/mlp_450_vs_mean/configs/pi05_task0_pix_mlp_latest_accelerated.py` | `work_dirs/pi05_libero10_task0_with_l_pixshuffle_mlp_v1/checkpoints/latest-checkpoint.safetensors` | `2,3,4,5,6,7` | bfloat16 | 32 | 100 |
