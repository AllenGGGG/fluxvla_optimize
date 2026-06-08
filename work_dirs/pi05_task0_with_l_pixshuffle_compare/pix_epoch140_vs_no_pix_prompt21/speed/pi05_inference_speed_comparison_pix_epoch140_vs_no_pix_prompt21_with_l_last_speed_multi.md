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
| accelerated | 40.942 | 40.284 | 40.869 | 54.162 | 122.149 | 1221.495 | 3449.985 | 60.357 |

## Multi-GPU Throughput

| Variant | Workers | Total Hz | Per-worker Hz | Total Action-step Hz | Per-worker Mean ms | Total Peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| accelerated | 5 | 122.149 | 24.430 | 1221.495 | 40.942 | 60.357 |

### Per-Worker Details

#### accelerated

| Worker | Device | Mean ms | P90 ms | Hz | Peak GiB |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | cuda:0 | 41.843 | 41.332 | 23.899 | 12.071 |
| 1 | cuda:1 | 40.349 | 40.301 | 24.784 | 12.071 |
| 2 | cuda:2 | 40.449 | 40.375 | 24.723 | 12.071 |
| 3 | cuda:3 | 40.655 | 40.390 | 24.597 | 12.071 |
| 4 | cuda:4 | 41.413 | 41.949 | 24.147 | 12.071 |


## Run Details

| Variant | Config | Checkpoint | CUDA_VISIBLE_DEVICES | Dtype | Prompt len | Iters |
| --- | --- | --- | --- | --- | ---: | ---: |
| accelerated | `work_dirs/pi05_task0_with_l_pixshuffle_compare/pix_epoch140_vs_no_pix_prompt21/configs/pi05_task0_with_l_last_accelerated_prompt21.py` | `work_dirs/pi05_libero10_task0_with_l/checkpoints/latest-checkpoint.safetensors` | `1,4,5,6,7` | bfloat16 | 21 | 100 |
