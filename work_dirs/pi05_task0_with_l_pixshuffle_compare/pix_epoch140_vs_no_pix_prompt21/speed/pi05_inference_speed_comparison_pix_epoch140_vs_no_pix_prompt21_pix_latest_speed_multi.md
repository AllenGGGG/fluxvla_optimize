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
| accelerated | 31.040 | 30.574 | 31.660 | 38.872 | 161.145 | 1611.453 | 2901.219 | 60.167 |

## Multi-GPU Throughput

| Variant | Workers | Total Hz | Per-worker Hz | Total Action-step Hz | Per-worker Mean ms | Total Peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| accelerated | 5 | 161.145 | 32.229 | 1611.453 | 31.040 | 60.167 |

### Per-Worker Details

#### accelerated

| Worker | Device | Mean ms | P90 ms | Hz | Peak GiB |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | cuda:0 | 32.243 | 35.793 | 31.014 | 12.033 |
| 1 | cuda:1 | 30.919 | 30.645 | 32.343 | 12.033 |
| 2 | cuda:2 | 30.553 | 30.564 | 32.730 | 12.033 |
| 3 | cuda:3 | 30.863 | 30.646 | 32.401 | 12.033 |
| 4 | cuda:4 | 30.621 | 30.655 | 32.657 | 12.033 |


## Run Details

| Variant | Config | Checkpoint | CUDA_VISIBLE_DEVICES | Dtype | Prompt len | Iters |
| --- | --- | --- | --- | --- | ---: | ---: |
| accelerated | `work_dirs/pi05_task0_with_l_pixshuffle_compare/pix_epoch140_vs_no_pix_prompt21/configs/pi05_task0_pix_latest_accelerated_prompt21.py` | `work_dirs/pi05_libero10_task0_with_l_pixshuffle/checkpoints/latest-checkpoint.safetensors` | `1,4,5,6,7` | bfloat16 | 21 | 100 |
