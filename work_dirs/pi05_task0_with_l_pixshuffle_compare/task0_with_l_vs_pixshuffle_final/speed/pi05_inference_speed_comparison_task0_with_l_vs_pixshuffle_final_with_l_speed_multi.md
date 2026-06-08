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
| accelerated | 38.859 | 38.724 | 38.899 | 41.863 | 128.674 | 1286.736 | 1904.188 | 60.373 |

## Multi-GPU Throughput

| Variant | Workers | Total Hz | Per-worker Hz | Total Action-step Hz | Per-worker Mean ms | Total Peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| accelerated | 5 | 128.674 | 25.735 | 1286.736 | 38.859 | 60.373 |

### Per-Worker Details

#### accelerated

| Worker | Device | Mean ms | P90 ms | Hz | Peak GiB |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | cuda:0 | 39.048 | 38.838 | 25.610 | 12.075 |
| 1 | cuda:1 | 38.817 | 38.822 | 25.762 | 12.075 |
| 2 | cuda:2 | 38.650 | 38.659 | 25.874 | 12.075 |
| 3 | cuda:3 | 39.038 | 39.424 | 25.616 | 12.075 |
| 4 | cuda:4 | 38.741 | 38.752 | 25.812 | 12.075 |


## Run Details

| Variant | Config | Checkpoint | CUDA_VISIBLE_DEVICES | Dtype | Prompt len | Iters |
| --- | --- | --- | --- | --- | ---: | ---: |
| accelerated | `configs/pi05/pi05_paligemma_libero_10_full_inference.py` | `work_dirs/pi05_libero10_task0_with_l/checkpoints/step-008460-epoch-60-loss=0.0910.safetensors` | `3,4,5,6,7` | bfloat16 | 32 | 100 |
