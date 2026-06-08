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
| accelerated | 30.496 | 30.302 | 30.485 | 33.761 | 163.958 | 1639.582 | 1517.871 | 60.178 |

## Multi-GPU Throughput

| Variant | Workers | Total Hz | Per-worker Hz | Total Action-step Hz | Per-worker Mean ms | Total Peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| accelerated | 5 | 163.958 | 32.792 | 1639.582 | 30.496 | 60.178 |

### Per-Worker Details

#### accelerated

| Worker | Device | Mean ms | P90 ms | Hz | Peak GiB |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | cuda:0 | 30.328 | 30.332 | 32.973 | 12.036 |
| 1 | cuda:1 | 30.351 | 30.361 | 32.948 | 12.036 |
| 2 | cuda:2 | 30.565 | 30.567 | 32.717 | 12.036 |
| 3 | cuda:3 | 30.636 | 30.533 | 32.641 | 12.036 |
| 4 | cuda:4 | 30.600 | 30.629 | 32.679 | 12.036 |


## Run Details

| Variant | Config | Checkpoint | CUDA_VISIBLE_DEVICES | Dtype | Prompt len | Iters |
| --- | --- | --- | --- | --- | ---: | ---: |
| accelerated | `configs/pi05/pi05_libero10_task0_with_l_pixshuffle_inference.py` | `work_dirs/pi05_libero10_task0_with_l_pixshuffle/checkpoints/step-008460-epoch-60-loss=0.0918.safetensors` | `3,4,5,6,7` | bfloat16 | 32 | 100 |
