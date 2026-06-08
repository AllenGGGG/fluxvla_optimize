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
| accelerated | 30.520 | 30.317 | 30.418 | 34.459 | 163.831 | 1638.307 | 1946.679 | 60.178 |

## Multi-GPU Throughput

| Variant | Workers | Total Hz | Per-worker Hz | Total Action-step Hz | Per-worker Mean ms | Total Peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| accelerated | 5 | 163.831 | 32.766 | 1638.307 | 30.520 | 60.178 |

### Per-Worker Details

#### accelerated

| Worker | Device | Mean ms | P90 ms | Hz | Peak GiB |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | cuda:0 | 30.782 | 30.505 | 32.487 | 12.036 |
| 1 | cuda:1 | 30.559 | 30.406 | 32.723 | 12.036 |
| 2 | cuda:2 | 30.503 | 30.339 | 32.784 | 12.036 |
| 3 | cuda:3 | 30.347 | 30.418 | 32.952 | 12.036 |
| 4 | cuda:4 | 30.409 | 30.423 | 32.885 | 12.036 |


## Run Details

| Variant | Config | Checkpoint | CUDA_VISIBLE_DEVICES | Dtype | Prompt len | Iters |
| --- | --- | --- | --- | --- | ---: | ---: |
| accelerated | `work_dirs/pi05_task0_with_l_pixshuffle_compare/pix_epoch140_vs_no_pix/configs/pi05_task0_pix_latest_accelerated.py` | `work_dirs/pi05_libero10_task0_with_l_pixshuffle/checkpoints/latest-checkpoint.safetensors` | `1,4,5,6,7` | bfloat16 | 32 | 100 |
