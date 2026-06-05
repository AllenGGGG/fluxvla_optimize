# PI0.5 Speed and Success Comparison Record

This document records the exact comparison design for PI0.5 baseline inference
and FluxVLA accelerated inference.

## Experiment Goal

Compare the same trained PI0.5 LIBERO-10 checkpoint under two inference
implementations:

| Variant | Config | Meaning |
| --- | --- | --- |
| baseline | `configs/pi05/pi05_paligemma_libero_10_full_finetune.py` | Standard PI0.5 PyTorch inference path |
| accelerated | `configs/pi05/pi05_paligemma_libero_10_full_inference.py` | FluxVLA accelerated inference path with Triton / CUDA Graph / inference-specific modules |

The checkpoint is identical for both variants:

```bash
checkpoints/pi05_paligemma_libero_10_full_finetune_bs64/pi05_paligemma_libero_10_full_finetune_bs64/checkpoints/step-038064-epoch-24-loss=0.0170.safetensors
```

Therefore, the controlled variable is the inference implementation, not the
trained weights, task suite, action chunk size, or dataset statistics.

## Metrics

| Metric | Meaning | Source |
| --- | --- | --- |
| `Mean ms` | Average latency of measured `predict_action` calls | Speed benchmark |
| `P50 ms` | Median latency | Speed benchmark |
| `P90 ms` | 90th percentile latency | Speed benchmark |
| `P99 ms` | 99th percentile latency | Speed benchmark |
| `Hz` | `1000 / Mean ms`; single-GPU frequency or multi-GPU total throughput | Speed benchmark |
| `Per-worker Hz` | Average frequency per GPU worker in multi-GPU mode | Speed benchmark |
| `Action-step Hz` | `Hz * 10`; each PI0.5 call outputs 10 action steps | Speed benchmark |
| `Cold start ms` | First `predict_action` latency, not included in `Mean ms` | Speed benchmark |
| `Peak GiB` | Peak allocated GPU memory during measured benchmark | Speed benchmark |
| `Success rate` | LIBERO task success percentage | LIBERO eval |
| `Successes / Episodes` | Number of successful episodes over total evaluated episodes | LIBERO eval |

## Report Table Template

| Scenario | Variant | GPUs | Workers | Mean ms | P90 ms | Hz | Per-worker Hz | Action-step Hz | Success rate | Successes / Episodes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| single | baseline | `4` | 1 |  |  |  |  |  |  |  |
| single | accelerated | `4` | 1 |  |  |  |  |  |  |  |
| multi | baseline | `5,6,7` | 3 |  |  |  |  |  |  |  |
| multi | accelerated | `5,6,7` | 3 |  |  |  |  |  |  |  |

Notes:

- `single` evaluates one model replica on physical GPU 4.
- `multi` evaluates three independent model replicas on physical GPUs 5,6,7.
- Multi-GPU `Hz` is total serving throughput, not single-robot latency.
- Single-robot inference frequency should be read from `single` or
  `Per-worker Hz`, not multi-GPU `Total Hz`.

## One-Command Combined Reports

Recommended one-click launcher:

```bash
bash scripts/run_pi05_speed_success_formal.sh
```

Run it in the background with full logging:

```bash
BACKGROUND=1 bash scripts/run_pi05_speed_success_formal.sh
```

The launcher writes `run.log`, `run_metadata.txt`, command logs, and the final
Markdown / CSV / JSON reports under one timestamped comparison directory.

Quick run for validating the full workflow:

```bash
python scripts/compare_pi05_speed_success.py \
  --single-gpus 4 \
  --multi-gpus 5,6,7 \
  --success-gpus 4 \
  --success-nproc-per-node 1 \
  --speed-warmup-iters 3 \
  --speed-bench-iters 10 \
  --success-trials-per-task 1 \
  --success-seeds 7 \
  --tag quick_speed_success
```

Formal run for reporting:

```bash
python scripts/compare_pi05_speed_success.py \
  --single-gpus 4 \
  --multi-gpus 5,6,7 \
  --success-gpus 5,6,7 \
  --success-nproc-per-node 3 \
  --speed-warmup-iters 10 \
  --speed-bench-iters 100 \
  --success-trials-per-task 50 \
  --success-seeds 7 \
  --tag formal_speed_success
```

Output directory:

```bash
checkpoints/pi05_paligemma_libero_10_full_finetune_bs64/pi05_paligemma_libero_10_full_finetune_bs64/comparisons/
```

The combined workflow writes:

```text
pi05_speed_success_comparison_<tag>.md
pi05_speed_success_comparison_<tag>.csv
pi05_speed_success_comparison_<tag>.json
```

## Speed-Only Commands

Single-GPU speed comparison on physical GPU 4:

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/benchmark_pi05_inference_speed.py \
  --mode both \
  --warmup-iters 10 \
  --bench-iters 100 \
  --tag formal_1gpu_gpu4
```

Three-GPU concurrent throughput comparison on physical GPUs 5,6,7:

```bash
CUDA_VISIBLE_DEVICES=5,6,7 python scripts/benchmark_pi05_inference_speed.py \
  --mode both \
  --num-workers 3 \
  --warmup-iters 10 \
  --bench-iters 100 \
  --tag formal_3gpu_567
```

## Success-Only Commands

Baseline success rate:

```bash
CUDA_VISIBLE_DEVICES=5,6,7 torchrun \
  --nnodes 1 \
  --nproc-per-node 3 \
  --master-addr 127.0.0.1 \
  --master-port 29500 \
  scripts/eval.py \
  --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
  --ckpt-path checkpoints/pi05_paligemma_libero_10_full_finetune_bs64/pi05_paligemma_libero_10_full_finetune_bs64/checkpoints/step-038064-epoch-24-loss=0.0170.safetensors \
  --cfg-options eval.num_trials_per_task=50 eval.seed=7
```

Accelerated success rate:

```bash
CUDA_VISIBLE_DEVICES=5,6,7 torchrun \
  --nnodes 1 \
  --nproc-per-node 3 \
  --master-addr 127.0.0.1 \
  --master-port 29501 \
  scripts/eval.py \
  --config configs/pi05/pi05_paligemma_libero_10_full_inference.py \
  --ckpt-path checkpoints/pi05_paligemma_libero_10_full_finetune_bs64/pi05_paligemma_libero_10_full_finetune_bs64/checkpoints/step-038064-epoch-24-loss=0.0170.safetensors \
  --cfg-options eval.num_trials_per_task=50 eval.seed=7
```

## Training Command

If retraining the PI0.5 LIBERO-10 checkpoint is required:

```bash
export WANDB_MODE=disabled

torchrun \
  --standalone \
  --nnodes 1 \
  --nproc-per-node 2 \
  scripts/train.py \
  --config configs/pi05/pi05_paligemma_libero_10_full_finetune.py \
  --work-dir ./work_dirs/pi05_paligemma_libero_10_full_finetune \
  --cfg-options train_dataloader.per_device_batch_size=2
```

After training, evaluate a checkpoint by replacing `--ckpt-path` in the commands
above with the generated checkpoint under:

```bash
./work_dirs/pi05_paligemma_libero_10_full_finetune/checkpoints/
```

## Recommended Wording

For speed:

```text
We compare the same trained PI0.5 checkpoint using the standard inference path
and the FluxVLA accelerated inference path. The accelerated path reduces
model-side predict_action latency and increases inference frequency.
```

For multi-GPU throughput:

```text
In the three-GPU setting, each GPU hosts one independent model replica. The
reported Total Hz is aggregate serving throughput across three concurrent
workers, while Per-worker Hz reflects average single-replica inference speed.
```

For success rate:

```text
LIBERO success rate is evaluated separately to verify that the accelerated
inference implementation preserves task performance.
```
