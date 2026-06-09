# PI0.5 Task0 Accelerated Variants

Both variants use accelerated `PI05FlowMatchingInference`.

## Checkpoints

- with_l: `work_dirs/pi05_libero10_task0_with_l/checkpoints/step-008460-epoch-60-loss=0.0910.safetensors`
- pixshuffle: `work_dirs/pi05_libero10_task0_with_l_pixshuffle/checkpoints/step-008460-epoch-60-loss=0.0918.safetensors`
- pixshuffle_mlp: `work_dirs/pi05_libero10_task0_with_l_pixshuffle_mlp/checkpoints/latest-checkpoint.safetensors`
- Selected variants: `with_l,pixshuffle`

## Speed

| Variant | Scenario | GPUs | Workers | Prompt len | Mean ms | P90 ms | Chunk Hz | Action-step Hz | Peak GiB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pix_lastest | single | `7` | 1 | 32 | 30.405 | 30.478 | 32.890 | 328.898 | 12.037 |
| pix_lastest | multi | `2,3,4,5,6,7` | 6 | 32 | 30.484 | 30.396 | 196.826 | 1968.257 | 72.220 |
| pix_mlp_latest | single | `7` | 1 | 32 | 30.313 | 30.320 | 32.989 | 329.891 | 12.061 |
| pix_mlp_latest | multi | `2,3,4,5,6,7` | 6 | 32 | 30.517 | 30.355 | 196.619 | 1966.188 | 72.368 |

## Success

| Variant | Seed | GPUs | Episodes | Successes | Success Rate | Eval File |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pix_lastest | 7 | `2,3,4,5,6,7` | 50 | 43 | 86.000% | `/data/guohao/FluxVLA_workdirs/pi05_libero10_task0_with_l_pixshuffle/EVAL-libero_10-pi0-2026_06_09-09_43_23.txt` |
| pix_mlp_latest | 7 | `2,3,4,5,6,7` | 50 | 44 | 88.000% | `/data/guohao/FluxVLA_workdirs/pi05_libero10_task0_with_l_pixshuffle_mlp/EVAL-libero_10-pi0-2026_06_09-09_48_13.txt` |

## Settings

- Task id: `4`
- Prompt len for speed: `32`
- Triton max prompt len override: `None`
- Success trials per task: `50`
- Success seeds: `7`
- Speed warmup/bench iters: `10` / `100`
- with_l config: `configs/pi05/pi05_paligemma_libero_10_full_inference.py`
- pixshuffle config: `configs/pi05/pi05_libero10_task0_with_l_pixshuffle_inference.py`
- pixshuffle_mlp config: `configs/pi05/pi05_libero10_task0_with_l_pixshuffle_mlp_inference.py`
- Base weights used for success eval construction: `./checkpoints/pi05_base/model.safetensors`
