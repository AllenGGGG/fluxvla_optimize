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
| pix_lastest | single | `7` | 1 | 32 | 30.406 | 30.502 | 32.889 | 328.886 | 12.037 |
| pix_lastest | multi | `2,3,4,5,6,7` | 6 | 32 | 30.419 | 30.374 | 197.251 | 1972.511 | 72.220 |
| pix_mlp_latest | single | `7` | 1 | 32 | 30.353 | 30.408 | 32.946 | 329.460 | 12.061 |
| pix_mlp_latest | multi | `2,3,4,5,6,7` | 6 | 32 | 30.551 | 30.408 | 196.402 | 1964.016 | 72.368 |

## Success

| Variant | Seed | GPUs | Episodes | Successes | Success Rate | Eval File |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| pix_lastest | 7 | `2,3,4,5,6,7` | 50 | 43 | 86.000% | `/data/guohao/FluxVLA_workdirs/pi05_libero10_task0_with_l_pixshuffle/EVAL-libero_10-pi0-2026_06_09-18_43_42.txt` |
| pix_mlp_latest | 7 | `2,3,4,5,6,7` | 50 | 46 | 92.000% | `/data/guohao/FluxVLA_workdirs/pi05_libero10_task0_with_l_pixshuffle_mlp_v1/EVAL-libero_10-pi0-2026_06_09-18_48_27.txt` |
| pix_lastest | 8 | `2,3,4,5,6,7` | 50 | 41 | 82.000% | `/data/guohao/FluxVLA_workdirs/pi05_libero10_task0_with_l_pixshuffle/EVAL-libero_10-pi0-2026_06_09-18_52_54.txt` |
| pix_mlp_latest | 8 | `2,3,4,5,6,7` | 50 | 44 | 88.000% | `/data/guohao/FluxVLA_workdirs/pi05_libero10_task0_with_l_pixshuffle_mlp_v1/EVAL-libero_10-pi0-2026_06_09-18_58_02.txt` |

## Settings

- Task id: `4`
- Prompt len for speed: `32`
- Triton max prompt len override: `None`
- Success trials per task: `50`
- Success seeds: `7,8`
- Speed warmup/bench iters: `10` / `100`
- with_l config: `configs/pi05/pi05_paligemma_libero_10_full_inference.py`
- pixshuffle config: `configs/pi05/pi05_libero10_task0_with_l_pixshuffle_inference.py`
- pixshuffle_mlp config: `configs/pi05/pi05_libero10_task0_with_l_pixshuffle_mlp_inference.py`
- Base weights used for success eval construction: `./checkpoints/pi05_base/model.safetensors`
