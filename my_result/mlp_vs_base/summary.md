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
| with_l_last | single | `7` | 1 | 32 | 38.891 | 38.991 | 25.713 | 257.132 | 12.076 |
| with_l_last | multi | `2,3,4,5,6,7` | 6 | 32 | 39.015 | 38.896 | 153.790 | 1537.898 | 72.454 |
| pix_mlp_latest | single | `7` | 1 | 32 | 30.419 | 30.537 | 32.874 | 328.740 | 12.061 |
| pix_mlp_latest | multi | `2,3,4,5,6,7` | 6 | 32 | 30.498 | 30.431 | 196.736 | 1967.361 | 72.368 |

## Success

| Variant | Seed | GPUs | Episodes | Successes | Success Rate | Eval File |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| with_l_last | 7 | `2,3,4,5,6,7` | 50 | 47 | 94.000% | `/data/guohao/FluxVLA_workdirs/pi05_libero10_task0_with_l/EVAL-libero_10-pi0-2026_06_08-14_56_17.txt` |
| pix_mlp_latest | 7 | `2,3,4,5,6,7` | 50 | 32 | 64.000% | `/data/guohao/FluxVLA_workdirs/pi05_libero10_task0_with_l_pixshuffle_mlp/EVAL-libero_10-pi0-2026_06_08-15_00_58.txt` |

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
