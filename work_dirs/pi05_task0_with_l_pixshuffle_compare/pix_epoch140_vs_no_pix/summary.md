# PI0.5 Task0 With-L vs Pixshuffle

Both variants use accelerated `PI05FlowMatchingInference`.

## Checkpoints

- with_l: `work_dirs/pi05_libero10_task0_with_l/checkpoints/step-008460-epoch-60-loss=0.0910.safetensors`
- pixshuffle: `work_dirs/pi05_libero10_task0_with_l_pixshuffle/checkpoints/step-008460-epoch-60-loss=0.0918.safetensors`
- Selected variants: `with_l,pixshuffle`

## Speed

| Variant | Scenario | GPUs | Workers | Prompt len | Mean ms | P90 ms | Chunk Hz | Action-step Hz | Peak GiB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| with_l_last | single | `7` | 1 | 32 | 38.831 | 38.869 | 25.753 | 257.526 | 12.075 |
| with_l_last | multi | `1,4,5,6,7` | 5 | 32 | 39.185 | 38.869 | 127.626 | 1276.264 | 60.373 |
| pix_latest | single | `7` | 1 | 32 | 30.353 | 30.417 | 32.945 | 329.454 | 12.036 |
| pix_latest | multi | `1,4,5,6,7` | 5 | 32 | 30.520 | 30.418 | 163.831 | 1638.307 | 60.178 |

## Success

| Variant | Seed | GPUs | Episodes | Successes | Success Rate | Eval File |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| with_l_last | 7 | `1,4,5,6,7` | 50 | 46 | 92.000% | `/home/guohao/FluxVLA/work_dirs/pi05_libero10_task0_with_l/EVAL-libero_10-pi0-2026_06_06-19_01_59.txt` |
| pix_latest | 7 | `1,4,5,6,7` | 50 | 43 | 86.000% | `/home/guohao/FluxVLA/work_dirs/pi05_libero10_task0_with_l_pixshuffle/EVAL-libero_10-pi0-2026_06_06-19_06_44.txt` |

## Settings

- Task id: `4`
- Prompt len for speed: `32`
- Triton max prompt len override: `None`
- Success trials per task: `50`
- Success seeds: `7`
- Speed warmup/bench iters: `10` / `100`
- with_l config: `configs/pi05/pi05_paligemma_libero_10_full_inference.py`
- pixshuffle config: `configs/pi05/pi05_libero10_task0_with_l_pixshuffle_inference.py`
- Base weights used for success eval construction: `./checkpoints/pi05_base/model.safetensors`
