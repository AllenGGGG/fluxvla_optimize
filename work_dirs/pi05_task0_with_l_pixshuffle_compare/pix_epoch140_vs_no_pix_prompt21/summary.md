# PI0.5 Task0 With-L vs Pixshuffle

Both variants use accelerated `PI05FlowMatchingInference`.

## Checkpoints

- with_l: `work_dirs/pi05_libero10_task0_with_l/checkpoints/step-008460-epoch-60-loss=0.0910.safetensors`
- pixshuffle: `work_dirs/pi05_libero10_task0_with_l_pixshuffle/checkpoints/step-008460-epoch-60-loss=0.0918.safetensors`
- Selected variants: `with_l,pixshuffle`

## Speed

| Variant | Scenario | GPUs | Workers | Prompt len | Mean ms | P90 ms | Chunk Hz | Action-step Hz | Peak GiB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| with_l_last | single | `7` | 1 | 21 | 40.427 | 40.728 | 24.736 | 247.359 | 12.071 |
| with_l_last | multi | `1,4,5,6,7` | 5 | 21 | 40.942 | 40.869 | 122.149 | 1221.495 | 60.357 |
| pix_latest | single | `7` | 1 | 21 | 31.678 | 31.436 | 31.568 | 315.677 | 12.033 |
| pix_latest | multi | `1,4,5,6,7` | 5 | 21 | 31.040 | 31.660 | 161.145 | 1611.453 | 60.167 |

## Success

| Variant | Seed | GPUs | Episodes | Successes | Success Rate | Eval File |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| with_l_last | 7 | `1,4,5,6,7` | 50 | 46 | 92.000% | `/home/guohao/FluxVLA/work_dirs/pi05_libero10_task0_with_l/EVAL-libero_10-pi0-2026_06_06-18_33_52.txt` |
| pix_latest | 7 | `1,4,5,6,7` | 50 | 41 | 82.000% | `/home/guohao/FluxVLA/work_dirs/pi05_libero10_task0_with_l_pixshuffle/EVAL-libero_10-pi0-2026_06_06-18_38_38.txt` |

## Settings

- Task id: `4`
- Prompt len for speed: `21`
- Triton max prompt len override: `21`
- Success trials per task: `50`
- Success seeds: `7`
- Speed warmup/bench iters: `10` / `100`
- with_l config: `configs/pi05/pi05_paligemma_libero_10_full_inference.py`
- pixshuffle config: `configs/pi05/pi05_libero10_task0_with_l_pixshuffle_inference.py`
- Base weights used for success eval construction: `./checkpoints/pi05_base/model.safetensors`
