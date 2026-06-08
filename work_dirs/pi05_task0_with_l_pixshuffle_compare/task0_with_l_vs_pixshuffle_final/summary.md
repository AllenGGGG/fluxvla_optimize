# PI0.5 Task0 With-L vs Pixshuffle

Both variants use accelerated `PI05FlowMatchingInference`.

## Checkpoints

- with_l: `work_dirs/pi05_libero10_task0_with_l/checkpoints/step-008460-epoch-60-loss=0.0910.safetensors`
- pixshuffle: `work_dirs/pi05_libero10_task0_with_l_pixshuffle/checkpoints/step-008460-epoch-60-loss=0.0918.safetensors`

## Speed

| Variant | Scenario | GPUs | Workers | Prompt len | Mean ms | P90 ms | Chunk Hz | Action-step Hz | Peak GiB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| with_l | single | `3` | 1 | 32 | 38.739 | 38.823 | 25.814 | 258.139 | 12.075 |
| with_l | multi | `3,4,5,6,7` | 5 | 32 | 38.859 | 38.899 | 128.674 | 1286.736 | 60.373 |
| pixshuffle | single | `3` | 1 | 32 | 30.338 | 30.391 | 32.962 | 329.625 | 12.036 |
| pixshuffle | multi | `3,4,5,6,7` | 5 | 32 | 30.496 | 30.485 | 163.958 | 1639.582 | 60.178 |

## Success

| Variant | Seed | GPUs | Episodes | Successes | Success Rate | Eval File |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| with_l | 7 | `3,4,5,6,7` | 50 | 46 | 92.000% | `/home/guohao/FluxVLA/work_dirs/pi05_libero10_task0_with_l/EVAL-libero_10-pi0-2026_06_06-10_07_29.txt` |
| pixshuffle | 7 | `3,4,5,6,7` | 50 | 38 | 76.000% | `/home/guohao/FluxVLA/work_dirs/pi05_libero10_task0_with_l_pixshuffle/EVAL-libero_10-pi0-2026_06_06-10_11_30.txt` |

## Settings

- Task id: `4`
- Prompt len for speed: `32`
- Success trials per task: `50`
- Success seeds: `7`
- Speed warmup/bench iters: `10` / `100`
- with_l config: `configs/pi05/pi05_paligemma_libero_10_full_inference.py`
- pixshuffle config: `configs/pi05/pi05_libero10_task0_with_l_pixshuffle_inference.py`
- Base weights used for success eval construction: `./checkpoints/pi05_base/model.safetensors`
