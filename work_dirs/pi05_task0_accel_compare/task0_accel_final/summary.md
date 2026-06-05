# PI0.5 Task0 Accelerated With-L vs No-L

Both variants use `PI05FlowMatchingInference` accelerated inference.
`no_l` is evaluated with `use_language=False` and `NoLanguagePrompt`.

## Checkpoints

- with_l: `work_dirs/pi05_libero10_task0_with_l/checkpoints/latest-checkpoint.safetensors`
- no_l: `work_dirs/pi05_libero10_task0_no_l/checkpoints/latest-checkpoint.safetensors`

## Speed

| Variant | Scenario | GPUs | Prompt len | Mean ms | P90 ms | Chunk Hz | Action-step Hz | Peak GiB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| with_l | single | `3` | 32 | 38.909 | 39.010 | 25.701 | 257.010 | 12.090 |
| with_l | multi | `3,4,5,6,7` | 32 | 39.085 | 39.002 | 127.930 | 1279.299 | 60.448 |
| no_l | single | `3` | 0 | 38.789 | 38.559 | 25.780 | 257.805 | 12.089 |
| no_l | multi | `3,4,5,6,7` | 0 | 38.901 | 38.779 | 128.533 | 1285.329 | 60.447 |

## Success

| Variant | Seed | GPUs | Episodes | Successes | Success Rate | Eval File |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| with_l | 7 | `3,4,5,6,7` | 50 | 46 | 92.000% | `/home/guohao/FluxVLA/work_dirs/pi05_libero10_task0_with_l/EVAL-libero_10-pi0-2026_06_05-17_14_59.txt` |
| no_l | 7 | `3,4,5,6,7` | 50 | 34 | 68.000% | `/home/guohao/FluxVLA/work_dirs/pi05_libero10_task0_no_l/EVAL-libero_10-pi0-2026_06_05-17_20_31.txt` |

## Settings

- Task id: `4`
- Success trials per task: `50`
- Success seeds: `7`
- Speed warmup/bench iters: `10` / `100`
- Base weights used for construction: `./checkpoints/pi05_base/model.safetensors`
