# RealTimeVLA Status

## Summary

The realtime-named TempoVLA config is now safe for existing FluxVLA eval runs.
It does not enable raw RealTimeVLA kernels by default.

## Working

- `configs/pi05/pi05_libero10_task0_tempovla_realtime_inference.py` defines
  `inference_model`.
- The config inherits the stable speed-modulated inference config.
- `--eval-speeds` updates `inference_model.default_tempo_speed`.
- Speed is not injected into the language prompt.
- FluxVLA checkpoints load through the normal `load_state_dict(strict=True)`
  path.

## Not Yet Enabled

- Raw RealTimeVLA kernels are not used for FluxVLA checkpoints.
- The copied `pi05_realtime_base.py` still requires a separate flattened
  checkpoint layout.
- No latency claim is made for raw RealTimeVLA in this repo yet.

## Reason

FluxVLA training checkpoints and the copied RealTimeVLA kernel code use
different weight names and layouts. Enabling raw kernels without a verified
converter would either fail at load time or silently run incorrect weights.

## Verification

Run:

```bash
python -m py_compile \
  fluxvla/models/vlas/pi0_infer.py \
  fluxvla/models/vlas/pi05_realtime_base.py \
  fluxvla/models/vlas/pi05_realtime_speed_modulated_inference.py \
  configs/pi05/pi05_libero10_task0_tempovla_realtime_inference.py \
  scripts/test_realtime_tempovla.py

/home/guohao/miniconda3/envs/fluxvla/bin/python scripts/test_realtime_tempovla.py
```

