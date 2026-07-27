"""Fused Triton + CUDA Graph inference with non-VJP guidance RTC."""

_base_ = ['./pi05_parcel_sort_none_triton_inference.py']

inference_model = dict(type='PI05FlowMatchingGuidanceInference')
# rtc_schedule override: base config sets 'linear'. Real-robot A/B data
# (2026-07-26, execution_horizon=10): with 'linear', 50-65% of large
# per-tick command jumps landed within 150ms of a chunk-switch marker (vs a
# ~35% random baseline) across two independent runs, with delay-estimation
# error ruled out separately (measured vs applied delay matched within
# ~0.1 step in steady state). linear holds guidance weight >=0.5 for
# roughly half the decay window then drops sharply near the end,
# concentrating the old/new-trajectory discrepancy release right at the
# window's tail (near the next switch); 'exp' -- the schedule
# fluxvla/engines/utils/rtc_guidance.py's compute_prefix_weights docstring
# marks as recommended -- releases it gradually from the start instead.
# Not yet re-validated on the real robot with 'exp'; if a follow-up run's
# jump-concentration doesn't drop below the random baseline, reconsider.
inference_options = dict(rtc_method='guidance', rtc_schedule='exp')
