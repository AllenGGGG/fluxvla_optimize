"""Fused Triton + CUDA Graph inference with hard prefix RTC."""

_base_ = ['../none/triton_inference.py']

inference_model = dict(type='PI05FlowMatchingRTCInference')
inference_options = dict(rtc_method='prefix')
