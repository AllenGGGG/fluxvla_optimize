"""Standard PyTorch PI0.5 inference with prefix RTC."""

_base_ = ['../none/pytorch_inference.py']

inference_options = dict(rtc_method='prefix')
