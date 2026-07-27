"""Standard PyTorch PI0.5 inference with prefix RTC."""

_base_ = ['./pi05_parcel_sort_none_pytorch_inference.py']

inference_options = dict(rtc_method='prefix')
