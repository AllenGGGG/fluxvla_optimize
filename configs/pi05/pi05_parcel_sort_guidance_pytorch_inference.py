"""Eager inference config with guidance RTC enabled."""

_base_ = ['./pi05_parcel_sort_none_pytorch_inference.py']

inference_options = dict(rtc_method='guidance')
