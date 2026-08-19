"""Eager inference config with guidance RTC enabled."""

_base_ = ['../none/pytorch_inference.py']

inference_options = dict(rtc_method='guidance')
