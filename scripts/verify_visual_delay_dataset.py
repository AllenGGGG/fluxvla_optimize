#!/usr/bin/env python
# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Regression checks for delayed visual-frame training samples."""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class _Registry:

    def register_module(self):

        def _decorator(cls):
            return cls

        return _decorator


def _install_import_stubs():
    datasets_mod = types.ModuleType('datasets')
    datasets_mod.concatenate_datasets = lambda datasets: datasets[0]
    datasets_mod.load_dataset = lambda *args, **kwargs: None
    sys.modules.setdefault('datasets', datasets_mod)

    fluxvla_mod = types.ModuleType('fluxvla')
    engines_mod = types.ModuleType('fluxvla.engines')
    engines_mod.DATASETS = _Registry()
    engines_mod.build_transform_from_cfg = lambda cfg: cfg
    sys.modules.setdefault('fluxvla', fluxvla_mod)
    sys.modules.setdefault('fluxvla.engines', engines_mod)


def _load_parquet_dataset_cls():
    _install_import_stubs()
    module_path = ROOT / 'fluxvla' / 'datasets' / 'parquet_dataset.py'
    spec = importlib.util.spec_from_file_location('parquet_dataset_under_test',
                                                  module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ParquetDataset


def _make_dataset(parquet_dataset_cls, episode_indices=None):
    if episode_indices is None:
        episode_indices = [0] * 12
    dataset = object.__new__(parquet_dataset_cls)
    dataset.dataset = [{
        'episode_index': episode_idx,
        'task_index': 0,
        'timestamp': float(i),
        'action': [float(i)],
    } for i, episode_idx in enumerate(episode_indices)]
    dataset.tasks = [[{
        'task': 'demo'
    }]]
    dataset.dataset_cumulative_sizes = None
    dataset.action_window_size = 3
    dataset.action_key = 'action'
    dataset.use_delta = False
    dataset.window_start_idx = 0
    dataset.frame_window_size = 1
    dataset.visual_delay_max_steps = 5
    dataset.visual_delay_min_steps = 0
    dataset.visual_delay_distribution = 'constant'
    dataset.visual_delay_probability = 1.0
    return dataset


def _assert_equal(actual, expected, message):
    if actual != expected:
        raise AssertionError(f'{message}: expected {expected}, got {actual}')


def test_visual_timestamp_delay():
    parquet_dataset_cls = _load_parquet_dataset_cls()
    dataset = _make_dataset(parquet_dataset_cls)

    timestamps, actual_delay = dataset._visual_timestamps_for_delay(5, 2, 0, 1)
    _assert_equal(timestamps, [3.0], 'single image should come from t-d')
    _assert_equal(actual_delay, 2, 'delay should be kept inside episode')

    timestamps, actual_delay = dataset._visual_timestamps_for_delay(5, 2, 0, 3)
    _assert_equal(timestamps, [3.0, 4.0, 5.0],
                  'temporal image window should shift by the same delay')
    _assert_equal(actual_delay, 2, 'window delay should be kept')


def test_delay_does_not_cross_episode_boundary():
    parquet_dataset_cls = _load_parquet_dataset_cls()
    dataset = _make_dataset(parquet_dataset_cls,
                            episode_indices=[0, 0, 0, 1, 1, 1, 1])

    timestamps, actual_delay = dataset._visual_timestamps_for_delay(3, 2, 0, 1)
    _assert_equal(timestamps, [3.0],
                  'delay should fall back at episode boundary')
    _assert_equal(actual_delay, 0,
                  'reported delay should match boundary fallback')


def test_sampling_modes():
    parquet_dataset_cls = _load_parquet_dataset_cls()
    dataset = _make_dataset(parquet_dataset_cls)

    dataset.visual_delay_distribution = 'constant'
    dataset.visual_delay_max_steps = 4
    dataset.visual_delay_probability = 1.0
    _assert_equal(dataset._sample_visual_delay(), 4,
                  'constant delay should return max steps')

    dataset.visual_delay_probability = 0.0
    _assert_equal(dataset._sample_visual_delay(), 0,
                  'zero probability should disable delay')

    dataset.visual_delay_probability = 1.0
    dataset.visual_delay_distribution = 'uniform'
    dataset.visual_delay_min_steps = 1
    dataset.visual_delay_max_steps = 3
    np.random.seed(0)
    samples = {dataset._sample_visual_delay() for _ in range(50)}
    if not samples.issubset({1, 2, 3}) or len(samples) < 2:
        raise AssertionError(f'uniform samples out of range: {samples}')


def test_action_window_stays_current():
    parquet_dataset_cls = _load_parquet_dataset_cls()
    dataset = _make_dataset(parquet_dataset_cls)
    index = 5
    actions = [
        dataset.dataset[index + offset][dataset.action_key]
        for offset in range(dataset.action_window_size)
    ]
    timestamps, actual_delay = dataset._visual_timestamps_for_delay(index, 2,
                                                                    0, 1)

    _assert_equal(actions, [[5.0], [6.0], [7.0]],
                  'action target should still start at current index')
    _assert_equal(timestamps, [3.0],
                  'visual input should be delayed independently')
    _assert_equal(actual_delay, 2, 'visual delay should remain nonzero')


def main():
    test_visual_timestamp_delay()
    test_delay_does_not_cross_episode_boundary()
    test_sampling_modes()
    test_action_window_stays_current()
    print('visual delay dataset regression ok')


if __name__ == '__main__':
    main()
