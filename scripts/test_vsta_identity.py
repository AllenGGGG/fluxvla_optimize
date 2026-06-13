#!/usr/bin/env python3
"""Sanity checks for OnlineVSTATempoParquetDataset.

Checks:
1. s=1.0 keeps baseline action windows identical.
2. merge/split conserves continuous motion inside each processed segment.
3. valid retimed observations do not duplicate chunk-start sources.
"""

import sys
import types
from collections import Counter
from importlib import util
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = './datasets/libero_10_no_noops_lerobotv2.1'


def load_dataset_classes():
    """Load only dataset modules, avoiding model-side CUDA/Triton imports."""

    class Registry:

        def register_module(self, *args, **kwargs):
            del kwargs
            if args and isinstance(args[0], type):
                return args[0]

            def deco(cls):
                return cls

            return deco

    engines = types.ModuleType('fluxvla.engines')
    engines.DATASETS = Registry()
    engines.build_transform_from_cfg = lambda cfg: (_ for _ in ()).throw(
        RuntimeError(f'Transforms are not used in this sanity script: {cfg}'))

    fluxvla_pkg = types.ModuleType('fluxvla')
    datasets_pkg = types.ModuleType('fluxvla.datasets')
    fluxvla_pkg.__path__ = [str(REPO_ROOT / 'fluxvla')]
    datasets_pkg.__path__ = [str(REPO_ROOT / 'fluxvla' / 'datasets')]

    sys.modules.setdefault('fluxvla', fluxvla_pkg)
    sys.modules.setdefault('fluxvla.datasets', datasets_pkg)
    sys.modules['fluxvla.engines'] = engines

    parquet_path = REPO_ROOT / 'fluxvla' / 'datasets' / 'parquet_dataset.py'
    parquet_spec = util.spec_from_file_location(
        'fluxvla.datasets.parquet_dataset', parquet_path)
    parquet_mod = util.module_from_spec(parquet_spec)
    sys.modules[parquet_spec.name] = parquet_mod
    parquet_spec.loader.exec_module(parquet_mod)

    vsta_path = REPO_ROOT / 'fluxvla' / 'datasets' / 'online_vsta_dataset.py'
    vsta_spec = util.spec_from_file_location(
        'fluxvla.datasets.online_vsta_dataset', vsta_path)
    vsta_mod = util.module_from_spec(vsta_spec)
    sys.modules[vsta_spec.name] = vsta_mod
    vsta_spec.loader.exec_module(vsta_mod)

    return parquet_mod.ParquetDataset, vsta_mod.OnlineVSTATempoParquetDataset


ParquetDataset, OnlineVSTATempoParquetDataset = load_dataset_classes()


def stats():
    return {
        'libero_10_no_noops': {
            'action': {
                'mean': [0] * 7,
                'std': [1] * 7,
            },
            'proprio': {
                'mean': [0] * 32,
                'std': [1] * 32,
            },
            'observation.state': {
                'mean': [0] * 32,
                'std': [1] * 32,
            },
        }
    }


def build_base():
    return ParquetDataset(
        data_root_path=DATA_ROOT,
        transforms=[],
        action_window_size=10,
        action_key='action',
        use_delta=False,
        statistic_name='libero_10_no_noops',
        window_start_idx=0,
        task_indices=[0],
    )


def build_vsta(speed, random_phase=False):
    return OnlineVSTATempoParquetDataset(
        data_root_path=DATA_ROOT,
        transforms=[],
        action_window_size=10,
        action_key='action',
        tempo_speeds=[speed],
        tempo_speed_key='tempo_speed',
        keep_last_action_dim_nearest=True,
        random_phase=random_phase,
        statistic_name='libero_10_no_noops',
        window_start_idx=0,
        task_indices=[0],
    )


def valid_indices(ds, max_count=32):
    out = []
    for idx in range(len(ds.dataset)):
        if ds._is_train_observation_index(idx):
            out.append(idx)
            if len(out) >= max_count:
                break
    return out


def check_identity():
    base = build_base()
    vsta = build_vsta(1.0)
    dataset_stats = stats()
    indices = valid_indices(vsta)
    max_diff = 0.0

    for idx in indices:
        base_sample = base.__getitem__(idx, dataset_stats)
        vsta_sample = vsta.__getitem__(idx, dataset_stats)
        diff = np.abs(base_sample['actions'] - vsta_sample['actions']).max()
        max_diff = max(max_diff, float(diff))
        if diff >= 1e-5:
            raise AssertionError(
                f's=1.0 identity failed at index={idx}, max_diff={diff}')

    print(f'[PASS] s=1.0 identity on {len(indices)} samples, max_diff={max_diff:.2e}')


def check_motion_conservation():
    vsta = build_vsta(2.0, random_phase=False)
    q, p = vsta._speed_to_coprime(2.0)
    checked = 0

    for segments in vsta.episode_segments.values():
        for seg in segments:
            if len(seg) < q:
                continue
            retimed = vsta._retime_segment(seg, q, p)
            original_sum = np.zeros_like(
                np.asarray(vsta.dataset[seg[0]][vsta.action_key], dtype=np.float32))
            retimed_sum = np.zeros_like(original_sum)

            chunk_len = (len(seg) // q) * q
            for idx in seg[:chunk_len]:
                original_sum[:-1] += np.asarray(
                    vsta.dataset[idx][vsta.action_key], dtype=np.float32)[:-1]
            for step in retimed[:chunk_len // q * p]:
                retimed_sum[:-1] += step.action[:-1]

            diff = np.abs(original_sum[:-1] - retimed_sum[:-1]).max()
            if diff >= 1e-5:
                raise AssertionError(
                    f'motion conservation failed, segment_len={len(seg)}, diff={diff}')
            checked += 1
            if checked >= 32:
                print(f'[PASS] motion conservation on {checked} retimed segments')
                return

    raise AssertionError('No segment long enough to check motion conservation.')


def check_valid_observations_unique():
    vsta = build_vsta(0.5, random_phase=False)
    q, p = vsta._speed_to_coprime(0.5)
    checked = 0

    for segments in vsta.episode_segments.values():
        all_retimed = []
        for seg in segments:
            all_retimed.extend(vsta._retime_segment(seg, q, p))
        sources = [
            step.source_idx for step in all_retimed
            if step.valid_observation and step.source_idx is not None
        ]
        duplicates = [idx for idx, count in Counter(sources).items() if count > 1]
        if duplicates:
            raise AssertionError(
                f'valid observation source duplicated: {duplicates[:5]}')
        checked += 1
        if checked >= 16:
            print(f'[PASS] no duplicated valid observation source in {checked} episodes')
            return

    raise AssertionError('No episode checked for valid observation uniqueness.')


def main():
    check_identity()
    check_motion_conservation()
    check_valid_observations_unique()


if __name__ == '__main__':
    main()
