#!/usr/bin/env python3
"""Compute observation.state/action normalization statistics from a recorded
LeRobot v2.1 or v3.0 dataset, and write them to meta/stats.json (v3.0) or
meta/episodes_stats.jsonl (v2.1) -- the format fluxvla's ParquetDataset /
ParquetDatasetV3 expect to find at training time.

Deliberately not part of the ros2_data_collection live recording path (see
that repo's recorder_core/lerobot_v21_writer.py, _write_meta_files()
docstring): computing these means reading back every recorded frame, which
would put an O(n) stall on the writer thread if done live, the exact thing
that method's per-episode meta writes exist to avoid. Run this once,
offline, after a recording session (or a batch of sessions) finishes.

ros2_data_collection/README.md's own "计算 norm_stats" section shows a
version of this built on the official `lerobot` pip package's
LeRobotDataset -- not available in either this repo's or
ros2_data_collection's environment (neither installs it; ros2_data_collection
deliberately avoids it as a heavy, torch-pulling dependency for an embedded
recording box). This script produces the same output shape (mean/std/min/max
/q01/q99, --max-frames sampling for speed on large datasets) using only
pyarrow/numpy, runnable wherever fluxvla itself already runs.

--exclude-joint-names / --pad-to: must exactly match the training config's
`SelectJointDims` (fluxvla/transforms/transform_states.py). It compacts the
raw vector by dropping named joints, shifting later dims -- stats computed
on the raw, un-selected columns would then be indexed to the wrong joints.

Usage:
    python scripts/compute_norm_stats.py <dataset_root> [<dataset_root> ...] \\
        [--features observation.state action] [--max-frames N] [--seed S] \\
        [--exclude-joint-names body_joint1 body_joint2 ...] [--pad-to N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text to path via a temp file + rename, matching the convention
    ros2_data_collection's own writer uses (lerobot_v21_writer.py's
    _atomic_write_text): a kill mid-write must leave the previous complete
    file in place, never a truncated one.
    """
    tmp_path = path.with_name(path.name + '.tmp')
    try:
        with tmp_path.open('w', encoding='utf-8') as f:
            f.write(text)
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _resolve_keep_indices(names: List[str],
                          exclude_joint_names: List[str]) -> List[int]:
    """Mirrors SelectJointDims._resolve_keep_indices."""
    name_set = set(names)
    missing = [
        name for name in exclude_joint_names if name not in name_set
    ]
    if missing:
        raise KeyError(
            f'exclude_joint_names {missing} not found in this dataset\'s '
            f'declared names {names}.')
    exclude_set = set(exclude_joint_names)
    return [i for i, name in enumerate(names) if name not in exclude_set]


def _select_and_pad(values: np.ndarray, keep_indices: List[int],
                    pad_to: int) -> np.ndarray:
    """Mirrors SelectJointDims._select_and_pad."""
    selected = values[..., keep_indices]
    if pad_to is None:
        return selected
    n_keep = selected.shape[-1]
    pad_amount = pad_to - n_keep
    if pad_amount < 0:
        raise ValueError(
            f'pad_to={pad_to} is smaller than the {n_keep} dims left after '
            'joint selection -- raise pad_to or drop fewer joints.')
    if pad_amount == 0:
        return selected
    pad_width = [(0, 0)] * (selected.ndim - 1) + [(0, pad_amount)]
    return np.pad(selected, pad_width, mode='constant', constant_values=0.0)


def _feature_stats(values: np.ndarray) -> Dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    return {
        'min': values.min(axis=0).tolist(),
        'max': values.max(axis=0).tolist(),
        'mean': values.mean(axis=0).tolist(),
        'std': values.std(axis=0).tolist(),
        'q01': np.quantile(values, 0.01, axis=0).tolist(),
        'q99': np.quantile(values, 0.99, axis=0).tolist(),
        'count': int(values.shape[0]),
    }


def _read_all_frames(dataset_root: Path) -> pa.Table:
    parquet_files = sorted((dataset_root / 'data').rglob('*.parquet'))
    if not parquet_files:
        raise FileNotFoundError(
            f'No parquet files found under {dataset_root}/data/')
    tables = [pq.read_table(p) for p in parquet_files]
    return pa.concat_tables(tables)


def _maybe_subsample(table: pa.Table, max_frames: int, seed: int) -> pa.Table:
    """Random row subsample for speed on large datasets (see --max-frames)."""
    if max_frames <= 0 or table.num_rows <= max_frames:
        return table
    idx = np.sort(
        np.random.default_rng(seed).choice(
            table.num_rows, max_frames, replace=False))
    return table.take(pa.array(idx))


def _keep_indices_by_feature(info: Dict[str, object], features: List[str],
                             exclude_joint_names: List[str]
                             ) -> Dict[str, List[int]]:
    """Resolve keep-indices per feature from info.json's joint names."""
    keep_indices = {}
    for feat in features:
        names = (info.get('features') or {}).get(feat, {}).get('names')
        if not names:
            raise KeyError(
                f"--exclude-joint-names requires info.json features['{feat}']"
                "['names'] to look up joints by name, but it is missing/empty.")
        keep_indices[feat] = _resolve_keep_indices(names, exclude_joint_names)
    return keep_indices


def compute_stats(dataset_root: Path, features: List[str], max_frames: int,
                  seed: int, exclude_joint_names: Optional[List[str]] = None,
                  pad_to: Optional[int] = None) -> None:
    info_path = dataset_root / 'meta' / 'info.json'
    if not info_path.exists():
        raise FileNotFoundError(f'{info_path} not found')
    info = json.loads(info_path.read_text(encoding='utf-8'))
    version = info.get('codebase_version')

    table = _read_all_frames(dataset_root)
    total_frames = table.num_rows
    table = _maybe_subsample(table, max_frames, seed)
    for feat in features:
        if feat not in table.column_names:
            raise KeyError(
                f"feature '{feat}' not found in parquet columns: "
                f'{table.column_names}')

    keep_indices = (
        _keep_indices_by_feature(info, features, exclude_joint_names)
        if exclude_joint_names else {})
    if exclude_joint_names:
        print(f'{dataset_root}: excluding {exclude_joint_names} '
              f'(pad_to={pad_to}) -- must match the training config\'s '
              'SelectJointDims exactly.', file=sys.stderr)

    def _feature_values(feat: str) -> np.ndarray:
        values = np.stack(table.column(feat).to_pylist())
        if feat in keep_indices:
            values = _select_and_pad(values, keep_indices[feat], pad_to)
        return values

    meta_dir = dataset_root / 'meta'
    feature_values = {feat: _feature_values(feat) for feat in features}

    if version == 'v3.0':
        # v3.0's meta/stats.json is a single aggregate dict (see
        # fluxvla/datasets/parquet_dataset_v3.py's _load_stats_records),
        # not per-episode -- one file, computed over every frame at once.
        agg_stats = {
            feat: _feature_stats(feature_values[feat])
            for feat in features
        }
        _atomic_write_text(meta_dir / 'stats.json',
                          json.dumps(agg_stats, indent=2))
        print(f'{dataset_root}: wrote meta/stats.json '
              f'({table.num_rows}/{total_frames} frames, features={features})')
        return

    # v2.1 (or anything without a recognized v3.0 codebase_version):
    # per-episode stats, one JSON line per episode, matching
    # ParquetDataset's meta/episodes_stats.jsonl.
    episode_index_col = np.asarray(table.column('episode_index').to_pylist())
    lines = []
    for episode_index in sorted(set(episode_index_col.tolist())):
        mask = episode_index_col == episode_index
        ep_stats = {
            feat: _feature_stats(feature_values[feat][mask])
            for feat in features
        }
        lines.append(
            json.dumps(
                {
                    'episode_index': int(episode_index),
                    'stats': ep_stats
                },
                ensure_ascii=False))
    _atomic_write_text(meta_dir / 'episodes_stats.jsonl',
                       ''.join(line + '\n' for line in lines))
    print(f'{dataset_root}: wrote meta/episodes_stats.jsonl '
          f'({len(lines)} episodes, {table.num_rows}/{total_frames} frames, '
          f'features={features})')


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'dataset_root',
        nargs='+',
        type=Path,
        help='One or more dataset root directories to compute stats for.')
    parser.add_argument(
        '--features',
        nargs='+',
        default=['observation.state', 'action'],
        help='Parquet columns to compute stats for (default: %(default)s).')
    parser.add_argument(
        '--max-frames',
        type=int,
        default=0,
        help='Randomly subsample to this many frames before computing '
             'stats, for speed on large datasets. 0 (default) uses every '
             'frame.')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='RNG seed for --max-frames subsampling (default: %(default)s).')
    parser.add_argument(
        '--exclude-joint-names',
        nargs='+',
        default=None,
        help='Joint names to drop before computing stats -- must match the '
             "training config's SelectJointDims(exclude_joint_names=...).")
    parser.add_argument(
        '--pad-to',
        type=int,
        default=None,
        help='Zero-pad after dropping, matching SelectJointDims(pad_to=...).')
    args = parser.parse_args()

    exit_code = 0
    for dataset_root in args.dataset_root:
        try:
            compute_stats(dataset_root, args.features, args.max_frames,
                         args.seed, args.exclude_joint_names, args.pad_to)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f'{dataset_root}: FAILED - {exc}', file=sys.stderr)
            exit_code = 1
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
