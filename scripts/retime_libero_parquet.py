#!/usr/bin/env python3
"""Create retimed LIBERO/LeRobot parquet datasets.

This script keeps the original task prompts and videos, but rewrites the
tabular trajectory rows at different temporal rates. It is intended for
experiments where a VLA learns from a single faster/slower retimed dataset, or
from multiple speed-conditioned variants generated from the same source.

For LIBERO EEF_POS-style actions, the default behavior treats actions as
absolute targets and interpolates the first action dimensions while keeping the
last gripper dimension nearest-neighbor. If your action is a delta command,
use this script as a starting point and implement task-specific action
composition before training.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


DEFAULT_SPEEDS = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)


def import_datasets():
    try:
        from datasets import Dataset, load_dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            'This script requires HuggingFace `datasets`. Run it in the same '
            'environment used for FluxVLA training, or install `datasets`.'
        ) from exc
    return Dataset, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate retimed LIBERO parquet dataset variants.')
    parser.add_argument(
        '--input-root',
        required=True,
        help='Original LeRobot dataset root containing meta/ and data/.')
    parser.add_argument(
        '--output-root',
        required=True,
        help='Directory that will contain speed_* dataset roots.')
    parser.add_argument(
        '--speeds',
        nargs='+',
        type=float,
        default=list(DEFAULT_SPEEDS),
        help='Temporal speed factors. >1 compresses, <1 slows down.')
    parser.add_argument(
        '--task-indices',
        nargs='*',
        type=int,
        default=[0],
        help='Task indices to keep. Default: task0 only.')
    parser.add_argument(
        '--state-key',
        default='observation.state',
        help='State/proprio column to interpolate.')
    parser.add_argument(
        '--action-key',
        default='action',
        help='Raw action column to interpolate.')
    parser.add_argument(
        '--speed-key',
        default='tempo_speed',
        help='Column name used to store the temporal speed factor.')
    parser.add_argument(
        '--split',
        default='train',
        help='Dataset split to load from input data/.')
    parser.add_argument(
        '--video-mode',
        choices=('symlink', 'copy', 'absolute', 'none'),
        default='symlink',
        help=(
            'How output roots access original videos. symlink/copy preserve '
            'the original relative info["video_path"]; absolute rewrites '
            'info["video_path"] to the input root; none leaves videos absent.'
        ))
    parser.add_argument(
        '--data-file',
        default='retimed.parquet',
        help='Parquet filename written under each output data/ directory.')
    parser.add_argument(
        '--interpolate-keys',
        nargs='*',
        default=None,
        help=(
            'Extra numeric columns to interpolate. By default only state, '
            'action, and timestamp are retimed.'
        ))
    parser.add_argument(
        '--nearest-keys',
        nargs='*',
        default=None,
        help='Extra columns copied from the nearest source row.')
    parser.add_argument(
        '--keep-last-action-dim-nearest',
        action='store_true',
        default=True,
        help='Keep final action dimension nearest-neighbor, useful for gripper.')
    parser.add_argument(
        '--interpolate-last-action-dim',
        action='store_false',
        dest='keep_last_action_dim_nearest',
        help='Also interpolate the final action dimension.')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing speed_* output directories.')
    return parser.parse_args()


def speed_name(speed: float) -> str:
    return f'speed_{str(speed).replace(".", "_")}'


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open('r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def as_float_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def is_numeric_value(value: Any) -> bool:
    try:
        arr = np.asarray(value)
    except Exception:
        return False
    return np.issubdtype(arr.dtype, np.number)


def interp_numeric(a: Any, b: Any, alpha: float) -> Any:
    arr_a = as_float_array(a)
    arr_b = as_float_array(b)
    out = (1.0 - alpha) * arr_a + alpha * arr_b
    if out.ndim == 0:
        return float(out)
    return out.tolist()


def nearest_value(a: Any, b: Any, alpha: float) -> Any:
    return deepcopy(a if alpha < 0.5 else b)


def interp_action(a: Any, b: Any, alpha: float,
                  keep_last_dim_nearest: bool) -> Any:
    arr_a = as_float_array(a)
    arr_b = as_float_array(b)
    out = (1.0 - alpha) * arr_a + alpha * arr_b
    if keep_last_dim_nearest and out.ndim > 0 and out.shape[-1] > 0:
        nearest = arr_a if alpha < 0.5 else arr_b
        out[..., -1] = nearest[..., -1]
    if out.ndim == 0:
        return float(out)
    return out.tolist()


def source_positions(num_rows: int, speed: float) -> np.ndarray:
    if num_rows <= 1:
        return np.array([0.0], dtype=np.float32)
    if speed <= 0:
        raise ValueError(f'speed must be positive, got {speed}')
    positions = np.arange(0.0, num_rows - 1 + 1e-6, speed, dtype=np.float32)
    if not np.isclose(positions[-1], num_rows - 1):
        positions = np.concatenate(
            [positions, np.array([num_rows - 1], dtype=np.float32)])
    return positions


def retime_episode(rows: Sequence[dict], speed: float, state_key: str,
                   action_key: str, interpolate_keys: Sequence[str],
                   nearest_keys: Sequence[str],
                   keep_last_action_dim_nearest: bool,
                   episode_index: int,
                   speed_key: str) -> List[dict]:
    positions = source_positions(len(rows), speed)
    retimed = []

    interpolate_set = set(interpolate_keys)
    nearest_set = set(nearest_keys)
    interpolate_set.update([state_key, action_key, 'timestamp'])

    for new_frame_idx, pos in enumerate(positions):
        lo = int(math.floor(float(pos)))
        hi = min(lo + 1, len(rows) - 1)
        alpha = float(pos - lo)
        row_lo = rows[lo]
        row_hi = rows[hi]
        nearest_row = row_lo if alpha < 0.5 else row_hi
        out = deepcopy(nearest_row)

        for key in interpolate_set:
            if key not in row_lo or key not in row_hi:
                continue
            if key == action_key:
                out[key] = interp_action(row_lo[key], row_hi[key], alpha,
                                         keep_last_action_dim_nearest)
            elif is_numeric_value(row_lo[key]) and is_numeric_value(row_hi[key]):
                out[key] = interp_numeric(row_lo[key], row_hi[key], alpha)

        for key in nearest_set:
            if key in row_lo and key in row_hi:
                out[key] = nearest_value(row_lo[key], row_hi[key], alpha)

        out['episode_index'] = int(episode_index)
        out[speed_key] = float(speed)
        if 'frame_index' in out:
            out['frame_index'] = int(new_frame_idx)
        if 'index' in out:
            out.pop('index', None)
        retimed.append(out)

    return retimed


def stat_block(values: Sequence[Any]) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    return {
        'min': arr.min(axis=0).tolist(),
        'max': arr.max(axis=0).tolist(),
        'mean': arr.mean(axis=0).tolist(),
        'std': arr.std(axis=0).tolist(),
        'q01': np.quantile(arr, 0.01, axis=0).tolist(),
        'q99': np.quantile(arr, 0.99, axis=0).tolist(),
        'count': [int(arr.shape[0])],
    }


def episode_stats(rows: Sequence[dict], state_key: str,
                  action_key: str) -> dict:
    stats = {}
    for key in [state_key, action_key, 'timestamp', 'task_index',
                'episode_index']:
        if key in rows[0]:
            stats[key] = stat_block([row[key] for row in rows])
    return {'episode_index': int(rows[0]['episode_index']), 'stats': stats}


def update_episode_record(template: dict, rows: Sequence[dict],
                          episode_index: int) -> dict:
    record = deepcopy(template) if template else {}
    record['episode_index'] = int(episode_index)
    record['length'] = len(rows)
    record['num_frames'] = len(rows)
    if 'tasks' not in record and rows and 'task_index' in rows[0]:
        record['tasks'] = [int(rows[0]['task_index'])]
    return record


def copy_meta(input_meta: Path, output_meta: Path, info: dict) -> None:
    output_meta.mkdir(parents=True, exist_ok=True)
    tasks_path = input_meta / 'tasks.jsonl'
    if tasks_path.exists():
        shutil.copy2(tasks_path, output_meta / 'tasks.jsonl')
    with (output_meta / 'info.json').open('w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


def prepare_videos(input_root: Path, output_root: Path, info: dict,
                   mode: str) -> dict:
    out_info = deepcopy(info)
    if mode == 'none':
        return out_info
    if mode == 'absolute':
        video_path = out_info.get('video_path')
        if video_path and not os.path.isabs(video_path):
            out_info['video_path'] = str((input_root / video_path).resolve())
        return out_info

    for name in ('videos', 'video'):
        src = input_root / name
        dst = output_root / name
        if not src.exists():
            continue
        if dst.exists():
            continue
        if mode == 'symlink':
            dst.symlink_to(src.resolve(), target_is_directory=True)
        elif mode == 'copy':
            shutil.copytree(src, dst)
    return out_info


def update_info_counts(info: dict, rows: Sequence[dict],
                       episodes: Sequence[dict], speed_key: str) -> dict:
    out_info = deepcopy(info)
    out_info['total_episodes'] = len(episodes)
    out_info['total_frames'] = len(rows)
    if 'total_videos' in out_info:
        video_feature_count = sum(
            1 for feature in out_info.get('features', {}).values()
            if isinstance(feature, dict) and feature.get('dtype') == 'video')
        if video_feature_count:
            out_info['total_videos'] = len(episodes) * video_feature_count
    out_info['splits'] = {'train': f'0:{len(episodes)}'}
    if rows:
        for key, value in rows[0].items():
            if key not in out_info.get('features', {}) and key == speed_key:
                out_info.setdefault('features', {})[key] = {
                    'dtype': 'float32',
                    'shape': [1],
                    'names': None,
                }
    return out_info


def load_info(input_root: Path) -> dict:
    with (input_root / 'meta' / 'info.json').open('r', encoding='utf-8') as f:
        return json.load(f)


def group_by_episode(rows: Iterable[dict]) -> Dict[int, List[dict]]:
    grouped: Dict[int, List[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row['episode_index'])].append(dict(row))
    for episode_rows in grouped.values():
        episode_rows.sort(
            key=lambda x: (x.get('frame_index', 0), x.get('timestamp', 0.0)))
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def write_dataset_root(dataset_cls, output_root: Path, rows: Sequence[dict],
                       stats_rows: Sequence[dict],
                       episode_rows: Sequence[dict], info: dict,
                       input_meta: Path, data_file: str) -> None:
    copy_meta(input_meta, output_root / 'meta', info)
    write_jsonl(output_root / 'meta' / 'episodes_stats.jsonl', stats_rows)
    write_jsonl(output_root / 'meta' / 'episodes.jsonl', episode_rows)

    data_dir = output_root / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_cls.from_list(list(rows)).to_parquet(str(data_dir / data_file))


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_base = Path(args.output_root).expanduser().resolve()
    input_meta = input_root / 'meta'
    input_data = input_root / 'data'
    dataset_cls, load_dataset = import_datasets()

    if not input_meta.exists() or not input_data.exists():
        raise FileNotFoundError(
            f'Expected {input_root} to contain meta/ and data/')

    task_indices = set(args.task_indices or [])
    dataset = load_dataset('parquet', data_dir=str(input_data),
                           split=args.split)
    raw_rows = [
        dict(row) for row in dataset
        if not task_indices or int(row.get('task_index', -1)) in task_indices
    ]
    if not raw_rows:
        raise ValueError(f'No rows matched task_indices={sorted(task_indices)}')

    source_by_episode = group_by_episode(raw_rows)
    source_episode_meta = {
        int(row.get('episode_index', idx)): row
        for idx, row in enumerate(read_jsonl(input_meta / 'episodes.jsonl'))
    }
    source_info = load_info(input_root)

    output_base.mkdir(parents=True, exist_ok=True)
    summary = []
    extra_interpolate = args.interpolate_keys or []
    extra_nearest = args.nearest_keys or []

    for speed in args.speeds:
        out_root = output_base / speed_name(speed)
        if out_root.exists():
            if not args.overwrite:
                raise FileExistsError(
                    f'{out_root} already exists; pass --overwrite to replace')
            shutil.rmtree(out_root)
        out_root.mkdir(parents=True, exist_ok=True)

        out_info = prepare_videos(input_root, out_root, source_info,
                                  args.video_mode)
        all_rows: List[dict] = []
        all_stats: List[dict] = []
        all_episodes: List[dict] = []

        for old_episode_index, episode_rows in source_by_episode.items():
            retimed_rows = retime_episode(
                episode_rows,
                speed=speed,
                state_key=args.state_key,
                action_key=args.action_key,
                interpolate_keys=extra_interpolate,
                nearest_keys=extra_nearest,
                keep_last_action_dim_nearest=args.keep_last_action_dim_nearest,
                episode_index=old_episode_index,
                speed_key=args.speed_key)
            all_rows.extend(retimed_rows)
            all_stats.append(
                episode_stats(retimed_rows, args.state_key, args.action_key))
            all_episodes.append(
                update_episode_record(
                    source_episode_meta.get(old_episode_index, {}),
                    retimed_rows,
                    old_episode_index))

        write_dataset_root(
            dataset_cls,
            out_root,
            rows=all_rows,
            stats_rows=all_stats,
            episode_rows=all_episodes,
            info=update_info_counts(out_info, all_rows, all_episodes,
                                    args.speed_key),
            input_meta=input_meta,
            data_file=args.data_file)

        record = {
            'speed': speed,
            'output_root': str(out_root),
            'num_rows': len(all_rows),
            'num_episodes': len(all_episodes),
        }
        summary.append(record)
        print(
            f'[retime] speed={speed:g} rows={len(all_rows)} '
            f'episodes={len(all_episodes)} -> {out_root}',
            flush=True)

    with (output_base / 'retime_summary.json').open('w',
                                                    encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
