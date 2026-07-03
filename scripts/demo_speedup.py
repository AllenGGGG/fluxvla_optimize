#!/usr/bin/env python
# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Build an entropy/importance downsampled LeRobot parquet dataset.

This script creates a new dataset directory without modifying the source
dataset. It keeps the original videos through symlinks and rewrites parquet
rows on a shorter time axis. For LIBERO delta-style actions, continuous action
dims are accumulated between kept frames while the gripper dim is copied from
the end of each interval.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


CONT_DIMS = 6
GRIPPER_DIM = 6


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create an entropy-guided speedup parquet dataset.')
    parser.add_argument('--src', required=True, help='Source dataset root.')
    parser.add_argument('--dst', required=True, help='Output dataset root.')
    parser.add_argument(
        '--task-indices',
        default=None,
        help='Comma-separated task indices to process, e.g. 0. '
        'Default processes all tasks.')
    parser.add_argument(
        '--target-speed',
        type=float,
        default=1.5,
        help='Approximate per-episode speedup T/K.')
    parser.add_argument(
        '--min-keep',
        type=int,
        default=12,
        help='Minimum kept rows per episode.')
    parser.add_argument(
        '--length-weight',
        type=float,
        default=1.0,
        help='Weight for action magnitude importance.')
    parser.add_argument(
        '--accel-weight',
        type=float,
        default=0.4,
        help='Weight for action change importance.')
    parser.add_argument(
        '--turn-weight',
        type=float,
        default=0.4,
        help='Weight for direction change importance.')
    parser.add_argument(
        '--gripper-weight',
        type=float,
        default=1.2,
        help='Weight for gripper change importance.')
    parser.add_argument(
        '--floor',
        type=float,
        default=0.05,
        help='Uniform density floor; larger keeps low-entropy regions denser.')
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite destination if it already exists.')
    return parser.parse_args()


def parse_task_indices(value):
    if value is None or value == '':
        return None
    return {int(x) for x in value.split(',') if x.strip()}


def normalize(values):
    values = np.asarray(values, dtype=np.float64)
    values = np.maximum(values, 0.0)
    scale = float(np.max(values, initial=0.0))
    if scale <= 1e-12:
        return np.zeros_like(values)
    return values / scale


def row_array(row, key):
    return np.asarray(row[key], dtype=np.float32)


def compute_importance(actions, args):
    cont = actions[:, :CONT_DIMS].astype(np.float64)
    gripper = actions[:, GRIPPER_DIM].astype(np.float64)
    n = len(actions)

    step_len = np.linalg.norm(cont, axis=1)

    accel = np.zeros(n, dtype=np.float64)
    if n > 1:
        accel_step = np.linalg.norm(np.diff(cont, axis=0), axis=1)
        accel[:-1] = np.maximum(accel[:-1], accel_step)
        accel[1:] = np.maximum(accel[1:], accel_step)

    turn = np.zeros(n, dtype=np.float64)
    if n > 1:
        prev = cont[:-1]
        nxt = cont[1:]
        prev_norm = np.linalg.norm(prev, axis=1)
        nxt_norm = np.linalg.norm(nxt, axis=1)
        valid = (prev_norm > 1e-8) & (nxt_norm > 1e-8)
        if np.any(valid):
            cos = np.zeros(n - 1, dtype=np.float64)
            cos[valid] = (
                np.sum(prev[valid] * nxt[valid], axis=1) /
                (prev_norm[valid] * nxt_norm[valid]))
            angle = np.arccos(np.clip(cos, -1.0, 1.0)) / np.pi
            turn[:-1] = np.maximum(turn[:-1], angle)
            turn[1:] = np.maximum(turn[1:], angle)

    grip = np.zeros(n, dtype=np.float64)
    if n > 1:
        grip_step = np.abs(np.diff(gripper))
        grip[:-1] = np.maximum(grip[:-1], grip_step)
        grip[1:] = np.maximum(grip[1:], grip_step)

    score = (
        args.length_weight * normalize(step_len) +
        args.accel_weight * normalize(accel) +
        args.turn_weight * normalize(turn) +
        args.gripper_weight * normalize(grip))
    score += float(args.floor)
    score[0] = max(score[0], 1.0)
    score[-1] = max(score[-1], 1.0)
    return score


def sample_keep_indices(score, target_count):
    n = len(score)
    target_count = int(np.clip(target_count, 2, n))
    if target_count >= n:
        return np.arange(n, dtype=np.int64)

    cumulative = np.concatenate([[0.0], np.cumsum(score)])
    targets = np.linspace(0.0, cumulative[-1], target_count)
    positions = np.searchsorted(cumulative, targets, side='right') - 1
    positions = np.clip(positions, 0, n - 1)

    keep = np.unique(np.concatenate([[0], positions, [n - 1]])).astype(np.int64)
    if len(keep) < target_count:
        missing = target_count - len(keep)
        candidates = np.setdiff1d(np.arange(n, dtype=np.int64), keep)
        if len(candidates):
            ranked = candidates[np.argsort(-score[candidates], kind='stable')]
            keep = np.unique(np.concatenate([keep, ranked[:missing]]))
    if len(keep) > target_count:
        protected = {0, n - 1}
        interior = np.array([i for i in keep if i not in protected],
                            dtype=np.int64)
        ranked = interior[np.argsort(-score[interior], kind='stable')]
        keep = np.unique(
            np.concatenate([[0], ranked[:target_count - 2], [n - 1]]))
    return np.sort(keep).astype(np.int64)


def retime_episode(df, keep):
    rows = []
    global_index = None
    for out_i, src_i in enumerate(keep):
        row = df.iloc[int(src_i)].copy()
        if out_i < len(keep) - 1:
            start = int(src_i)
            end = int(keep[out_i + 1])
            action_block = np.stack(
                [row_array(df.iloc[j], 'action') for j in range(start, end)],
                axis=0)
            new_action = np.zeros_like(action_block[0])
            new_action[:CONT_DIMS] = np.sum(action_block[:, :CONT_DIMS], axis=0)
            new_action[GRIPPER_DIM] = action_block[-1, GRIPPER_DIM]
        else:
            new_action = np.zeros_like(row_array(row, 'action'))
            new_action[GRIPPER_DIM] = row_array(row, 'action')[GRIPPER_DIM]

        row['action'] = new_action.astype(np.float32).tolist()
        row['frame_index'] = out_i
        if global_index is None:
            global_index = int(row['index'])
        row['index'] = global_index + out_i
        rows.append(row)
    return pd.DataFrame(rows)


def copy_meta(src, dst):
    src_meta = src / 'meta'
    dst_meta = dst / 'meta'
    dst_meta.mkdir(parents=True, exist_ok=True)
    for name in ('info.json', 'tasks.jsonl'):
        shutil.copy2(src_meta / name, dst_meta / name)


def update_info(dst, num_episodes, total_frames):
    info_path = dst / 'meta' / 'info.json'
    with open(info_path, 'r', encoding='utf-8') as f:
        info = json.load(f)
    info['total_episodes'] = int(num_episodes)
    info['total_frames'] = int(total_frames)
    info['total_chunks'] = 1 if num_episodes else 0
    info['splits'] = {'train': f'0:{int(num_episodes)}'}

    video_features = [
        feature for feature in info.get('features', {}).values()
        if feature.get('dtype') == 'video'
    ]
    if video_features:
        info['total_videos'] = int(num_episodes * len(video_features))

    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)


def symlink_videos(src, dst):
    src_videos = src / 'videos'
    dst_videos = dst / 'videos'
    if dst_videos.exists():
        return
    rel = os.path.relpath(src_videos, dst_videos.parent)
    os.symlink(rel, dst_videos)


def write_jsonl(path, rows):
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def numeric_stats(values):
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    values = values.astype(np.float64)
    return {
        'min': np.min(values, axis=0).tolist(),
        'max': np.max(values, axis=0).tolist(),
        'mean': np.mean(values, axis=0).tolist(),
        'std': np.std(values, axis=0).tolist(),
        'count': [int(len(values))],
    }


def episode_stats(df):
    state = np.stack(
        [np.asarray(v, dtype=np.float64) for v in df['observation.state']],
        axis=0)
    action = np.stack(
        [np.asarray(v, dtype=np.float64) for v in df['action']],
        axis=0)
    timestamp = df['timestamp'].to_numpy(dtype=np.float64)[:, None]
    frame_index = df['frame_index'].to_numpy(dtype=np.float64)[:, None]
    episode_index = df['episode_index'].to_numpy(dtype=np.float64)[:, None]
    index = df['index'].to_numpy(dtype=np.float64)[:, None]
    task_index = df['task_index'].to_numpy(dtype=np.float64)[:, None]
    return {
        'observation.state': numeric_stats(state),
        'action': numeric_stats(action),
        'timestamp': numeric_stats(timestamp),
        'frame_index': numeric_stats(frame_index),
        'episode_index': numeric_stats(episode_index),
        'index': numeric_stats(index),
        'task_index': numeric_stats(task_index),
    }


def main():
    args = parse_args()
    if args.target_speed <= 0:
        raise ValueError('--target-speed must be positive')

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()
    task_indices = parse_task_indices(args.task_indices)

    if dst.exists():
        if not args.force:
            raise FileExistsError(f'{dst} exists; pass --force to overwrite.')
        shutil.rmtree(dst)
    (dst / 'data' / 'chunk-000').mkdir(parents=True, exist_ok=True)
    copy_meta(src, dst)
    symlink_videos(src, dst)

    with open(src / 'meta' / 'tasks.jsonl', 'r', encoding='utf-8') as f:
        tasks = [json.loads(line) for line in f]
    with open(src / 'meta' / 'episodes.jsonl', 'r', encoding='utf-8') as f:
        episodes = [json.loads(line) for line in f]

    out_episodes = []
    out_stats = []
    summary = []
    total_frames = 0

    for ep in episodes:
        ep_idx = int(ep['episode_index'])
        src_file = src / 'data' / 'chunk-000' / f'episode_{ep_idx:06d}.parquet'
        df = pd.read_parquet(src_file)
        if df.empty:
            continue
        task_idx = int(df.iloc[0]['task_index'])
        if task_indices is not None and task_idx not in task_indices:
            continue

        actions = np.stack([row_array(row, 'action') for _, row in df.iterrows()])
        score = compute_importance(actions, args)
        target_count = max(args.min_keep, int(round(len(df) / args.target_speed)))
        keep = sample_keep_indices(score, target_count)
        out_df = retime_episode(df, keep)

        out_file = dst / 'data' / 'chunk-000' / f'episode_{ep_idx:06d}.parquet'
        out_df.to_parquet(out_file, index=False)

        out_episodes.append({
            'episode_index': ep_idx,
            'tasks': ep.get('tasks', [tasks[task_idx]['task']]),
            'length': int(len(out_df)),
        })
        out_stats.append({
            'episode_index': ep_idx,
            'stats': episode_stats(out_df),
        })
        summary.append({
            'episode_index': ep_idx,
            'task_index': task_idx,
            'input_len': int(len(df)),
            'output_len': int(len(out_df)),
            'speedup': float(len(df) / max(len(out_df), 1)),
        })
        total_frames += int(len(out_df))

    write_jsonl(dst / 'meta' / 'episodes.jsonl', out_episodes)
    write_jsonl(dst / 'meta' / 'episodes_stats.jsonl', out_stats)
    update_info(dst, len(summary), total_frames)
    with open(dst / 'demo_speedup_summary.json', 'w', encoding='utf-8') as f:
        json.dump({
            'source': str(src),
            'target_speed': args.target_speed,
            'num_episodes': len(summary),
            'mean_speedup': (
                float(np.mean([row['speedup'] for row in summary]))
                if summary else 0.0),
            'episodes': summary,
        }, f, indent=2)

    print(f'Wrote {len(summary)} episodes to {dst}')
    if summary:
        print('mean speedup:', np.mean([row['speedup'] for row in summary]))


if __name__ == '__main__':
    main()
