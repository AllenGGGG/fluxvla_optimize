#!/usr/bin/env python
# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Preview phase-aware demo speedup videos.

This keeps gripper-change windows and the final placement segment dense, then
spends the requested compression mostly on low-risk frames.
"""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd

from demo_speedup import compute_importance, parse_task_indices


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create safe phase-aware speedup preview videos.')
    parser.add_argument('--src', required=True, help='Source dataset root.')
    parser.add_argument('--out-dir', required=True, help='Preview output dir.')
    parser.add_argument('--task-indices', default='0')
    parser.add_argument('--target-speed', type=float, default=2.0)
    parser.add_argument('--num-videos', type=int, default=5)
    parser.add_argument('--video-key', default='observation.images.image')
    parser.add_argument('--fps', type=float, default=None)
    parser.add_argument('--quality', type=int, default=8)
    parser.add_argument('--min-keep', type=int, default=12)
    parser.add_argument(
        '--final-protect-ratio',
        type=float,
        default=0.22,
        help='Final episode fraction kept at original frame density.')
    parser.add_argument(
        '--gripper-window',
        type=int,
        default=8,
        help='Frames protected on each side of gripper changes.')
    parser.add_argument(
        '--gripper-threshold',
        type=float,
        default=1e-4,
        help='Minimum absolute gripper delta treated as a change.')
    parser.add_argument(
        '--max-gap',
        type=int,
        default=4,
        help='Maximum original-frame gap between kept frames. This prevents '
        'low-risk regions from being compressed into teleport-like jumps.')
    parser.add_argument('--length-weight', type=float, default=1.0)
    parser.add_argument('--accel-weight', type=float, default=0.5)
    parser.add_argument('--turn-weight', type=float, default=0.5)
    parser.add_argument('--gripper-weight', type=float, default=2.0)
    parser.add_argument('--floor', type=float, default=0.04)
    return parser.parse_args()


def protected_mask(actions, args):
    n = len(actions)
    protected = np.zeros(n, dtype=bool)
    if n == 0:
        return protected

    protected[0] = True
    protected[-1] = True

    final_start = int(np.floor(n * (1.0 - args.final_protect_ratio)))
    final_start = int(np.clip(final_start, 0, n - 1))
    protected[final_start:] = True

    gripper = actions[:, 6].astype(np.float64)
    if n > 1:
        changes = np.where(np.abs(np.diff(gripper)) > args.gripper_threshold)[0]
        for idx in changes:
            lo = max(0, int(idx) - args.gripper_window)
            hi = min(n, int(idx) + args.gripper_window + 2)
            protected[lo:hi] = True

    return protected


def sample_safe_keep(score, protected, target_count):
    n = len(score)
    target_count = int(np.clip(target_count, 2, n))
    protected = protected.copy()
    protected[0] = True
    protected[-1] = True

    protected_idx = np.where(protected)[0]
    target_count = max(target_count, len(protected_idx))
    if target_count >= n:
        return np.arange(n, dtype=np.int64)

    keep = set(int(i) for i in protected_idx)
    remaining = target_count - len(keep)
    candidates = np.where(~protected)[0]
    if remaining <= 0 or len(candidates) == 0:
        return np.array(sorted(keep), dtype=np.int64)

    weights = np.asarray(score[candidates], dtype=np.float64)
    weights = np.maximum(weights, 1e-8)
    cumulative = np.concatenate([[0.0], np.cumsum(weights)])
    targets = np.linspace(0.0, cumulative[-1], remaining + 2)[1:-1]
    picked_pos = np.searchsorted(cumulative, targets, side='right') - 1
    picked_pos = np.clip(picked_pos, 0, len(candidates) - 1)
    picked = candidates[picked_pos]
    keep.update(int(i) for i in picked)

    if len(keep) < target_count:
        missing = target_count - len(keep)
        unused = np.array([i for i in candidates if int(i) not in keep],
                          dtype=np.int64)
        if len(unused):
            ranked = unused[np.argsort(-score[unused], kind='stable')]
            keep.update(int(i) for i in ranked[:missing])

    return np.array(sorted(keep), dtype=np.int64)


def enforce_max_gap(keep, n, max_gap):
    if max_gap <= 1 or len(keep) == 0:
        return np.arange(n, dtype=np.int64)

    expanded = []
    keep = np.asarray(sorted(set(int(i) for i in keep)), dtype=np.int64)
    for start, end in zip(keep[:-1], keep[1:]):
        expanded.append(int(start))
        gap = int(end - start)
        if gap > max_gap:
            extra = np.arange(start + max_gap, end, max_gap, dtype=np.int64)
            expanded.extend(int(i) for i in extra)
    expanded.append(int(keep[-1]))
    return np.array(sorted(set(expanded)), dtype=np.int64)


def write_video(src_path, dst_path, frame_indices, fps, quality):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    reader = imageio.get_reader(str(src_path))
    try:
        frames = [frame for frame in reader.iter_data()]
    finally:
        reader.close()
    if not frames:
        raise ValueError(f'No frames decoded from {src_path}')

    with imageio.get_writer(
            str(dst_path),
            fps=fps,
            codec='libx264',
            quality=quality,
            macro_block_size=None,
            pixelformat='yuv420p') as writer:
        for idx in frame_indices:
            idx = int(np.clip(idx, 0, len(frames) - 1))
            writer.append_data(frames[idx])


def main():
    args = parse_args()
    if args.target_speed <= 0:
        raise ValueError('--target-speed must be positive')

    src = Path(args.src).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    task_indices = parse_task_indices(args.task_indices)

    with open(src / 'meta' / 'info.json', 'r', encoding='utf-8') as f:
        info = json.load(f)
    with open(src / 'meta' / 'episodes.jsonl', 'r', encoding='utf-8') as f:
        episodes = [json.loads(line) for line in f]

    fps = float(args.fps if args.fps is not None else info.get('fps', 20))
    video_template = info['video_path']
    chunks_size = int(info.get('chunks_size', 1000))

    summary = []
    for ep in episodes:
        if len(summary) >= args.num_videos:
            break
        ep_idx = int(ep['episode_index'])
        parquet_path = src / 'data' / 'chunk-000' / f'episode_{ep_idx:06d}.parquet'
        df = pd.read_parquet(parquet_path)
        if df.empty:
            continue
        task_idx = int(df.iloc[0]['task_index'])
        if task_indices is not None and task_idx not in task_indices:
            continue

        actions = np.stack(
            [np.asarray(row['action'], dtype=np.float32) for _, row in df.iterrows()],
            axis=0)
        score = compute_importance(actions, args)
        protected = protected_mask(actions, args)
        target_count = max(args.min_keep, int(round(len(df) / args.target_speed)))
        keep = sample_safe_keep(score, protected, target_count)
        keep = enforce_max_gap(keep, len(df), args.max_gap)
        frame_indices = df.iloc[keep]['frame_index'].to_numpy(dtype=np.int64)

        episode_chunk = ep_idx // chunks_size
        rel_video = video_template.format(
            episode_chunk=episode_chunk,
            video_key=args.video_key,
            episode_index=ep_idx)
        src_video = src / rel_video
        if not src_video.exists():
            raise FileNotFoundError(f'Video not found: {src_video}')

        dst_video = out_dir / (
            f'episode_{ep_idx:06d}_safe_speed{args.target_speed:g}_'
            f'{args.video_key.split(".")[-1]}.mp4')
        write_video(src_video, dst_video, frame_indices, fps, args.quality)

        protected_kept = int(np.count_nonzero(protected[keep]))
        summary.append({
            'episode_index': ep_idx,
            'task_index': task_idx,
            'input_frames': int(len(df)),
            'output_frames': int(len(frame_indices)),
            'speedup': float(len(df) / max(len(frame_indices), 1)),
            'protected_frames': int(np.count_nonzero(protected)),
            'protected_kept': protected_kept,
            'video': str(dst_video),
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'preview_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f'Wrote {len(summary)} preview videos to {out_dir}')
    for row in summary:
        print(
            f"episode_{row['episode_index']:06d}: "
            f"{row['input_frames']} -> {row['output_frames']} "
            f"({row['speedup']:.3f}x), "
            f"protected {row['protected_kept']}/{row['protected_frames']} "
            f"{row['video']}")


if __name__ == '__main__':
    main()
