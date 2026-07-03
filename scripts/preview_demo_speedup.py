#!/usr/bin/env python
# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Write preview videos for the entropy/importance demo speedup script."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd

from demo_speedup import compute_importance, parse_task_indices, sample_keep_indices


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create short mp4 previews of demo_speedup frame selection.')
    parser.add_argument('--src', required=True, help='Source dataset root.')
    parser.add_argument('--out-dir', required=True, help='Preview output dir.')
    parser.add_argument('--task-indices', default='0')
    parser.add_argument('--target-speed', type=float, default=1.5)
    parser.add_argument('--num-videos', type=int, default=5)
    parser.add_argument(
        '--video-key',
        default='observation.images.image',
        help='Video feature to preview.')
    parser.add_argument('--fps', type=float, default=None)
    parser.add_argument('--quality', type=int, default=8)
    parser.add_argument('--length-weight', type=float, default=1.0)
    parser.add_argument('--accel-weight', type=float, default=0.4)
    parser.add_argument('--turn-weight', type=float, default=0.4)
    parser.add_argument('--gripper-weight', type=float, default=1.2)
    parser.add_argument('--floor', type=float, default=0.05)
    parser.add_argument('--min-keep', type=int, default=12)
    return parser.parse_args()


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

    written = 0
    summary = []
    for ep in episodes:
        if written >= args.num_videos:
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
        target_count = max(args.min_keep, int(round(len(df) / args.target_speed)))
        keep = sample_keep_indices(score, target_count)
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
            f'episode_{ep_idx:06d}_speed{args.target_speed:g}_'
            f'{args.video_key.split(".")[-1]}.mp4')
        write_video(src_video, dst_video, frame_indices, fps, args.quality)
        summary.append({
            'episode_index': ep_idx,
            'task_index': task_idx,
            'input_frames': int(len(df)),
            'output_frames': int(len(frame_indices)),
            'speedup': float(len(df) / max(len(frame_indices), 1)),
            'video': str(dst_video),
        })
        written += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'preview_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f'Wrote {written} preview videos to {out_dir}')
    for row in summary:
        print(
            f"episode_{row['episode_index']:06d}: "
            f"{row['input_frames']} -> {row['output_frames']} "
            f"({row['speedup']:.3f}x) {row['video']}")


if __name__ == '__main__':
    main()
