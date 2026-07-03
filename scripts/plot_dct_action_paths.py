#!/usr/bin/env python
# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from libero.libero import benchmark  # noqa: E402
from mmengine import Config, DictAction  # noqa: E402

from fluxvla.engines import build_runner_from_cfg  # noqa: E402
from fluxvla.engines.utils.eval_utils import (  # noqa: E402
    get_libero_dummy_action,
    get_libero_env,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot raw vs DCT-compressed PI0.5 action chunks.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt-path', required=True)
    parser.add_argument('--output-dir', default='work_dirs/action_path_plots')
    parser.add_argument('--task-id', type=int, default=4)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--episodes', type=int, default=2)
    parser.add_argument('--chunks-per-episode', type=int, default=4)
    parser.add_argument('--max-steps', type=int, default=120)
    parser.add_argument(
        '--execute',
        choices=('compressed', 'raw'),
        default='compressed',
        help='Which action chunk to execute while collecting plots.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config options in xxx=yyy format.')
    return parser.parse_args()


def cumulative_xyz(actions):
    return np.cumsum(actions[:, :3], axis=0)


def cumulative_cont(actions):
    return np.cumsum(actions[:, :6], axis=0)


def plot_chunk(raw_actions, compressed_actions, out_png, title):
    raw_xyz = cumulative_xyz(raw_actions)
    comp_xyz = cumulative_xyz(compressed_actions)
    raw_cont = cumulative_cont(raw_actions)
    comp_cont = cumulative_cont(compressed_actions)

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 4)

    ax3d = fig.add_subplot(gs[:, 0], projection='3d')
    ax3d.plot(raw_xyz[:, 0], raw_xyz[:, 1], raw_xyz[:, 2],
              'o-', label=f'raw 10 ({len(raw_xyz)} pts)', linewidth=1.8)
    ax3d.plot(comp_xyz[:, 0], comp_xyz[:, 1], comp_xyz[:, 2],
              's-', label=f'DCT ({len(comp_xyz)} pts)', linewidth=1.8)
    ax3d.scatter(raw_xyz[0, 0], raw_xyz[0, 1], raw_xyz[0, 2],
                 marker='^', s=80, c='tab:blue')
    ax3d.scatter(raw_xyz[-1, 0], raw_xyz[-1, 1], raw_xyz[-1, 2],
                 marker='x', s=80, c='tab:blue')
    ax3d.scatter(comp_xyz[0, 0], comp_xyz[0, 1], comp_xyz[0, 2],
                 marker='^', s=80, c='tab:orange')
    ax3d.scatter(comp_xyz[-1, 0], comp_xyz[-1, 1], comp_xyz[-1, 2],
                 marker='x', s=80, c='tab:orange')
    ax3d.set_xlabel('cum dx')
    ax3d.set_ylabel('cum dy')
    ax3d.set_zlabel('cum dz')
    ax3d.set_title('XYZ cumulative path')
    ax3d.legend(loc='best')

    labels = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
    for dim in range(6):
        ax = fig.add_subplot(gs[dim // 3, dim % 3 + 1])
        ax.plot(np.arange(len(raw_cont)), raw_cont[:, dim],
                'o-', label='raw 10', linewidth=1.5)
        ax.plot(
            np.linspace(0, len(raw_cont) - 1, len(comp_cont)),
            comp_cont[:, dim],
            's-',
            label='DCT',
            linewidth=1.5,
        )
        ax.set_title(f'cumulative {labels[dim]}')
        ax.grid(True, alpha=0.25)
        if dim == 0:
            ax.legend(loc='best')

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib_fluxvla')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.eval.cfg = cfg
    cfg.eval.ckpt_path = args.ckpt_path
    cfg.eval.seed = args.seed
    cfg.eval.eval_task_ids = [args.task_id]
    cfg.eval.num_trials_per_task = max(args.episodes, 1)
    if hasattr(cfg.eval,
               'processor') and not hasattr(cfg.eval.processor, 'model_path'):
        cfg.eval.processor.model_path = args.ckpt_path

    runner = build_runner_from_cfg(cfg.eval)
    runner.run_setup()

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[runner.task_suite_name]()
    task = task_suite.get_task(args.task_id)
    initial_states = task_suite.get_task_init_states(args.task_id)
    unnorm_key = runner.task_suite_name
    if (unnorm_key not in runner.vla.norm_stats
            and f'{unnorm_key}_no_noops' in runner.vla.norm_stats):
        unnorm_key = f'{unnorm_key}_no_noops'

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    for ep in range(args.episodes):
        env, task_description = get_libero_env(task, resolution=256)
        env.reset()
        obs = env.set_init_state(initial_states[ep])
        is_new_episode = True
        next_batch = None
        t = 0
        chunks = 0
        done = False

        while t < runner.num_steps_wait:
            obs, _, _, _ = env.step(get_libero_dummy_action())
            t += 1

        while chunks < args.chunks_per_episode and t < args.max_steps and not done:
            if next_batch is None:
                obs['task_description'] = task_description
                obs['is_new_episode'] = is_new_episode
                batch, _ = runner.dataset(obs)
            else:
                batch = next_batch
                next_batch = None
            is_new_episode = False
            batch['unnorm_key'] = unnorm_key

            with torch.autocast(
                    'cuda',
                    dtype=runner.mixed_precision_dtype,
                    enabled=runner.enable_mixed_precision_training):
                with torch.no_grad():
                    actions = runner.vla.predict_action(**batch)

            if len(actions.shape) == 3:
                raw_norm = actions[0, :runner.eval_chunk_size, :]
            else:
                raw_norm = actions[0, None, :]
            raw_np = raw_norm.float().cpu().numpy()

            raw_denorm = []
            for action in raw_np:
                raw_denorm.append(
                    runner.denormalize_action(
                        dict(
                            action=action,
                            task_suite_name=runner.task_suite_name,
                            norm_stats_key=runner.norm_stats_key,
                        )))
            raw_denorm = np.asarray(raw_denorm, dtype=np.float32)
            compressed = runner._postprocess_action_chunk(raw_denorm)

            stem = f'task{args.task_id}_seed{args.seed}_ep{ep:02d}_chunk{chunks:02d}'
            plot_chunk(
                raw_denorm,
                compressed,
                out_dir / f'{stem}.png',
                f'{stem}: raw 10 vs DCT-compressed {len(compressed)}',
            )
            np.savez_compressed(
                out_dir / f'{stem}.npz',
                raw_actions=raw_denorm,
                compressed_actions=compressed,
            )
            metadata.append(
                dict(
                    episode=ep,
                    chunk=chunks,
                    raw_steps=int(len(raw_denorm)),
                    compressed_steps=int(len(compressed)),
                    png=str(out_dir / f'{stem}.png'),
                    npz=str(out_dir / f'{stem}.npz'),
                ))

            exec_actions = compressed if args.execute == 'compressed' else raw_denorm
            for action_denormed in exec_actions:
                obs, _, done, _ = env.step(action_denormed.tolist())
                obs['task_description'] = task_description
                batch, _ = runner.dataset(obs)
                next_batch = batch
                t += 1
                if done or t >= args.max_steps:
                    break
            chunks += 1

        env.close()

    summary_path = out_dir / 'summary.txt'
    with summary_path.open('w') as f:
        f.write(f'config: {args.config}\n')
        f.write(f'ckpt: {args.ckpt_path}\n')
        f.write(f'action_postprocess: {runner.action_postprocess}\n')
        f.write(f'eval_chunk_size: {runner.eval_chunk_size}\n')
        for item in metadata:
            f.write(
                f"episode={item['episode']} chunk={item['chunk']} "
                f"raw={item['raw_steps']} compressed={item['compressed_steps']} "
                f"png={item['png']}\n")
    print(f'Wrote {len(metadata)} plots to {out_dir}')
    print(f'Summary: {summary_path}')


if __name__ == '__main__':
    main()
