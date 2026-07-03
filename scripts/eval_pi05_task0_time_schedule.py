#!/usr/bin/env python3
"""Evaluate PI0.5 task0 baseline vs custom denoising schedules.

The dataset task_index=0 config maps to LIBERO benchmark task id 4. This script
therefore defaults to ``--task-id 4`` and runs 50 LIBERO initial states per
variant unless overridden.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from mmengine import Config


DEFAULT_CONFIG = (
    'configs/pi05/pi05_libero10_task0_with_l_pixshuffle_mlp_inference.py')
DEFAULT_CKPT = (
    'work_dirs/pi05_libero10_task0_with_l_pixshuffle_mlp/checkpoints/'
    'step-016560-epoch-240-loss=0.0315.safetensors')
DEFAULT_BASE_WEIGHTS = './checkpoints/pi05_base/model.safetensors'

SUCCESS_RE = re.compile(r'# successes:\s*([0-9.]+)\s*\(([0-9.]+)%\)')
EPISODES_RE = re.compile(r'# episodes completed so far:\s*([0-9.]+)')
PREDICT_CALLS_RE = re.compile(r'# predict_action calls:\s*([0-9.]+)')
PREDICT_MEAN_MS_RE = re.compile(r'# predict_action mean ms:\s*([0-9.]+)')
PREDICT_CALL_HZ_RE = re.compile(r'# predict_action call Hz:\s*([0-9.]+)')
PREDICT_ACTION_STEP_HZ_RE = re.compile(
    r'# predict_action action-step Hz:\s*([0-9.]+)')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Run LIBERO task0/task-id-4 success and rollout-frequency '
            'comparison for PI0.5 time schedules.'))
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--ckpt', default=DEFAULT_CKPT)
    parser.add_argument('--base-weights', default=DEFAULT_BASE_WEIGHTS)
    parser.add_argument(
        '--scenario',
        action='append',
        default=None,
        help=(
            'Scenario to evaluate. Built-ins: baseline10, uniform5. Custom: '
            'name:t0,t1,... or just t0,t1,... . Repeat for multiple.'))
    parser.add_argument('--task-id', type=int, default=4)
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--gpus', default='1')
    parser.add_argument('--nproc-per-node', type=int, default=1)
    parser.add_argument('--max-rollout-videos', type=int, default=10)
    parser.add_argument('--master-addr', default='127.0.0.1')
    parser.add_argument('--master-port', type=int, default=29710)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--continue-on-error', action='store_true')
    return parser.parse_args()


def split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(',') if item.strip()]


def parse_float_list(value: str) -> List[float]:
    items = [item.strip() for item in value.split(',') if item.strip()]
    if not items:
        raise ValueError('empty time schedule')
    return [float(item) for item in items]


def parse_scenarios(values: Optional[List[str]]) -> List[Tuple[str, Optional[List[float]]]]:
    if not values:
        return [
            ('baseline10', None),
            ('uniform5', [1.0, 0.8, 0.6, 0.4, 0.2]),
        ]
    scenarios = []
    for value in values:
        if value == 'baseline10':
            scenarios.append(('baseline10', None))
        elif value == 'uniform5':
            scenarios.append(('uniform5', [1.0, 0.8, 0.6, 0.4, 0.2]))
        elif ':' in value:
            name, raw = value.split(':', 1)
            scenarios.append((name.strip() or 'custom', parse_float_list(raw)))
        else:
            scenarios.append(
                ('custom_' + value.replace(',', '_').replace('.', '_'),
                 parse_float_list(value)))
    return scenarios


def repo_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def ckpt_root(ckpt_path: str) -> Path:
    return repo_path(ckpt_path).parent.parent


def rollout_paths(work_root: Path) -> set[Path]:
    rollout_root = work_root / 'rollouts'
    if not rollout_root.exists():
        return set()
    return {path.resolve() for path in rollout_root.rglob('*.mp4')}


def base_env(gpus: str, output_dir: Path) -> Dict[str, str]:
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpus
    env['MUJOCO_GL'] = 'osmesa'
    env.pop('MUJOCO_EGL_DEVICE_ID', None)
    env['TOKENIZERS_PARALLELISM'] = 'false'
    env['WANDB_MODE'] = 'disabled'
    env['PYTHONUNBUFFERED'] = '1'
    env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    env.setdefault('NUMBA_CACHE_DIR', '/tmp/numba_cache_fluxvla')
    env.setdefault('MPLCONFIGDIR', '/tmp/matplotlib_fluxvla')
    return env


def materialize_config(base_config: str, output_dir: Path, name: str,
                       schedule: Optional[List[float]]) -> Path:
    cfg = Config.fromfile(base_config)
    cfg.inference_model.type = 'PI05FlowMatchingTimeScheduleInference'
    if schedule is None:
        cfg.inference_model.num_steps = 10
        cfg.inference_model.pop('time_schedule', None)
    else:
        cfg.inference_model.num_steps = len(schedule)
        cfg.inference_model.time_schedule = schedule
    cfg.custom_imports = dict(
        imports=[
            'fluxvla.models.vlas.pi05_flowmatching_time_schedule_inference'
        ],
        allow_failed_imports=False,
    )
    config_dir = output_dir / 'configs'
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f'{name}.py'
    cfg.dump(str(path))
    return path


def parse_success_text(text: str, source: Path) -> Dict[str, float]:
    episodes = successes = success_rate = None
    predict_calls = predict_mean_ms = predict_call_hz = None
    predict_action_step_hz = None
    for line in text.splitlines():
        ep_match = EPISODES_RE.search(line)
        if ep_match:
            episodes = float(ep_match.group(1))
        success_match = SUCCESS_RE.search(line)
        if success_match:
            successes = float(success_match.group(1))
            success_rate = float(success_match.group(2))
        calls_match = PREDICT_CALLS_RE.search(line)
        if calls_match:
            predict_calls = float(calls_match.group(1))
        mean_match = PREDICT_MEAN_MS_RE.search(line)
        if mean_match:
            predict_mean_ms = float(mean_match.group(1))
        call_hz_match = PREDICT_CALL_HZ_RE.search(line)
        if call_hz_match:
            predict_call_hz = float(call_hz_match.group(1))
        action_hz_match = PREDICT_ACTION_STEP_HZ_RE.search(line)
        if action_hz_match:
            predict_action_step_hz = float(action_hz_match.group(1))
    if episodes is None or successes is None or success_rate is None:
        raise ValueError(f'Could not parse success stats from {source}')
    result = dict(
        episodes=episodes,
        successes=successes,
        success_rate_pct=success_rate,
    )
    if predict_calls is not None:
        result.update(
            task_predict_calls=predict_calls,
            task_predict_mean_ms=predict_mean_ms,
            task_predict_call_hz=predict_call_hz,
            task_predict_action_step_hz=predict_action_step_hz,
        )
    return result


def parse_best_success(candidates: Iterable[Path], fallback_log: Path):
    ordered = sorted(candidates, key=lambda path: path.stat().st_mtime)
    errors = []
    for path in reversed(ordered):
        try:
            return path, parse_success_text(path.read_text(errors='ignore'), path)
        except Exception as exc:
            errors.append(f'{path}: {exc}')
    return fallback_log, parse_success_text(
        fallback_log.read_text(errors='ignore'), fallback_log)


def copy_rollout_subset(paths: Iterable[Path], target_dir: Path,
                        limit: int) -> List[str]:
    if limit <= 0:
        return []
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for src in sorted(paths, key=lambda path: path.stat().st_mtime)[:limit]:
        dst = target_dir / src.name
        if dst.exists():
            dst = target_dir / f'{src.stem}-{len(copied)}{src.suffix}'
        shutil.copy2(src, dst)
        copied.append(str(dst))
    return copied


def run_logged(cmd: List[str], env: Dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(' '.join(cmd), flush=True)
    with log_path.open('w') as log_file:
        proc = subprocess.run(
            cmd,
            cwd=Path(__file__).resolve().parent.parent,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f'Command failed with exit code {proc.returncode}. See {log_path}')


def schedule_and_deltas(schedule: Optional[List[float]]) -> Tuple[List[float], List[float]]:
    active_schedule = (
        schedule if schedule is not None else
        [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    deltas = [
        float(next_time - current_time)
        for current_time, next_time in zip(active_schedule,
                                           active_schedule[1:] + [0.0])
    ]
    return list(active_schedule), deltas


def run_scenario(args: argparse.Namespace, output_dir: Path, name: str,
                 schedule: Optional[List[float]], config_path: Path,
                 command_index: int) -> Dict:
    work_root = ckpt_root(args.ckpt)
    before_eval = {path.resolve() for path in work_root.glob('EVAL-*.txt')}
    before_rollouts = rollout_paths(work_root)
    log_path = output_dir / 'logs' / f'{name}.log'
    cmd = [
        sys.executable,
        '-m',
        'torch.distributed.run',
        '--nnodes',
        '1',
        '--nproc-per-node',
        str(args.nproc_per_node),
        '--master-addr',
        args.master_addr,
        '--master-port',
        str(args.master_port + command_index),
        'scripts/eval.py',
        '--config',
        str(config_path),
        '--ckpt-path',
        args.ckpt,
        '--cfg-options',
        f'eval.num_trials_per_task={args.trials}',
        f'eval.seed={args.seed}',
        f'eval.eval_task_ids=[{args.task_id}]',
        'eval.measure_predict_latency=True',
        'inference_model.use_language=True',
        f'inference_model.pretrained_name_or_path={args.base_weights}',
    ]
    run_logged(cmd, base_env(args.gpus, output_dir), log_path)

    after_eval = list(work_root.glob('EVAL-*.txt'))
    candidates = [path for path in after_eval if path.resolve() not in before_eval]
    if not candidates:
        candidates = after_eval
    eval_file, parsed = parse_best_success(candidates, log_path)
    new_rollouts = rollout_paths(work_root) - before_rollouts
    copied = copy_rollout_subset(
        new_rollouts,
        output_dir / 'rollouts_10' / name,
        args.max_rollout_videos,
    )
    parsed.update(
        scenario=name,
        time_schedule=schedule_and_deltas(schedule)[0],
        time_deltas=schedule_and_deltas(schedule)[1],
        config_path=str(config_path),
        ckpt_path=args.ckpt,
        seed=args.seed,
        task_id=args.task_id,
        trials=args.trials,
        gpus=args.gpus,
        nproc_per_node=args.nproc_per_node,
        eval_file=str(eval_file),
        log_path=str(log_path),
        copied_rollout_count=len(copied),
        copied_rollout_dir=str(output_dir / 'rollouts_10' / name),
    )
    return parsed


def write_csv(path: Path, rows: List[Dict]) -> None:
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return 'n/a'
    if isinstance(value, float):
        return f'{value:.{digits}f}'
    return str(value)


def write_markdown(path: Path, rows: List[Dict], args: argparse.Namespace) -> None:
    lines = [
        '# PI0.5 Task0 Time Schedule Eval',
        '',
        f'- LIBERO task id: `{args.task_id}` (repo task0 config maps to id 4)',
        f'- Trials/statistics per scenario: `{args.trials}`',
        f'- Seed: `{args.seed}`',
        f'- Rollout videos copied per scenario: `{args.max_rollout_videos}`',
        '',
        '| Scenario | t schedule | dt | Episodes | Successes | Success Rate | Mean ms | Call Hz | Action-step Hz | Rollout Videos | Eval File |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |',
    ]
    for row in rows:
        schedule_s = ','.join(f'{value:g}' for value in row['time_schedule'])
        deltas_s = ','.join(f'{value:g}' for value in row['time_deltas'])
        lines.append(
            '| {scenario} | `{schedule}` | `{deltas}` | {episodes:.0f} | '
            '{successes:.0f} | {rate:.1f}% | {mean_ms} | {call_hz} | '
            '{action_hz} | {videos} | `{eval_file}` |'
            .format(
                scenario=row['scenario'],
                schedule=schedule_s,
                deltas=deltas_s,
                episodes=row['episodes'],
                successes=row['successes'],
                rate=row['success_rate_pct'],
                mean_ms=fmt(row.get('task_predict_mean_ms')),
                call_hz=fmt(row.get('task_predict_call_hz')),
                action_hz=fmt(row.get('task_predict_action_step_hz')),
                videos=row.get('copied_rollout_count', 0),
                eval_file=row.get('eval_file', ''),
            ))
    path.write_text('\n'.join(lines) + '\n')


def main() -> int:
    args = parse_args()
    if args.task_id != 4:
        print(
            f'WARNING: task0 configs in this repo map to LIBERO task id 4; '
            f'you requested task id {args.task_id}.',
            flush=True,
        )
    if args.trials > 50:
        raise ValueError('LIBERO task id 4 has 50 initial states; use <= 50.')
    if args.nproc_per_node != len(split_csv(args.gpus)):
        print(
            'WARNING: --nproc-per-node differs from number of --gpus entries; '
            'make sure this is intentional.',
            flush=True,
        )

    tag = args.tag or time.strftime('time_schedule_%Y%m%d_%H%M%S')
    output_dir = (
        Path(args.output_dir) if args.output_dir else
        Path('work_dirs/pi05_task0_time_schedule_eval') / tag)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = parse_scenarios(args.scenario)
    config_paths = [
        (name, schedule,
         materialize_config(args.config, output_dir, name, schedule))
        for name, schedule in scenarios
    ]

    rows = []
    errors = []
    for idx, (name, schedule, config_path) in enumerate(config_paths):
        try:
            rows.append(
                run_scenario(args, output_dir, name, schedule, config_path,
                             idx))
        except Exception as exc:
            errors.append({'scenario': name, 'error': repr(exc)})
            print(f'[error] {name}: {exc}', flush=True)
            if not args.continue_on_error:
                break

    if rows:
        write_csv(output_dir / 'summary.csv', rows)
        (output_dir / 'summary.json').write_text(
            json.dumps(rows, indent=2, ensure_ascii=False) + '\n')
        write_markdown(output_dir / 'summary.md', rows, args)
        print((output_dir / 'summary.md').read_text(), flush=True)
    if errors:
        (output_dir / 'errors.json').write_text(
            json.dumps(errors, indent=2, ensure_ascii=False) + '\n')
    return 0 if rows and not errors else 1


if __name__ == '__main__':
    raise SystemExit(main())
