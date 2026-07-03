#!/usr/bin/env python3
"""Rebuild speed-time-schedule summaries from per-case logs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from eval_pi05_task0_speed_time_schedule import (
    parse_success_text,
    schedule_and_deltas,
    write_csv,
    write_markdown,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir')
    parser.add_argument('--config', default='configs/pi05/pi05_libero10_task0_tempovla_speed_modulated_inference.py')
    parser.add_argument('--ckpt', default='work_dirs/libero10_full_tempovla_speed_modulated/checkpoints/latest-checkpoint.safetensors')
    parser.add_argument('--task-id', type=int, default=4)
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--seeds', default='7,8')
    parser.add_argument('--gpus', default='')
    return parser.parse_args()


def schedule_for_case(case: str):
    if case.endswith('_t5_108642'):
        return [1.0, 0.8, 0.6, 0.4, 0.2]
    if case.endswith('_t4_10741'):
        return [1.0, 0.7, 0.4, 0.1]
    if case.endswith('_t3_1062'):
        return [1.0, 0.6, 0.2]
    return None


def speed_for_case(case: str) -> float:
    match = re.match(r'speed_([0-9_]+)_t[0-9]_', case)
    if not match:
        raise ValueError(f'cannot parse speed from {case}')
    return float(match.group(1).replace('_', '.'))


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    rows = []
    for log_path in sorted((output_dir / 'logs').glob('*.log')):
        match = re.match(r'(.+)_seed([0-9]+)\.log$', log_path.name)
        if not match:
            continue
        case = match.group(1)
        seed = int(match.group(2))
        parsed = parse_success_text(log_path.read_text(errors='ignore'), log_path)
        schedule = schedule_for_case(case)
        active_schedule, deltas = schedule_and_deltas(schedule)
        parsed.update(
            case=case,
            tempo_speed=speed_for_case(case),
            time_schedule=active_schedule,
            time_deltas=deltas,
            config_path='',
            ckpt_path=args.ckpt,
            seed=seed,
            task_id=args.task_id,
            trials=args.trials,
            gpus=args.gpus,
            nproc_per_node=1,
            eval_file=str(log_path),
            log_path=str(log_path),
            copied_rollout_count=0,
            copied_rollout_dir='',
        )
        rows.append(parsed)
    if not rows:
        raise RuntimeError(f'no logs found under {output_dir / "logs"}')
    rows.sort(key=lambda row: (row['tempo_speed'], row['case'], row['seed']))
    write_csv(output_dir / 'summary.csv', rows)
    (output_dir / 'summary.json').write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + '\n')
    write_markdown(output_dir / 'summary.md', rows, args)
    print(output_dir / 'summary.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
