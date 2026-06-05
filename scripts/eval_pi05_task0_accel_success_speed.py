#!/usr/bin/env python3
"""Compare PI0.5 task0 with_l/no_l using accelerated inference only."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from mmengine import Config


DEFAULT_ACCEL_CONFIG = 'configs/pi05/pi05_paligemma_libero_10_full_inference.py'
DEFAULT_WITH_L_CKPT = (
    'work_dirs/pi05_libero10_task0_with_l/checkpoints/'
    'latest-checkpoint.safetensors')
DEFAULT_NO_L_CKPT = (
    'work_dirs/pi05_libero10_task0_no_l/checkpoints/'
    'latest-checkpoint.safetensors')
DEFAULT_BASE_WEIGHTS = './checkpoints/pi05_base/model.safetensors'

SUCCESS_RE = re.compile(r'# successes:\s*([0-9.]+)\s*\(([0-9.]+)%\)')
EPISODES_RE = re.compile(r'# episodes completed so far:\s*([0-9.]+)')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Evaluate PI0.5 task0 with_l and no_l with accelerated inference, '
            'recording LIBERO success rate and predict_action frequency.'))
    parser.add_argument('--with-l-ckpt', default=DEFAULT_WITH_L_CKPT)
    parser.add_argument('--no-l-ckpt', default=DEFAULT_NO_L_CKPT)
    parser.add_argument('--accelerated-config', default=DEFAULT_ACCEL_CONFIG)
    parser.add_argument('--base-weights', default=DEFAULT_BASE_WEIGHTS)
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Defaults to work_dirs/pi05_task0_accel_compare/<timestamp>.')
    parser.add_argument('--tag', default=None)
    parser.add_argument('--task-id', type=int, default=4)
    parser.add_argument('--success-trials-per-task', type=int, default=50)
    parser.add_argument('--success-seeds', default='7')
    parser.add_argument('--success-gpus', default='3,4,5,6,7')
    parser.add_argument('--success-nproc-per-node', type=int, default=None)
    parser.add_argument('--speed-single-gpus', default='3')
    parser.add_argument('--speed-multi-gpus', default='3,4,5,6,7')
    parser.add_argument('--speed-warmup-iters', type=int, default=10)
    parser.add_argument('--speed-bench-iters', type=int, default=100)
    parser.add_argument('--master-addr', default='127.0.0.1')
    parser.add_argument('--master-port', type=int, default=29640)
    parser.add_argument('--skip-success', action='store_true')
    parser.add_argument('--skip-speed', action='store_true')
    parser.add_argument('--skip-speed-multi', action='store_true')
    return parser.parse_args()


def split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(',') if item.strip()]


def repo_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def ckpt_root(ckpt_path: str) -> Path:
    return repo_path(ckpt_path).parent.parent


def run_logged(cmd: List[str], env: Dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print('[Run] ' + ' '.join(cmd), flush=True)
    print(f'[Run] CUDA_VISIBLE_DEVICES={env.get("CUDA_VISIBLE_DEVICES", "")}',
          flush=True)
    with log_path.open('w') as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=Path.cwd(),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end='', flush=True)
            log_file.write(line)
        return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(
            f'Command failed with exit code {return_code}: {" ".join(cmd)}')


def prompt_transform(use_language: bool) -> Dict:
    if not use_language:
        return dict(type='NoLanguagePrompt')
    return dict(
        type='LiberoPromptFromInputs',
        use_conversation=False,
        tokenizer=dict(type='PaligemmaTokenizer'))


def set_eval_prompt_transform(cfg: Config, use_language: bool) -> None:
    transforms = cfg.eval.dataset.transforms
    replaced = False
    for idx, transform in enumerate(transforms):
        if transform.get('type') in ('LiberoPromptFromInputs',
                                     'NoLanguagePrompt'):
            transforms[idx] = prompt_transform(use_language)
            replaced = True
    if not replaced:
        transforms.append(prompt_transform(use_language))


def write_accel_config(args: argparse.Namespace, output_dir: Path,
                       variant: str, use_language: bool) -> Path:
    cfg = Config.fromfile(args.accelerated_config)
    cfg.inference_model.use_language = use_language
    cfg.inference_model.pretrained_name_or_path = args.base_weights
    if hasattr(cfg, 'model'):
        cfg.model.use_language = use_language
        cfg.model.pretrained_name_or_path = args.base_weights
    cfg.eval.eval_task_ids = [args.task_id]
    cfg.eval.num_trials_per_task = args.success_trials_per_task
    set_eval_prompt_transform(cfg, use_language)

    config_dir = output_dir / 'configs'
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f'pi05_task0_{variant}_accelerated.py'
    cfg.dump(str(path))
    return path


def parse_success_text(text: str, source: Path) -> Dict[str, float]:
    episodes: Optional[float] = None
    successes: Optional[float] = None
    success_rate: Optional[float] = None
    for line in text.splitlines():
        ep_match = EPISODES_RE.search(line)
        if ep_match:
            episodes = float(ep_match.group(1))
        success_match = SUCCESS_RE.search(line)
        if success_match:
            successes = float(success_match.group(1))
            success_rate = float(success_match.group(2))
    if episodes is None or successes is None or success_rate is None:
        raise ValueError(f'Could not parse success stats from {source}')
    return dict(
        episodes=episodes,
        successes=successes,
        success_rate_pct=success_rate,
    )


def parse_success_file(path: Path) -> Dict[str, float]:
    return parse_success_text(path.read_text(errors='replace'), path)


def parse_best_success(paths: Iterable[Path],
                       log_path: Path) -> Tuple[Path, Dict[str, float]]:
    parsed = []
    for path in paths:
        try:
            parsed.append((path, parse_success_file(path)))
        except ValueError:
            continue
    if log_path.exists():
        try:
            parsed.append((log_path, parse_success_file(log_path)))
        except ValueError:
            pass
    if not parsed:
        raise ValueError(f'No success stats found for {log_path}')
    return max(parsed, key=lambda item: item[1]['episodes'])


def run_speed(args: argparse.Namespace, output_dir: Path, logs_dir: Path,
              variant: str, ckpt_path: str, config_path: Path,
              use_language: bool, scenario: str, gpus: str) -> Dict:
    speed_dir = output_dir / 'speed'
    speed_dir.mkdir(parents=True, exist_ok=True)
    tag = f'{args.tag}_{variant}_speed_{scenario}'
    cmd = [
        sys.executable,
        'scripts/benchmark_pi05_inference_speed.py',
        '--ckpt-path',
        ckpt_path,
        '--accelerated-config',
        str(config_path),
        '--mode',
        'accelerated',
        '--warmup-iters',
        str(args.speed_warmup_iters),
        '--bench-iters',
        str(args.speed_bench_iters),
        '--prompt-len',
        '32' if use_language else '0',
        '--tag',
        tag,
        '--output-dir',
        str(speed_dir),
    ]
    if scenario == 'multi':
        cmd += ['--num-workers', str(len(split_csv(gpus)))]

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpus
    env['WANDB_MODE'] = 'disabled'
    env['PYTHONUNBUFFERED'] = '1'
    run_logged(cmd, env, logs_dir / f'{tag}.log')

    result_path = speed_dir / f'pi05_inference_speed_{tag}.json'
    with result_path.open('r') as f:
        data = json.load(f)
    result = data['results'][0]
    result.update(
        requested_variant=variant,
        scenario=scenario,
        result_path=str(result_path),
        use_language=use_language,
    )
    return result


def run_success(args: argparse.Namespace, output_dir: Path, logs_dir: Path,
                variant: str, ckpt_path: str, config_path: Path,
                use_language: bool, seed: int, command_index: int) -> Dict:
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = args.success_gpus
    env['WANDB_MODE'] = 'disabled'
    env['MUJOCO_GL'] = 'egl'
    env['PYTHONUNBUFFERED'] = '1'
    nproc = args.success_nproc_per_node or len(split_csv(args.success_gpus))
    port = args.master_port + command_index
    work_root = ckpt_root(ckpt_path)
    before = {p.resolve() for p in work_root.glob('EVAL-*.txt')}
    tag = f'{args.tag}_{variant}_success_seed{seed}'
    log_path = logs_dir / f'{tag}.log'
    cmd = [
        sys.executable,
        '-m',
        'torch.distributed.run',
        '--nnodes',
        '1',
        '--nproc-per-node',
        str(nproc),
        '--master-addr',
        args.master_addr,
        '--master-port',
        str(port),
        'scripts/eval.py',
        '--config',
        str(config_path),
        '--ckpt-path',
        ckpt_path,
        '--cfg-options',
        f'eval.num_trials_per_task={args.success_trials_per_task}',
        f'eval.seed={seed}',
        f'eval.eval_task_ids=[{args.task_id}]',
    ]
    run_logged(cmd, env, log_path)
    after = list(work_root.glob('EVAL-*.txt'))
    candidates = [p for p in after if p.resolve() not in before]
    if not candidates:
        candidates = after
    eval_file, parsed = parse_best_success(candidates, log_path)
    parsed.update(
        requested_variant=variant,
        seed=seed,
        ckpt_path=ckpt_path,
        config_path=str(config_path),
        use_language=use_language,
        cuda_visible_devices=args.success_gpus,
        nproc_per_node=nproc,
        eval_file=str(eval_file),
        log_path=str(log_path),
    )
    return parsed


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, speed_rows: List[Dict],
                   success_rows: List[Dict], args: argparse.Namespace) -> None:
    lines = [
        '# PI0.5 Task0 Accelerated With-L vs No-L',
        '',
        'Both variants use `PI05FlowMatchingInference` accelerated inference.',
        '`no_l` is evaluated with `use_language=False` and `NoLanguagePrompt`.',
        '',
        '## Checkpoints',
        '',
        f'- with_l: `{args.with_l_ckpt}`',
        f'- no_l: `{args.no_l_ckpt}`',
        '',
        '## Speed',
        '',
        '| Variant | Scenario | GPUs | Prompt len | Mean ms | P90 ms | Chunk Hz | Action-step Hz | Peak GiB |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in speed_rows:
        lines.append(
            '| {variant} | {scenario} | `{gpus}` | {prompt_len} | '
            '{mean_ms:.3f} | {p90_ms:.3f} | {chunk_hz:.3f} | '
            '{action_step_hz:.3f} | {peak:.3f} |'.format(
                variant=row['requested_variant'],
                scenario=row['scenario'],
                gpus=row.get('cuda_visible_devices', ''),
                prompt_len=row.get('prompt_len', ''),
                mean_ms=float(row['mean_ms']),
                p90_ms=float(row['p90_ms']),
                chunk_hz=float(row['chunk_hz']),
                action_step_hz=float(row['action_step_hz']),
                peak=float(row['peak_allocated_gib_during_benchmark']),
            ))

    lines += [
        '',
        '## Success',
        '',
        '| Variant | Seed | GPUs | Episodes | Successes | Success Rate | Eval File |',
        '| --- | ---: | --- | ---: | ---: | ---: | --- |',
    ]
    for row in success_rows:
        lines.append(
            '| {variant} | {seed} | `{gpus}` | {episodes:.0f} | '
            '{successes:.0f} | {rate:.3f}% | `{eval_file}` |'.format(
                variant=row['requested_variant'],
                seed=row['seed'],
                gpus=row['cuda_visible_devices'],
                episodes=float(row['episodes']),
                successes=float(row['successes']),
                rate=float(row['success_rate_pct']),
                eval_file=row['eval_file'],
            ))

    lines += [
        '',
        '## Settings',
        '',
        f'- Task id: `{args.task_id}`',
        f'- Success trials per task: `{args.success_trials_per_task}`',
        f'- Success seeds: `{args.success_seeds}`',
        f'- Speed warmup/bench iters: `{args.speed_warmup_iters}` / `{args.speed_bench_iters}`',
        f'- Base weights used for construction: `{args.base_weights}`',
    ]
    path.write_text('\n'.join(lines) + '\n')


def validate_inputs(args: argparse.Namespace) -> None:
    for label, path in [
            ('with_l checkpoint', args.with_l_ckpt),
            ('no_l checkpoint', args.no_l_ckpt),
            ('accelerated config', args.accelerated_config),
            ('base weights', args.base_weights),
    ]:
        if not Path(path).exists():
            raise FileNotFoundError(f'{label} not found: {path}')


def main() -> None:
    args = parse_args()
    args.tag = args.tag or time.strftime('%Y%m%d-%H%M%S')
    validate_inputs(args)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir else
        Path('work_dirs/pi05_task0_accel_compare') / args.tag)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)

    variants = [
        ('with_l', args.with_l_ckpt, True),
        ('no_l', args.no_l_ckpt, False),
    ]
    generated_configs = {
        variant: write_accel_config(args, output_dir, variant, use_language)
        for variant, _, use_language in variants
    }

    speed_rows: List[Dict] = []
    if not args.skip_speed:
        for variant, ckpt_path, use_language in variants:
            speed_rows.append(
                run_speed(args, output_dir, logs_dir, variant, ckpt_path,
                          generated_configs[variant], use_language, 'single',
                          args.speed_single_gpus))
            if not args.skip_speed_multi:
                speed_rows.append(
                    run_speed(args, output_dir, logs_dir, variant, ckpt_path,
                              generated_configs[variant], use_language,
                              'multi', args.speed_multi_gpus))

    success_rows: List[Dict] = []
    if not args.skip_success:
        command_index = 0
        for seed_str in split_csv(args.success_seeds):
            seed = int(seed_str)
            for variant, ckpt_path, use_language in variants:
                success_rows.append(
                    run_success(args, output_dir, logs_dir, variant, ckpt_path,
                                generated_configs[variant], use_language, seed,
                                command_index))
                command_index += 1

    payload = dict(
        tag=args.tag,
        output_dir=str(output_dir),
        generated_configs={
            key: str(value)
            for key, value in generated_configs.items()
        },
        speed=speed_rows,
        success=success_rows,
        args=vars(args),
    )
    (output_dir / 'summary.json').write_text(
        json.dumps(payload, indent=2) + '\n')
    write_csv(output_dir / 'speed.csv', speed_rows)
    write_csv(output_dir / 'success.csv', success_rows)
    write_markdown(output_dir / 'summary.md', speed_rows, success_rows, args)
    print(f'[Done] Wrote {output_dir / "summary.json"}', flush=True)
    print(f'[Done] Wrote {output_dir / "summary.md"}', flush=True)


if __name__ == '__main__':
    main()
