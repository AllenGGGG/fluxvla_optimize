#!/usr/bin/env python3
"""Compare PI0.5 task0 with_l vs pixshuffle using accelerated inference."""

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

DEFAULT_WITH_L_CONFIG = 'configs/pi05/pi05_paligemma_libero_10_full_inference.py'
DEFAULT_PIXSHUFFLE_CONFIG = (
    'configs/pi05/pi05_libero10_task0_with_l_pixshuffle_inference.py')
DEFAULT_WITH_L_CKPT = (
    'work_dirs/pi05_libero10_task0_with_l/checkpoints/'
    'step-008460-epoch-60-loss=0.0910.safetensors')
DEFAULT_PIXSHUFFLE_CKPT = (
    'work_dirs/pi05_libero10_task0_with_l_pixshuffle/checkpoints/'
    'step-008460-epoch-60-loss=0.0918.safetensors')

SUCCESS_RE = re.compile(r'# successes:\s*([0-9.]+)\s*\(([0-9.]+)%\)')
EPISODES_RE = re.compile(r'# episodes completed so far:\s*([0-9.]+)')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Compare trained PI0.5 task0 with_l and pixshuffle checkpoints '
            'with accelerated inference speed and LIBERO success rate.'))
    parser.add_argument('--with-l-config', default=DEFAULT_WITH_L_CONFIG)
    parser.add_argument('--pixshuffle-config', default=DEFAULT_PIXSHUFFLE_CONFIG)
    parser.add_argument('--with-l-ckpt', default=DEFAULT_WITH_L_CKPT)
    parser.add_argument('--pixshuffle-ckpt', default=DEFAULT_PIXSHUFFLE_CKPT)
    parser.add_argument(
        '--variants',
        default='with_l,pixshuffle',
        help='Comma-separated built-in variants to run: with_l,pixshuffle. Ignored when --variant is used.')
    parser.add_argument(
        '--variant',
        nargs=3,
        action='append',
        metavar=('NAME', 'CONFIG', 'CKPT'),
        default=[],
        help='Custom variant triple. Can be repeated: --variant NAME CONFIG CKPT.')
    parser.add_argument(
        '--base-weights',
        default='./checkpoints/pi05_base/model.safetensors',
        help='Base weights used when constructing inference_model before ckpt load.')
    parser.add_argument(
        '--triton-max-prompt-len',
        type=int,
        default=None,
        help='Override inference_model.triton_max_prompt_len for speed and success runs.')
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Defaults to work_dirs/pi05_task0_with_l_pixshuffle_compare/<tag>.')
    parser.add_argument('--tag', default=None)
    parser.add_argument('--task-id', type=int, default=4)
    parser.add_argument('--prompt-len', type=int, default=32)
    parser.add_argument('--success-trials-per-task', type=int, default=50)
    parser.add_argument('--success-seeds', default='7')
    parser.add_argument('--success-gpus', default='3,4,5,6,7')
    parser.add_argument('--success-nproc-per-node', type=int, default=None)
    parser.add_argument('--speed-single-gpus', default='3')
    parser.add_argument('--speed-multi-gpus', default='3,4,5,6,7')
    parser.add_argument('--speed-warmup-iters', type=int, default=10)
    parser.add_argument('--speed-bench-iters', type=int, default=100)
    parser.add_argument('--master-addr', default='127.0.0.1')
    parser.add_argument('--master-port', type=int, default=29680)
    parser.add_argument('--skip-speed', action='store_true')
    parser.add_argument('--skip-success', action='store_true')
    parser.add_argument('--skip-speed-multi', action='store_true')
    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='Keep running remaining variants/phases if one benchmark/eval command fails.')
    parser.add_argument(
        '--debug-token-counts',
        action='store_true',
        help='Print accelerated path token counts during benchmark/eval.')
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


def base_env(gpus: str, args: argparse.Namespace) -> Dict[str, str]:
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpus
    env['WANDB_MODE'] = 'disabled'
    env['PYTHONUNBUFFERED'] = '1'
    env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    if args.debug_token_counts:
        env['FLUXVLA_DEBUG_TOKEN_COUNTS'] = '1'
        env['FLUXVLA_DEBUG_TOKEN_COUNTS_LIMIT'] = '4'
    return env


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
    return {
        'episodes': episodes,
        'successes': successes,
        'success_rate_pct': success_rate,
    }


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


def requested_variants(args: argparse.Namespace) -> List[str]:
    allowed = {'with_l', 'pixshuffle'}
    requested = split_csv(args.variants)
    if not requested:
        raise ValueError('--variants must select at least one variant.')
    unknown = sorted(set(requested) - allowed)
    if unknown:
        raise ValueError(f'Unknown variants: {unknown}. Allowed: {sorted(allowed)}')
    return requested


def materialize_config(args: argparse.Namespace, output_dir: Path,
                       item: Dict[str, str]) -> Dict[str, str]:
    cfg = Config.fromfile(item['source_config'])
    cfg.inference_model.use_language = True
    cfg.inference_model.pretrained_name_or_path = args.base_weights
    if args.triton_max_prompt_len is not None:
        cfg.inference_model.triton_max_prompt_len = args.triton_max_prompt_len
    if hasattr(cfg, 'model'):
        cfg.model.use_language = True
        cfg.model.pretrained_name_or_path = args.base_weights

    config_dir = output_dir / 'configs'
    config_dir.mkdir(parents=True, exist_ok=True)
    suffix = (
        f'_prompt{args.triton_max_prompt_len}'
        if args.triton_max_prompt_len is not None else '')
    path = config_dir / f'pi05_task0_{item["variant"]}_accelerated{suffix}.py'
    cfg.dump(str(path))

    item = dict(item)
    item['config'] = str(path)
    return item


def variants(args: argparse.Namespace, output_dir: Path) -> List[Dict[str, str]]:
    if args.variant:
        custom_items = [
            {
                'variant': name,
                'source_config': config,
                'ckpt': ckpt,
            } for name, config, ckpt in args.variant
        ]
        return [
            materialize_config(args, output_dir, item)
            for item in custom_items
        ]

    all_items = [
        {
            'variant': 'with_l',
            'source_config': args.with_l_config,
            'ckpt': args.with_l_ckpt,
        },
        {
            'variant': 'pixshuffle',
            'source_config': args.pixshuffle_config,
            'ckpt': args.pixshuffle_ckpt,
        },
    ]
    selected = set(requested_variants(args))
    return [
        materialize_config(args, output_dir, item)
        for item in all_items if item['variant'] in selected
    ]


def run_speed(args: argparse.Namespace, output_dir: Path, logs_dir: Path,
              item: Dict[str, str], scenario: str, gpus: str) -> Dict:
    speed_dir = output_dir / 'speed'
    speed_dir.mkdir(parents=True, exist_ok=True)
    tag = f'{args.tag}_{item["variant"]}_speed_{scenario}'
    cmd = [
        sys.executable,
        'scripts/benchmark_pi05_inference_speed.py',
        '--ckpt-path',
        item['ckpt'],
        '--accelerated-config',
        item['config'],
        '--mode',
        'accelerated',
        '--warmup-iters',
        str(args.speed_warmup_iters),
        '--bench-iters',
        str(args.speed_bench_iters),
        '--prompt-len',
        str(args.prompt_len),
        '--tag',
        tag,
        '--output-dir',
        str(speed_dir),
    ]
    if scenario == 'multi':
        cmd += ['--num-workers', str(len(split_csv(gpus)))]

    run_logged(cmd, base_env(gpus, args), logs_dir / f'{tag}.log')

    result_path = speed_dir / f'pi05_inference_speed_{tag}.json'
    with result_path.open('r') as f:
        data = json.load(f)
    result = data['results'][0]
    result.update({
        'requested_variant': item['variant'],
        'scenario': scenario,
        'result_path': str(result_path),
        'config_path': item['config'],
        'source_config_path': item['source_config'],
        'ckpt_path': item['ckpt'],
        'triton_max_prompt_len': args.triton_max_prompt_len,
    })
    return result


def run_success(args: argparse.Namespace, output_dir: Path, logs_dir: Path,
                item: Dict[str, str], seed: int, command_index: int) -> Dict:
    nproc = args.success_nproc_per_node or len(split_csv(args.success_gpus))
    port = args.master_port + command_index
    work_root = ckpt_root(item['ckpt'])
    before = {p.resolve() for p in work_root.glob('EVAL-*.txt')}
    tag = f'{args.tag}_{item["variant"]}_success_seed{seed}'
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
        item['config'],
        '--ckpt-path',
        item['ckpt'],
        '--cfg-options',
        f'eval.num_trials_per_task={args.success_trials_per_task}',
        f'eval.seed={seed}',
        f'eval.eval_task_ids=[{args.task_id}]',
        'inference_model.use_language=True',
        f'inference_model.pretrained_name_or_path={args.base_weights}',
    ]
    if args.triton_max_prompt_len is not None:
        cmd.append(f'inference_model.triton_max_prompt_len={args.triton_max_prompt_len}')
    env = base_env(args.success_gpus, args)
    env['MUJOCO_GL'] = 'egl'
    run_logged(cmd, env, log_path)

    after = list(work_root.glob('EVAL-*.txt'))
    candidates = [p for p in after if p.resolve() not in before]
    if not candidates:
        candidates = after
    eval_file, parsed = parse_best_success(candidates, log_path)
    parsed.update({
        'requested_variant': item['variant'],
        'seed': seed,
        'ckpt_path': item['ckpt'],
        'config_path': item['config'],
        'source_config_path': item['source_config'],
        'cuda_visible_devices': args.success_gpus,
        'nproc_per_node': nproc,
        'triton_max_prompt_len': args.triton_max_prompt_len,
        'eval_file': str(eval_file),
        'log_path': str(log_path),
    })
    return parsed


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
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


def write_markdown(path: Path, speed_rows: List[Dict],
                   success_rows: List[Dict], errors: List[Dict],
                   args: argparse.Namespace) -> None:
    lines = [
        '# PI0.5 Task0 With-L vs Pixshuffle',
        '',
        'Both variants use accelerated `PI05FlowMatchingInference`.',
        '',
        '## Checkpoints',
        '',
        f'- with_l: `{args.with_l_ckpt}`',
        f'- pixshuffle: `{args.pixshuffle_ckpt}`',
        f'- Selected variants: `{args.variants}`',
        '',
        '## Speed',
        '',
        '| Variant | Scenario | GPUs | Workers | Prompt len | Mean ms | P90 ms | Chunk Hz | Action-step Hz | Peak GiB |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in speed_rows:
        lines.append(
            '| {variant} | {scenario} | `{gpus}` | {workers} | {prompt_len} | '
            '{mean_ms} | {p90_ms} | {chunk_hz} | {action_hz} | {peak} |'.
            format(
                variant=row['requested_variant'],
                scenario=row['scenario'],
                gpus=row.get('cuda_visible_devices', ''),
                workers=row.get('num_workers', 1),
                prompt_len=row.get('prompt_len', args.prompt_len),
                mean_ms=fmt(float(row['mean_ms'])),
                p90_ms=fmt(float(row['p90_ms'])),
                chunk_hz=fmt(float(row['chunk_hz'])),
                action_hz=fmt(float(row['action_step_hz'])),
                peak=fmt(float(row['peak_allocated_gib_during_benchmark'])),
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
            '| {variant} | {seed} | `{gpus}` | {episodes} | {successes} | '
            '{rate}% | `{eval_file}` |'.format(
                variant=row['requested_variant'],
                seed=row['seed'],
                gpus=row['cuda_visible_devices'],
                episodes=fmt(float(row['episodes']), 0),
                successes=fmt(float(row['successes']), 0),
                rate=fmt(float(row['success_rate_pct'])),
                eval_file=row['eval_file'],
            ))

    if errors:
        lines += [
            '',
            '## Errors',
            '',
            '| Phase | Variant | Scenario/Seed | Error |',
            '| --- | --- | --- | --- |',
        ]
        for row in errors:
            detail = row.get('scenario', row.get('seed', ''))
            error = str(row['error']).replace('|', '\\|')
            lines.append(
                f"| {row['phase']} | {row['variant']} | {detail} | {error} |")

    lines += [
        '',
        '## Settings',
        '',
        f'- Task id: `{args.task_id}`',
        f'- Prompt len for speed: `{args.prompt_len}`',
        f'- Triton max prompt len override: `{args.triton_max_prompt_len}`',
        f'- Success trials per task: `{args.success_trials_per_task}`',
        f'- Success seeds: `{args.success_seeds}`',
        f'- Speed warmup/bench iters: `{args.speed_warmup_iters}` / `{args.speed_bench_iters}`',
        f'- with_l config: `{args.with_l_config}`',
        f'- pixshuffle config: `{args.pixshuffle_config}`',
        f'- Base weights used for success eval construction: `{args.base_weights}`',
    ]
    path.write_text('\n'.join(lines) + '\n')


def main() -> None:
    args = parse_args()
    if args.tag is None:
        args.tag = time.strftime('%Y%m%d_%H%M%S')
    output_dir = (
        Path(args.output_dir) if args.output_dir else
        Path('work_dirs/pi05_task0_with_l_pixshuffle_compare') / args.tag)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / 'logs'
    variant_items = variants(args, output_dir)

    speed_results: List[Dict] = []
    success_results: List[Dict] = []
    errors: List[Dict] = []

    def handle_error(phase: str, item: Dict[str, str], exc: Exception,
                     scenario=None, seed=None) -> None:
        errors.append({
            'phase': phase,
            'variant': item['variant'],
            'scenario': scenario,
            'seed': seed,
            'error': str(exc),
        })
        print(
            f"[Error] phase={phase} variant={item['variant']} "
            f"scenario={scenario} seed={seed}: {exc}",
            flush=True)
        if not args.continue_on_error:
            raise exc

    if not args.skip_speed:
        for item in variant_items:
            try:
                speed_results.append(
                    run_speed(args, output_dir, logs_dir, item, 'single',
                              args.speed_single_gpus))
            except Exception as exc:
                handle_error('speed', item, exc, scenario='single')
            if not args.skip_speed_multi:
                try:
                    speed_results.append(
                        run_speed(args, output_dir, logs_dir, item, 'multi',
                                  args.speed_multi_gpus))
                except Exception as exc:
                    handle_error('speed', item, exc, scenario='multi')

    command_index = 0
    if not args.skip_success:
        for seed in [int(seed) for seed in split_csv(args.success_seeds)]:
            for item in variant_items:
                try:
                    success_results.append(
                        run_success(args, output_dir, logs_dir, item, seed,
                                    command_index))
                except Exception as exc:
                    handle_error('success', item, exc, seed=seed)
                command_index += 1

    write_csv(output_dir / 'speed.csv', speed_results)
    write_csv(output_dir / 'success.csv', success_results)
    write_csv(output_dir / 'errors.csv', errors)
    write_markdown(output_dir / 'summary.md', speed_results, success_results,
                   errors, args)
    with (output_dir / 'summary.json').open('w') as f:
        json.dump(
            {
                'speed': speed_results,
                'success': success_results,
                'errors': errors,
                'args': vars(args),
            },
            f,
            indent=2)

    print(f'[Done] Wrote comparison to {output_dir}', flush=True)


if __name__ == '__main__':
    main()
