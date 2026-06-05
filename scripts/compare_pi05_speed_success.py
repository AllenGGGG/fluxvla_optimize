# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run PI0.5 speed and LIBERO success comparisons in one workflow."""

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


DEFAULT_BASELINE_CONFIG = 'configs/pi05/pi05_paligemma_libero_10_full_finetune.py'
DEFAULT_ACCELERATED_CONFIG = 'configs/pi05/pi05_paligemma_libero_10_full_inference.py'
DEFAULT_CKPT = (
    'checkpoints/pi05_paligemma_libero_10_full_finetune_bs64/'
    'pi05_paligemma_libero_10_full_finetune_bs64/checkpoints/'
    'step-038064-epoch-24-loss=0.0170.safetensors')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Compare PI0.5 speed and LIBERO success rate.')
    parser.add_argument('--ckpt-path', default=DEFAULT_CKPT)
    parser.add_argument('--baseline-config', default=DEFAULT_BASELINE_CONFIG)
    parser.add_argument('--accelerated-config', default=DEFAULT_ACCELERATED_CONFIG)
    parser.add_argument(
        '--tag',
        default=None,
        help='Tag used in generated filenames.')
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Directory for combined reports. Defaults next to checkpoint.')

    parser.add_argument(
        '--skip-speed',
        action='store_true',
        help='Skip predict_action latency/throughput benchmarks.')
    parser.add_argument(
        '--skip-success',
        action='store_true',
        help='Skip LIBERO success-rate evaluation.')
    parser.add_argument(
        '--reuse-speed-json',
        action='append',
        default=[],
        help='Reuse an existing speed benchmark JSON file. Can be repeated.')
    parser.add_argument(
        '--reuse-success-file',
        action='append',
        default=[],
        help='Reuse success stats as variant=path. Can be repeated.')
    parser.add_argument(
        '--success-variants',
        default='baseline,accelerated',
        help='Comma-separated success variants to run: baseline,accelerated.')
    parser.add_argument(
        '--speed-scenarios',
        default='single,multi',
        help='Comma-separated speed scenarios: single,multi.')
    parser.add_argument(
        '--single-gpus',
        default='4',
        help='Physical GPU list for single-GPU speed benchmark.')
    parser.add_argument(
        '--multi-gpus',
        default='5,6,7',
        help='Physical GPU list for multi-GPU throughput benchmark.')
    parser.add_argument('--speed-warmup-iters', type=int, default=3)
    parser.add_argument('--speed-bench-iters', type=int, default=10)

    parser.add_argument(
        '--success-gpus',
        default='4',
        help='Physical GPU list for LIBERO success evaluation.')
    parser.add_argument(
        '--success-trials-per-task',
        type=int,
        default=1,
        help='LIBERO trials per task. Use 50 for a formal LIBERO run.')
    parser.add_argument(
        '--success-seeds',
        default='7',
        help='Comma-separated LIBERO eval seeds.')
    parser.add_argument(
        '--success-nproc-per-node',
        type=int,
        default=None,
        help='torchrun processes for success eval. Defaults to GPU count.')
    parser.add_argument(
        '--master-addr',
        default='127.0.0.1',
        help='Master address for torchrun success eval.')
    parser.add_argument(
        '--master-port',
        type=int,
        default=29500,
        help='Base port for torchrun eval commands.')
    return parser.parse_args()


def split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(',') if item.strip()]


def ckpt_root(ckpt_path: str) -> Path:
    return Path(ckpt_path).resolve().parent.parent


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


def load_json(path: Path) -> Dict:
    with path.open('r') as f:
        return json.load(f)


def load_reused_speed(path: Path) -> Dict:
    data = load_json(path)
    name = path.name
    if '_speed_single' in name:
        scenario = 'single'
    elif '_speed_multi' in name:
        scenario = 'multi'
    else:
        scenario = data.get('scenario', path.stem)
    data['scenario'] = scenario
    if 'cuda_visible_devices' not in data:
        devices = {
            result.get('cuda_visible_devices', '')
            for result in data.get('results', [])
            if result.get('cuda_visible_devices')
        }
        data['cuda_visible_devices'] = ','.join(sorted(devices))
    data['result_path'] = str(path)
    return data


def parse_variant_path(value: str) -> Tuple[str, Path]:
    if '=' not in value:
        raise ValueError(
            f'Expected variant=path for --reuse-success-file, got: {value}')
    variant, path = value.split('=', 1)
    variant = variant.strip()
    if not variant:
        raise ValueError(f'Missing variant in --reuse-success-file {value}')
    return variant, Path(path)


def run_speed_scenario(args: argparse.Namespace, scenario: str, tag: str,
                       output_dir: Path, logs_dir: Path) -> Dict:
    if scenario == 'single':
        visible_gpus = args.single_gpus
        num_workers_args: List[str] = []
    elif scenario == 'multi':
        visible_gpus = args.multi_gpus
        num_workers_args = ['--num-workers', str(len(split_csv(args.multi_gpus)))]
    else:
        raise ValueError(f'Unknown speed scenario: {scenario}')

    scenario_tag = f'{tag}_speed_{scenario}'
    speed_output_dir = output_dir / 'speed_benchmarks'
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = visible_gpus
    cmd = [
        sys.executable,
        'scripts/benchmark_pi05_inference_speed.py',
        '--ckpt-path',
        args.ckpt_path,
        '--baseline-config',
        args.baseline_config,
        '--accelerated-config',
        args.accelerated_config,
        '--mode',
        'both',
        '--warmup-iters',
        str(args.speed_warmup_iters),
        '--bench-iters',
        str(args.speed_bench_iters),
        '--tag',
        scenario_tag,
        '--output-dir',
        str(speed_output_dir),
    ] + num_workers_args
    run_logged(cmd, env, logs_dir / f'{scenario_tag}.log')

    result_path = (
        speed_output_dir / f'pi05_inference_speed_{scenario_tag}.json')
    data = load_json(result_path)
    data['scenario'] = scenario
    data['cuda_visible_devices'] = visible_gpus
    data['result_path'] = str(result_path)
    return data


SUCCESS_RE = re.compile(r'# successes:\s*([0-9.]+)\s*\(([0-9.]+)%\)')
EPISODES_RE = re.compile(r'# episodes completed so far:\s*([0-9.]+)')


def parse_success_file(path: Path) -> Dict[str, float]:
    return parse_success_text(path.read_text(errors='replace'), path)


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


def parse_best_success_file(paths: Iterable[Path], log_path: Path) -> Tuple[Path, Dict[str, float]]:
    parsed_candidates = []
    for path in paths:
        try:
            parsed_candidates.append((path, parse_success_file(path)))
        except ValueError:
            continue
    if log_path.exists():
        try:
            parsed_candidates.append((log_path, parse_success_file(log_path)))
        except ValueError:
            pass
    if not parsed_candidates:
        raise ValueError(
            f'Could not parse success stats from EVAL files or {log_path}')
    return max(parsed_candidates, key=lambda item: item[1]['episodes'])


def latest_new_eval_file(before: Iterable[Path], after: Iterable[Path]) -> Path:
    before_set = {p.resolve() for p in before}
    candidates = [p for p in after if p.resolve() not in before_set]
    if not candidates:
        candidates = list(after)
    if not candidates:
        raise FileNotFoundError('No EVAL-*.txt file was produced.')
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_success_eval(args: argparse.Namespace, variant: str, config_path: str,
                     seed: int, tag: str, logs_dir: Path,
                     command_index: int) -> Dict:
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = args.success_gpus
    nproc = args.success_nproc_per_node or len(split_csv(args.success_gpus))
    work_root = ckpt_root(args.ckpt_path)
    before = list(work_root.glob('EVAL-*.txt'))
    port = args.master_port + command_index
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
        config_path,
        '--ckpt-path',
        args.ckpt_path,
        '--cfg-options',
        f'eval.num_trials_per_task={args.success_trials_per_task}',
        f'eval.seed={seed}',
    ]
    log_name = f'{tag}_success_{variant}_seed{seed}.log'
    run_logged(cmd, env, logs_dir / log_name)
    after = list(work_root.glob('EVAL-*.txt'))
    before_set = {p.resolve() for p in before}
    candidates = [p for p in after if p.resolve() not in before_set]
    if not candidates:
        candidates = list(after)
    if not candidates:
        raise FileNotFoundError('No EVAL-*.txt file was produced.')
    eval_file, parsed = parse_best_success_file(candidates, logs_dir / log_name)
    parsed.update({
        'variant': variant,
        'config': config_path,
        'seed': seed,
        'cuda_visible_devices': args.success_gpus,
        'nproc_per_node': nproc,
        'eval_file': str(eval_file),
    })
    return parsed


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float('nan')


def aggregate_success(rows: List[Dict]) -> List[Dict]:
    variants = sorted({row['variant'] for row in rows})
    output = []
    for variant in variants:
        subset = [row for row in rows if row['variant'] == variant]
        output.append({
            'variant': variant,
            'success_rate_pct': mean([row['success_rate_pct'] for row in subset]),
            'episodes': sum(row['episodes'] for row in subset),
            'successes': sum(row['successes'] for row in subset),
            'num_seeds': len(subset),
            'seeds': ','.join(str(row['seed']) for row in subset),
        })
    return output


def speed_rows(speed_results: List[Dict]) -> List[Dict]:
    rows = []
    for scenario_result in speed_results:
        scenario = scenario_result['scenario']
        for result in scenario_result['results']:
            rows.append({
                'scenario': scenario,
                'variant': result['variant'],
                'mean_ms': result['mean_ms'],
                'p50_ms': result['p50_ms'],
                'p90_ms': result['p90_ms'],
                'hz': result['chunk_hz'],
                'action_step_hz': result['action_step_hz'],
                'num_workers': result.get('num_workers', 1),
                'per_worker_hz': result.get('per_worker_chunk_hz',
                                            result['chunk_hz']),
                'peak_gib': result['peak_allocated_gib_during_benchmark'],
                'cuda_visible_devices':
                scenario_result.get('cuda_visible_devices', ''),
            })
    return rows


def comparison_rows(speed_results: List[Dict], success_summary: List[Dict]
                    ) -> List[Dict]:
    speed = speed_rows(speed_results)
    success_by_variant = {row['variant']: row for row in success_summary}
    rows = []
    for speed_row in speed:
        success = success_by_variant.get(speed_row['variant'], {})
        rows.append({
            **speed_row,
            'success_rate_pct': success.get('success_rate_pct'),
            'successes': success.get('successes'),
            'episodes': success.get('episodes'),
            'success_seeds': success.get('seeds'),
        })
    return rows


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return 'n/a'
    if isinstance(value, float):
        return f'{value:.{digits}f}'
    return str(value)


def write_markdown(path: Path, rows: List[Dict], success_details: List[Dict],
                   args: argparse.Namespace) -> None:
    lines = [
        '# PI0.5 Speed + Success Comparison',
        '',
        'This report combines model-side inference speed and LIBERO success rate.',
        '',
        '## Run Size',
        '',
        f'- Speed warmup iterations: `{args.speed_warmup_iters}`',
        f'- Speed measured iterations per worker: `{args.speed_bench_iters}`',
        f'- Success trials per LIBERO task: `{args.success_trials_per_task}`',
        f'- Success seeds: `{args.success_seeds}`',
        '- PI0.5 action chunk size: `10` action steps per `predict_action` call',
        '',
        '## Combined Table',
        '',
        '| Scenario | Variant | GPUs | Workers | Mean ms | P90 ms | Hz | Per-worker Hz | Action-step Hz | Success rate | Successes / Episodes |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in rows:
        lines.append(
            '| {scenario} | {variant} | `{gpus}` | {workers} | {mean_ms} | '
            '{p90_ms} | {hz} | {per_worker_hz} | {action_hz} | '
            '{success_rate} | {successes}/{episodes} |'.format(
                scenario=row['scenario'],
                variant=row['variant'],
                gpus=row['cuda_visible_devices'],
                workers=row['num_workers'],
                mean_ms=fmt(row['mean_ms']),
                p90_ms=fmt(row['p90_ms']),
                hz=fmt(row['hz']),
                per_worker_hz=fmt(row['per_worker_hz']),
                action_hz=fmt(row['action_step_hz']),
                success_rate=fmt(row.get('success_rate_pct')),
                successes=fmt(row.get('successes'), 0),
                episodes=fmt(row.get('episodes'), 0),
            ))

    lines += [
        '',
        '## Success Eval Details',
        '',
        '| Variant | Seed | GPUs | Episodes | Successes | Success rate | Eval file |',
        '| --- | ---: | --- | ---: | ---: | ---: | --- |',
    ]
    for row in success_details:
        lines.append(
            f"| {row['variant']} | {row['seed']} | "
            f"`{row['cuda_visible_devices']}` | {fmt(row['episodes'], 0)} | "
            f"{fmt(row['successes'], 0)} | {fmt(row['success_rate_pct'])} | "
            f"`{row['eval_file']}` |")

    lines += [
        '',
        '## Settings',
        '',
        f'- Checkpoint: `{args.ckpt_path}`',
        f'- Baseline config: `{args.baseline_config}`',
        f'- Accelerated config: `{args.accelerated_config}`',
        f'- Speed warmup / bench iters: {args.speed_warmup_iters} / {args.speed_bench_iters}',
        f'- Success trials per task: {args.success_trials_per_task}',
        f'- Success seeds: `{args.success_seeds}`',
    ]
    path.write_text('\n'.join(lines) + '\n')


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    tag = args.tag or time.strftime('%Y%m%d-%H%M%S')
    output_dir = Path(args.output_dir) if args.output_dir else (
        ckpt_root(args.ckpt_path) / 'comparisons')
    logs_dir = output_dir / 'logs'
    output_dir.mkdir(parents=True, exist_ok=True)

    speed_results: List[Dict] = []
    for path_str in args.reuse_speed_json:
        speed_results.append(load_reused_speed(Path(path_str)))
    if not args.skip_speed:
        scenarios = split_csv(args.speed_scenarios)
        for scenario in scenarios:
            speed_results.append(
                run_speed_scenario(args, scenario, tag, output_dir, logs_dir))

    success_details: List[Dict] = []
    for reused in args.reuse_success_file:
        variant, path = parse_variant_path(reused)
        parsed = parse_success_file(path)
        parsed.update({
            'variant': variant,
            'config': (args.baseline_config if variant == 'baseline' else
                       args.accelerated_config),
            'seed': None,
            'cuda_visible_devices': args.success_gpus,
            'nproc_per_node': args.success_nproc_per_node,
            'eval_file': str(path),
        })
        success_details.append(parsed)
    if not args.skip_success:
        seeds = [int(seed) for seed in split_csv(args.success_seeds)]
        success_variants = split_csv(args.success_variants)
        variant_configs = {
            'baseline': args.baseline_config,
            'accelerated': args.accelerated_config,
        }
        command_index = 0
        for seed in seeds:
            for variant in success_variants:
                if variant not in variant_configs:
                    raise ValueError(f'Unknown success variant: {variant}')
                config = variant_configs[variant]
                success_details.append(
                    run_success_eval(args, variant, config, seed, tag,
                                     logs_dir, command_index))
                command_index += 1

    success_summary = aggregate_success(success_details) if success_details else []
    rows = comparison_rows(speed_results, success_summary)

    payload = {
        'combined_rows': rows,
        'speed_results': speed_results,
        'success_details': success_details,
        'success_summary': success_summary,
    }
    json_path = output_dir / f'pi05_speed_success_comparison_{tag}.json'
    csv_path = output_dir / f'pi05_speed_success_comparison_{tag}.csv'
    md_path = output_dir / f'pi05_speed_success_comparison_{tag}.md'
    json_path.write_text(json.dumps(payload, indent=2) + '\n')
    write_csv(csv_path, rows)
    write_markdown(md_path, rows, success_details, args)
    print(f'[Compare] Wrote {json_path}', flush=True)
    print(f'[Compare] Wrote {csv_path}', flush=True)
    print(f'[Compare] Wrote {md_path}', flush=True)


if __name__ == '__main__':
    main()
