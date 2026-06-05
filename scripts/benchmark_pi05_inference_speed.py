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

"""Benchmark PI0.5 baseline vs accelerated predict_action latency.

This script intentionally measures only model-side ``predict_action`` latency.
It avoids LIBERO environment stepping, video writing, and dataset I/O so the
reported Hz reflects inference speed instead of simulator throughput.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import multiprocessing as py_mp
import os
import queue
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.multiprocessing as mp
from mmengine import Config
from safetensors.torch import load_file


DEFAULT_BASELINE_CONFIG = 'configs/pi05/pi05_paligemma_libero_10_full_finetune.py'
DEFAULT_ACCELERATED_CONFIG = 'configs/pi05/pi05_paligemma_libero_10_full_inference.py'
DEFAULT_CKPT = (
    'checkpoints/pi05_paligemma_libero_10_full_finetune_bs64/'
    'pi05_paligemma_libero_10_full_finetune_bs64/checkpoints/'
    'step-038064-epoch-24-loss=0.0170.safetensors')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Benchmark PI0.5 baseline and accelerated inference.')
    parser.add_argument(
        '--ckpt-path',
        default=DEFAULT_CKPT,
        help='Fine-tuned PI0.5 checkpoint path.')
    parser.add_argument(
        '--baseline-config',
        default=DEFAULT_BASELINE_CONFIG,
        help='Config whose cfg.model is used as the baseline model.')
    parser.add_argument(
        '--accelerated-config',
        default=DEFAULT_ACCELERATED_CONFIG,
        help='Config whose cfg.inference_model is used as the accelerated model.')
    parser.add_argument(
        '--mode',
        choices=('baseline', 'accelerated', 'both'),
        default='both',
        help='Which model variant to benchmark.')
    parser.add_argument(
        '--device',
        default='cuda:0',
        help='CUDA device for single-worker mode after CUDA_VISIBLE_DEVICES filtering.')
    parser.add_argument(
        '--num-workers',
        type=int,
        default=1,
        help=(
            'Number of concurrent GPU worker processes. Worker i uses cuda:i '
            'after CUDA_VISIBLE_DEVICES filtering. Use 3 for a three-GPU '
            'throughput comparison.'))
    parser.add_argument(
        '--dtype',
        choices=('bf16', 'fp16', 'fp32'),
        default='bf16',
        help='Model and autocast dtype.')
    parser.add_argument(
        '--warmup-iters',
        type=int,
        default=10,
        help='Warmup iterations excluded from latency statistics.')
    parser.add_argument(
        '--bench-iters',
        type=int,
        default=100,
        help='Measured iterations.')
    parser.add_argument(
        '--prompt-len',
        type=int,
        default=32,
        help='Synthetic prompt length. PI0.5 accelerated default max is 48.')
    parser.add_argument(
        '--num-views',
        type=int,
        default=2,
        help='Number of camera views.')
    parser.add_argument(
        '--image-size',
        type=int,
        default=224,
        help='Square image size.')
    parser.add_argument(
        '--state-dim',
        type=int,
        default=32,
        help='State vector dimension.')
    parser.add_argument(
        '--action-dim',
        type=int,
        default=32,
        help='Synthetic noise/action dimension.')
    parser.add_argument(
        '--seed',
        type=int,
        default=7,
        help='Random seed for synthetic inputs.')
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Directory for JSON/CSV/Markdown results. Defaults next to ckpt.')
    parser.add_argument(
        '--tag',
        default=None,
        help='Optional filename tag for result files.')
    parser.add_argument(
        '--skip-cold-start',
        action='store_true',
        help='Do not measure the first predict_action call separately.')
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    if name == 'bf16':
        return torch.bfloat16
    if name == 'fp16':
        return torch.float16
    if name == 'fp32':
        return torch.float32
    raise ValueError(f'Unsupported dtype: {name}')


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return float('nan')
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(index)
    hi = math.ceil(index)
    if lo == hi:
        return ordered[int(index)]
    return ordered[lo] * (hi - index) + ordered[hi] * (index - lo)


def bytes_to_gib(value: int) -> float:
    return value / (1024**3)


def load_state_dict(ckpt_path: str) -> Dict[str, torch.Tensor]:
    if ckpt_path.endswith('.safetensors'):
        return load_file(ckpt_path, device='cpu')

    checkpoint = torch.load(ckpt_path, map_location='cpu', mmap=True)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        return checkpoint['model']
    return checkpoint


def build_model(config_path: str, variant: str, ckpt_path: str,
                device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    # Import here so --help does not trigger LIBERO/user-directory setup.
    # The package root import populates all registries used by build_vla_from_cfg.
    import fluxvla  # noqa: F401
    from fluxvla.engines import build_vla_from_cfg

    cfg = Config.fromfile(config_path)
    if variant == 'baseline':
        model_cfg = cfg.model
    elif variant == 'accelerated':
        if not hasattr(cfg, 'inference_model'):
            raise ValueError(
                f'{config_path} does not define cfg.inference_model.')
        model_cfg = cfg.inference_model
    else:
        raise ValueError(f'Unsupported variant: {variant}')

    model = build_vla_from_cfg(model_cfg).eval()
    state_dict = load_state_dict(ckpt_path)
    model.load_state_dict(state_dict, strict=True)
    del state_dict
    gc.collect()

    model.to(device=device, dtype=dtype)
    model.eval()
    return model


def make_synthetic_batch(args: argparse.Namespace, action_steps: int,
                         device: torch.device,
                         dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    images = torch.randn(
        1,
        args.num_views * 3,
        args.image_size,
        args.image_size,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    lang_tokens = torch.randint(
        low=100,
        high=1000,
        size=(1, args.prompt_len),
        generator=generator,
        device=device,
        dtype=torch.long,
    )
    states = torch.randn(
        1,
        args.state_dim,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    img_masks = torch.ones(
        1, args.num_views, device=device, dtype=torch.bool)
    lang_masks = torch.ones(
        1, args.prompt_len, device=device, dtype=torch.bool)
    noise = torch.randn(
        1,
        action_steps,
        args.action_dim,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    return dict(
        images=images,
        lang_tokens=lang_tokens,
        states=states,
        img_masks=img_masks,
        lang_masks=lang_masks,
        noise=noise,
    )


def timed_predict(model: torch.nn.Module, batch: Dict[str, torch.Tensor],
                  dtype: torch.dtype) -> tuple[torch.Tensor, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.inference_mode(), torch.autocast(
            'cuda', dtype=dtype, enabled=dtype != torch.float32):
        output = model.predict_action(**batch)
    end.record()
    torch.cuda.synchronize()
    return output, start.elapsed_time(end)


def benchmark_variant(variant: str, config_path: str, args: argparse.Namespace,
                      device: torch.device,
                      dtype: torch.dtype,
                      start_barrier: Optional[Any] = None,
                      worker_rank: int = 0,
                      num_workers: int = 1) -> Dict[str, object]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    t_load_start = time.perf_counter()
    model = build_model(config_path, variant, args.ckpt_path, device, dtype)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - t_load_start
    model_allocated_gib = bytes_to_gib(torch.cuda.memory_allocated(device))

    action_steps = int(getattr(model, 'n_action_steps', 10))
    batch = make_synthetic_batch(args, action_steps, device, dtype)

    cold_start_ms: Optional[float] = None
    last_output: Optional[torch.Tensor] = None
    if not args.skip_cold_start:
        last_output, cold_start_ms = timed_predict(model, batch, dtype)

    for _ in range(args.warmup_iters):
        last_output, _ = timed_predict(model, batch, dtype)

    if start_barrier is not None:
        start_barrier.wait()

    torch.cuda.reset_peak_memory_stats(device)
    latencies_ms: List[float] = []
    for _ in range(args.bench_iters):
        last_output, latency_ms = timed_predict(model, batch, dtype)
        latencies_ms.append(latency_ms)

    peak_allocated_gib = bytes_to_gib(torch.cuda.max_memory_allocated(device))
    mean_ms = statistics.fmean(latencies_ms)
    std_ms = statistics.pstdev(latencies_ms) if len(latencies_ms) > 1 else 0.0
    chunk_hz = 1000.0 / mean_ms

    result: Dict[str, object] = {
        'variant': variant,
        'worker_rank': worker_rank,
        'num_workers': num_workers,
        'config': config_path,
        'ckpt_path': args.ckpt_path,
        'device': str(device),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
        'dtype': str(dtype).replace('torch.', ''),
        'warmup_iters': args.warmup_iters,
        'bench_iters': args.bench_iters,
        'prompt_len': args.prompt_len,
        'num_views': args.num_views,
        'image_size': args.image_size,
        'action_steps_per_call': action_steps,
        'load_seconds': load_seconds,
        'cold_start_ms': cold_start_ms,
        'mean_ms': mean_ms,
        'std_ms': std_ms,
        'min_ms': min(latencies_ms),
        'p50_ms': percentile(latencies_ms, 50),
        'p90_ms': percentile(latencies_ms, 90),
        'p95_ms': percentile(latencies_ms, 95),
        'p99_ms': percentile(latencies_ms, 99),
        'max_ms': max(latencies_ms),
        'chunk_hz': chunk_hz,
        'action_step_hz': chunk_hz * action_steps,
        'model_allocated_gib_after_load': model_allocated_gib,
        'peak_allocated_gib_during_benchmark': peak_allocated_gib,
        'output_shape': list(last_output.shape) if last_output is not None else None,
        'output_mean': float(last_output.float().mean().item())
        if last_output is not None else None,
        'output_std': float(last_output.float().std().item())
        if last_output is not None else None,
        '_output_cpu': last_output.detach().float().cpu()
        if last_output is not None else None,
    }

    del model, batch, last_output
    gc.collect()
    torch.cuda.empty_cache()
    return result


def benchmark_worker(variant: str, config_path: str, args: argparse.Namespace,
                     worker_rank: int, num_workers: int,
                     start_barrier: py_mp.synchronize.Barrier,
                     result_queue: py_mp.Queue) -> None:
    try:
        device = torch.device(f'cuda:{worker_rank}')
        torch.cuda.set_device(device)
        dtype = dtype_from_name(args.dtype)
        torch.manual_seed(args.seed + worker_rank)
        torch.cuda.manual_seed_all(args.seed + worker_rank)
        result = benchmark_variant(
            variant,
            config_path,
            args,
            device,
            dtype,
            start_barrier=start_barrier,
            worker_rank=worker_rank,
            num_workers=num_workers)
        result.pop('_output_cpu', None)
        result_queue.put(('ok', result))
    except Exception as exc:  # pragma: no cover - child process reporting.
        result_queue.put(('error', worker_rank, repr(exc)))


def aggregate_worker_results(variant: str, config_path: str,
                             worker_results: List[Dict[str, object]]
                             ) -> Dict[str, object]:
    if not worker_results:
        raise ValueError(f'No worker results for {variant}.')

    def mean_key(key: str) -> float:
        return statistics.fmean(float(r[key]) for r in worker_results)

    def max_key(key: str) -> float:
        return max(float(r[key]) for r in worker_results)

    def min_key(key: str) -> float:
        return min(float(r[key]) for r in worker_results)

    total_chunk_hz = sum(float(r['chunk_hz']) for r in worker_results)
    total_action_step_hz = sum(
        float(r['action_step_hz']) for r in worker_results)
    first = worker_results[0]

    return {
        'variant': variant,
        'worker_rank': 'aggregate',
        'num_workers': len(worker_results),
        'config': config_path,
        'ckpt_path': first['ckpt_path'],
        'device': ','.join(str(r['device']) for r in worker_results),
        'cuda_visible_devices': first.get('cuda_visible_devices', ''),
        'dtype': first['dtype'],
        'warmup_iters': first['warmup_iters'],
        'bench_iters': first['bench_iters'],
        'prompt_len': first['prompt_len'],
        'num_views': first['num_views'],
        'image_size': first['image_size'],
        'action_steps_per_call': first['action_steps_per_call'],
        'load_seconds': max_key('load_seconds'),
        'cold_start_ms': max_key('cold_start_ms')
        if first['cold_start_ms'] is not None else None,
        'mean_ms': mean_key('mean_ms'),
        'std_ms': mean_key('std_ms'),
        'min_ms': min_key('min_ms'),
        'p50_ms': mean_key('p50_ms'),
        'p90_ms': mean_key('p90_ms'),
        'p95_ms': mean_key('p95_ms'),
        'p99_ms': mean_key('p99_ms'),
        'max_ms': max_key('max_ms'),
        'chunk_hz': total_chunk_hz,
        'per_worker_chunk_hz': mean_key('chunk_hz'),
        'action_step_hz': total_action_step_hz,
        'per_worker_action_step_hz': mean_key('action_step_hz'),
        'model_allocated_gib_after_load':
        sum(float(r['model_allocated_gib_after_load'])
            for r in worker_results),
        'peak_allocated_gib_during_benchmark':
        sum(float(r['peak_allocated_gib_during_benchmark'])
            for r in worker_results),
        'output_shape': first['output_shape'],
        'output_mean': mean_key('output_mean'),
        'output_std': mean_key('output_std'),
        'worker_results': [public_result(r) for r in worker_results],
    }


def benchmark_variant_multi(variant: str, config_path: str,
                            args: argparse.Namespace
                            ) -> Dict[str, object]:
    ctx = mp.get_context('spawn')
    start_barrier = ctx.Barrier(args.num_workers)
    result_queue = ctx.Queue()
    processes = []

    for worker_rank in range(args.num_workers):
        proc = ctx.Process(
            target=benchmark_worker,
            args=(variant, config_path, args, worker_rank, args.num_workers,
                  start_barrier, result_queue))
        proc.start()
        processes.append(proc)

    worker_results: List[Dict[str, object]] = []
    errors = []
    while len(worker_results) + len(errors) < args.num_workers:
        try:
            item = result_queue.get(timeout=5)
        except queue.Empty:
            failed_now = [
                (idx, proc.exitcode)
                for idx, proc in enumerate(processes)
                if proc.exitcode not in (None, 0)
            ]
            if failed_now:
                errors.extend(failed_now)
                break
            continue
        if item[0] == 'ok':
            worker_results.append(item[1])
        else:
            errors.append(item[1:])

    for proc in processes:
        proc.join(timeout=5)

    failed = [proc.exitcode for proc in processes if proc.exitcode != 0]
    if errors or failed:
        raise RuntimeError(
            f'Multi-worker benchmark failed: errors={errors}, '
            f'exitcodes={failed}')

    worker_results.sort(key=lambda r: int(r['worker_rank']))
    return aggregate_worker_results(variant, config_path, worker_results)


def public_result(result: Dict[str, object]) -> Dict[str, object]:
    return {k: v for k, v in result.items() if not k.startswith('_')}


def add_pairwise_comparison(results: List[Dict[str, object]]) -> Optional[Dict[str, float]]:
    by_variant = {r['variant']: r for r in results}
    if 'baseline' not in by_variant or 'accelerated' not in by_variant:
        return None

    baseline = by_variant['baseline']
    accelerated = by_variant['accelerated']
    comparison = {
        'mean_latency_speedup': baseline['mean_ms'] / accelerated['mean_ms'],
        'p50_latency_speedup': baseline['p50_ms'] / accelerated['p50_ms'],
        'hz_speedup': accelerated['chunk_hz'] / baseline['chunk_hz'],
        'mean_latency_reduction_pct':
        (1.0 - accelerated['mean_ms'] / baseline['mean_ms']) * 100.0,
        'peak_memory_delta_gib':
        accelerated['peak_allocated_gib_during_benchmark'] -
        baseline['peak_allocated_gib_during_benchmark'],
    }
    if ('per_worker_chunk_hz' in baseline
            and 'per_worker_chunk_hz' in accelerated):
        comparison['per_worker_hz_speedup'] = (
            accelerated['per_worker_chunk_hz'] /
            baseline['per_worker_chunk_hz'])

    base_out = baseline.get('_output_cpu')
    accel_out = accelerated.get('_output_cpu')
    if isinstance(base_out, torch.Tensor) and isinstance(accel_out, torch.Tensor):
        if list(base_out.shape) == list(accel_out.shape):
            diff = (base_out - accel_out).abs()
            comparison['output_max_abs_diff'] = float(diff.max().item())
            comparison['output_mean_abs_diff'] = float(diff.mean().item())
    return comparison


def format_float(value: object, digits: int = 3) -> str:
    if value is None:
        return 'n/a'
    if isinstance(value, float):
        return f'{value:.{digits}f}'
    return str(value)


def write_csv(path: Path, results: List[Dict[str, object]]) -> None:
    rows = [{
        key: value
        for key, value in public_result(r).items()
        if key != 'worker_results'
    } for r in results]
    fieldnames = sorted({key for row in rows for key in row.keys()
                         if key != 'worker_results'})
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, results: List[Dict[str, object]],
                   comparison: Optional[Dict[str, float]]) -> None:
    lines = [
        '# PI0.5 Inference Speed Comparison',
        '',
        'This benchmark measures model-side `predict_action` latency only.',
        '',
        '## Benchmark Settings',
        '',
        f"- Warmup iterations: `{results[0]['warmup_iters']}`",
        f"- Measured iterations per worker: `{results[0]['bench_iters']}`",
        f"- Prompt length: `{results[0]['prompt_len']}`",
        f"- Camera views: `{results[0]['num_views']}`",
        f"- Action chunk size: `{results[0]['action_steps_per_call']}` action steps per `predict_action` call",
        '',
        'Only measured iterations are used for Mean / P50 / P90 / P99 / Hz.',
        '',
        '## Summary',
        '',
        '| Variant | Mean ms | P50 ms | P90 ms | P99 ms | Hz | Action-step Hz | Cold start ms | Peak GiB |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for r in results:
        lines.append(
            '| {variant} | {mean_ms} | {p50_ms} | {p90_ms} | {p99_ms} | '
            '{chunk_hz} | {action_step_hz} | {cold_start_ms} | {peak_gib} |'
            .format(
                variant=r['variant'],
                mean_ms=format_float(r['mean_ms']),
                p50_ms=format_float(r['p50_ms']),
                p90_ms=format_float(r['p90_ms']),
                p99_ms=format_float(r['p99_ms']),
                chunk_hz=format_float(r['chunk_hz']),
                action_step_hz=format_float(r['action_step_hz']),
                cold_start_ms=format_float(r['cold_start_ms']),
                peak_gib=format_float(
                    r['peak_allocated_gib_during_benchmark']),
            ))

    if any('per_worker_chunk_hz' in r for r in results):
        lines += [
            '',
            '## Multi-GPU Throughput',
            '',
            '| Variant | Workers | Total Hz | Per-worker Hz | Total Action-step Hz | Per-worker Mean ms | Total Peak GiB |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
        ]
        for r in results:
            lines.append(
                '| {variant} | {workers} | {total_hz} | {worker_hz} | '
                '{total_action_hz} | {mean_ms} | {peak_gib} |'.format(
                    variant=r['variant'],
                    workers=r['num_workers'],
                    total_hz=format_float(r['chunk_hz']),
                    worker_hz=format_float(r.get('per_worker_chunk_hz')),
                    total_action_hz=format_float(r['action_step_hz']),
                    mean_ms=format_float(r['mean_ms']),
                    peak_gib=format_float(
                        r['peak_allocated_gib_during_benchmark']),
                ))

        lines += ['', '### Per-Worker Details', '']
        for r in results:
            worker_results = r.get('worker_results')
            if not worker_results:
                continue
            lines += [
                f"#### {r['variant']}",
                '',
                '| Worker | Device | Mean ms | P90 ms | Hz | Peak GiB |',
                '| ---: | --- | ---: | ---: | ---: | ---: |',
            ]
            for wr in worker_results:
                lines.append(
                    '| {worker} | {device} | {mean_ms} | {p90_ms} | '
                    '{hz} | {peak_gib} |'.format(
                        worker=wr['worker_rank'],
                        device=wr['device'],
                        mean_ms=format_float(wr['mean_ms']),
                        p90_ms=format_float(wr['p90_ms']),
                        hz=format_float(wr['chunk_hz']),
                        peak_gib=format_float(
                            wr['peak_allocated_gib_during_benchmark']),
                    ))
            lines.append('')

    if comparison:
        lines += [
            '',
            '## Pairwise Comparison',
            '',
            '| Metric | Value |',
            '| --- | ---: |',
        ]
        for key, value in comparison.items():
            lines.append(f'| {key} | {format_float(value)} |')

    lines += [
        '',
        '## Run Details',
        '',
        '| Variant | Config | Checkpoint | CUDA_VISIBLE_DEVICES | Dtype | Prompt len | Iters |',
        '| --- | --- | --- | --- | --- | ---: | ---: |',
    ]
    for r in results:
        lines.append(
            f"| {r['variant']} | `{r['config']}` | `{r['ckpt_path']}` | "
            f"`{r.get('cuda_visible_devices', '')}` | {r['dtype']} | "
            f"{r['prompt_len']} | {r['bench_iters']} |")

    path.write_text('\n'.join(lines) + '\n')


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this benchmark.')
    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {args.ckpt_path}')

    if args.num_workers < 1:
        raise ValueError('--num-workers must be at least 1.')
    if args.num_workers > torch.cuda.device_count():
        raise ValueError(
            f'--num-workers={args.num_workers} exceeds visible CUDA device '
            f'count {torch.cuda.device_count()}. Set CUDA_VISIBLE_DEVICES.')

    device = torch.device(args.device)
    if args.num_workers == 1:
        if device.type != 'cuda':
            raise ValueError('Only CUDA devices are supported.')
        torch.cuda.set_device(device)
    dtype = dtype_from_name(args.dtype)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(args.ckpt_path).resolve().parent.parent / 'benchmarks')
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or time.strftime('%Y%m%d-%H%M%S')

    variants = []
    if args.mode in ('baseline', 'both'):
        variants.append(('baseline', args.baseline_config))
    if args.mode in ('accelerated', 'both'):
        variants.append(('accelerated', args.accelerated_config))

    results: List[Dict[str, object]] = []
    for variant, config_path in variants:
        print(f'[Benchmark] Running {variant}: {config_path}', flush=True)
        if args.num_workers == 1:
            result = benchmark_variant(variant, config_path, args, device,
                                       dtype)
        else:
            result = benchmark_variant_multi(variant, config_path, args)
        results.append(result)
        print(
            f"[Benchmark] {variant}: mean={result['mean_ms']:.2f}ms, "
            f"hz={result['chunk_hz']:.2f}, "
            f"p90={result['p90_ms']:.2f}ms",
            flush=True)

    comparison = add_pairwise_comparison(results)
    if comparison:
        print(
            '[Benchmark] Speedup: '
            f"{comparison['hz_speedup']:.2f}x, "
            f"latency reduction="
            f"{comparison['mean_latency_reduction_pct']:.1f}%",
            flush=True)

    json_path = output_dir / f'pi05_inference_speed_{tag}.json'
    csv_path = output_dir / f'pi05_inference_speed_{tag}.csv'
    md_path = output_dir / f'pi05_inference_speed_comparison_{tag}.md'

    payload = {
        'results': [public_result(r) for r in results],
        'comparison': comparison,
    }
    json_path.write_text(json.dumps(payload, indent=2) + '\n')
    write_csv(csv_path, results)
    write_markdown(md_path, results, comparison)

    print(f'[Benchmark] Wrote {json_path}', flush=True)
    print(f'[Benchmark] Wrote {csv_path}', flush=True)
    print(f'[Benchmark] Wrote {md_path}', flush=True)


if __name__ == '__main__':
    main()
