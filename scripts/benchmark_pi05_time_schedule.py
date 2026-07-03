#!/usr/bin/env python3
"""Benchmark PI0.5 accelerated inference with custom denoising time schedules.

Examples:
    CUDA_VISIBLE_DEVICES=1 python scripts/benchmark_pi05_time_schedule.py

    CUDA_VISIBLE_DEVICES=1 python scripts/benchmark_pi05_time_schedule.py \
        --ckpt /path/to/pi05_libero10_task0_with_l_pixshuffle_mlp_v1.safetensors \
        --scenario custom:1,0.9,0.5,0.4
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from mmengine import Config
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/numba_cache_fluxvla')

import fluxvla.models  # noqa: F401
from fluxvla.engines.utils.builder import build_vla_from_cfg
from fluxvla.models.vlas.pi05_flowmatching_time_schedule_inference import (  # noqa: F401,E501
    PI05FlowMatchingTimeScheduleInference,
)


DEFAULT_CONFIG = (
    'configs/pi05/pi05_libero10_task0_with_l_pixshuffle_mlp_inference.py')
DEFAULT_CKPT = (
    'work_dirs/pi05_libero10_task0_with_l_pixshuffle_mlp/checkpoints/'
    'step-016560-epoch-240-loss=0.0315.safetensors')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Benchmark custom PI0.5 denoising time schedules.')
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--ckpt', default=DEFAULT_CKPT)
    parser.add_argument(
        '--scenario',
        action='append',
        default=None,
        help=(
            'Schedule scenario. Built-ins: baseline10, uniform5. Custom: '
            'name:t0,t1,... or just t0,t1,... . Can be repeated.'))
    parser.add_argument('--warmup-iters', type=int, default=5)
    parser.add_argument('--bench-iters', type=int, default=30)
    parser.add_argument('--prompt-len', type=int, default=32)
    parser.add_argument('--num-views', type=int, default=2)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--state-dim', type=int, default=32)
    parser.add_argument('--action-dim', type=int, default=32)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--output-json', default=None)
    return parser.parse_args()


def parse_float_list(value: str) -> List[float]:
    items = [item.strip() for item in value.split(',') if item.strip()]
    if not items:
        raise ValueError('empty time schedule')
    return [float(item) for item in items]


def normalize_schedule(schedule: List[float]) -> Tuple[List[float], List[float]]:
    deltas = [
        float(next_time - current_time)
        for current_time, next_time in zip(schedule, schedule[1:] + [0.0])
    ]
    return schedule, deltas


def builtin_scenario(name: str) -> Tuple[str, Optional[List[float]]]:
    if name == 'baseline10':
        return 'baseline10', None
    if name == 'uniform5':
        return 'uniform5', [1.0, 0.8, 0.6, 0.4, 0.2]
    raise ValueError(f'Unknown built-in scenario: {name}')


def parse_scenarios(values: Optional[List[str]]) -> List[Tuple[str, Optional[List[float]]]]:
    if not values:
        return [
            ('baseline10', None),
            ('uniform5', [1.0, 0.8, 0.6, 0.4, 0.2]),
            ('custom_1_0_9_0_5_0_4', [1.0, 0.9, 0.5, 0.4]),
        ]

    scenarios = []
    for value in values:
        if ',' not in value and ':' not in value:
            scenarios.append(builtin_scenario(value))
            continue
        if ':' in value:
            name, raw_schedule = value.split(':', 1)
            name = name.strip() or 'custom'
        else:
            raw_schedule = value
            name = 'custom_' + value.replace(',', '_').replace('.', '_')
        scenarios.append((name, parse_float_list(raw_schedule)))
    return scenarios


def bytes_to_gib(value: int) -> float:
    return value / (1024**3)


def load_state_dict(ckpt_path: str) -> Dict[str, torch.Tensor]:
    if ckpt_path.endswith('.safetensors'):
        return load_file(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location='cpu', mmap=True)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        return checkpoint['model']
    return checkpoint


def make_batch(args: argparse.Namespace, action_steps: int,
               device: torch.device) -> Dict[str, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(args.seed)
    images = torch.randn(
        1,
        args.num_views * 3,
        args.image_size,
        args.image_size,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
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
        dtype=torch.bfloat16,
    )
    img_masks = torch.ones(1, args.num_views, device=device, dtype=torch.bool)
    lang_masks = torch.ones(1, args.prompt_len, device=device, dtype=torch.bool)
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


def build_model(config_path: str, ckpt_path: str,
                device: torch.device) -> torch.nn.Module:
    cfg = Config.fromfile(config_path)
    model_cfg = cfg.inference_model.copy()
    model_cfg.type = 'PI05FlowMatchingTimeScheduleInference'
    model = build_vla_from_cfg(model_cfg).to(device).eval()
    state_dict = load_state_dict(ckpt_path)
    model.load_state_dict(state_dict, strict=True)
    del state_dict
    gc.collect()
    return model


def prepare_model_for_schedule(model: torch.nn.Module,
                               schedule: Optional[List[float]]) -> None:
    num_steps = len(schedule) if schedule is not None else 10
    model.num_steps = num_steps
    model.time_schedule = schedule
    model.time_deltas = None
    model.prepare_triton_inference(
        num_views=getattr(model, 'num_views', 2),
        max_prompt_len=getattr(model, 'triton_max_prompt_len', 48),
        chunk_size=model.n_action_steps,
        num_steps=num_steps,
    )
    model._triton_ready = True


def timed_predict(model: torch.nn.Module,
                  batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.inference_mode():
        output = model.predict_action(**batch)
    end.record()
    torch.cuda.synchronize()
    return output, start.elapsed_time(end)


def run_scenario(args: argparse.Namespace, model: torch.nn.Module, name: str,
                 schedule: Optional[List[float]],
                 device: torch.device,
                 reference: Optional[torch.Tensor]) -> Dict[str, object]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    print(f'[scenario] preparing {name}', flush=True)
    prepare_model_for_schedule(model, schedule)
    model_allocated = bytes_to_gib(torch.cuda.memory_allocated(device))
    action_steps = int(getattr(model, 'n_action_steps', 10))
    batch = make_batch(args, action_steps, device)

    print(f'[scenario] running {name}', flush=True)
    output, cold_start_ms = timed_predict(model, batch)
    for _ in range(args.warmup_iters):
        output, _ = timed_predict(model, batch)

    torch.cuda.reset_peak_memory_stats(device)
    latencies = []
    for _ in range(args.bench_iters):
        output, latency_ms = timed_predict(model, batch)
        latencies.append(latency_ms)

    peak_allocated = bytes_to_gib(torch.cuda.max_memory_allocated(device))
    total_mem = bytes_to_gib(torch.cuda.get_device_properties(device).total_memory)
    mean_ms = statistics.fmean(latencies)
    std_ms = statistics.pstdev(latencies) if len(latencies) > 1 else 0.0

    active_schedule = (
        list(getattr(model, '_decoder_time_schedule', ())) or
        [1.0 - i / int(getattr(model, '_num_steps', 10))
         for i in range(int(getattr(model, '_num_steps', 10)))]
    )
    active_deltas = list(getattr(model, '_decoder_time_deltas', ()))

    rel_err = None
    if reference is not None and reference.shape == output.shape:
        diff = (output.detach().float().cpu() - reference).abs()
        rel_err = float(diff.max() / (reference.abs().max() + 1e-6))

    result = {
        'name': name,
        'enabled': True,
        'num_steps': int(getattr(model, '_num_steps', getattr(model, 'num_steps', 10))),
        'time_schedule': active_schedule,
        'time_deltas': active_deltas,
        'cold_start_ms': cold_start_ms,
        'mean_ms': mean_ms,
        'std_ms': std_ms,
        'hz': 1000.0 / mean_ms,
        'model_allocated_gib': model_allocated,
        'peak_allocated_gib': peak_allocated,
        'mem_pct': peak_allocated / total_mem * 100.0,
        'output_shape': list(output.shape),
        'relative_error_vs_baseline10': rel_err,
        '_output_cpu': output.detach().float().cpu(),
    }

    del batch, output
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    args = parse_args()
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    scenarios = parse_scenarios(args.scenario)
    print('[setup] loading model once', flush=True)
    model = build_model(args.config, args.ckpt, device)
    print('[setup] model loaded', flush=True)
    rows = []
    reference = None
    for name, schedule in scenarios:
        try:
            row = run_scenario(args, model, name, schedule, device, reference)
            if reference is None and name == 'baseline10':
                reference = row['_output_cpu']
                row['relative_error_vs_baseline10'] = 0.0
            rows.append(row)
        except Exception as exc:
            rows.append({'name': name, 'enabled': False, 'error': repr(exc)})

    print('\nPI0.5 time schedule benchmark')
    print(f'config: {args.config}')
    print(f'ckpt:   {args.ckpt}')
    print(f'gpu:    {torch.cuda.get_device_name(device)}')
    print()
    print(f'{"scenario":<24} {"ok":<4} {"steps":<5} {"schedule":<24} '
          f'{"peak mem":<14} {"latency":<17} {"rel err":<10}')
    print('-' * 108)
    for row in rows:
        if not row['enabled']:
            print(f'{row["name"]:<24} no   -     -                        '
                  f'-              -                 {row["error"]}')
            continue
        schedule_s = ','.join(f'{value:g}' for value in row['time_schedule'])
        if len(schedule_s) > 23:
            schedule_s = schedule_s[:20] + '...'
        rel = row['relative_error_vs_baseline10']
        rel_s = 'baseline' if rel == 0.0 else (
            f'{rel * 100:.2f}%' if rel is not None else 'shape diff')
        print(f'{row["name"]:<24} yes  {row["num_steps"]:<5} '
              f'{schedule_s:<24} '
              f'{row["peak_allocated_gib"]:.1f}GB ({row["mem_pct"]:.0f}%)   '
              f'{row["mean_ms"]:.2f}ms / {row["hz"]:.1f}Hz   {rel_s:<10}')

    serializable = []
    for row in rows:
        clean = dict(row)
        clean.pop('_output_cpu', None)
        serializable.append(clean)
    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(serializable, indent=2, ensure_ascii=False) + '\n')
        print(f'\nwrote {output_path}')
    return 0 if all(row['enabled'] for row in rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
