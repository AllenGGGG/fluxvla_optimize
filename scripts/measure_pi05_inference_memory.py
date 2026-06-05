# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Measure PI0.5 inference GPU memory for different synthetic batch sizes.

This script is intentionally standalone and does not save predictions. It loads
one checkpoint, builds synthetic inputs, repeatedly calls ``predict_action``,
and keeps the process alive long enough to inspect it with ``nvidia-smi``.
"""

from __future__ import annotations

import argparse
import gc
import os
import time
from typing import Dict

import torch
from mmengine import Config
from safetensors.torch import load_file


DEFAULT_CONFIG = 'configs/pi05/pi05_paligemma_libero_10_full_finetune.py'
DEFAULT_CKPT = (
    'checkpoints/pi05_paligemma_libero_10_full_finetune_bs64/'
    'pi05_paligemma_libero_10_full_finetune_bs64/checkpoints/'
    'step-038064-epoch-24-loss=0.0170.safetensors')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Measure PI0.5 predict_action memory at one batch size.')
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--ckpt-path', default=DEFAULT_CKPT)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument(
        '--dtype',
        choices=('bf16', 'fp16', 'fp32'),
        default='bf16',
        help='Model/input dtype used for inference.')
    parser.add_argument('--prompt-len', type=int, default=32)
    parser.add_argument('--num-views', type=int, default=2)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--state-dim', type=int, default=32)
    parser.add_argument('--action-dim', type=int, default=32)
    parser.add_argument('--warmup-iters', type=int, default=2)
    parser.add_argument(
        '--loop-seconds',
        type=float,
        default=120.0,
        help='Keep running inference for this many seconds.')
    parser.add_argument(
        '--hold-seconds',
        type=float,
        default=30.0,
        help='Sleep after the loop so nvidia-smi can still see the process.')
    parser.add_argument('--seed', type=int, default=7)
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    if name == 'bf16':
        return torch.bfloat16
    if name == 'fp16':
        return torch.float16
    if name == 'fp32':
        return torch.float32
    raise ValueError(f'Unsupported dtype: {name}')


def bytes_to_gib(value: int) -> float:
    return value / (1024**3)


def load_state_dict(ckpt_path: str) -> Dict[str, torch.Tensor]:
    if ckpt_path.endswith('.safetensors'):
        return load_file(ckpt_path, device='cpu')
    checkpoint = torch.load(ckpt_path, map_location='cpu', mmap=True)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        return checkpoint['model']
    return checkpoint


def build_model(config_path: str, ckpt_path: str, device: torch.device,
                dtype: torch.dtype) -> torch.nn.Module:
    import fluxvla  # noqa: F401
    from fluxvla.engines import build_vla_from_cfg

    cfg = Config.fromfile(config_path)
    model = build_vla_from_cfg(cfg.model).eval()
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
    batch_size = args.batch_size

    images = torch.randn(
        batch_size,
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
        size=(batch_size, args.prompt_len),
        generator=generator,
        device=device,
        dtype=torch.long,
    )
    states = torch.randn(
        batch_size,
        args.state_dim,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    img_masks = torch.ones(
        batch_size, args.num_views, device=device, dtype=torch.bool)
    lang_masks = torch.ones(
        batch_size, args.prompt_len, device=device, dtype=torch.bool)
    noise = torch.randn(
        batch_size,
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


def run_predict(model: torch.nn.Module, batch: Dict[str, torch.Tensor],
                dtype: torch.dtype) -> torch.Tensor:
    with torch.inference_mode(), torch.autocast(
            'cuda', dtype=dtype, enabled=dtype != torch.float32):
        return model.predict_action(**batch)


def print_memory(prefix: str, device: torch.device) -> None:
    torch.cuda.synchronize(device)
    allocated = bytes_to_gib(torch.cuda.memory_allocated(device))
    reserved = bytes_to_gib(torch.cuda.memory_reserved(device))
    peak_allocated = bytes_to_gib(torch.cuda.max_memory_allocated(device))
    peak_reserved = bytes_to_gib(torch.cuda.max_memory_reserved(device))
    print(
        f'{prefix} allocated={allocated:.3f}GiB '
        f'reserved={reserved:.3f}GiB '
        f'peak_allocated={peak_allocated:.3f}GiB '
        f'peak_reserved={peak_reserved:.3f}GiB',
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError('--batch-size must be >= 1')
    if args.loop_seconds < 0:
        raise ValueError('--loop-seconds must be >= 0')

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dtype = dtype_from_name(args.dtype)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print(
        f'[Setup] pid={os.getpid()} CUDA_VISIBLE_DEVICES='
        f'{os.environ.get("CUDA_VISIBLE_DEVICES", "")} device={device} '
        f'batch_size={args.batch_size} dtype={args.dtype}',
        flush=True,
    )

    model = build_model(args.config, args.ckpt_path, device, dtype)
    print_memory('[After model load]', device)

    action_steps = int(getattr(model, 'n_action_steps', 10))
    batch = make_synthetic_batch(args, action_steps, device, dtype)
    print_memory('[After batch alloc]', device)

    for idx in range(args.warmup_iters):
        output = run_predict(model, batch, dtype)
        print(f'[Warmup] iter={idx + 1} output_shape={tuple(output.shape)}',
              flush=True)

    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    iters = 0
    last_report = start
    output = None
    while time.perf_counter() - start < args.loop_seconds:
        output = run_predict(model, batch, dtype)
        iters += 1
        now = time.perf_counter()
        if now - last_report >= 10.0:
            elapsed = now - start
            print_memory(f'[Loop] elapsed={elapsed:.1f}s iters={iters}',
                         device)
            last_report = now

    elapsed = time.perf_counter() - start
    if output is not None:
        print(f'[Done] output_shape={tuple(output.shape)}', flush=True)
    print(f'[Done] elapsed={elapsed:.1f}s iters={iters}', flush=True)
    print_memory('[Done]', device)

    if args.hold_seconds > 0:
        print(f'[Hold] sleeping {args.hold_seconds:.1f}s for nvidia-smi',
              flush=True)
        time.sleep(args.hold_seconds)


if __name__ == '__main__':
    main()
