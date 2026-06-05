# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0

"""Launch PI0.5 inference-memory tests on multiple GPUs.

Edit the CONFIG / RUNS section below, then run this file directly with the
project conda environment. The child processes do not save prediction results.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


# =========================
# Edit this section only.
# =========================

CONFIG = 'configs/pi05/pi05_paligemma_libero_10_full_finetune.py'
CKPT_PATH = (
    'checkpoints/pi05_paligemma_libero_10_full_finetune_bs64/'
    'pi05_paligemma_libero_10_full_finetune_bs64/checkpoints/'
    'step-038064-epoch-24-loss=0.0170.safetensors'
)

DTYPE = 'bf16'
LOOP_SECONDS = 120
HOLD_SECONDS = 30
WARMUP_ITERS = 2

# One process per entry. Change GPU ids / batch sizes here.
RUNS = [
    {'gpu': 0, 'batch_size': 8},
    {'gpu': 1, 'batch_size': 12},
    {'gpu': 2, 'batch_size': 16},
    {'gpu': 3, 'batch_size': 24},
    {'gpu': 4, 'batch_size': 32},
]

# =========================
# Usually no need to edit below.
# =========================


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    worker_script = repo_root / 'scripts' / 'measure_pi05_inference_memory.py'
    python_bin = sys.executable

    processes = []
    for run in RUNS:
        gpu = str(run['gpu'])
        batch_size = str(run['batch_size'])
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = gpu

        cmd = [
            python_bin,
            str(worker_script),
            '--config',
            CONFIG,
            '--ckpt-path',
            CKPT_PATH,
            '--batch-size',
            batch_size,
            '--device',
            'cuda:0',
            '--dtype',
            DTYPE,
            '--warmup-iters',
            str(WARMUP_ITERS),
            '--loop-seconds',
            str(LOOP_SECONDS),
            '--hold-seconds',
            str(HOLD_SECONDS),
        ]
        print(
            f'[Launch] physical_gpu={gpu} batch_size={batch_size} '
            f'cmd={" ".join(cmd)}',
            flush=True,
        )
        processes.append(subprocess.Popen(cmd, cwd=repo_root, env=env))

    print('[Launch] All jobs started. Use: watch -n 0.5 nvidia-smi',
          flush=True)

    return_codes = []
    for proc in processes:
        return_codes.append(proc.wait())

    failed = [code for code in return_codes if code != 0]
    if failed:
        raise SystemExit(f'Some jobs failed: return_codes={return_codes}')
    print('[Launch] All jobs finished.', flush=True)


if __name__ == '__main__':
    main()
