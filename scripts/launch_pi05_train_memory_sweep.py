# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0

"""Launch short PI0.5 full-finetune jobs to inspect training GPU memory.

Edit the CONFIG / RUNS section below, then run this file directly. Each entry
starts one normal single-GPU training job with a different per-device batch
size. Use ``nvidia-smi`` in another terminal to compare GPU memory.
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
WORK_DIR_ROOT = 'work_dirs/pi05_train_memory_sweep'

# Short normal-training run. Increase this if you want more time to observe.
MAX_STEPS = 20

# Keep this high so the short memory test does not write large checkpoints.
SAVE_ITER_INTERVAL = 100000

# Set to 0 to reduce CPU/DataLoader noise during a GPU-memory sweep. Change to
# 4 if you want to match the config's normal dataloader worker count.
PER_DEVICE_NUM_WORKERS = 0

# One process per entry. Change GPU ids / per-device batch sizes here.
RUNS = [
    {'gpu': 3, 'batch_size': 8},
    {'gpu': 4, 'batch_size': 12},
    {'gpu': 5, 'batch_size': 16},
    {'gpu': 6, 'batch_size': 20},
    {'gpu': 7, 'batch_size': 24},
]

# =========================
# Usually no need to edit below.
# =========================


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    train_script = repo_root / 'scripts' / 'train.py'
    python_bin = sys.executable

    processes = []
    for idx, run in enumerate(RUNS):
        gpu = str(run['gpu'])
        batch_size = str(run['batch_size'])
        master_port = str(29600 + idx)
        work_dir = str(
            repo_root / WORK_DIR_ROOT / f'gpu{gpu}_bs{batch_size}')

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = gpu
        env.setdefault('WANDB_MODE', 'disabled')
        env.setdefault('TOKENIZERS_PARALLELISM', 'false')

        cmd = [
            python_bin,
            '-m',
            'torch.distributed.run',
            '--nnodes',
            '1',
            '--nproc-per-node',
            '1',
            '--master-addr',
            '127.0.0.1',
            '--master-port',
            master_port,
            str(train_script),
            '--config',
            CONFIG,
            '--work-dir',
            work_dir,
            '--cfg-options',
            f'train_dataloader.per_device_batch_size={batch_size}',
            f'train_dataloader.per_device_num_workers={PER_DEVICE_NUM_WORKERS}',
            'runner.max_epochs=None',
            f'runner.max_steps={MAX_STEPS}',
            f'runner.save_iter_interval={SAVE_ITER_INTERVAL}',
            'runner.max_keep_ckpts=1',
        ]

        print(
            f'[Launch] physical_gpu={gpu} batch_size={batch_size} '
            f'master_port={master_port} work_dir={work_dir}',
            flush=True,
        )
        processes.append(subprocess.Popen(cmd, cwd=repo_root, env=env))

    print('[Launch] All training jobs started. Use: watch -n 0.5 nvidia-smi',
          flush=True)

    return_codes = []
    for proc in processes:
        return_codes.append(proc.wait())

    failed = [code for code in return_codes if code != 0]
    if failed:
        raise SystemExit(f'Some jobs failed: return_codes={return_codes}')
    print('[Launch] All training jobs finished.', flush=True)


if __name__ == '__main__':
    main()
