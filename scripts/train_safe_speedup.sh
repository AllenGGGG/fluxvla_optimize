#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/pi05/pi05_task0_safe_speedup.py}
WORK_DIR=${2:-work_dirs/pi05_task0_safe_speedup}

setsid nohup env \
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
NPROC_PER_NODE=6 \
MASTER_ADDR=127.0.0.1 \
MASTER_PORT=29725 \
WANDB_MODE=disabled \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash scripts/train.sh \
"${CONFIG}" \
"${WORK_DIR}" \
--cfg-options \
model.pretrained_name_or_path=checkpoints/pi05_base/model.safetensors \
train_dataloader.per_device_batch_size=24 \
runner.learning_rate=5e-5 \
runner.max_epochs=300 \
> logs/pi05_task0_safe_speedup.log 2>&1 < /dev/null &

echo "Started safe-speedup training."
echo "Config: ${CONFIG}"
echo "Work dir: ${WORK_DIR}"
echo "Log: logs/pi05_task0_safe_speedup.log"
