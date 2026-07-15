#!/bin/bash
# Single-node, 8-GPU launcher for pi05_parcel_sort training.

set -euo pipefail

export WANDB_API_KEY=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
if [[ -z "${WANDB_API_KEY}" ]]; then
  echo "WANDB_API_KEY is empty -- set it in scripts/train.sh before running." >&2
  exit 1
fi
export WANDB_MODE=online

FLUXVLA_ENV_PREFIX=/data/wqz/miniconda/envs/fluxvla_p1p99
export PATH="${FLUXVLA_ENV_PREFIX}/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

CONFIG="configs/pi05/pi05_parcel_sort.py"
WORK_DIR="work_dirs/pi05_parcel_sort"

NPROC_PER_NODE=8
WORLD_SIZE=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=29500

RESUME_ARGS=()
if [[ -n "${RESUME_FROM:-}" ]]; then
  RESUME_ARGS=(--resume-from "${RESUME_FROM}")
fi

torchrun \
  --nproc-per-node="${NPROC_PER_NODE}" \
  --nnodes="${WORLD_SIZE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "scripts/train.py" \
  --config "${CONFIG}" \
  --work-dir "${WORK_DIR}" \
  "${RESUME_ARGS[@]}" \
  --cfg-options 'runner.metric.active_trackers=("jsonl","wandb")' \
  'model.pretrained_name_or_path=./checkpoints/pi05_base/model.safetensors'
