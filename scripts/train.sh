#!/bin/bash
# Unified launcher for single-node or multi-node distributed training.
# Auto-detects common distributed environment variable conventions:
#   - Standard torchrun / ali-style: NPROC_PER_NODE, WORLD_SIZE, RANK,
#     MASTER_ADDR, MASTER_PORT
#   - Vol-platform: MLP_WORKER_GPU, MLP_WORKER_NUM, MLP_ROLE_INDEX,
#     MLP_WORKER_0_HOST, MLP_WORKER_0_PORT
# Falls back to a sensible single-node default when none are set.
#
# WANDB_API_KEY must be set in the calling shell environment (e.g. in your
# ~/.bashrc or CI secrets) -- never hardcode it here, this file is meant
# to be committed -- unless WANDB_MODE is set to something other than
# "online" (e.g. "disabled" or "offline").

set -euo pipefail

WANDB_MODE="${WANDB_MODE:-online}"
if [[ "${WANDB_MODE}" == "online" ]]; then
  : "${WANDB_API_KEY:?WANDB_API_KEY is not set. Export it in your shell before running this script, or set WANDB_MODE=offline/disabled to skip wandb.}"
  export WANDB_API_KEY
fi
export WANDB_MODE

export PATH=/data/wqz/miniconda/envs/fluxvla/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONFIG=${1:-"configs/gr00t/gr00t_eagle_3b_libero_10_full_finetune.py"}
WORK_DIR=${2:-"work_dirs/gr00t_eagle_3b_libero_10_full_finetune"}

NPROC_PER_NODE="${NPROC_PER_NODE:-${MLP_WORKER_GPU:-1}}"
WORLD_SIZE="${WORLD_SIZE:-${MLP_WORKER_NUM:-1}}"
NODE_RANK="${RANK:-${MLP_ROLE_INDEX:-0}}"
MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-localhost}}"
MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-29500}}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES

torchrun \
  --nproc-per-node="${NPROC_PER_NODE}" \
  --nnodes="${WORLD_SIZE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "scripts/train.py" \
  --config "${CONFIG}" \
  --work-dir "${WORK_DIR}" \
  ${@:3}
