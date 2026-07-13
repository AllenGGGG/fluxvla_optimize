#!/bin/bash
# Launch pi05_parcel_sort FSDP training (8 GPUs, per-device bs=32) with
# wandb logging enabled.
#
# WANDB_API_KEY must be set in the calling shell environment (e.g. in your
# ~/.bashrc or CI secrets) -- never hardcode it here, this file is meant
# to be committed.

set -euo pipefail

: "${WANDB_API_KEY:?WANDB_API_KEY is not set. Export it in your shell before running this script.}"
export WANDB_API_KEY
export WANDB_MODE=online

export PATH=/data/wqz/miniconda/envs/fluxvla/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export CUDA_VISIBLE_DEVICES

CONFIG=${1:-configs/pi05/pi05_parcel_sort.py}
WORK_DIR=${2:-work_dirs/pi05_parcel_sort}

NPROC_PER_NODE="${NPROC_PER_NODE}" WORLD_SIZE=1 \
  bash scripts/train.sh "${CONFIG}" "${WORK_DIR}" "${@:3}"
