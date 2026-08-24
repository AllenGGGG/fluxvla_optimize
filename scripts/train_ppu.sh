#!/usr/bin/env bash
# Launch PI0.5 training on Alibaba Cloud PPU. A100 settings stay in train.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage: bash scripts/train_ppu.sh [options] [--cfg-options key=value ...]

Defaults:
  dataset roots       /mnt/cpfs_data/allen/dataset/2026_07_23
                      /mnt/cpfs_data/allen/dataset/2026_07_24
                      /mnt/cpfs_data/allen/dataset/2026_07_25
                      /mnt/cpfs_data/allen/dataset/2026_07_27
                      /mnt/cpfs_data/allen/dataset/2026_07_29
  base model dir      <repo>/checkpoints/pi05_base
  pretrained weights /mnt/cpfs_data/allen/checkpoints/2026_07_29/checkpoints/step-020088-epoch-04-loss=0.0041.safetensors
  checkpoint dir      /mnt/cpfs_data/allen/checkpoints/2026_07_30
  log dir             /mnt/cpfs_data/allen/logs/2026_07_30
  training            16 PPUs, batch 20/device, 7 epochs, lr 3e-5, BF16/SDPA

Options:
  --dataset PATH[,PATH...]    Dataset root(s). Each root must contain meta/.
  --base-model-dir PATH       PI0.5 tokenizer/config directory.
  --base-weights PATH         Initial model checkpoint (.pt/.safetensors).
  --work-dir PATH             Checkpoint/config/statistics directory.
  --log-dir PATH              Metric/jsonl/wandb log directory.
  --batch-size N              Per-device batch size.
  --epochs N                  Max training epochs.
  --lr LR, --learning-rate LR Optimizer learning rate.
  --max-keep-ckpts N          Number of recent checkpoints to keep.
  --save-pt                   Also save legacy .pt checkpoints.
  --nproc-per-node N          PPU devices/processes per node.
  --cuda-visible-devices CSV  CUDA-compatible PPU device list.
  --master-port PORT          torchrun master port.
  --resume-from PATH          Resume from a .pt checkpoint.
  --wandb-mode MODE           online, offline, or disabled.
  --background                Run detached and save PID (default).
  --foreground                Run in the current terminal for debugging.
  --help                      Show this help.

Everything after --cfg-options is passed to scripts/train.py --cfg-options
after these defaults, so it can override any config key.

Environment overrides:
  PPU_ENV_FILE                Default: /usr/local/PPU_SDK/envsetup.sh
  FLUXVLA_ENV_PREFIX          Default: /opt/conda/envs/fluxvla
EOF
}

CONFIG="configs/pi05/pi05_parcel_sort_ppu.py"
DATASET_ROOTS=(
  "/mnt/cpfs_data/allen/dataset/2026_07_23"
  "/mnt/cpfs_data/allen/dataset/2026_07_24"
  "/mnt/cpfs_data/allen/dataset/2026_07_25"
  "/mnt/cpfs_data/allen/dataset/2026_07_27"
  "/mnt/cpfs_data/allen/dataset/2026_07_29"
)
BASE_MODEL_DIR="${REPO_ROOT}/checkpoints/pi05_base"
BASE_WEIGHTS="/mnt/cpfs_data/allen/checkpoints/2026_07_29/checkpoints/step-020088-epoch-04-loss=0.0041.safetensors"
WORK_DIR="/mnt/cpfs_data/allen/checkpoints/2026_07_30"
LOG_DIR="/mnt/cpfs_data/allen/logs/2026_07_30"
BATCH_SIZE=20
MAX_EPOCHS=7
LEARNING_RATE="3e-5"
MAX_KEEP_CKPTS=5
SAVE_PT=false
SAVE_EPOCH_INTERVAL=1
SAVE_ITER_INTERVAL=5000

NPROC_PER_NODE=16
WORLD_SIZE=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=29500
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
FLUXVLA_ENV_PREFIX="${FLUXVLA_ENV_PREFIX:-/opt/conda/envs/fluxvla}"
PPU_ENV_FILE="${PPU_ENV_FILE:-/usr/local/PPU_SDK/envsetup.sh}"
WANDB_MODE_VALUE="${WANDB_MODE:-offline}"

RESUME_FROM="${RESUME_FROM:-}"
BACKGROUND=true
ORIGINAL_ARGS=("$@")
EXTRA_CFG_OPTIONS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --dataset) IFS=',' read -r -a DATASET_ROOTS <<< "$2"; shift 2 ;;
    --base-model-dir) BASE_MODEL_DIR="$2"; shift 2 ;;
    --base-weights) BASE_WEIGHTS="$2"; shift 2 ;;
    --work-dir) WORK_DIR="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --batch-size|--bs) BATCH_SIZE="$2"; shift 2 ;;
    --epochs|--max-epochs) MAX_EPOCHS="$2"; shift 2 ;;
    --lr|--learning-rate) LEARNING_RATE="$2"; shift 2 ;;
    --max-keep-ckpts) MAX_KEEP_CKPTS="$2"; shift 2 ;;
    --save-pt) SAVE_PT=true; shift ;;
    --save-epoch-interval) SAVE_EPOCH_INTERVAL="$2"; shift 2 ;;
    --save-iter-interval) SAVE_ITER_INTERVAL="$2"; shift 2 ;;
    --nproc-per-node) NPROC_PER_NODE="$2"; shift 2 ;;
    --cuda-visible-devices|--devices) CUDA_VISIBLE_DEVICES_VALUE="$2"; shift 2 ;;
    --master-port) MASTER_PORT="$2"; shift 2 ;;
    --master-addr) MASTER_ADDR="$2"; shift 2 ;;
    --resume-from) RESUME_FROM="$2"; shift 2 ;;
    --wandb-mode) WANDB_MODE_VALUE="$2"; shift 2 ;;
    --background) BACKGROUND=true; shift ;;
    --foreground) BACKGROUND=false; shift ;;
    --cfg-options)
      shift
      EXTRA_CFG_OPTIONS+=("$@")
      break
      ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${BASE_WEIGHTS}" ]]; then
  BASE_WEIGHTS="${BASE_MODEL_DIR}/model.safetensors"
elif [[ -d "${BASE_WEIGHTS}" ]]; then
  BASE_MODEL_DIR="${BASE_WEIGHTS}"
  BASE_WEIGHTS="${BASE_WEIGHTS}/model.safetensors"
fi

if [[ ! -f "${BASE_WEIGHTS}" ]]; then
  echo "Base weights not found: ${BASE_WEIGHTS}" >&2
  exit 1
fi
if [[ ! -d "${BASE_MODEL_DIR}" ]]; then
  echo "Base model dir not found: ${BASE_MODEL_DIR}" >&2
  exit 1
fi
for DATASET_ROOT in "${DATASET_ROOTS[@]}"; do
  if [[ ! -d "${DATASET_ROOT}/meta" ]]; then
    echo "Dataset root must contain meta/: ${DATASET_ROOT}" >&2
    exit 1
  fi
done
if [[ ! -x "${FLUXVLA_ENV_PREFIX}/bin/python" ]]; then
  echo "PPU environment Python not executable: ${FLUXVLA_ENV_PREFIX}/bin/python" >&2
  exit 1
fi
if [[ ! -x "${FLUXVLA_ENV_PREFIX}/bin/torchrun" ]]; then
  echo "PPU environment torchrun not executable: ${FLUXVLA_ENV_PREFIX}/bin/torchrun" >&2
  exit 1
fi
if [[ ! -f "${PPU_ENV_FILE}" ]]; then
  echo "PPU SDK environment not found: ${PPU_ENV_FILE}" >&2
  exit 1
fi

# The vendor script references unset variables, so temporarily relax -u.
set +u
# shellcheck disable=SC1090
source "${PPU_ENV_FILE}" "" >/dev/null
set -u

export PATH="${FLUXVLA_ENV_PREFIX}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"
export WANDB_MODE="${WANDB_MODE_VALUE}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"

CONDA_LIBSTDCXX="${FLUXVLA_ENV_PREFIX}/lib/libstdc++.so.6"
if [[ ! -f "${CONDA_LIBSTDCXX}" ]]; then
  echo "Conda C++ runtime not found: ${CONDA_LIBSTDCXX}" >&2
  exit 1
fi
case ":${LD_PRELOAD:-}:" in
  *":${CONDA_LIBSTDCXX}:"*) ;;
  *) export LD_PRELOAD="${CONDA_LIBSTDCXX}${LD_PRELOAD:+:${LD_PRELOAD}}" ;;
esac

EXPECTED_DEVICE_COUNT="${NPROC_PER_NODE}" \
  "${FLUXVLA_ENV_PREFIX}/bin/python" - <<'PY'
import os

import av
import torch
import torch.nn.functional as F

expected = int(os.environ['EXPECTED_DEVICE_COUNT'])
count = torch.cuda.device_count()
if not torch.cuda.is_available() or count < expected:
    raise SystemExit(
        f'PPU preflight failed: requested {expected} process(es), but the '
        f'PyTorch CUDA compatibility API reports {count} device(s).')
if not torch.distributed.is_nccl_available():
    raise SystemExit('PPU preflight failed: NCCL/PCCL backend is unavailable.')
if not hasattr(F, 'scaled_dot_product_attention'):
    raise SystemExit('PPU preflight failed: PyTorch SDPA is unavailable.')
bf16_check = getattr(torch.cuda, 'is_bf16_supported', None)
if callable(bf16_check) and not bf16_check():
    raise SystemExit('PPU preflight failed: BF16 is unavailable.')
print(
    f'PPU preflight OK: torch={torch.__version__}, devices={count}, '
    f'device={torch.cuda.get_device_name(0)}, precision=bf16 mixed, '
    f'attention=sdpa, nccl/pccl=yes, pyav={av.__version__}')
PY

USE_PI05_BASE_MAPPING=false
if [[ "$(readlink -f "${BASE_WEIGHTS}")" == "$(readlink -f "${BASE_MODEL_DIR}/model.safetensors" 2>/dev/null || true)" ]]; then
  USE_PI05_BASE_MAPPING=true
fi

RESUME_ARGS=()
if [[ -n "${RESUME_FROM}" ]]; then
  RESUME_ARGS=(--resume-from "${RESUME_FROM}")
fi

DATASET_CFG_VALUE="["
for DATASET_ROOT in "${DATASET_ROOTS[@]}"; do
  DATASET_CFG_VALUE+="\"${DATASET_ROOT}\","
done
DATASET_CFG_VALUE="${DATASET_CFG_VALUE%,}]"

CFG_OPTIONS=(
  'runner.metric.active_trackers=("jsonl","wandb")'
  "model.pretrained_name_or_path=${BASE_WEIGHTS}"
  "train_dataloader.per_device_batch_size=${BATCH_SIZE}"
  "train_dataloader.dataset.datasets.0.data_root_path=${DATASET_CFG_VALUE}"
  "train_dataloader.dataset.datasets.0.transforms.3.tokenizer.model_path=${BASE_MODEL_DIR}"
  "runner.tokenizer.model_path=${BASE_MODEL_DIR}"
  "runner.max_epochs=${MAX_EPOCHS}"
  "runner.optimizer.lr=${LEARNING_RATE}"
  "runner.max_keep_ckpts=${MAX_KEEP_CKPTS}"
  "runner.save_epoch_interval=${SAVE_EPOCH_INTERVAL}"
  "runner.save_iter_interval=${SAVE_ITER_INTERVAL}"
)
if [[ "${USE_PI05_BASE_MAPPING}" != "true" ]]; then
  CFG_OPTIONS+=("model.name_mapping=None")
  CFG_OPTIONS+=("model.strict_mapping=True")
fi
if [[ "${SAVE_PT}" == "true" ]]; then
  CFG_OPTIONS+=("runner.save_pt_checkpoints=True")
else
  CFG_OPTIONS+=("runner.save_pt_checkpoints=False")
fi
CFG_OPTIONS+=("${EXTRA_CFG_OPTIONS[@]}")

mkdir -p "${WORK_DIR}" "${LOG_DIR}"

if [[ "${BACKGROUND}" == "true" && -z "${FLUXVLA_PPU_TRAIN_DAEMON_CHILD:-}" ]]; then
  CONSOLE_LOG="${LOG_DIR}/console.log"
  PID_FILE="${LOG_DIR}/train.pid"
  setsid nohup env FLUXVLA_PPU_TRAIN_DAEMON_CHILD=1 \
    bash "${SCRIPT_DIR}/train_ppu.sh" "${ORIGINAL_ARGS[@]}" \
    >"${CONSOLE_LOG}" 2>&1 < /dev/null &
  TRAIN_PID=$!
  printf '%s\n' "${TRAIN_PID}" > "${PID_FILE}"
  echo "Detached PPU training started."
  echo "PID: ${TRAIN_PID}"
  echo "Console log: ${CONSOLE_LOG}"
  echo "PID file: ${PID_FILE}"
  exit 0
fi

echo "Config: ${CONFIG}"
echo "Datasets:"
printf '  %s\n' "${DATASET_ROOTS[@]}"
echo "Base weights: ${BASE_WEIGHTS}"
echo "Use PI0.5 base name mapping: ${USE_PI05_BASE_MAPPING}"
echo "Checkpoint dir: ${WORK_DIR}"
echo "Log dir: ${LOG_DIR}"
echo "Batch size: ${BATCH_SIZE}"
echo "Max epochs: ${MAX_EPOCHS}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Keep checkpoints: ${MAX_KEEP_CKPTS}"
echo "Save .pt checkpoints: ${SAVE_PT}"

torchrun \
  --nproc-per-node="${NPROC_PER_NODE}" \
  --nnodes="${WORLD_SIZE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "scripts/train.py" \
  --config "${CONFIG}" \
  --work-dir "${WORK_DIR}" \
  --log-dir "${LOG_DIR}" \
  "${RESUME_ARGS[@]}" \
  --cfg-options "${CFG_OPTIONS[@]}"
