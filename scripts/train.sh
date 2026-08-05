#!/usr/bin/env bash
# Launch pi05 training. Defaults are intentionally overridable from CLI.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage: bash scripts/train.sh [options] [--cfg-options key=value ...]

Defaults:
  Defaults are defined in the variables below.

Options:
  --dataset PATH[,PATH...]    Dataset root(s). Each root must contain meta/.
                              Comma-separated paths are supported.
  --base-model-dir PATH       pi05 tokenizer/config directory.
  --base-weights PATH         pi05 model safetensors file or containing dir.
  --work-dir PATH             Checkpoint/config/statistics directory.
  --log-dir PATH              Metric/jsonl/wandb log directory.
  --batch-size N              Per-device batch size.
  --epochs N                  Max training epochs.
  --lr LR, --learning-rate LR Optimizer learning rate.
  --max-keep-ckpts N          Number of recent checkpoints to keep.
  --save-pt                  Also save .pt checkpoints.
  --nproc-per-node N          GPUs/processes per node.
  --cuda-visible-devices CSV  CUDA_VISIBLE_DEVICES value.
  --master-port PORT          torchrun master port.
  --resume-from PATH          Resume from a .pt checkpoint.
  --wandb-mode MODE           online, offline, or disabled.
  --background                Run detached with setsid/nohup and save PID.
                              This is the default.
  --foreground                Run in the current terminal for debugging.
  --help                      Show this help.

Everything after --cfg-options is passed to scripts/train.py --cfg-options
after these defaults, so it can override any config key.
EOF
}

CONFIG="configs/pi05/pi05_parcel_sort.py"
DATASET_ROOTS=(
  "/data/guohao/FluxVLA_workdirs/dataset/2026_07_24"
  "/data/guohao/FluxVLA_workdirs/dataset/2026_07_25"
  "/data/guohao/FluxVLA_workdirs/dataset/2026_07_27"
  "/data/guohao/FluxVLA_workdirs/dataset/2026_07_28"
  "/data/guohao/FluxVLA_workdirs/dataset/2026_07_30"
  "/data/guohao/FluxVLA_workdirs/dataset/2026_08_03"
)
BASE_MODEL_DIR="/home/guohao/FluxVLA/checkpoints/pi05_base"
BASE_WEIGHTS="/home/guohao/FluxVLA/workdirs/checkpoints/2026_08_01/checkpoints/step-034220-epoch-05-loss=0.0028.safetensors"
WORK_DIR="/data/guohao/FluxVLA_workdirs/checkpoints/2026_08_03"
LOG_DIR="/data/guohao/FluxVLA_workdirs/logs/2026_08_03"
BATCH_SIZE=32
MAX_EPOCHS=5
LEARNING_RATE="1e-5"
MAX_KEEP_CKPTS=5
SAVE_PT=false
SAVE_EPOCH_INTERVAL=1
SAVE_ITER_INTERVAL=5000

NPROC_PER_NODE=8
WORLD_SIZE=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=29500
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
FLUXVLA_ENV_PREFIX="${FLUXVLA_ENV_PREFIX:-/home/guohao/miniconda3/envs/fluxvla_train}"
WANDB_MODE_VALUE="${WANDB_MODE:-offline}"

RESUME_FROM="${RESUME_FROM:-}"
BACKGROUND=true
ORIGINAL_ARGS=("$@")
EXTRA_CFG_OPTIONS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      usage
      exit 0
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --dataset)
      IFS=',' read -r -a DATASET_ROOTS <<< "$2"
      shift 2
      ;;
    --base-model-dir)
      BASE_MODEL_DIR="$2"
      shift 2
      ;;
    --base-weights)
      BASE_WEIGHTS="$2"
      shift 2
      ;;
    --work-dir)
      WORK_DIR="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --batch-size|--bs)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --epochs|--max-epochs)
      MAX_EPOCHS="$2"
      shift 2
      ;;
    --lr|--learning-rate)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --max-keep-ckpts)
      MAX_KEEP_CKPTS="$2"
      shift 2
      ;;
    --save-pt)
      SAVE_PT=true
      shift
      ;;
    --save-epoch-interval)
      SAVE_EPOCH_INTERVAL="$2"
      shift 2
      ;;
    --save-iter-interval)
      SAVE_ITER_INTERVAL="$2"
      shift 2
      ;;
    --nproc-per-node)
      NPROC_PER_NODE="$2"
      shift 2
      ;;
    --cuda-visible-devices)
      CUDA_VISIBLE_DEVICES_VALUE="$2"
      shift 2
      ;;
    --master-port)
      MASTER_PORT="$2"
      shift 2
      ;;
    --master-addr)
      MASTER_ADDR="$2"
      shift 2
      ;;
    --resume-from)
      RESUME_FROM="$2"
      shift 2
      ;;
    --wandb-mode)
      WANDB_MODE_VALUE="$2"
      shift 2
      ;;
    --background)
      BACKGROUND=true
      shift
      ;;
    --foreground)
      BACKGROUND=false
      shift
      ;;
    --cfg-options)
      shift
      EXTRA_CFG_OPTIONS+=("$@")
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${BASE_WEIGHTS}" ]]; then
  BASE_WEIGHTS="${BASE_MODEL_DIR}/model.safetensors"
elif [[ -d "${BASE_WEIGHTS}" ]]; then
  BASE_MODEL_DIR="${BASE_WEIGHTS}"
  BASE_WEIGHTS="${BASE_WEIGHTS}/model.safetensors"
fi

TRAIN_DATASET_ROOTS=()
for DATASET_ROOT in "${DATASET_ROOTS[@]}"; do
  TRAIN_DATASET_ROOT="${DATASET_ROOT}"
  if [[ ! -d "${TRAIN_DATASET_ROOT}/meta" && -d "${DATASET_ROOT}/2026_07_15/meta" ]]; then
    TRAIN_DATASET_ROOT="${DATASET_ROOT}/2026_07_15"
  fi
  TRAIN_DATASET_ROOTS+=("${TRAIN_DATASET_ROOT}")
done

if [[ ! -f "${BASE_WEIGHTS}" ]]; then
  echo "Base weights not found: ${BASE_WEIGHTS}" >&2
  exit 1
fi
if [[ ! -d "${BASE_MODEL_DIR}" ]]; then
  echo "Base model dir not found: ${BASE_MODEL_DIR}" >&2
  exit 1
fi
for TRAIN_DATASET_ROOT in "${TRAIN_DATASET_ROOTS[@]}"; do
  if [[ ! -d "${TRAIN_DATASET_ROOT}/meta" ]]; then
    echo "Dataset root must contain meta/: ${TRAIN_DATASET_ROOT}" >&2
    exit 1
  fi
done
if [[ ! -x "${FLUXVLA_ENV_PREFIX}/bin/python" ]]; then
  echo "Conda env python not executable: ${FLUXVLA_ENV_PREFIX}/bin/python" >&2
  exit 1
fi
if [[ ! -x "${FLUXVLA_ENV_PREFIX}/bin/torchrun" ]]; then
  echo "Conda env torchrun not executable: ${FLUXVLA_ENV_PREFIX}/bin/torchrun" >&2
  exit 1
fi

USE_PI05_BASE_MAPPING=false
if [[ "$(readlink -f "${BASE_WEIGHTS}")" == "$(readlink -f "${BASE_MODEL_DIR}/model.safetensors" 2>/dev/null || true)" ]]; then
  USE_PI05_BASE_MAPPING=true
fi

export PATH="${FLUXVLA_ENV_PREFIX}/bin:${PATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"
export WANDB_MODE="${WANDB_MODE_VALUE}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"

RESUME_ARGS=()
if [[ -n "${RESUME_FROM}" ]]; then
  RESUME_ARGS=(--resume-from "${RESUME_FROM}")
fi

DATASET_CFG_VALUE="["
for TRAIN_DATASET_ROOT in "${TRAIN_DATASET_ROOTS[@]}"; do
  DATASET_CFG_VALUE+="\"${TRAIN_DATASET_ROOT}\","
done
DATASET_CFG_VALUE="${DATASET_CFG_VALUE%,}]"

CFG_OPTIONS=(
  'runner.metric.active_trackers=("jsonl","wandb")'
  "model.pretrained_name_or_path=${BASE_WEIGHTS}"
  "inference_model.pretrained_name_or_path=${BASE_WEIGHTS}"
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
  CFG_OPTIONS+=("inference_model.name_mapping=None")
  CFG_OPTIONS+=("model.strict_mapping=True")
  CFG_OPTIONS+=("inference_model.strict_mapping=True")
fi
if [[ "${SAVE_PT}" == "true" ]]; then
  CFG_OPTIONS+=("runner.save_pt_checkpoints=True")
else
  CFG_OPTIONS+=("runner.save_pt_checkpoints=False")
fi
CFG_OPTIONS+=("${EXTRA_CFG_OPTIONS[@]}")

mkdir -p "${WORK_DIR}" "${LOG_DIR}"

if [[ "${BACKGROUND}" == "true" && -z "${FLUXVLA_TRAIN_DAEMON_CHILD:-}" ]]; then
  CONSOLE_LOG="${LOG_DIR}/console.log"
  PID_FILE="${LOG_DIR}/train.pid"
  setsid nohup env FLUXVLA_TRAIN_DAEMON_CHILD=1 bash "${SCRIPT_DIR}/train.sh" \
    "${ORIGINAL_ARGS[@]}" >"${CONSOLE_LOG}" 2>&1 < /dev/null &
  TRAIN_PID=$!
  printf '%s\n' "${TRAIN_PID}" > "${PID_FILE}"
  echo "Detached training started."
  echo "PID: ${TRAIN_PID}"
  echo "Console log: ${CONSOLE_LOG}"
  echo "PID file: ${PID_FILE}"
  exit 0
fi

echo "Config: ${CONFIG}"
echo "Datasets:"
printf '  %s\n' "${TRAIN_DATASET_ROOTS[@]}"
echo "Base weights: ${BASE_WEIGHTS}"
echo "Use pi05 base name mapping: ${USE_PI05_BASE_MAPPING}"
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
