#!/usr/bin/env bash
# One-click PI0.5 baseline-vs-accelerated speed + success comparison.
#
# Default plan:
# - single-GPU speed benchmark on physical GPU 4
# - three-GPU concurrent speed benchmark on physical GPUs 5,6,7
# - LIBERO success evaluation on physical GPUs 5,6,7
# - write Markdown / CSV / JSON reports and full logs under one run directory

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CKPT_PATH="${CKPT_PATH:-checkpoints/pi05_paligemma_libero_10_full_finetune_bs64/pi05_paligemma_libero_10_full_finetune_bs64/checkpoints/step-038064-epoch-24-loss=0.0170.safetensors}"
BASELINE_CONFIG="${BASELINE_CONFIG:-configs/pi05/pi05_paligemma_libero_10_full_finetune.py}"
ACCELERATED_CONFIG="${ACCELERATED_CONFIG:-configs/pi05/pi05_paligemma_libero_10_full_inference.py}"

SINGLE_GPUS="${SINGLE_GPUS:-4}"
MULTI_GPUS="${MULTI_GPUS:-5,6,7}"
SUCCESS_GPUS="${SUCCESS_GPUS:-5,6,7}"
SPEED_SCENARIOS="${SPEED_SCENARIOS:-single,multi}"

SPEED_WARMUP_ITERS="${SPEED_WARMUP_ITERS:-10}"
SPEED_BENCH_ITERS="${SPEED_BENCH_ITERS:-100}"
SUCCESS_TRIALS_PER_TASK="${SUCCESS_TRIALS_PER_TASK:-50}"
SUCCESS_SEEDS="${SUCCESS_SEEDS:-7}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
PYTHON_BIN="${PYTHON_BIN:-python}"

TAG="${TAG:-formal_speed_success_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/checkpoints/pi05_paligemma_libero_10_full_finetune_bs64/pi05_paligemma_libero_10_full_finetune_bs64/comparisons/${TAG}}"
MAIN_LOG="${OUTPUT_DIR}/run.log"
METADATA_PATH="${OUTPUT_DIR}/run_metadata.txt"

csv_count() {
  local value="$1"
  if [[ -z "${value}" ]]; then
    echo 0
    return
  fi
  awk -F',' '{print NF}' <<<"${value}"
}

SUCCESS_NPROC_PER_NODE="${SUCCESS_NPROC_PER_NODE:-$(csv_count "${SUCCESS_GPUS}")}"

if [[ "${BACKGROUND:-0}" == "1" && "${_PI05_BACKGROUND_CHILD:-0}" != "1" ]]; then
  mkdir -p "${OUTPUT_DIR}"
  export _PI05_BACKGROUND_CHILD=1
  export CKPT_PATH BASELINE_CONFIG ACCELERATED_CONFIG
  export SINGLE_GPUS MULTI_GPUS SUCCESS_GPUS SPEED_SCENARIOS
  export SPEED_WARMUP_ITERS SPEED_BENCH_ITERS
  export SUCCESS_TRIALS_PER_TASK SUCCESS_SEEDS SUCCESS_NPROC_PER_NODE
  export MASTER_ADDR MASTER_PORT PYTHON_BIN TAG OUTPUT_DIR
  nohup bash "$0" "$@" >"${OUTPUT_DIR}/nohup.log" 2>&1 &
  echo "Started PI0.5 comparison in background."
  echo "PID: $!"
  echo "Output dir: ${OUTPUT_DIR}"
  echo "Main log: ${MAIN_LOG}"
  exit 0
fi

mkdir -p "${OUTPUT_DIR}"
exec > >(tee -a "${MAIN_LOG}") 2>&1

echo "[Run] PI0.5 speed + success comparison"
echo "[Run] Started at: $(date -Is)"
echo "[Run] Repo root: ${REPO_ROOT}"
echo "[Run] Output dir: ${OUTPUT_DIR}"
echo "[Run] Tag: ${TAG}"

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "[Error] Checkpoint not found: ${CKPT_PATH}" >&2
  exit 1
fi
if [[ ! -f "${BASELINE_CONFIG}" ]]; then
  echo "[Error] Baseline config not found: ${BASELINE_CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${ACCELERATED_CONFIG}" ]]; then
  echo "[Error] Accelerated config not found: ${ACCELERATED_CONFIG}" >&2
  exit 1
fi

{
  echo "started_at=$(date -Is)"
  echo "repo_root=${REPO_ROOT}"
  echo "tag=${TAG}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "ckpt_path=${CKPT_PATH}"
  echo "baseline_config=${BASELINE_CONFIG}"
  echo "accelerated_config=${ACCELERATED_CONFIG}"
  echo "speed_scenarios=${SPEED_SCENARIOS}"
  echo "single_gpus=${SINGLE_GPUS}"
  echo "multi_gpus=${MULTI_GPUS}"
  echo "success_gpus=${SUCCESS_GPUS}"
  echo "success_nproc_per_node=${SUCCESS_NPROC_PER_NODE}"
  echo "speed_warmup_iters=${SPEED_WARMUP_ITERS}"
  echo "speed_bench_iters=${SPEED_BENCH_ITERS}"
  echo "success_trials_per_task=${SUCCESS_TRIALS_PER_TASK}"
  echo "success_seeds=${SUCCESS_SEEDS}"
  echo "master_addr=${MASTER_ADDR}"
  echo "master_port=${MASTER_PORT}"
  echo "python_bin=${PYTHON_BIN}"
  echo
  echo "git_status_short:"
  git status --short || true
  echo
  echo "gpu_status:"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader || true
} >"${METADATA_PATH}"

echo "[Run] Metadata written to: ${METADATA_PATH}"

CMD=(
  "${PYTHON_BIN}" scripts/compare_pi05_speed_success.py
  --ckpt-path "${CKPT_PATH}"
  --baseline-config "${BASELINE_CONFIG}"
  --accelerated-config "${ACCELERATED_CONFIG}"
  --tag "${TAG}"
  --output-dir "${OUTPUT_DIR}"
  --speed-scenarios "${SPEED_SCENARIOS}"
  --single-gpus "${SINGLE_GPUS}"
  --multi-gpus "${MULTI_GPUS}"
  --success-gpus "${SUCCESS_GPUS}"
  --success-nproc-per-node "${SUCCESS_NPROC_PER_NODE}"
  --speed-warmup-iters "${SPEED_WARMUP_ITERS}"
  --speed-bench-iters "${SPEED_BENCH_ITERS}"
  --success-trials-per-task "${SUCCESS_TRIALS_PER_TASK}"
  --success-seeds "${SUCCESS_SEEDS}"
  --master-addr "${MASTER_ADDR}"
  --master-port "${MASTER_PORT}"
)

echo "[Run] Command:"
printf ' %q' "${CMD[@]}"
echo

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[Run] DRY_RUN=1, command not executed."
  exit 0
fi

"${CMD[@]}"

echo "[Run] Finished at: $(date -Is)"
echo "[Run] Reports:"
echo "  ${OUTPUT_DIR}/pi05_speed_success_comparison_${TAG}.md"
echo "  ${OUTPUT_DIR}/pi05_speed_success_comparison_${TAG}.csv"
echo "  ${OUTPUT_DIR}/pi05_speed_success_comparison_${TAG}.json"
echo "[Run] Logs:"
echo "  ${MAIN_LOG}"
echo "  ${OUTPUT_DIR}/logs/"
