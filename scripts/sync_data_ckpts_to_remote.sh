#!/usr/bin/env bash
# Sync local datasets/ and checkpoints/ to a remote FluxVLA workspace.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

REMOTE="${REMOTE:-root@8.160.187.101}"
REMOTE_DIR="${REMOTE_DIR:-/root/fluxvla_optimize}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
PID_FILE="${PID_FILE:-${LOG_DIR}/rsync_data_ckpts.pid}"
BACKGROUND=1
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/sync_data_ckpts_to_remote.sh [options]

Default:
  Start in the background and write rsync output to logs/rsync_data_ckpts_*.log.

Options:
  --foreground       Run in the current shell instead of nohup background mode.
  --dry-run          Show what would be copied without transferring files.
  --remote HOST      Remote SSH target. Default: root@8.160.187.101
  --remote-dir DIR   Remote root directory. Default: /root/fluxvla_optimize
  --log-dir DIR      Local log directory. Default: ./logs
  -h, --help         Show this help.

Environment overrides:
  REMOTE=root@8.160.187.101
  REMOTE_DIR=/root/fluxvla_optimize
  LOG_DIR=./logs

Examples:
  bash scripts/sync_data_ckpts_to_remote.sh
  tail -f logs/rsync_data_ckpts_YYYYmmdd_HHMMSS.log
  bash scripts/sync_data_ckpts_to_remote.sh --foreground --dry-run
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --foreground)
      BACKGROUND=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --remote)
      REMOTE="${2:?missing value for --remote}"
      shift 2
      ;;
    --remote-dir)
      REMOTE_DIR="${2:?missing value for --remote-dir}"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="${2:?missing value for --log-dir}"
      PID_FILE="${LOG_DIR}/rsync_data_ckpts.pid"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

run_rsync() {
  local src="$1"
  local dst="$2"
  local opts=(-avP)

  if [[ "${DRY_RUN}" == "1" ]]; then
    opts+=(--dry-run)
  fi

  [[ -d "${src}" ]] || {
    echo "error: source directory not found: ${src}" >&2
    return 1
  }

  run_cmd rsync "${opts[@]}" "${src}/" "${REMOTE}:${dst}/"
}

main() {
  cd "${REPO_ROOT}"
  mkdir -p "${LOG_DIR}"

  local started_at
  started_at="$(date '+%Y-%m-%d %H:%M:%S %z')"

  echo "Started at: ${started_at}"
  echo "Repo root: ${REPO_ROOT}"
  echo "Remote: ${REMOTE}:${REMOTE_DIR}"
  echo

  run_cmd ssh "${REMOTE}" "mkdir -p '${REMOTE_DIR}/datasets' '${REMOTE_DIR}/checkpoints'"
  echo

  run_rsync "${REPO_ROOT}/datasets" "${REMOTE_DIR}/datasets"
  echo
  run_rsync "${REPO_ROOT}/checkpoints" "${REMOTE_DIR}/checkpoints"
  echo

  echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S %z')"
}

if [[ "${BACKGROUND}" == "1" ]]; then
  mkdir -p "${LOG_DIR}"
  LOG_FILE="${LOG_DIR}/rsync_data_ckpts_$(date '+%Y%m%d_%H%M%S').log"
  extra_args=()

  if [[ "${DRY_RUN}" == "1" ]]; then
    extra_args+=(--dry-run)
  fi

  nohup "$0" --foreground \
    --remote "${REMOTE}" \
    --remote-dir "${REMOTE_DIR}" \
    --log-dir "${LOG_DIR}" \
    "${extra_args[@]}" \
    >"${LOG_FILE}" 2>&1 &

  pid="$!"
  echo "${pid}" >"${PID_FILE}"
  echo "Started background rsync: PID ${pid}"
  echo "Log: ${LOG_FILE}"
  echo "Watch: tail -f ${LOG_FILE}"
  exit 0
fi

main
