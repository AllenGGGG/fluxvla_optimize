#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PPU_ENV_FILE="${PPU_ENV_FILE:-/usr/local/PPU_SDK/envsetup.sh}"
CONDA_BIN="${CONDA_BIN:-/opt/conda/bin/conda}"
ENV_NAME="${FLUXVLA_PPU_ENV_NAME:-fluxvla}"
PYTHON_VERSION="${FLUXVLA_PPU_PYTHON_VERSION:-3.12}"
TORCH_VERSION="${FLUXVLA_PPU_TORCH_VERSION:-${PYTORCH_VERSION:-2.8.0}}"
TORCHVISION_VERSION="${FLUXVLA_PPU_TORCHVISION_VERSION:-${TORCHVISION_VERSION:-0.23.0}}"
TRITON_VERSION="${FLUXVLA_PPU_TRITON_VERSION:-${TRITON_VERSION:-3.3.0}}"
AV_VERSION="${FLUXVLA_PPU_AV_VERSION:-18.0.0}"
DRY_RUN=0
SKIP_PROJECT=0

usage() {
  cat <<'EOF'
Usage: bash scripts/install_env_ppu.sh [options]

Install the dedicated PPU training environment. This installer requires the
vendor PPU SDK and its authenticated PIP_INDEX_URL.

Options:
  --env-name NAME             Conda environment name (default: fluxvla).
  --python-version VERSION    Python version (default: 3.12).
  --pytorch-version VERSION   PPU PyTorch version (default: 2.8.0).
  --torchvision-version VER   torchvision version (default: 0.23.0).
  --triton-version VERSION    Triton version (default: 3.3.0).
  --av-version VERSION        PyAV version (default: 18.0.0).
  --skip-project              Skip editable FluxVLA installation.
  --dry-run                   Print commands without running them.
  -h, --help                  Show this help.

Environment variables:
  PPU_ENV_FILE                Default: /usr/local/PPU_SDK/envsetup.sh
  CONDA_BIN                   Default: /opt/conda/bin/conda
  FLUXVLA_PPU_ENV_NAME        Same as --env-name.
  FLUXVLA_PPU_PYTHON_VERSION  Same as --python-version.
  FLUXVLA_PPU_TORCH_VERSION   Same as --pytorch-version.
  FLUXVLA_PPU_TORCHVISION_VERSION
                              Same as --torchvision-version.
  FLUXVLA_PPU_TRITON_VERSION  Same as --triton-version.
  FLUXVLA_PPU_AV_VERSION      Same as --av-version.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name) ENV_NAME="$2"; shift 2 ;;
    --python-version) PYTHON_VERSION="$2"; shift 2 ;;
    --pytorch-version|--torch-version) TORCH_VERSION="$2"; shift 2 ;;
    --torchvision-version) TORCHVISION_VERSION="$2"; shift 2 ;;
    --triton-version) TRITON_VERSION="$2"; shift 2 ;;
    --av-version) AV_VERSION="$2"; shift 2 ;;
    --skip-project) SKIP_PROJECT=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

run() {
  if [[ ${DRY_RUN} -eq 1 ]]; then
    printf '+ '
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

if [[ ! -f "${PPU_ENV_FILE}" ]]; then
  echo "PPU SDK setup file not found: ${PPU_ENV_FILE}" >&2
  exit 1
fi
if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "Conda executable not found: ${CONDA_BIN}" >&2
  exit 1
fi

set +u
# shellcheck disable=SC1090
source "${PPU_ENV_FILE}" "" >/dev/null
set -u

if [[ -z "${PIP_INDEX_URL:-}" ]]; then
  echo "The PPU SDK did not provide PIP_INDEX_URL." >&2
  echo "Check the authenticated SDK setup in ${PPU_ENV_FILE}." >&2
  exit 1
fi

if "${CONDA_BIN}" env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "Using existing Conda environment: ${ENV_NAME}"
else
  run "${CONDA_BIN}" create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip
fi

CONDA_ROOT="$(cd "$(dirname "${CONDA_BIN}")/.." && pwd)"
ENV_PREFIX="${CONDA_ROOT}/envs/${ENV_NAME}"
PYTHON_BIN="${ENV_PREFIX}/bin/python"

ppu_pip_install() {
  # pip reads PIP_INDEX_URL from the PPU SDK environment. Do not put the
  # possibly credential-bearing URL in argv or command output.
  run "${PYTHON_BIN}" -m pip install "$@"
}

run "${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
ppu_pip_install "torch==${TORCH_VERSION}"
ppu_pip_install --no-build-isolation "torchvision==${TORCHVISION_VERSION}"
ppu_pip_install --no-deps "triton==${TRITON_VERSION}"
run "${PYTHON_BIN}" -m pip install --index-url https://pypi.org/simple \
  "av==${AV_VERSION}"
ppu_pip_install cmake ninja
ppu_pip_install -r "${REPO_ROOT}/environment/pip-requirements-train.txt"

if [[ ${SKIP_PROJECT} -eq 0 ]]; then
  if [[ ${DRY_RUN} -eq 0 ]]; then
    TORCH_CUDA_ARCH_LIST="$("${PYTHON_BIN}" - <<'PY'
import torch

major, minor = torch.cuda.get_device_capability(0)
print(f'{major}.{minor}')
PY
)"
    export TORCH_CUDA_ARCH_LIST
  fi
  run "${PYTHON_BIN}" -m pip install --no-build-isolation -e "${REPO_ROOT}"
fi

if [[ ${DRY_RUN} -eq 0 ]]; then
  "${PYTHON_BIN}" - <<'PY'
import av
import torch
import torchvision

print(f'torch={torch.__version__}')
print(f'torchvision={torchvision.__version__}')
print(f'av={av.__version__}')
print(f'PPU devices through CUDA compatibility API: {torch.cuda.device_count()}')
if not torch.cuda.is_available():
    raise SystemExit('PPU is unavailable after installation.')
if not torch.distributed.is_nccl_available():
    raise SystemExit('NCCL/PCCL distributed backend is unavailable.')
PY
fi

echo "PPU environment ready: ${ENV_PREFIX}"
echo "Activate it with: source /opt/conda/bin/activate ${ENV_NAME}"
echo "Start training with: bash scripts/train_ppu.sh"
