#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-fluxvla}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
MINICONDA_DIR="${MINICONDA_DIR:-$HOME/miniconda3}"
INSTALLER="/tmp/Miniconda3-latest-Linux-x86_64.sh"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export LANG=C.UTF-8
export LC_ALL=C.UTF-8
unset LANGUAGE

log() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

log "Repository: ${REPO_DIR}"
log "Miniconda target: ${MINICONDA_DIR}"
log "Environment: ${ENV_NAME}"

CONDA="${MINICONDA_DIR}/bin/conda"
PIP="${MINICONDA_DIR}/envs/${ENV_NAME}/bin/pip"
PYTHON="${MINICONDA_DIR}/envs/${ENV_NAME}/bin/python"

if [ ! -x "${CONDA}" ]; then
  if [ -e "${MINICONDA_DIR}" ]; then
    BACKUP="${MINICONDA_DIR}.bak.$(date '+%Y%m%d_%H%M%S')"
    log "Existing non-functional miniconda path found; moving it to ${BACKUP}"
    mv "${MINICONDA_DIR}" "${BACKUP}"
  fi

  log "Downloading Miniconda installer"
  curl -L -o "${INSTALLER}" \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

  log "Installing Miniconda"
  bash "${INSTALLER}" -b -p "${MINICONDA_DIR}"
else
  log "Reusing existing Miniconda at ${MINICONDA_DIR}"
fi

log "Configuring conda"
"${CONDA}" config --set auto_activate_base false
"${CONDA}" init bash

if "${CONDA}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  log "Reusing existing ${ENV_NAME} environment"
else
  log "Creating ${ENV_NAME} with Python ${PYTHON_VERSION}"
  "${CONDA}" create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

log "Installing CUDA 12.4 compiler/runtime headers into the conda env"
"${CONDA}" install -n "${ENV_NAME}" -c nvidia/label/cuda-12.4.0 \
  cuda-nvcc=12.4.99 cuda-cudart-dev=12.4.99 libcublas-dev=12.4.5.8 -y

log "Installing PyTorch CUDA 12.4 wheels"
"${PIP}" install \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

log "Installing flash-attn build prerequisites"
"${PIP}" install psutil ninja packaging

log "Installing flash-attn 2.5.5"
CUDA_HOME="${MINICONDA_DIR}/envs/${ENV_NAME}" \
PATH="${MINICONDA_DIR}/envs/${ENV_NAME}/bin:${PATH}" \
MAX_JOBS="${MAX_JOBS:-8}" \
"${PIP}" install flash-attn==2.5.5 --no-build-isolation \
  --find-links https://github.com/Dao-AILab/flash-attention/releases

log "Installing av 14.4.0 from conda-forge"
"${CONDA}" install -n "${ENV_NAME}" -c conda-forge av=14.4.0 -y

log "Installing project requirements"
cd "${REPO_DIR}"
"${PIP}" install -r requirements.txt

log "Pinning TensorFlow/protobuf-compatible Google packages"
"${PIP}" install \
  tensorflow-metadata==1.14.0 \
  protobuf==3.20.3 \
  google-api-core==2.11.1 \
  proto-plus==1.22.3 \
  google-cloud-storage==2.10.0

log "Installing conda activation environment hooks"
ACTIVATE_DIR="${MINICONDA_DIR}/envs/${ENV_NAME}/etc/conda/activate.d"
DEACTIVATE_DIR="${MINICONDA_DIR}/envs/${ENV_NAME}/etc/conda/deactivate.d"
mkdir -p "${ACTIVATE_DIR}" "${DEACTIVATE_DIR}"
cat > "${ACTIVATE_DIR}/fluxvla_env.sh" <<EOF
export _FLUXVLA_OLD_CUDA_HOME="\${CUDA_HOME:-}"
export _FLUXVLA_OLD_CPATH="\${CPATH:-}"
export _FLUXVLA_OLD_LIBRARY_PATH="\${LIBRARY_PATH:-}"
export _FLUXVLA_OLD_LD_LIBRARY_PATH="\${LD_LIBRARY_PATH:-}"
export _FLUXVLA_OLD_NUMBA_CACHE_DIR="\${NUMBA_CACHE_DIR:-}"
export CUDA_HOME="${MINICONDA_DIR}/envs/${ENV_NAME}"
export CPATH="${MINICONDA_DIR}/envs/${ENV_NAME}/targets/x86_64-linux/include:${MINICONDA_DIR}/envs/${ENV_NAME}/lib/python3.10/site-packages/nvidia/cublas/include:\${CPATH:-}"
export LIBRARY_PATH="${MINICONDA_DIR}/envs/${ENV_NAME}/targets/x86_64-linux/lib:${MINICONDA_DIR}/envs/${ENV_NAME}/lib/python3.10/site-packages/nvidia/cublas/lib:\${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${MINICONDA_DIR}/envs/${ENV_NAME}/targets/x86_64-linux/lib:${MINICONDA_DIR}/envs/${ENV_NAME}/lib/python3.10/site-packages/nvidia/cublas/lib:\${LD_LIBRARY_PATH:-}"
export NUMBA_CACHE_DIR="\${NUMBA_CACHE_DIR:-/tmp/numba_cache_fluxvla}"
EOF
cat > "${DEACTIVATE_DIR}/fluxvla_env.sh" <<'EOF'
if [ -n "${_FLUXVLA_OLD_CUDA_HOME:-}" ]; then export CUDA_HOME="${_FLUXVLA_OLD_CUDA_HOME}"; else unset CUDA_HOME; fi
if [ -n "${_FLUXVLA_OLD_CPATH:-}" ]; then export CPATH="${_FLUXVLA_OLD_CPATH}"; else unset CPATH; fi
if [ -n "${_FLUXVLA_OLD_LIBRARY_PATH:-}" ]; then export LIBRARY_PATH="${_FLUXVLA_OLD_LIBRARY_PATH}"; else unset LIBRARY_PATH; fi
if [ -n "${_FLUXVLA_OLD_LD_LIBRARY_PATH:-}" ]; then export LD_LIBRARY_PATH="${_FLUXVLA_OLD_LD_LIBRARY_PATH}"; else unset LD_LIBRARY_PATH; fi
if [ -n "${_FLUXVLA_OLD_NUMBA_CACHE_DIR:-}" ]; then export NUMBA_CACHE_DIR="${_FLUXVLA_OLD_NUMBA_CACHE_DIR}"; else unset NUMBA_CACHE_DIR; fi
unset _FLUXVLA_OLD_CUDA_HOME _FLUXVLA_OLD_CPATH _FLUXVLA_OLD_LIBRARY_PATH _FLUXVLA_OLD_LD_LIBRARY_PATH _FLUXVLA_OLD_NUMBA_CACHE_DIR
EOF

log "Installing FluxVLA editable package"
CUDA_HOME="${MINICONDA_DIR}/envs/${ENV_NAME}" \
CPATH="${MINICONDA_DIR}/envs/${ENV_NAME}/targets/x86_64-linux/include:${MINICONDA_DIR}/envs/${ENV_NAME}/lib/python3.10/site-packages/nvidia/cublas/include:${CPATH:-}" \
LIBRARY_PATH="${MINICONDA_DIR}/envs/${ENV_NAME}/targets/x86_64-linux/lib:${MINICONDA_DIR}/envs/${ENV_NAME}/lib/python3.10/site-packages/nvidia/cublas/lib:${LIBRARY_PATH:-}" \
LD_LIBRARY_PATH="${MINICONDA_DIR}/envs/${ENV_NAME}/targets/x86_64-linux/lib:${MINICONDA_DIR}/envs/${ENV_NAME}/lib/python3.10/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}" \
PATH="${MINICONDA_DIR}/envs/${ENV_NAME}/bin:${PATH}" \
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}" \
MAX_JOBS="${FLUXVLA_MAX_JOBS:-1}" \
"${PIP}" install --no-build-isolation -e .

log "Verifying environment"
"${PYTHON}" - <<'PY'
import sys
import torch

print("python", sys.executable)
print("python_version", sys.version.split()[0])
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device", torch.cuda.get_device_name(0))

import flash_attn
print("flash_attn", getattr(flash_attn, "__version__", "unknown"))

import fluxvla
import fluxvla.transforms
import fluxvla.models
print("fluxvla import ok")
PY

log "Rebuild complete"
