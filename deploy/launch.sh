#!/usr/bin/env bash
# Launch guarded FluxVLA PI0.5 joint-space inference.

set -euo pipefail
# Control
AUTO_START=true

PYTHON_BIN="$HOME/runtime/miniforge3/envs/fluxvla_infer/bin/python"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_NAME="$(basename "$SCRIPT_DIR")"
PACKAGE_PARENT="$(dirname "$SCRIPT_DIR")"
CHECKPOINT_ROOT="${FLUXVLA_CHECKPOINT_ROOT:-$HOME/Downloads/load/base_pi05}"


# Paths
MODEL_ID="$CHECKPOINT_ROOT/checkpoints/step-094029-epoch-13-loss=0.0074.safetensors"

# Inference configs:
#   pi05_parcel_sort_inference.py              complete plain eager config
#   pi05_parcel_sort_inference_prefix_rtc.py   prefix RTC Triton variant
#   pi05_parcel_sort_inference_guidance_rtc.py guidance RTC eager variant
# The deployment uses the complete base config; its inference_options selects
# rtc_method='none'.
INFERENCE_CONFIG="$PACKAGE_PARENT/configs/pi05/pi05_parcel_sort_inference.py"

# Empty = use BaseInferenceRunner's default (MODEL_ID's grandparent directory).
NORM_STATS_PATH="$CHECKPOINT_ROOT/dataset_statistics.json"
TOKENIZER_PATH="$CHECKPOINT_ROOT/tokenizer"

# Model and execution
ROBOT_HZ=30.0
TASK="Pick up the parcel with the left hand, then move it onto the conveyor belt with the right hand."
DEVICE="cuda"
DTYPE="bf16"

# Debug
DEBUG=false
DEBUG_DIR="/tmp/rtc_debug"
RERUN_ENABLED=true

# ROS environment
ROS_SETUP="/opt/ros/jazzy/setup.bash"
CONTROL_WS_SETUP="$HOME/fa_w2_ws/install/setup.bash"

export PYTHONNOUSERSITE=1
#export FLUXVLA_CHECKPOINT_ROOT="$CHECKPOINT_ROOT"
#export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
unset http_proxy https_proxy all_proxy no_proxy
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY
set +u
source "$ROS_SETUP"
source "$CONTROL_WS_SETUP"
set -u

ROS_PARAMS=(
  --ros-args

  # Paths
  -p "model_id:=$MODEL_ID"
  -p "inference_config:=$INFERENCE_CONFIG"
  -p "norm_stats_path:=$NORM_STATS_PATH"

  # Model and execution
  -p "robot_exec_hz:=$ROBOT_HZ"
  -p "task:=$TASK"
  -p "device:=$DEVICE"
  -p "dtype:=$DTYPE"

  # Control
  -p "auto_start:=$AUTO_START"

  # Debug
  -p "debug:=$DEBUG"
  -p "debug_dir:=$DEBUG_DIR"
  -p "rerun_enabled:=$RERUN_ENABLED"
)

echo "model      : $MODEL_ID"
echo "config     : $INFERENCE_CONFIG"
echo "control    : auto_start=$AUTO_START; send /xr/controller_state=15/16 to start/pause"
echo "rerun     : live camera and joint visualization"
echo "WBC stream : temporarily sets movej_interpolation_type='none'"

# pi05_parcel_sort_inference.py resolves pretrained_name_or_path/tokenizer
# paths (e.g. 'checkpoints/pi05_base') relative to the process cwd, matching
# training's convention - run from the repo root, not deploy/.
cd "$PACKAGE_PARENT"
export PYTHONPATH="$PACKAGE_PARENT${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" -m "$PACKAGE_NAME.main" "${ROS_PARAMS[@]}"
