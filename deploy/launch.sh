#!/usr/bin/env bash
# Launch guarded FluxVLA PI0.5 joint-space inference.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_NAME="$(basename "$SCRIPT_DIR")"
PACKAGE_PARENT="$(dirname "$SCRIPT_DIR")"
CHECKPOINT_ROOT="$SCRIPT_DIR/checkpoints/07_13"

# Paths
MODEL_ID="$CHECKPOINT_ROOT/checkpoints/step-072330-epoch-10-loss=0.0142.safetensors"
# Must match RTC_METHOD below:
#   RTC_METHOD=none     -> pi05_parcel_sort_inference.py               (Triton)
#   RTC_METHOD=prefix   -> pi05_parcel_sort_inference_prefix_rtc.py    (Triton)
#   RTC_METHOD=guidance -> pi05_parcel_sort_inference_guidance_rtc.py  (eager, slower)
# A mismatch (e.g. guidance against the plain/prefix Triton configs) fails
# fast at startup -- see model.py's _RTC_METHOD_SUPPORTED_TYPES check.
INFERENCE_CONFIG="$PACKAGE_PARENT/configs/pi05/pi05_parcel_sort_inference.py"
PYTHON_BIN="$HOME/runtime/miniforge3/envs/fluxvla_infer/bin/python"
# Empty = use BaseInferenceRunner's default (MODEL_ID's grandparent directory).
NORM_STATS_PATH=""

# Model and execution
ROBOT_HZ=30.0
NUM_STEPS_OVERRIDE=0  # 0 = use the inference_config's own denoising step count
TASK="Pick up the parcel with the left hand, then move it onto the conveyor belt with the right hand."
DEVICE="cuda"
DTYPE="bf16"
NORM_TYPE="min_max"  # "quantile" or "min_max"; must match dataset_statistics.json
#NORM_TYPE="quantile"  # "quantile" or "min_max"; must match dataset_statistics.json

# fluxvla's own RTC guidance, fed by deploy/exec_engine's ChunkScheduler
# (the async scheduler; see deploy/README.md for the RTC_METHOD/INFERENCE_CONFIG
# pairing table).
RTC_METHOD="none"  # "none", "prefix", or "guidance"
RTC_EXECUTION_HORIZON=10
RTC_MAX_GUIDANCE_WEIGHT=10.0
RTC_SCHEDULE="linear"  # "linear", "exp", "ones", or "zeros"

# CFG / advantage-conditioning (inert until a checkpoint is trained on
# advantage-tagged data; infrastructure for collecting that signal now).
ADVANTAGE_ENABLED=false
CFG_ENABLED=false
CFG_SCALE=2.0
CFG_SCALE_JOINT=-1.0
CFG_SCALE_GRIPPER=-1.0
CFG_COND_ADVANTAGE_TAG="positive"
CFG_UNCOND_ADVANTAGE_TAG=""
ARM_DOF=7

# Safety gates
AUTO_START=true
REQUIRE_DIRECT_MOVEJ=true
MANAGE_WBC_INTERPOLATION=true

# Debug
DEBUG=false
DEBUG_DIR="/tmp/rtc_debug"
RERUN_ENABLED=true

# ROS environment
ROS_SETUP="/opt/ros/jazzy/setup.bash"
CONTROL_WS_SETUP="$HOME/fa_w2_ws/install/setup.bash"

export PYTHONNOUSERSITE=1
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
  -p "num_inference_steps_override:=$NUM_STEPS_OVERRIDE"
  -p "task:=$TASK"
  -p "device:=$DEVICE"
  -p "dtype:=$DTYPE"
  -p "norm_type:=$NORM_TYPE"

  # fluxvla RTC guidance
  -p "rtc_method:=$RTC_METHOD"
  -p "rtc_execution_horizon:=$RTC_EXECUTION_HORIZON"
  -p "rtc_max_guidance_weight:=$RTC_MAX_GUIDANCE_WEIGHT"
  -p "rtc_schedule:=$RTC_SCHEDULE"

  # CFG / advantage-conditioning
  -p "advantage_enabled:=$ADVANTAGE_ENABLED"
  -p "cfg_enabled:=$CFG_ENABLED"
  -p "cfg_scale:=$CFG_SCALE"
  -p "cfg_scale_joint:=$CFG_SCALE_JOINT"
  -p "cfg_scale_gripper:=$CFG_SCALE_GRIPPER"
  -p "cfg_cond_advantage_tag:=$CFG_COND_ADVANTAGE_TAG"
  -p "cfg_uncond_advantage_tag:=$CFG_UNCOND_ADVANTAGE_TAG"
  -p "arm_dof:=$ARM_DOF"

  # Safety gates
  -p "auto_start:=$AUTO_START"
  -p "require_direct_movej:=$REQUIRE_DIRECT_MOVEJ"
  -p "manage_wbc_interpolation:=$MANAGE_WBC_INTERPOLATION"

  # Debug
  -p "debug:=$DEBUG"
  -p "debug_dir:=$DEBUG_DIR"
  -p "rerun_enabled:=$RERUN_ENABLED"
)

echo "model      : $MODEL_ID"
echo "config     : $INFERENCE_CONFIG"
echo "rtc method : $RTC_METHOD"
echo "control    : auto-start after safety gates pass"
echo "rerun     : live camera and joint visualization"
echo "WBC stream : temporarily sets movej_interpolation_type='none'"

# pi05_parcel_sort_inference.py resolves pretrained_name_or_path/tokenizer
# paths (e.g. 'checkpoints/pi05_base') relative to the process cwd, matching
# training's convention - run from the repo root, not deploy/.
cd "$PACKAGE_PARENT"
export PYTHONPATH="$PACKAGE_PARENT${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" -m "$PACKAGE_NAME.main" "${ROS_PARAMS[@]}"
