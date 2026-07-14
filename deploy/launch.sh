#!/usr/bin/env bash
# Launch guarded FluxVLA PI0.5 joint-space inference.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_NAME="$(basename "$SCRIPT_DIR")"
PACKAGE_PARENT="$(dirname "$SCRIPT_DIR")"
CHECKPOINT_ROOT="$SCRIPT_DIR/checkpoints/07_13"

# Paths
MODEL_ID="$CHECKPOINT_ROOT/checkpoints/step-072330-epoch-10-loss=0.0142.safetensors"
INFERENCE_CONFIG="$PACKAGE_PARENT/configs/pi05/pi05_parcel_sort_inference.py"
PYTHON_BIN="$HOME/runtime/miniforge3/envs/fluxvla_infer/bin/python"

# Model and execution
RTC_MODE="plain"
ROBOT_HZ=30.0
NUM_STEPS_OVERRIDE=0
MAX_NORMALIZED_ACTION_ABS=1.25
TASK="Pick up the parcel with the left hand, then move it onto the conveyor belt with the right hand."
DEVICE="cuda"
DTYPE="bf16"

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
ALLOW_SHARED_CONTROL=false
REQUIRE_DIRECT_MOVEJ=true
MANAGE_WBC_INTERPOLATION=true
MAX_HELD_POSE_ERROR_RAD=0.10

# Chunking
PLAIN_EXEC_FRACTION=0.5
OVERLAP_STEPS=15
ACTION_BLEND_STEPS=10
NOTIFY_EVERY=12
DISCARD_LATENCY=true
MIN_QUEUE_KEEP=1
STATE_FUSION_ALPHA=0.5
RTC_HORIZON=10
RTC_GUIDANCE_WEIGHT=10.0

# Debug
DEBUG=false
DEBUG_DIR="/tmp/plain_debug"
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
  -p "model_id:=$MODEL_ID"
  -p "inference_config:=$INFERENCE_CONFIG"
  -p "device:=$DEVICE"
  -p "dtype:=$DTYPE"
  -p "rtc_mode:=$RTC_MODE"
  -p "robot_exec_hz:=$ROBOT_HZ"
  -p "num_inference_steps_override:=$NUM_STEPS_OVERRIDE"
  -p "max_normalized_action_abs:=$MAX_NORMALIZED_ACTION_ABS"
  -p "advantage_enabled:=$ADVANTAGE_ENABLED"
  -p "cfg_enabled:=$CFG_ENABLED"
  -p "cfg_scale:=$CFG_SCALE"
  -p "cfg_scale_joint:=$CFG_SCALE_JOINT"
  -p "cfg_scale_gripper:=$CFG_SCALE_GRIPPER"
  -p "cfg_cond_advantage_tag:=$CFG_COND_ADVANTAGE_TAG"
  -p "cfg_uncond_advantage_tag:=$CFG_UNCOND_ADVANTAGE_TAG"
  -p "arm_dof:=$ARM_DOF"
  -p "auto_start:=$AUTO_START"
  -p "allow_shared_control:=$ALLOW_SHARED_CONTROL"
  -p "require_direct_movej:=$REQUIRE_DIRECT_MOVEJ"
  -p "manage_wbc_interpolation:=$MANAGE_WBC_INTERPOLATION"
  -p "max_held_pose_error_rad:=$MAX_HELD_POSE_ERROR_RAD"
  -p "plain_execution_fraction:=$PLAIN_EXEC_FRACTION"
  -p "overlap_steps:=$OVERLAP_STEPS"
  -p "action_blend_steps:=$ACTION_BLEND_STEPS"
  -p "notify_every:=$NOTIFY_EVERY"
  -p "discard_latency_steps:=$DISCARD_LATENCY"
  -p "min_queue_keep_steps:=$MIN_QUEUE_KEEP"
  -p "state_fusion_alpha:=$STATE_FUSION_ALPHA"
  -p "rtc_infer_execution_horizon:=$RTC_HORIZON"
  -p "rtc_infer_max_guidance_weight:=$RTC_GUIDANCE_WEIGHT"
  -p "debug:=$DEBUG"
  -p "debug_dir:=$DEBUG_DIR"
  -p "rerun_enabled:=$RERUN_ENABLED"
  -p "task:=$TASK"
)

echo "model      : $MODEL_ID"
echo "config     : $INFERENCE_CONFIG"
echo "mode       : $RTC_MODE"
echo "control    : auto-start after safety gates pass"
echo "rerun     : live camera and joint visualization"
echo "WBC stream : temporarily sets movej_interpolation_type='none'"

# pi05_parcel_sort_inference.py resolves pretrained_name_or_path/tokenizer
# paths (e.g. 'checkpoints/pi05_base') relative to the process cwd, matching
# training's convention - run from the repo root, not deploy/.
cd "$PACKAGE_PARENT"
export PYTHONPATH="$PACKAGE_PARENT${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" -m "$PACKAGE_NAME.main" "${ROS_PARAMS[@]}"
