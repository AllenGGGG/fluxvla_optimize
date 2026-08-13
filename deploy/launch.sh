#!/usr/bin/env bash
# Launch guarded FluxVLA PI0.5 joint-space inference.

set -euo pipefail
# Control (environment overrides are useful for isolated startup checks).
AUTO_START="${PISTAR_AUTO_START:-true}"
HAND_CLOSE_THRESHOLD="${PISTAR_HAND_CLOSE_THRESHOLD:-0.25}"
HAND_CLOSED_POSITIONS="${PISTAR_HAND_CLOSED_POSITIONS:-[0.99,1.39,0.504,0.504,0.504,0.504]}"

PYTHON_BIN="$HOME/runtime/miniforge3/envs/fluxvla_infer/bin/python"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_NAME="$(basename "$SCRIPT_DIR")"
PACKAGE_PARENT="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$PACKAGE_PARENT/configs/pi05"

# Checkpoint selection. Change these two values to use another root/default run.
CHECKPOINT_BASE_DIR="${PISTAR_CHECKPOINT_BASE_DIR:-/home/fiveages/Downloads/load}"
DEFAULT_CHECKPOINT_NAME="${PISTAR_CHECKPOINT_NAME:-best_model}"
DEFAULT_MODEL_BASENAME="${PISTAR_MODEL_BASENAME:-step-020238-epoch-03-loss=0.0043.safetensors}"
DEFAULT_EXECUTION_MODE="${PISTAR_EXECUTION_MODE:-async}"
DEFAULT_RTC_MODE="${PISTAR_RTC_MODE:-guidance}"
DEFAULT_ACCELERATION="${PISTAR_ACCELERATION:-triton}"
DEFAULT_ROBOT_HZ="${PISTAR_ROBOT_HZ:-30.0}"
DEFAULT_RTC_EXECUTION_HORIZON="${PISTAR_RTC_EXECUTION_HORIZON:-10}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

is_interactive() {
  [[ -t 0 && -t 1 ]]
}

prompt_with_default() {
  local variable_name="$1"
  local prompt="$2"
  local default_value="$3"
  local value=""
  if is_interactive; then
    if ! read -r -p "$prompt [$default_value]: " value; then
      value=""
    fi
  fi
  printf -v "$variable_name" '%s' "${value:-$default_value}"
}

CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR/#\~/$HOME}"
[[ -d "$CHECKPOINT_BASE_DIR" ]] || \
  die "checkpoint 根目录不存在: $CHECKPOINT_BASE_DIR"
CHECKPOINT_BASE_DIR="$(cd -- "$CHECKPOINT_BASE_DIR" && pwd)"

mapfile -t CHECKPOINT_DIR_CANDIDATES < <(
  find "$CHECKPOINT_BASE_DIR" -mindepth 1 -maxdepth 1 -type d \
    -printf '%f\n' | sort -V
)

if is_interactive; then
  echo "Checkpoint 根目录：$CHECKPOINT_BASE_DIR"
  if (( ${#CHECKPOINT_DIR_CANDIDATES[@]} > 0 )); then
    echo "发现以下 ${#CHECKPOINT_DIR_CANDIDATES[@]} 个文件夹（输入文件夹名即可，* 为默认选择）："
    for checkpoint_dir_name in "${CHECKPOINT_DIR_CANDIDATES[@]}"; do
      marker=" "
      [[ "$checkpoint_dir_name" == "$DEFAULT_CHECKPOINT_NAME" ]] && marker="*"
      printf '  %s %s\n' "$marker" "$checkpoint_dir_name"
    done
  else
    echo "根目录下暂时没有子文件夹，也可以输入绝对路径。"
  fi
fi

# PISTAR_CHECKPOINT_ROOT remains supported as a full-path override.
DEFAULT_CHECKPOINT_INPUT="${PISTAR_CHECKPOINT_ROOT:-$DEFAULT_CHECKPOINT_NAME}"
prompt_with_default CHECKPOINT_INPUT "Checkpoint 文件夹名" "$DEFAULT_CHECKPOINT_INPUT"
CHECKPOINT_INPUT="${CHECKPOINT_INPUT/#\~/$HOME}"
if [[ "$CHECKPOINT_INPUT" == /* ]]; then
  WEIGHT_DIR="$CHECKPOINT_INPUT"
else
  WEIGHT_DIR="$CHECKPOINT_BASE_DIR/$CHECKPOINT_INPUT"
fi
[[ -d "$WEIGHT_DIR" ]] || die "checkpoint 文件夹不存在: $WEIGHT_DIR"
WEIGHT_DIR="$(cd -- "$WEIGHT_DIR" && pwd)"

# Accept either the run root (.../2026_07_20) or its checkpoints/ directory.
CHECKPOINT_ROOT="$WEIGHT_DIR"
if [[ "$(basename "$WEIGHT_DIR")" == "checkpoints" ]]; then
  CHECKPOINT_ROOT="$(dirname "$WEIGHT_DIR")"
fi

MODEL_SEARCH_DIR="$WEIGHT_DIR"
MODEL_SEARCH_DEPTH=2
if [[ -d "$CHECKPOINT_ROOT/checkpoints" ]]; then
  MODEL_SEARCH_DIR="$CHECKPOINT_ROOT/checkpoints"
  MODEL_SEARCH_DEPTH=1
fi

mapfile -t MODEL_CANDIDATES < <(
  find "$MODEL_SEARCH_DIR" -maxdepth "$MODEL_SEARCH_DEPTH" -type f \
    \( -name '*.safetensors' -o -name '*.pt' -o -name '*.pth' \) \
    -print | sort -V
)
(( ${#MODEL_CANDIDATES[@]} > 0 )) || \
  die "没有在 $MODEL_SEARCH_DIR 中找到模型权重"

DEFAULT_MODEL_INDEX=$((${#MODEL_CANDIDATES[@]} - 1))
for index in "${!MODEL_CANDIDATES[@]}"; do
  if [[ "$(basename "${MODEL_CANDIDATES[$index]}")" == "$DEFAULT_MODEL_BASENAME" ]]; then
    DEFAULT_MODEL_INDEX="$index"
    break
  fi
done

MODEL_INDEX="$DEFAULT_MODEL_INDEX"
if is_interactive && (( ${#MODEL_CANDIDATES[@]} > 1 )); then
  echo
  echo "发现以下模型权重："
  for index in "${!MODEL_CANDIDATES[@]}"; do
    marker=" "
    [[ "$index" -eq "$DEFAULT_MODEL_INDEX" ]] && marker="*"
    printf '  %s %d) %s\n' "$marker" "$((index + 1))" \
      "$(basename "${MODEL_CANDIDATES[$index]}")"
  done
  while true; do
    choice=""
    if ! read -r -p "选择模型 [$((DEFAULT_MODEL_INDEX + 1))]: " choice; then
      choice=""
    fi
    choice="${choice:-$((DEFAULT_MODEL_INDEX + 1))}"
    if [[ "$choice" =~ ^[0-9]+$ ]] && \
       (( choice >= 1 && choice <= ${#MODEL_CANDIDATES[@]} )); then
      MODEL_INDEX=$((choice - 1))
      break
    fi
    echo "请输入 1-${#MODEL_CANDIDATES[@]}"
  done
fi
MODEL_ID="${MODEL_CANDIDATES[$MODEL_INDEX]}"

NORM_STATS_PATH="$CHECKPOINT_ROOT/dataset_statistics.json"
if [[ ! -f "$NORM_STATS_PATH" ]]; then
  mapfile -t STATS_CANDIDATES < <(
    find "$CHECKPOINT_ROOT" -maxdepth 2 -type f \
      -name 'dataset_statistics.json' -print | sort
  )
  (( ${#STATS_CANDIDATES[@]} > 0 )) || \
    die "没有在 $CHECKPOINT_ROOT 中找到 dataset_statistics.json"
  NORM_STATS_PATH="${STATS_CANDIDATES[0]}"
fi

TOKENIZER_PATH="$CHECKPOINT_ROOT/tokenizer"
if [[ ! -d "$TOKENIZER_PATH" ]]; then
  mapfile -t TOKENIZER_CANDIDATES < <(
    find "$CHECKPOINT_ROOT" -maxdepth 3 -type f \
      \( -name 'tokenizer.json' -o -name 'tokenizer.model' \) \
      -printf '%h\n' | sort -u
  )
  (( ${#TOKENIZER_CANDIDATES[@]} > 0 )) || \
    die "没有在 $CHECKPOINT_ROOT 中找到 tokenizer"
  TOKENIZER_PATH="${TOKENIZER_CANDIDATES[0]}"
fi

EXECUTION_MODE="$DEFAULT_EXECUTION_MODE"
case "$EXECUTION_MODE" in
  async|serial) ;;
  *) die "PISTAR_EXECUTION_MODE 必须是 async 或 serial" ;;
esac
if is_interactive; then
  default_choice=1
  [[ "$DEFAULT_EXECUTION_MODE" == "serial" ]] && default_choice=2
  echo
  echo "执行方式（* 为默认选择）："
  [[ "$default_choice" -eq 1 ]] && marker="*" || marker=" "
  printf '  %s 1) async  - 异步推理（后台生成 chunk，控制循环不中断）\n' "$marker"
  [[ "$default_choice" -eq 2 ]] && marker="*" || marker=" "
  printf '  %s 2) serial - 串行推理（执行完 chunk 后，在控制线程生成下一块）\n' "$marker"
  while true; do
    choice=""
    if ! read -r -p "选择执行方式 [$default_choice]: " choice; then
      choice=""
    fi
    choice="${choice:-$default_choice}"
    case "$choice" in
      1) EXECUTION_MODE="async"; break ;;
      2) EXECUTION_MODE="serial"; break ;;
      *) echo "请输入 1-2" ;;
    esac
  done
fi

RTC_MODE="$DEFAULT_RTC_MODE"
case "$RTC_MODE" in
  none|guidance|prefix) ;;
  *) die "PISTAR_RTC_MODE 必须是 none、guidance 或 prefix" ;;
esac
if [[ "$EXECUTION_MODE" == "serial" ]]; then
  RTC_MODE="none"
  is_interactive && echo "串行推理没有未执行队列前缀，RTC 固定关闭。"
elif is_interactive; then
  default_choice=1
  [[ "$DEFAULT_RTC_MODE" == "guidance" ]] && default_choice=2
  [[ "$DEFAULT_RTC_MODE" == "prefix" ]] && default_choice=3
  echo
  echo "RTC 方法（* 为默认选择）："
  [[ "$default_choice" -eq 1 ]] && marker="*" || marker=" "
  printf '  %s 1) none     - 关闭 RTC\n' "$marker"
  [[ "$default_choice" -eq 2 ]] && marker="*" || marker=" "
  printf '  %s 2) guidance - guidance RTC\n' "$marker"
  [[ "$default_choice" -eq 3 ]] && marker="*" || marker=" "
  printf '  %s 3) prefix   - prefix RTC\n' "$marker"
  while true; do
    choice=""
    if ! read -r -p "选择 RTC 方法 [$default_choice]: " choice; then
      choice=""
    fi
    choice="${choice:-$default_choice}"
    case "$choice" in
      1) RTC_MODE="none"; break ;;
      2) RTC_MODE="guidance"; break ;;
      3) RTC_MODE="prefix"; break ;;
      *) echo "请输入 1-3" ;;
    esac
  done
fi

ACCELERATION="$DEFAULT_ACCELERATION"
case "$ACCELERATION" in
  triton|python|custom) ;;
  *) die "PISTAR_ACCELERATION 必须是 triton、python 或 custom" ;;
esac
if is_interactive; then
  default_choice=1
  [[ "$DEFAULT_ACCELERATION" == "python" ]] && default_choice=2
  [[ "$DEFAULT_ACCELERATION" == "custom" ]] && default_choice=3
  echo
  echo "模型加速（* 为默认选择）："
  [[ "$default_choice" -eq 1 ]] && marker="*" || marker=" "
  printf '  %s 1) triton - Triton fused kernels + CUDA Graph\n' "$marker"
  [[ "$default_choice" -eq 2 ]] && marker="*" || marker=" "
  printf '  %s 2) python - 普通 PyTorch\n' "$marker"
  [[ "$default_choice" -eq 3 ]] && marker="*" || marker=" "
  printf '  %s 3) custom - 自定义 inference config\n' "$marker"
  while true; do
    choice=""
    if ! read -r -p "选择模型加速 [$default_choice]: " choice; then
      choice=""
    fi
    choice="${choice:-$default_choice}"
    case "$choice" in
      1) ACCELERATION="triton"; break ;;
      2) ACCELERATION="python"; break ;;
      3) ACCELERATION="custom"; break ;;
      *) echo "请输入 1-3" ;;
    esac
  done
fi

if [[ "$ACCELERATION" == "custom" ]]; then
    CUSTOM_CONFIG=""
    prompt_with_default CUSTOM_CONFIG "Inference config 路径" \
      "$CONFIG_DIR/pi05_parcel_sort_none_triton_inference.py"
    CUSTOM_CONFIG="${CUSTOM_CONFIG/#\~/$HOME}"
    if [[ "$CUSTOM_CONFIG" != /* ]]; then
      CUSTOM_CONFIG="$PACKAGE_PARENT/$CUSTOM_CONFIG"
    fi
    INFERENCE_CONFIG="$CUSTOM_CONFIG"
elif [[ "$RTC_MODE" == "guidance" && "$ACCELERATION" == "triton" ]]; then
  INFERENCE_CONFIG="$CONFIG_DIR/pi05_parcel_sort_guidance_triton_inference.py"
elif [[ "$RTC_MODE" == "guidance" ]]; then
  INFERENCE_CONFIG="$CONFIG_DIR/pi05_parcel_sort_guidance_pytorch_inference.py"
elif [[ "$RTC_MODE" == "prefix" && "$ACCELERATION" == "triton" ]]; then
  INFERENCE_CONFIG="$CONFIG_DIR/pi05_parcel_sort_prefix_triton_inference.py"
elif [[ "$RTC_MODE" == "prefix" ]]; then
  INFERENCE_CONFIG="$CONFIG_DIR/pi05_parcel_sort_prefix_pytorch_inference.py"
elif [[ "$ACCELERATION" == "triton" ]]; then
  INFERENCE_CONFIG="$CONFIG_DIR/pi05_parcel_sort_none_triton_inference.py"
else
  INFERENCE_CONFIG="$CONFIG_DIR/pi05_parcel_sort_none_pytorch_inference.py"
fi

case "$RTC_MODE" in
  none) RTC_LABEL="关闭" ;;
  guidance) RTC_LABEL="guidance" ;;
  prefix) RTC_LABEL="prefix" ;;
esac
case "$ACCELERATION" in
  triton) ACCELERATION_LABEL="Triton + CUDA Graph" ;;
  python) ACCELERATION_LABEL="普通 PyTorch" ;;
  custom) ACCELERATION_LABEL="自定义配置" ;;
esac
[[ "$EXECUTION_MODE" == "async" ]] && EXECUTION_LABEL="异步" || EXECUTION_LABEL="串行"

[[ -f "$MODEL_ID" ]] || die "模型权重不存在: $MODEL_ID"
[[ -f "$NORM_STATS_PATH" ]] || die "统计文件不存在: $NORM_STATS_PATH"
[[ -d "$TOKENIZER_PATH" ]] || die "tokenizer 不存在: $TOKENIZER_PATH"
[[ -f "$INFERENCE_CONFIG" ]] || die "推理配置不存在: $INFERENCE_CONFIG"

# Model and execution
prompt_with_default ROBOT_HZ "机器人控制频率 Hz" "$DEFAULT_ROBOT_HZ"
if ! [[ "$ROBOT_HZ" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || \
   [[ "$ROBOT_HZ" =~ ^0*([.]0*)?$ ]]; then
  die "机器人控制频率必须是正数，当前值: $ROBOT_HZ"
fi

# rtc_execution_horizon is consumed only by the asynchronous RTC scheduler.
# Serial execution forces RTC off, and async+none uses the plain scheduler, so
# overriding the value in either mode would be misleading and has no effect.
RTC_HORIZON_ACTIVE=false
RTC_EXECUTION_HORIZON="$DEFAULT_RTC_EXECUTION_HORIZON"
if [[ "$EXECUTION_MODE" == "async" && "$RTC_MODE" != "none" ]]; then
  RTC_HORIZON_ACTIVE=true
  prompt_with_default RTC_EXECUTION_HORIZON \
    "RTC 衔接/衰减 horizon（控制步数）" \
    "$DEFAULT_RTC_EXECUTION_HORIZON"
  if ! [[ "$RTC_EXECUTION_HORIZON" =~ ^[0-9]+$ ]] || \
     (( 10#$RTC_EXECUTION_HORIZON <= 0 )); then
    die "RTC horizon 必须是正整数，当前值: $RTC_EXECUTION_HORIZON"
  fi
fi
TASK="Pick up the parcel with the left hand, then move it onto the conveyor belt with the right hand."
DEVICE="cuda"
DTYPE="bf16"

RERUN_ENABLED="${PISTAR_RERUN_ENABLED:-true}"
case "$AUTO_START" in
  true|false) ;;
  *) die "PISTAR_AUTO_START 必须是 true 或 false" ;;
esac
case "$RERUN_ENABLED" in
  true|false) ;;
  *) die "PISTAR_RERUN_ENABLED 必须是 true 或 false" ;;
esac
[[ "$RERUN_ENABLED" == "true" ]] && RERUN_LABEL="开启" || RERUN_LABEL="关闭"

# ROS environment
ROS_SETUP="/opt/ros/jazzy/setup.bash"
CONTROL_WS_SETUP="$HOME/fa_w2_ws/install/setup.bash"

[[ -x "$PYTHON_BIN" ]] || die "Python 不可执行: $PYTHON_BIN"
[[ -f "$ROS_SETUP" ]] || die "ROS setup 不存在: $ROS_SETUP"
[[ -f "$CONTROL_WS_SETUP" ]] || die "控制工作区 setup 不存在: $CONTROL_WS_SETUP"

echo
echo "========== FluxVLA 启动配置 =========="
echo "checkpoint : $CHECKPOINT_ROOT"
echo "model      : $MODEL_ID"
echo "stats      : $NORM_STATS_PATH"
echo "tokenizer  : $TOKENIZER_PATH"
echo "execution  : $EXECUTION_LABEL"
echo "rtc        : $RTC_LABEL"
echo "accel      : $ACCELERATION_LABEL"
echo "config     : $INFERENCE_CONFIG"
echo "robot hz   : $ROBOT_HZ"
if [[ "$RTC_HORIZON_ACTIVE" == "true" ]]; then
  echo "rtc horizon: $RTC_EXECUTION_HORIZON steps"
else
  echo "rtc horizon: inactive (RTC is off)"
fi
echo "control    : auto_start=$AUTO_START"
echo "hand snap  : threshold=$HAND_CLOSE_THRESHOLD closed=$HAND_CLOSED_POSITIONS"
echo "rerun      : $RERUN_LABEL"
echo "======================================="

if is_interactive; then
  echo "可选值提示："
  echo "  execution : async（异步） | serial（串行）"
  if [[ "$EXECUTION_MODE" == "serial" ]]; then
    echo "  rtc       : none（串行模式下固定关闭）"
  else
    echo "  rtc       : none（关闭） | guidance | prefix"
  fi
  echo "  accel     : triton（Triton + CUDA Graph） | python（普通 PyTorch） | custom（自定义配置）"
  echo "  直接回车会采用方括号中的默认选择。"
  echo
  confirm=""
  if ! read -r -p "确认启动？[Y/n]: " confirm; then
    confirm=""
  fi
  case "${confirm:-y}" in
    y|Y|yes|YES) ;;
    *) echo "已取消"; exit 0 ;;
  esac
fi

# Test/configuration inspection without sourcing ROS or starting the node.
if [[ "${PISTAR_DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

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
  -p "tokenizer_path:=$TOKENIZER_PATH"

  # Model and execution
  -p "robot_exec_hz:=$ROBOT_HZ"
  -p "execution_mode:=$EXECUTION_MODE"
  -p "task:=$TASK"
  -p "device:=$DEVICE"
  -p "dtype:=$DTYPE"

  # Control
  -p "auto_start:=$AUTO_START"
  -p "hand_close_threshold:=$HAND_CLOSE_THRESHOLD"
  -p "hand_closed_positions:=$HAND_CLOSED_POSITIONS"

  # Visualization
  -p "rerun_enabled:=$RERUN_ENABLED"

)

if [[ "$RTC_HORIZON_ACTIVE" == "true" ]]; then
  ROS_PARAMS+=(
    -p "rtc_execution_horizon:=$RTC_EXECUTION_HORIZON"
  )
fi

echo "control    : auto_start=$AUTO_START; send /xr/controller_state=15/16 to start/pause"
echo "rerun      : $RERUN_LABEL"

# The selected config resolves pretrained_name_or_path/tokenizer
# paths (e.g. 'checkpoints/pi05_base') relative to the process cwd, matching
# training's convention - run from the repo root, not deploy/.
cd "$PACKAGE_PARENT"
export PYTHONPATH="$PACKAGE_PARENT${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" -m "$PACKAGE_NAME.main" "${ROS_PARAMS[@]}"
