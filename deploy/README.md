# fluxvla_deploy

Guarded ROS2 deployment for the AccVLA PI0.5 parcel-sorting policy trained by
`~/accvla/configs/pi05/pi05_parcel_sort.py`.

## Joint contract

The LeRobot state and action tensors have 32 values. Only the first 28 are
robot joints; the final four values are zero padding:

```text
0:2    body_joint3, body_joint4
2:9    left_joint1 .. left_joint7
9:16   right_joint1 .. right_joint7
16:22  left hand (thumb1, thumb2, index, middle, ring, pinky)
22:28  right hand (thumb1, thumb2, index, middle, ring, pinky)
28:32  zero padding
```

At inference start, `body_joint1`, `body_joint2`, `head_joint1`, and
`head_joint2` are latched from the current joint state. Every 28-D model action
is combined with those four held targets and published as:

```text
/ocs2_wbc_controller/target_joint_position  Float64MultiArray[20]
  body_joint1..4, left_joint1..7, right_joint1..7, head_joint1..2

/left_hand_controller/target_joint_position  Float64MultiArray[6]
/right_hand_controller/target_joint_position Float64MultiArray[6]
```

This order matches the live WBC `joints` parameter and the controller source in
`~/fa_w2_ws`.

The fixed training posture measured from MCAP is approximately
`[-0.50, -1.10, 0.00, 0.26] rad` for body 1/2 and head 1/2. The node requires
the start pose to be within `0.10 rad` per joint, then latches the measured pose;
it never commands an automatic move into that posture.

## Inputs

The default camera inputs match the currently available compressed topics:

```text
/camera_head/color/image_raw/compressed
/camera_left_wrist/color/image_raw/compressed
/camera_right_wrist/color/image_raw/compressed
/joint_states
```

Images are decoded as RGB and preprocessed exactly as in training: resize to
224 x 224, CHW concatenation in head/left/right order, and normalization to
`[-1, 1]`. State normalization, the 32-value PI0.5 state prompt, and action
denormalization use the training statistics.

## Controller requirements

The node starts with command publication disabled. It refuses to enable unless:

- `/fsm_state` is `4` (`MOVEJ` in the WBC controller's published state codes);
- the WBC and both hand-controller `joints` parameters exactly match the
  deployment order;
- no other node publishes any of the three command topics;
- all three cameras and all 32 physical joints are present and fresh;
- model outputs are finite and within the configured normalized envelope.

The deployment node does not read or modify WBC `movej_interpolation_type`.
Interpolation behavior is controlled entirely by the WBC controller's own
ros2_control YAML configuration. This package never sends `/fsm_command`.
If the controller leaves MOVEJ, an input becomes stale, or an action fails
validation, publication pauses immediately.

## Run

Use one Python environment containing ROS2 Jazzy `rclpy`, AccVLA dependencies,
PyTorch, OpenCV, MMEngine, safetensors, and the pinned visualization packages.

**x86_64 (with an x86 ROS2 install to set up):**

```bash
conda activate fluxvla_infer
bash scripts/install_env.sh real-only --with-ros2
```

`--with-ros2` installs the same ML dependencies as `scripts/install_env.sh
real-only`, then adds the pinned visualization packages above and installs
ROS2 Jazzy (`ros-jazzy-ros-base`) via apt if `rclpy` isn't already importable.
Set `ROS2_INSTALL=never` to only check, or `ROS2_INSTALL=always` to force a
reinstall; installing ROS2 runs `apt-get` as root and adds the ros2.org apt
source system-wide. There used to be a separate `deploy/install_env.sh` for
this; it's been folded into `scripts/install_env.sh` so there's a single
installer for both training and deploy.

**ARM/Jetson AGX Orin (`allen_agx_orin` container, ROS2 Jazzy already
installed system-wide):**

```bash
bash scripts/install_env_orin.sh
```

`scripts/install_env.sh` pulls PyTorch from `download.pytorch.org`, which has
no aarch64 wheels — it cannot install on Jetson. `install_env_orin.sh` is a
separate script for this platform: it provisions a Python 3.12 conda env
(matching the system ROS2 install's `rclpy` ABI — `rclpy` cannot be installed
via pip, so the interpreter version must match exactly), installs
PyTorch/torchvision/torchaudio/triton from the Jetson community wheel index
(`pypi.jetson-ai-lab.io`), installs cuDNN/libopenblas into the container (the
default cuDNN install path usually isn't bind-mounted from the host, unlike
the CUDA Toolkit), and builds FluxVLA's custom CUDA extensions against the
host's mounted CUDA Toolkit. See the root [`README.md`](../README.md#本分支allen_infer快速上手)
for the full breakdown and the host-side CUDA Toolkit prerequisite.

All deployment parameters are grouped at the top of `launch.sh`. Edit those
values directly when changing the model, inference mode, or safety thresholds,
then run:

```bash
./launch.sh
```

The launch script opens a live Rerun Viewer. It shows every compressed frame
from the head and wrist cameras plus measured and commanded joint-position
curves grouped into body, arms, hands, and head. Rerun is best-effort: closing
the Viewer or a visualization error does not stop inference or command safety
checks. Recordings are not written to disk.

The default task text is the parcel-sorting instruction used by the dataset.
With `AUTO_START=true`, inference starts when all camera and joint inputs are
ready. Set it to `false` to require `/xr/controller_state=15`; use value `16`
to pause.

Inference runs asynchronously: `deploy/exec_engine`'s `ChunkScheduler`
keeps a background thread producing action chunks while the 30 Hz control
loop drains one action per tick from a queue, so the control rate is never
gated on a single (much slower) model forward pass. When a new chunk lands,
the scheduler hands `FluxVLAPolicy.predict_chunk` (`deploy/model.py`) the
queue's real unconsumed tail as RTC prefix context, so consecutive chunks
splice together instead of jumping.

The interactive launcher defaults to asynchronous guidance RTC. The async
engine supplies the unconsumed queue context and the selected model applies
the guidance update. Available built-in combinations are:

| RTC | Acceleration | `INFERENCE_CONFIG` |
|---|---|---|
| `guidance` | Triton + CUDA Graph | `pi05_parcel_sort_guidance_triton_inference.py` |
| `guidance` | PyTorch | `pi05_parcel_sort_guidance_pytorch_inference.py` |
| `prefix` | Triton + CUDA Graph | `pi05_parcel_sort_prefix_triton_inference.py` |
| `prefix` | PyTorch | `pi05_parcel_sort_prefix_pytorch_inference.py` |
| off | Triton + CUDA Graph | `pi05_parcel_sort_none_triton_inference.py` |
| off | PyTorch | `pi05_parcel_sort_none_pytorch_inference.py` |

A mismatched custom config fails fast at startup instead of silently running
unguided.

## Dataset warning

The first 310 LeRobot episodes map to the original MCAP data and have the
expected joint order. The current appended dataset is incomplete:
`data/chunk-000/file-002.parquet` has no valid Parquet footer, and temporary
video files remain. Do not train or recompute statistics from the 358-episode
tree until that conversion is repaired or the dataset is rolled back to the
last complete version.
