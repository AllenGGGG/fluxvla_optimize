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
- WBC `movej_interpolation_type` is `none` for 30 Hz streaming targets;
- the WBC and both hand-controller `joints` parameters exactly match the
  deployment order;
- no other node publishes any of the three command topics;
- all three cameras and all 32 physical joints are present and fresh;
- model outputs are finite and within the configured normalized envelope.

With `MANAGE_WBC_INTERPOLATION=true`, the node saves the live WBC interpolation
value, temporarily sets it to `none` through the ROS parameter service, and
verifies the read-back before publishing. On a normal shutdown it restores the
original value. No files under `~/fa_w2_ws` are modified, and this package never
sends `/fsm_command`. If the controller leaves MOVEJ, its interpolation mode
changes, an input becomes stale, or an action fails validation, publication
pauses immediately.

For a learned 30 Hz position stream, `none` is required because each new sample
would otherwise restart interpolation. The runtime override is intentionally
temporary so the control stack's persistent configuration remains available to
other workflows.

## Run

Use one Python environment containing ROS2 Jazzy `rclpy`, AccVLA dependencies,
PyTorch, OpenCV, MMEngine, safetensors, and the pinned visualization packages:

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

`RTC_METHOD` (default `none`) is fluxvla's own RTC guidance -- the async
engine above only supplies the prefix context, it does not reimplement the
guidance math. It needs a matching `INFERENCE_CONFIG`, since which model
class actually implements it depends on the method:

| `RTC_METHOD` | `INFERENCE_CONFIG` | Backend |
|---|---|---|
| `none` | `pi05_parcel_sort_inference.py` | Triton (no RTC) |
| `prefix` | `pi05_parcel_sort_inference_prefix_rtc.py` | Triton (`PI05FlowMatchingRTCInference`) |
| `guidance` | `pi05_parcel_sort_inference_guidance_rtc.py` | eager PyTorch (`PI05FlowMatching`), no CUDA graph, slower |

Pointing `RTC_METHOD` at a config that doesn't implement it fails fast at
startup instead of silently running un-guided.

## Dataset warning

The first 310 LeRobot episodes map to the original MCAP data and have the
expected joint order. The current appended dataset is incomplete:
`data/chunk-000/file-002.parquet` has no valid Parquet footer, and temporary
video files remain. Do not train or recompute statistics from the 358-episode
tree until that conversion is repaired or the dataset is rolled back to the
last complete version.
