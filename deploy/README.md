# fluxvla_deploy

Guarded ROS2 deployment for the AccVLA PI0.5 parcel-sorting policy. Runtime
model and preprocessing settings live in
`configs/pi05/{none,guidance,prefix}/`.

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

After action denormalization, each hand joint predicted above
`hand_close_threshold` (default `0.25`) is snapped to its corresponding fully
closed target (default `[0.99, 1.39, 0.504, 0.504, 0.504, 0.504]`). Values at
or below the threshold remain continuous, so opening is unaffected. The
launcher exposes these as `PISTAR_HAND_CLOSE_THRESHOLD` and
`PISTAR_HAND_CLOSED_POSITIONS`; the snapping happens before action queueing so
RTC sees the same commands that the controllers receive.

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

新设备的 CUDA、Python、FluxVLA 扩展和 ROS 2 装机步骤统一维护在根目录
[`README.md`](../README.md#环境安装)。完成安装并 source ROS 2 与机器人工作区后，
从项目根目录运行 `./deploy/launch.sh`。

The launcher prompts for the robot control frequency and, when asynchronous RTC
is enabled, the RTC execution horizon. They can also be supplied without
editing the script through `PISTAR_ROBOT_HZ` and
`PISTAR_RTC_EXECUTION_HORIZON`, for example:

```bash
PISTAR_ROBOT_HZ=45 PISTAR_RTC_EXECUTION_HORIZON=15 ./deploy/launch.sh
```

`PISTAR_RTC_EXECUTION_HORIZON` is ignored in serial execution and when RTC is
disabled, because those paths do not use an RTC handoff window.

The launch script opens a live Rerun Viewer. It shows every compressed frame
from the head and wrist cameras plus measured and commanded joint-position
curves grouped into body, arms, hands, and head. Rerun is best-effort: closing
the Viewer or a visualization error does not stop inference or command safety
checks. Recordings are not written to disk.

The default task text is the parcel-sorting instruction used by the dataset.
With `PISTAR_AUTO_START=true`, inference starts when all camera and joint inputs are
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
| `guidance` | Triton + CUDA Graph | `guidance/triton_inference.py` |
| `guidance` | PyTorch | `guidance/pytorch_inference.py` |
| `prefix` | Triton + CUDA Graph | `prefix/triton_inference.py` |
| `prefix` | PyTorch | `prefix/pytorch_inference.py` |
| off | Triton + CUDA Graph | `none/triton_inference.py` |
| off | PyTorch | `none/pytorch_inference.py` |

A mismatched custom config fails fast at startup instead of silently running
unguided.

## Dataset warning

The first 310 LeRobot episodes map to the original MCAP data and have the
expected joint order. The current appended dataset is incomplete:
`data/chunk-000/file-002.parquet` has no valid Parquet footer, and temporary
video files remain. Do not train or recompute statistics from the 358-episode
tree until that conversion is repaired or the dataset is rolled back to the
last complete version.
