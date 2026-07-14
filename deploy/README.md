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
~/runtime/miniforge3/envs/fluxvla_infer/bin/python -m pip install \
  -r requirements-visualization.txt
```

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
After the node reports that all gates pass, enable it with the existing control
signal `/xr/controller_state=15`; use value `16` to pause. Keep `AUTO_START`,
`ALLOW_SHARED_CONTROL`, and `REQUIRE_DIRECT_MOVEJ` at their safe defaults for
the first robot test.

Start with `plain` mode. Validate a full episode and command traces before using
`overlap` or `rtc_infer`; RTC uses the native AccVLA guidance path and preserves
the latency alignment between the old action tail and the new chunk.

## Dataset warning

The first 310 LeRobot episodes map to the original MCAP data and have the
expected joint order. The current appended dataset is incomplete:
`data/chunk-000/file-002.parquet` has no valid Parquet footer, and temporary
video files remain. Do not train or recompute statistics from the 358-episode
tree until that conversion is repaired or the dataset is rolled back to the
last complete version.
