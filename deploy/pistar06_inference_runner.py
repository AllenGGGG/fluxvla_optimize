"""pistar06 real-robot inference runner: joint layout, model loading, and
training-native preprocessing/postprocessing via fluxvla.engines.

Requirement #2 (see .claude/plans/fluxvla-deploy-merge.plan.md): no
hand-rolled normalize/denormalize/prompt-building code lives here or in
model.py. ``PIStar06InferenceRunner`` subclasses
``fluxvla.engines.runners.base_inference_runner.BaseInferenceRunner``, whose
``__init__`` already builds the model, dataset transform pipeline, and
denormalize transform straight from ``configs/pi05/pi05_parcel_sort_inference.py``
via ``build_vla_from_cfg``/``build_dataset_from_cfg``/``build_transform_from_cfg``.

This runner's ``run()``/``_run_episode()`` (the ROS1 blocking loop inherited
from ``BaseInferenceRunner``) is intentionally never called: pistar06 is
ROS2/rclpy and timer-driven (see ``ros_node.AsyncJointInferenceNode``), which
owns all ROS I/O and safety gates directly and pushes observations into
``model.FluxVLAPolicyWorker``, which in turn calls this runner's
``dataset``/``vla``/``denormalize_action`` directly as phases. Consequently
``get_ros_observation``/``update_observation_window``/``_execute_actions``
below are not exercised by the real control path; they raise
``NotImplementedError`` rather than silently duplicating logic that already
lives, verbatim, in ``ros_node.py``.

The joint-name mapping/composition logic formerly in a standalone
``joint_layout.py`` lives inline below (requirement #4) since this runner is
where it is used (``ros_node.py`` and ``rerun_visualizer.py`` import it from
here).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import numpy as np

from fluxvla.engines.runners.base_inference_runner import BaseInferenceRunner
from fluxvla.engines.utils.root import OPERATORS, RUNNERS

# ---------------------------------------------------------------------------
# Joint layout (formerly joint_layout.py; canonical PI0.5 / WBC joint order
# for pistar06 - folded in here since this runner is its only real user).
# ---------------------------------------------------------------------------

BODY_JOINT_NAMES = [f"body_joint{i}" for i in range(1, 5)]
LEFT_ARM_JOINT_NAMES = [f"left_joint{i}" for i in range(1, 8)]
RIGHT_ARM_JOINT_NAMES = [f"right_joint{i}" for i in range(1, 8)]
HEAD_JOINT_NAMES = [f"head_joint{i}" for i in range(1, 3)]
LEFT_HAND_JOINT_NAMES = [
    "left_hand_thumb_joint1",
    "left_hand_thumb_joint2",
    "left_hand_index_joint",
    "left_hand_middle_joint",
    "left_hand_ring_joint",
    "left_hand_pinky_joint",
]
RIGHT_HAND_JOINT_NAMES = [
    "right_hand_thumb_joint1",
    "right_hand_thumb_joint2",
    "right_hand_index_joint",
    "right_hand_middle_joint",
    "right_hand_ring_joint",
    "right_hand_pinky_joint",
]

# This order is shared by the LeRobot feature metadata and pi05_parcel_sort.py.
MODEL_JOINT_NAMES = (
    "body_joint3", "body_joint4",
    "left_joint1", "left_joint2", "left_joint3", "left_joint4", "left_joint5", "left_joint6", "left_joint7",
    "right_joint1", "right_joint2", "right_joint3", "right_joint4", "right_joint5", "right_joint6", "right_joint7",
    "left_hand_thumb_joint1", "left_hand_thumb_joint2", "left_hand_index_joint", "left_hand_middle_joint", "left_hand_ring_joint", "left_hand_pinky_joint",
    "right_hand_thumb_joint1", "right_hand_thumb_joint2", "right_hand_index_joint", "right_hand_middle_joint", "right_hand_ring_joint", "right_hand_pinky_joint",
)
MODEL_JOINT_DIM = len(MODEL_JOINT_NAMES)
MODEL_TENSOR_DIM = 32

# Verified against /ocs2_wbc_controller's live `joints` parameter.
WBC_JOINT_NAMES = (
    BODY_JOINT_NAMES
    + LEFT_ARM_JOINT_NAMES
    + RIGHT_ARM_JOINT_NAMES
    + HEAD_JOINT_NAMES
)
HELD_JOINT_NAMES = BODY_JOINT_NAMES[:2] + HEAD_JOINT_NAMES
TRAINING_HELD_JOINT_POSITIONS = {
    "body_joint1": -0.50,
    "body_joint2": -1.10,
    "head_joint1": 0.00,
    "head_joint2": 0.26,
}
ALL_JOINT_NAMES = WBC_JOINT_NAMES + LEFT_HAND_JOINT_NAMES + RIGHT_HAND_JOINT_NAMES


@dataclass(frozen=True)
class JointCommands:
    """Position commands in the exact order expected by each controller."""

    wbc: np.ndarray
    left_hand: np.ndarray
    right_hand: np.ndarray


def require_joint_positions(
    joint_positions: Mapping[str, float],
    names: Sequence[str] = ALL_JOINT_NAMES,
) -> None:
    """Reject partial or non-finite joint-state snapshots."""

    missing = [name for name in names if name not in joint_positions]
    if missing:
        raise ValueError(f"joint_states is missing: {missing}")
    invalid = [name for name in names if not np.isfinite(joint_positions[name])]
    if invalid:
        raise ValueError(f"joint_states contains NaN or Inf for: {invalid}")


def build_model_state(joint_positions: Mapping[str, float]) -> np.ndarray:
    """Build `[28 trained joints, 4 zero pads]` exactly as used in training."""

    require_joint_positions(joint_positions, MODEL_JOINT_NAMES)
    state = np.zeros(MODEL_TENSOR_DIM, dtype=np.float32)
    state[:MODEL_JOINT_DIM] = [joint_positions[name] for name in MODEL_JOINT_NAMES]
    return state


def validate_held_training_pose(
    joint_positions: Mapping[str, float], *, max_error_rad: float
) -> None:
    """Require unobserved joints to start near their fixed training posture."""

    require_joint_positions(joint_positions, HELD_JOINT_NAMES)
    if not np.isfinite(max_error_rad) or max_error_rad <= 0.0:
        raise ValueError("max held-pose error must be finite and greater than zero")
    errors = {
        name: abs(float(joint_positions[name]) - expected)
        for name, expected in TRAINING_HELD_JOINT_POSITIONS.items()
    }
    exceeded = {
        name: error for name, error in errors.items() if error > max_error_rad
    }
    if exceeded:
        details = ", ".join(
            f"{name} error={error:.3f} rad" for name, error in exceeded.items()
        )
        raise ValueError(
            f"unobserved joints are outside the fixed training pose: {details}"
        )


def compose_joint_commands(
    action: Sequence[float],
    joint_positions: Mapping[str, float],
) -> JointCommands:
    """Combine a 28-D model action with the four held WBC joints."""

    predicted = np.asarray(action, dtype=np.float64)
    if predicted.shape != (MODEL_JOINT_DIM,):
        raise ValueError(
            f"model action must have shape ({MODEL_JOINT_DIM},), got {predicted.shape}"
        )
    if not np.isfinite(predicted).all():
        raise ValueError("model action contains NaN or Inf")
    require_joint_positions(joint_positions)
    predicted_by_name = dict(zip(MODEL_JOINT_NAMES, predicted, strict=True))

    def value(name: str) -> float:
        return float(predicted_by_name.get(name, joint_positions[name]))

    wbc = np.asarray([value(name) for name in WBC_JOINT_NAMES], dtype=np.float64)
    left_hand = np.asarray(
        [predicted_by_name[name] for name in LEFT_HAND_JOINT_NAMES],
        dtype=np.float64,
    )
    right_hand = np.asarray(
        [predicted_by_name[name] for name in RIGHT_HAND_JOINT_NAMES],
        dtype=np.float64,
    )
    return JointCommands(wbc=wbc, left_hand=left_hand, right_hand=right_hand)


# ---------------------------------------------------------------------------
# BaseInferenceRunner unconditionally builds a ROS operator
# (fluxvla/engines/runners/base_inference_runner.py:154); pistar06's ROS2
# node (ros_node.AsyncJointInferenceNode) owns all ROS I/O itself, so this
# runner needs an operator that does nothing.
# ---------------------------------------------------------------------------


@OPERATORS.register_module()
class NullOperator:
    """No-op stand-in for BaseInferenceRunner's ROS1 operator abstraction.

    pistar06 is ROS2/rclpy; ``ros_node.AsyncJointInferenceNode`` owns all
    subscriptions/publishers directly rather than through this abstraction.
    """


@RUNNERS.register_module()
class PIStar06InferenceRunner(BaseInferenceRunner):
    """pistar06 real-robot inference runner.

    Used by ``model.FluxVLAPolicyWorker`` purely for its
    ``dataset``/``vla``/``denormalize_action`` phases (built by
    ``BaseInferenceRunner.__init__`` from ``pi05_parcel_sort_inference.py``).
    See module docstring for why the ROS-facing abstract methods below are
    not implemented against real ROS I/O.
    """

    def get_ros_observation(self):
        raise NotImplementedError(
            "PIStar06InferenceRunner is driven by push-based observations "
            "built in ros_node.AsyncJointInferenceNode._build_sample(), not "
            "BaseInferenceRunner's pull-based run()/_run_episode() loop."
        )

    def update_observation_window(self) -> Dict:
        raise NotImplementedError(
            "PIStar06InferenceRunner is driven by push-based observations "
            "built in ros_node.AsyncJointInferenceNode._build_sample(), not "
            "BaseInferenceRunner's pull-based run()/_run_episode() loop."
        )

    def _execute_actions(self, actions: np.ndarray, rate):
        raise NotImplementedError(
            "pistar06 publishes joint commands from "
            "ros_node.AsyncJointInferenceNode._publish_joint_action(), which "
            "owns the WBC/hand publishers; it calls compose_joint_commands() "
            "from this module directly rather than through this method."
        )
