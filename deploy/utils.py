"""pistar06 joint layout: names, and WBC/hand command composition.

Shared by ``model.py``, ``ros_node.py``, and ``rerun_visualizer.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Joint layout: canonical PI0.5 / WBC joint order for pistar06.
# ---------------------------------------------------------------------------

BODY_JOINT_NAMES = ("body_joint1", "body_joint2", "body_joint3", "body_joint4")
LEFT_ARM_JOINT_NAMES = (
    "left_joint1", "left_joint2", "left_joint3", "left_joint4",
    "left_joint5", "left_joint6", "left_joint7",
)
RIGHT_ARM_JOINT_NAMES = (
    "right_joint1", "right_joint2", "right_joint3", "right_joint4",
    "right_joint5", "right_joint6", "right_joint7",
)
HEAD_JOINT_NAMES = ("head_joint1", "head_joint2")
LEFT_HAND_JOINT_NAMES = (
    "left_hand_thumb_joint1",
    "left_hand_thumb_joint2",
    "left_hand_index_joint",
    "left_hand_middle_joint",
    "left_hand_ring_joint",
    "left_hand_pinky_joint",
)
RIGHT_HAND_JOINT_NAMES = (
    "right_hand_thumb_joint1",
    "right_hand_thumb_joint2",
    "right_hand_index_joint",
    "right_hand_middle_joint",
    "right_hand_ring_joint",
    "right_hand_pinky_joint",
)

# Fixed model action/state order shared by the parcel-sort inference configs.
# ros_node.py's _build_sample/build_joint_commands use this as their source of
# truth.
MODEL_JOINT_NAMES = (
    "body_joint3", "body_joint4",
    "left_joint1", "left_joint2", "left_joint3", "left_joint4",
    "left_joint5", "left_joint6", "left_joint7",
    "right_joint1", "right_joint2", "right_joint3", "right_joint4",
    "right_joint5", "right_joint6", "right_joint7",
    "left_hand_thumb_joint1", "left_hand_thumb_joint2", "left_hand_index_joint",
    "left_hand_middle_joint", "left_hand_ring_joint", "left_hand_pinky_joint",
    "right_hand_thumb_joint1", "right_hand_thumb_joint2", "right_hand_index_joint",
    "right_hand_middle_joint", "right_hand_ring_joint", "right_hand_pinky_joint",
)
MODEL_JOINT_DIM = len(MODEL_JOINT_NAMES)
MODEL_TENSOR_DIM = 32
HAND_JOINT_DIM = len(LEFT_HAND_JOINT_NAMES)

# The fully closed targets used by the parcel-sort demonstrations.  These are
# robot-space joint positions (not normalized model values) in the hand joint
# order above.  Keeping one canonical vector for both hands also avoids using
# isolated dataset outliers as physical joint limits.
DEFAULT_HAND_CLOSED_POSITIONS = (
    0.99,
    1.39,
    0.504,
    0.504,
    0.504,
    0.504,
)

# Verified against /ocs2_wbc_controller's live `joints` parameter.
WBC_JOINT_NAMES = (
    BODY_JOINT_NAMES
    + LEFT_ARM_JOINT_NAMES
    + RIGHT_ARM_JOINT_NAMES
    + HEAD_JOINT_NAMES
)
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


def snap_hand_joints_closed(
    actions: np.ndarray,
    *,
    threshold: float,
    closed_positions: Sequence[float] = DEFAULT_HAND_CLOSED_POSITIONS,
) -> np.ndarray:
    """Snap confident hand-joint predictions to their fully closed targets.

    ``actions`` may be one action or an arbitrary leading batch/horizon shape;
    only its final 12 values (left hand x6, right hand x6) are changed.  Values
    at or below the threshold remain continuous so the model can still open a
    hand and make small approach adjustments.
    """

    result = np.asarray(actions).copy()
    if result.shape[-1] != MODEL_JOINT_DIM:
        raise ValueError(
            f"actions must end in {MODEL_JOINT_DIM} joints, got {result.shape}"
        )
    if not np.isfinite(threshold):
        raise ValueError(f"hand close threshold must be finite, got {threshold}")

    closed = np.asarray(closed_positions, dtype=result.dtype)
    if closed.shape != (HAND_JOINT_DIM,):
        raise ValueError(
            f"closed hand positions must have shape ({HAND_JOINT_DIM},), "
            f"got {closed.shape}"
        )
    if not np.isfinite(closed).all():
        raise ValueError("closed hand positions contain NaN or Inf")

    for hand_slice in (slice(16, 22), slice(22, 28)):
        hand = result[..., hand_slice]
        result[..., hand_slice] = np.where(hand > threshold, closed, hand)
    return result
