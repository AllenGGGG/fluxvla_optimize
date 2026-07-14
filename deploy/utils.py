"""pistar06 joint layout: names, and WBC/hand command composition.

Shared by ``model.py``, ``ros_node.py``, and ``rerun_visualizer.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

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

# Fixed model action/state order, shared by the LeRobot feature metadata and
# pi05_parcel_sort.py -- the single source of truth ros_node.py's
# _build_sample/build_joint_commands read from.
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
