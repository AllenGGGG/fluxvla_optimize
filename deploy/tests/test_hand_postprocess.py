import numpy as np
import pytest

from deploy.utils import DEFAULT_HAND_CLOSED_POSITIONS, snap_hand_joints_closed


def test_snaps_each_hand_joint_above_threshold() -> None:
    actions = np.zeros((2, 28), dtype=np.float32)
    actions[0, 16:22] = [0.24, 0.26, 0.50, 0.10, 0.30, 0.25]
    actions[1, 22:28] = [0.90, 1.20, 0.20, 0.40, 0.01, 0.60]

    result = snap_hand_joints_closed(actions, threshold=0.25)

    np.testing.assert_allclose(
        result[0, 16:22], [0.24, 1.39, 0.504, 0.10, 0.504, 0.25]
    )
    np.testing.assert_allclose(
        result[1, 22:28], [0.99, 1.39, 0.20, 0.504, 0.01, 0.504]
    )
    np.testing.assert_allclose(
        actions[0, 16:22], [0.24, 0.26, 0.50, 0.10, 0.30, 0.25]
    )


def test_supports_a_single_action_and_custom_closed_positions() -> None:
    action = np.zeros(28, dtype=np.float64)
    action[16:28] = 0.3
    closed = [1, 2, 3, 4, 5, 6]

    result = snap_hand_joints_closed(
        action, threshold=0.25, closed_positions=closed
    )

    np.testing.assert_array_equal(result[16:22], closed)
    np.testing.assert_array_equal(result[22:28], closed)


@pytest.mark.parametrize(
    ("actions", "threshold", "closed_positions"),
    [
        (np.zeros(27), 0.25, DEFAULT_HAND_CLOSED_POSITIONS),
        (np.zeros(28), float("nan"), DEFAULT_HAND_CLOSED_POSITIONS),
        (np.zeros(28), 0.25, [1.0] * 5),
    ],
)
def test_rejects_invalid_configuration(actions, threshold, closed_positions) -> None:
    with pytest.raises(ValueError):
        snap_hand_joints_closed(
            actions, threshold=threshold, closed_positions=closed_positions
        )
