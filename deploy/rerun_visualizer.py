"""Live Rerun visualization for camera inputs and joint commands."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import math
from pathlib import Path
import sys
from typing import Any

try:
    import rerun as rr
    import rerun.blueprint as rrb
except ImportError:  # Keep the control node importable when visualization is disabled.
    rr = None
    rrb = None

from .pistar06_inference_runner import (
    ALL_JOINT_NAMES,
    BODY_JOINT_NAMES,
    HEAD_JOINT_NAMES,
    LEFT_ARM_JOINT_NAMES,
    LEFT_HAND_JOINT_NAMES,
    RIGHT_ARM_JOINT_NAMES,
    RIGHT_HAND_JOINT_NAMES,
    WBC_JOINT_NAMES,
    JointCommands,
)


JOINT_GROUPS = {
    "body": BODY_JOINT_NAMES,
    "left_arm": LEFT_ARM_JOINT_NAMES,
    "right_arm": RIGHT_ARM_JOINT_NAMES,
    "left_hand": LEFT_HAND_JOINT_NAMES,
    "right_hand": RIGHT_HAND_JOINT_NAMES,
    "head": HEAD_JOINT_NAMES,
}
JOINT_GROUP_BY_NAME = {
    joint_name: group_name
    for group_name, joint_names in JOINT_GROUPS.items()
    for joint_name in joint_names
}


def _blueprint() -> Any:
    image_views = rrb.Vertical(
        rrb.Spatial2DView(name="Head camera", origin="/images/head"),
        rrb.Spatial2DView(name="Left wrist camera", origin="/images/left_wrist"),
        rrb.Spatial2DView(name="Right wrist camera", origin="/images/right_wrist"),
        name="Camera inputs",
    )
    joint_views = rrb.Vertical(
        rrb.Horizontal(
            rrb.TimeSeriesView(name="Body", origin="/joints/body"),
            rrb.TimeSeriesView(name="Head", origin="/joints/head"),
        ),
        rrb.Horizontal(
            rrb.TimeSeriesView(name="Left arm", origin="/joints/left_arm"),
            rrb.TimeSeriesView(name="Right arm", origin="/joints/right_arm"),
        ),
        rrb.Horizontal(
            rrb.TimeSeriesView(name="Left hand", origin="/joints/left_hand"),
            rrb.TimeSeriesView(name="Right hand", origin="/joints/right_hand"),
        ),
        name="Joint positions",
    )
    return rrb.Blueprint(
        rrb.Horizontal(image_views, joint_views, column_shares=[1.0, 1.5]),
        collapse_panels=True,
    )


def _media_type(image_format: str) -> str | None:
    normalized = image_format.strip().lower()
    if "jpeg" in normalized or "jpg" in normalized:
        return "image/jpeg"
    if "png" in normalized:
        return "image/png"
    return None


class RerunVisualizer:
    """Best-effort Rerun logger that never interrupts robot control."""

    def __init__(
        self,
        *,
        enabled: bool,
        warn: Callable[[str], None],
        info: Callable[[str], None] | None = None,
    ) -> None:
        self._warn = warn
        self._info = info or (lambda _message: None)
        self._recording: Any | None = None
        self.enabled = False
        if not enabled:
            return
        if rr is None or rrb is None:
            self._warn("Rerun visualization disabled: rerun-sdk is not installed")
            return
        try:
            self._recording = rr.RecordingStream("fluxvla_deploy")
            layout = _blueprint()
            rr.spawn(
                memory_limit="2GB",
                hide_welcome_screen=True,
                default_blueprint=layout,
                executable_path=str(Path(sys.executable).with_name("rerun")),
                recording=self._recording,
            )
            self._recording.send_blueprint(layout, make_active=True, make_default=True)
            self.enabled = True
            self._log_series_styles()
            self._info("[startup] Rerun Viewer initialized: app=fluxvla_deploy")
        except Exception as exc:
            self._disable(f"Rerun initialization failed: {exc}")

    def _disable(self, reason: str) -> None:
        recording = self._recording
        self._recording = None
        self.enabled = False
        self._warn(f"{reason}; inference will continue without visualization")
        if recording is not None:
            try:
                recording.disconnect()
            except Exception:
                pass

    def _log_series_styles(self) -> None:
        assert self._recording is not None
        for joint_name in ALL_JOINT_NAMES:
            base = self._joint_path(joint_name)
            self._recording.log(
                f"{base}/measured",
                rr.SeriesLines(
                    colors=[64, 190, 255],
                    widths=1.5,
                    names=f"{joint_name} measured",
                ),
                static=True,
            )
            self._recording.log(
                f"{base}/command",
                rr.SeriesLines(
                    colors=[255, 145, 48],
                    widths=2.0,
                    names=f"{joint_name} command",
                ),
                static=True,
            )

    @staticmethod
    def _joint_path(joint_name: str) -> str:
        return f"joints/{JOINT_GROUP_BY_NAME[joint_name]}/{joint_name}"

    def _run(self, operation: Callable[[Any], None]) -> None:
        if not self.enabled or self._recording is None:
            return
        try:
            operation(self._recording)
        except Exception as exc:
            self._disable(f"Rerun logging failed: {exc}")

    @staticmethod
    def _set_time(recording: Any, timestamp_s: float) -> None:
        if not math.isfinite(timestamp_s):
            raise ValueError(f"invalid ROS timestamp: {timestamp_s}")
        recording.set_time("ros_time", timestamp=timestamp_s)

    def log_image(
        self,
        camera_name: str,
        contents: bytes,
        image_format: str,
        *,
        timestamp_s: float,
    ) -> None:
        def log(recording: Any) -> None:
            self._set_time(recording, timestamp_s)
            recording.log(
                f"images/{camera_name}",
                rr.EncodedImage(
                    contents=bytes(contents),
                    media_type=_media_type(image_format),
                ),
            )

        self._run(log)

    def log_measured_joints(
        self, joint_positions: Mapping[str, float], *, timestamp_s: float
    ) -> None:
        values = {
            name: float(joint_positions[name])
            for name in ALL_JOINT_NAMES
            if name in joint_positions
        }
        self._log_joint_values(values, "measured", timestamp_s)

    def log_command(self, commands: JointCommands, *, timestamp_s: float) -> None:
        values = dict(zip(WBC_JOINT_NAMES, commands.wbc, strict=True))
        values.update(zip(LEFT_HAND_JOINT_NAMES, commands.left_hand, strict=True))
        values.update(zip(RIGHT_HAND_JOINT_NAMES, commands.right_hand, strict=True))
        self._log_joint_values(values, "command", timestamp_s)

    def _log_joint_values(
        self, values: Mapping[str, float], kind: str, timestamp_s: float
    ) -> None:
        def log(recording: Any) -> None:
            self._set_time(recording, timestamp_s)
            for joint_name, value in values.items():
                if joint_name not in JOINT_GROUP_BY_NAME:
                    continue
                if not math.isfinite(float(value)):
                    raise ValueError(f"{joint_name} has non-finite {kind} value")
                recording.log(
                    f"{self._joint_path(joint_name)}/{kind}",
                    rr.Scalars(float(value)),
                )

        self._run(log)

    def close(self) -> None:
        recording = self._recording
        self._recording = None
        self.enabled = False
        if recording is None:
            return
        try:
            recording.flush(timeout_sec=2.0)
            recording.disconnect()
        except Exception as exc:
            self._warn(f"Failed to close Rerun visualization cleanly: {exc}")
