"""Live Rerun visualization for camera inputs and joint commands."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any

try:
    import rerun as rr
    import rerun.blueprint as rrb
except ImportError:  # Keep the control node importable when visualization is disabled.
    rr = None
    rrb = None

from .utils import (
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
        self._joint_ranges: dict[str, list[float]] = {}
        self._inference_marker_logged = False
        self.enabled = False
        # Logging calls come from ROS subscriber/timer callbacks (30Hz+ control
        # loop). Rerun's own client-side buffering doesn't protect against a
        # slow/stalled viewer backpressuring recording.log(), so operations are
        # queued here and executed on a dedicated worker thread -- a full queue
        # drops the oldest entries rather than blocking the control loop.
        self._queue: queue.Queue[Callable[[Any], None] | None] = queue.Queue(maxsize=512)
        self._worker: threading.Thread | None = None
        self._dropped_since_warn = 0
        self._last_drop_warn = 0.0
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
            self._worker = threading.Thread(
                target=self._worker_loop, name="rerun-visualizer", daemon=True
            )
            self._worker.start()
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
        for group_name in JOINT_GROUPS:
            self._recording.log(
                f"joints/{group_name}/inference_marker",
                rr.SeriesPoints(
                    colors=[255, 215, 0],
                    markers=rr.components.MarkerShape.Circle,
                    marker_sizes=8.0,
                    names="inference start",
                ),
                static=True,
            )
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
        """Queue a logging operation for the worker thread.

        Called from ROS callbacks on the control loop, so this must never
        block: on a full queue, the oldest queued entry is dropped in favor
        of the new one (staying current matters more than completeness for
        a live view).
        """
        if not self.enabled or self._recording is None:
            return
        try:
            self._queue.put_nowait(operation)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(operation)
            except queue.Full:
                pass
            self._dropped_since_warn += 1
            now = time.monotonic()
            if now - self._last_drop_warn > 5.0:
                self._warn(
                    f"Rerun log queue full; dropped {self._dropped_since_warn} "
                    "entries (viewer too slow to keep up)"
                )
                self._dropped_since_warn = 0
                self._last_drop_warn = now

    def _worker_loop(self) -> None:
        while True:
            operation = self._queue.get()
            if operation is None:
                return
            try:
                operation(self._recording)
            except Exception as exc:
                self._disable(f"Rerun logging failed: {exc}")
                return

    @staticmethod
    def _set_time(recording: Any, timestamp_s: float) -> None:
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
        for joint_name, value in values.items():
            group_name = JOINT_GROUP_BY_NAME.get(joint_name)
            if group_name is None:
                continue
            bounds = self._joint_ranges.setdefault(
                group_name, [float(value), float(value)]
            )
            bounds[0] = min(bounds[0], float(value))
            bounds[1] = max(bounds[1], float(value))

        def log(recording: Any) -> None:
            self._set_time(recording, timestamp_s)
            for joint_name, value in values.items():
                if joint_name not in JOINT_GROUP_BY_NAME:
                    continue
                recording.log(
                    f"{self._joint_path(joint_name)}/{kind}",
                    rr.Scalars(float(value)),
                )

        self._run(log)

    def log_inference_marker(self, *, timestamp_s: float) -> None:
        """Mark the moment a new inference chunk starts in every joint plot.

        Logs one point per joint group at a fixed height above the observed
        range, rather than the two-point-spike-plus-Clear trick used
        previously to fake a vertical separator on a scalar time series
        (unreliable — depended on Clear not retroactively hiding the points
        just logged, and on sub-millisecond time deltas rendering as a
        visible near-vertical segment).
        """
        if not self._inference_marker_logged:
            self._inference_marker_logged = True
            self._info("[rerun] logging inference-start markers")

        def log(recording: Any) -> None:
            for group_name in JOINT_GROUPS:
                lower, upper = self._joint_ranges.get(group_name, [-1.0, 1.0])
                span = max(upper - lower, 0.2)
                margin = 0.05 * span
                path = f"joints/{group_name}/inference_marker"
                self._set_time(recording, timestamp_s)
                recording.log(path, rr.Scalars(upper + margin))

        self._run(log)

    def close(self) -> None:
        # Stop new work first, but keep self._recording alive until the worker
        # has drained its queue -- clearing it early would hand queued
        # operations a None recording (they call operation(self._recording)).
        self.enabled = False
        recording = self._recording
        if self._worker is not None:
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
            self._worker.join(timeout=2.0)
            self._worker = None
        self._recording = None
        if recording is None:
            return
        try:
            recording.flush(timeout_sec=2.0)
            recording.disconnect()
        except Exception as exc:
            self._warn(f"Failed to close Rerun visualization cleanly: {exc}")
