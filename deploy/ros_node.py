"""ROS2 sensor input, selectable inference scheduling, and guarded control."""

from __future__ import annotations

import io
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
import rclpy
from PIL import Image

from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.parameter_client import AsyncParameterClient
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Float64MultiArray, Int32

from .config import (
    DEFAULT_INFERENCE_CONFIG,
    DEFAULT_MODEL_ID,
    DEFAULT_TASK,
    default_device,
    load_inference_options,
)
from .utils import (
    MODEL_JOINT_DIM,
    MODEL_JOINT_NAMES,
    MODEL_TENSOR_DIM,
    TRAINING_HELD_JOINT_POSITIONS,
    WBC_JOINT_NAMES,
    JointCommands,
)
from .exec_engine import (
    ChunkScheduler,
    ChunkSchedulerConfig,
    InferBackend,
    SerialInferenceBackend,
)
from .rerun_visualizer import RerunVisualizer


CTRL_START = 15
CTRL_STOP = 16
WBC_FSM_MOVEJ = 4


class JointInferenceNode(Node):
    """Joint inference node with fail-closed command publication."""

    @staticmethod
    def build_joint_commands(
        action: Sequence[float],
        joint_positions: Mapping[str, float],
    ) -> JointCommands:
        """Combine a 28-D model action with the four held WBC joints.

        `action` is already in MODEL_JOINT_NAMES order (body3,4 + left arm x7 +
        right arm x7 + left hand x6 + right hand x6), so this is pure slicing --
        action[:16] (body3,4+both arms) is a contiguous, order-matching block of
        WBC_JOINT_NAMES too, missing only the two held joints on each end.
        """

        action = np.asarray(action, dtype=np.float64)
        if action.shape != (MODEL_JOINT_DIM,):
            raise ValueError(
                f"model action must have shape ({MODEL_JOINT_DIM},), got {action.shape}"
            )
        if not np.isfinite(action).all():
            raise ValueError("model action contains NaN or Inf")

        wbc = np.concatenate([
            [joint_positions["body_joint1"], joint_positions["body_joint2"]],
            action[:16],
            [joint_positions["head_joint1"], joint_positions["head_joint2"]],
        ])
        return JointCommands(wbc=wbc, left_hand=action[16:22], right_hand=action[22:28])

    def __init__(self) -> None:
        super().__init__("joint_inference_node")
        startup_started = time.perf_counter()

        self._declare_parameters()
        self._read_parameters()

        self.head_image: Image.Image | None = None
        self.left_wrist_image: Image.Image | None = None
        self.right_wrist_image: Image.Image | None = None
        self.joint_positions: dict[str, float] = {}
        self._last_joint_state_time: float | None = None
        self._last_head_rgb_time: float | None = None
        self._last_left_wrist_rgb_time: float | None = None
        self._last_right_wrist_rgb_time: float | None = None
        self._logged_rgb_topics: set[str] = set()
        self._last_skip_log_time = 0.0
        self._shutdown_started = False
        self._inference_enabled = False
        self._auto_start_pending = self.auto_start
        self._action_step_count = 0
        self._last_rerun_chunk_id: int | None = None
        self._wbc_fsm_state: int | None = None
        self._held_joint_positions: dict[str, float] | None = None
        self._wbc_original_interpolation_type: str | None = None
        self._wbc_override_applied = False
        self._wbc_interpolation_ready = self._configure_wbc_interpolation()
        self._backend: InferBackend = self._create_backend()
        self.get_logger().info(
            f"[startup] inference backend created: "
            f"type={type(self._backend).__name__} "
            f"action_horizon={self._backend.n_action_steps}"
        )
        self._rerun = RerunVisualizer(
            enabled=self.rerun_enabled,
            warn=lambda message: self.get_logger().warn(message),
            info=lambda message: self.get_logger().info(message),
        )

        self.wbc_pub = self.create_publisher(
            Float64MultiArray, self.wbc_command_topic, 10
        )
        self.left_hand_pub = self.create_publisher(
            Float64MultiArray, self.left_hand_topic, 10
        )
        self.right_hand_pub = self.create_publisher(
            Float64MultiArray, self.right_hand_topic, 10
        )

        self.create_subscription(
            CompressedImage, self.head_camera_topic, self._on_head_rgb, 10
        )
        self.create_subscription(
            CompressedImage, self.left_camera_topic, self._on_left_wrist_rgb, 10
        )
        self.create_subscription(
            CompressedImage, self.right_camera_topic, self._on_right_wrist_rgb, 10
        )
        self.create_subscription(JointState, self.joint_state_topic, self._on_joint_states, 10)
        self.create_subscription(Int32, self.control_topic, self._on_controller_state, 10)
        fsm_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            Int32, self.fsm_state_topic, self._on_wbc_fsm_state, fsm_qos
        )

        self._backend.start()
        self.get_logger().info(
            f"[startup] inference backend started: "
            f"type={type(self._backend).__name__}"
        )
        self.timer = self.create_timer(self.step_interval_s, self._control_step)
        start_mode = (
            "auto-start when camera and joint inputs are ready"
            if self.auto_start
            else f"send {self.control_topic}={CTRL_START} to start control_loop"
        )
        self.get_logger().info(
            f"[startup] Inference node ready: "
            f"model_dim={MODEL_JOINT_DIM} wbc_dim={len(WBC_JOINT_NAMES)}; "
            f"{start_mode}; "
            f"total={time.perf_counter() - startup_started:.2f}s"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("model_id", DEFAULT_MODEL_ID)
        self.declare_parameter("inference_config", DEFAULT_INFERENCE_CONFIG)
        self.declare_parameter("device", default_device())
        self.declare_parameter("dtype", "bf16")
        self.declare_parameter("task", DEFAULT_TASK)
        self.declare_parameter("norm_stats_path", "")
        self.declare_parameter("tokenizer_path", "")

        # RTC/algorithm knobs default to whatever the selected inference_config
        # file's inference_options dict says (see configs/pi05/
        # pi05_parcel_sort_none_pytorch_inference.py); still declared as ROS2
        # parameters so `-p name:=value` can override them at launch.
        inference_options = load_inference_options(
            str(self.get_parameter("inference_config").value)
        )
        self._inference_option_names = list(inference_options)
        for name, default in inference_options.items():
            self.declare_parameter(name, default)

        # robot_exec_hz is a ROS2/control-loop fact -- it must match the real
        # control timer frequency below, independent of which model/RTC
        # config is loaded, so it is never sourced from inference_options.
        self.declare_parameter("robot_exec_hz", 30.0)
        self.declare_parameter("execution_mode", "async")
        self.declare_parameter("auto_start", False)
        self.declare_parameter("wbc_controller_node", "/ocs2_wbc_controller")

        self.declare_parameter("debug", False)
        self.declare_parameter("debug_dir", "")
        self.declare_parameter("rerun_enabled", True)

        self.declare_parameter(
            "head_camera_topic", "/camera_head/color/image_raw/compressed"
        )
        self.declare_parameter(
            "left_camera_topic", "/camera_left_wrist/color/image_raw/compressed"
        )
        self.declare_parameter(
            "right_camera_topic", "/camera_right_wrist/color/image_raw/compressed"
        )
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("control_topic", "/xr/controller_state")
        self.declare_parameter("fsm_state_topic", "/fsm_state")
        self.declare_parameter(
            "wbc_command_topic", "/ocs2_wbc_controller/target_joint_position"
        )
        self.declare_parameter(
            "left_hand_topic", "/left_hand_controller/target_joint_position"
        )
        self.declare_parameter(
            "right_hand_topic", "/right_hand_controller/target_joint_position"
        )

    def _read_parameters(self) -> None:
        def value(name: str) -> Any:
            return self.get_parameter(name).value

        self.model_id = str(value("model_id"))
        self.task = str(value("task"))
        self.robot_exec_hz = float(value("robot_exec_hz"))
        self.step_interval_s = 1.0 / max(self.robot_exec_hz, 1e-6)
        self.execution_mode = str(value("execution_mode")).strip().lower()
        if self.execution_mode not in {"async", "serial"}:
            raise ValueError(
                "execution_mode must be 'async' or 'serial', got "
                f"{self.execution_mode!r}"
            )
        self.auto_start = bool(value("auto_start"))
        self.wbc_controller_node = str(value("wbc_controller_node"))

        self.rerun_enabled = bool(value("rerun_enabled"))

        self.head_camera_topic = str(value("head_camera_topic"))
        self.left_camera_topic = str(value("left_camera_topic"))
        self.right_camera_topic = str(value("right_camera_topic"))
        self.joint_state_topic = str(value("joint_state_topic"))
        self.control_topic = str(value("control_topic"))
        self.fsm_state_topic = str(value("fsm_state_topic"))
        self.wbc_command_topic = str(value("wbc_command_topic"))
        self.left_hand_topic = str(value("left_hand_topic"))
        self.right_hand_topic = str(value("right_hand_topic"))

        self.inference_config = str(value("inference_config"))
        self.device = str(value("device"))
        self.dtype = str(value("dtype"))
        self.norm_stats_path = str(value("norm_stats_path")).strip() or None
        self.tokenizer_path = str(value("tokenizer_path")).strip() or None

        for name in self._inference_option_names:
            setattr(self, name, value(name))

    def _create_backend(self) -> InferBackend:
        # Keep ROS protocol inspection importable without a PyTorch installation.
        from .model import build_predict_fn

        debug = bool(self.get_parameter("debug").value)
        debug_dir = str(self.get_parameter("debug_dir").value)
        # inference_options-derived kwargs (rtc_method, rtc_execution_horizon,
        # etc.) are forwarded dynamically from self._inference_option_names --
        # the same list _read_parameters() used to set these attributes --
        # so adding a new inference_options field never requires touching
        # this call site to keep it wired through.
        bundle = build_predict_fn(
            model_id=self.model_id,
            inference_config=self.inference_config,
            device=self.device,
            dtype=self.dtype,
            norm_stats_path=self.norm_stats_path,
            tokenizer_path=self.tokenizer_path,
            log_info=self.get_logger().info,
            **{name: getattr(self, name) for name in self._inference_option_names},
        )
        predict_fn = bundle.fn

        def logged_predict(*args: Any, **kwargs: Any) -> Any:
            try:
                return predict_fn(*args, **kwargs)
            except Exception:
                self.get_logger().error(
                    "Inference predict_chunk failed:\n"
                    f"{traceback.format_exc()}"
                )
                raise

        bundle.fn = logged_predict
        if self.execution_mode == "serial":
            if self.rtc_method != "none":
                raise ValueError(
                    "serial execution requires rtc_method='none'; RTC needs "
                    "an unconsumed async queue prefix"
                )
            return SerialInferenceBackend(
                bundle,
                publish_horizon=self.chunk_publish_horizon,
            )

        return ChunkScheduler(
            bundle,
            ChunkSchedulerConfig(
                rtc_enabled=self.rtc_method != "none",
                execution_horizon=self.rtc_execution_horizon,
                replan_remaining=self.rtc_replan_remaining,
                publish_horizon=self.chunk_publish_horizon,
                rtc_publish_horizon=self.rtc_publish_horizon,
                debug=debug,
                debug_dir=debug_dir or "/tmp/rtc_debug",
            ),
            robot_exec_hz=self.robot_exec_hz,
        )

    def _on_controller_state(self, msg: Int32) -> None:
        if msg.data == CTRL_START:
            self._set_inference_enabled(True)
        elif msg.data == CTRL_STOP:
            self._auto_start_pending = False
            self._set_inference_enabled(False)

    def _maybe_auto_start(self) -> None:
        if self._auto_start_pending and self._wbc_fsm_state == WBC_FSM_MOVEJ:
            self._set_inference_enabled(True)

    def _on_wbc_fsm_state(self, msg: Int32) -> None:
        self._wbc_fsm_state = int(msg.data)
        if self._inference_enabled and self._wbc_fsm_state != WBC_FSM_MOVEJ:
            self.get_logger().error(
                f"WBC left MOVEJ (fsm_state={self._wbc_fsm_state}); pausing publication"
            )
            self._set_inference_enabled(False)
        elif self._wbc_fsm_state == WBC_FSM_MOVEJ:
            self._maybe_auto_start()

    def _set_inference_enabled(self, enabled: bool) -> None:
        if enabled:
            if self._inference_enabled:
                return
            error = None
            if not self._wbc_interpolation_ready:
                error = "WBC interpolation setup failed"
            elif self._wbc_fsm_state != WBC_FSM_MOVEJ:
                error = (
                    "WBC is not in MOVEJ: "
                    f"fsm_state={self._wbc_fsm_state}, expected={WBC_FSM_MOVEJ}"
                )
            else:
                try:
                    self._build_sample()
                except Exception as exc:
                    error = f"inputs are not ready: {exc}"
            if error is not None:
                now = time.monotonic()
                if now - self._last_skip_log_time > 1.0:
                    self.get_logger().error(f"Refusing to enable inference: {error}")
                    self._last_skip_log_time = now
                return
            self._held_joint_positions = dict(TRAINING_HELD_JOINT_POSITIONS)
            self._backend.reset()
            if hasattr(self._backend, "resume"):
                self._backend.resume()
            self._action_step_count = 0
            self._inference_enabled = True
            self._auto_start_pending = False
            self.get_logger().warn("Inference command publication ENABLED")
            return

        if not self._inference_enabled:
            return
        self._inference_enabled = False
        self._held_joint_positions = None
        self._backend.reset()
        if hasattr(self._backend, "pause"):
            self._backend.pause()
        self.get_logger().info("Inference command publication paused; WBC holds last target")

    @staticmethod
    def _decode_compressed_image(msg: CompressedImage) -> Image.Image:
        with Image.open(io.BytesIO(msg.data)) as image:
            return image.convert("RGB").copy()

    def _message_timestamp_s(self, msg: CompressedImage | JointState) -> float:
        stamp = msg.header.stamp
        timestamp_s = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        if timestamp_s <= 0.0:
            return self.get_clock().now().nanoseconds * 1e-9
        return timestamp_s

    def _store_image(
        self, topic: str, camera_name: str, msg: CompressedImage
    ) -> tuple[Image.Image, float]:
        image = self._decode_compressed_image(msg)
        self._rerun.log_image(
            camera_name,
            msg.data,
            msg.format,
            timestamp_s=self._message_timestamp_s(msg),
        )
        if topic not in self._logged_rgb_topics:
            self.get_logger().info(
                f"{topic}: decoded RGB size={image.size} format={msg.format!r}"
            )
            self._logged_rgb_topics.add(topic)
        return image, time.monotonic()

    def _on_head_rgb(self, msg: CompressedImage) -> None:
        try:
            self.head_image, self._last_head_rgb_time = self._store_image(
                self.head_camera_topic, "head", msg
            )
            self._maybe_auto_start()
        except Exception as exc:
            self.get_logger().error(f"Failed to decode head image: {exc}")

    def _on_left_wrist_rgb(self, msg: CompressedImage) -> None:
        try:
            self.left_wrist_image, self._last_left_wrist_rgb_time = self._store_image(
                self.left_camera_topic, "left_wrist", msg
            )
            self._maybe_auto_start()
        except Exception as exc:
            self.get_logger().error(f"Failed to decode left wrist image: {exc}")

    def _on_right_wrist_rgb(self, msg: CompressedImage) -> None:
        try:
            self.right_wrist_image, self._last_right_wrist_rgb_time = self._store_image(
                self.right_camera_topic, "right_wrist", msg
            )
            self._maybe_auto_start()
        except Exception as exc:
            self.get_logger().error(f"Failed to decode right wrist image: {exc}")

    def _on_joint_states(self, msg: JointState) -> None:
        if len(msg.name) != len(msg.position):
            self.get_logger().error(
                f"Ignoring malformed joint_states: {len(msg.name)} names, "
                f"{len(msg.position)} positions"
            )
            return
        updates = {
            name: float(position) for name, position in zip(msg.name, msg.position)
        }
        self.joint_positions.update(updates)
        self._last_joint_state_time = time.monotonic()
        self._rerun.log_measured_joints(
            updates, timestamp_s=self._message_timestamp_s(msg)
        )
        self._maybe_auto_start()

    def _build_sample(self) -> dict[str, Any]:
        image_sources = {
            "observation.images.base_0_rgb": self.head_image,
            "observation.images.left_wrist_0_rgb": self.left_wrist_image,
            "observation.images.right_wrist_0_rgb": self.right_wrist_image,
        }
        images: dict[str, np.ndarray] = {}
        for key, image in image_sources.items():
            if image is None:
                raise ValueError(f"missing inference image: {key}")
            array = np.asarray(image)
            if array.ndim != 3 or array.shape[-1] != 3:
                raise ValueError(
                    f"inference image {key} must be HWC RGB, got {array.shape}"
                )
            images[key] = array

        missing_joints = [
            name for name in MODEL_JOINT_NAMES if name not in self.joint_positions
        ]
        if missing_joints:
            raise ValueError(f"missing inference joints: {missing_joints}")

        state = np.zeros(MODEL_TENSOR_DIM, dtype=np.float32)
        state[:MODEL_JOINT_DIM] = [
            self.joint_positions[name] for name in MODEL_JOINT_NAMES
        ]
        # PrivateInferenceDataset.__call__ (fluxvla/datasets/parquet_dataset.py)
        # hard-requires data['qpos'] and data.get('task_description', ...), and
        # calls data[img_key].transpose(2, 0, 1) -- a numpy HWC array, not PIL.
        return {
            **images,
            "qpos": state,
            "task_description": self.task,
        }

    def _control_step(self) -> None:
        if not self._inference_enabled:
            return
        if self._backend.failed:
            self.get_logger().error(
                "Inference backend thread exited; pausing publication. "
                "See the preceding predict_chunk traceback."
            )
            self._set_inference_enabled(False)
            return

        self._action_step_count += 1
        # ChunkScheduler's background thread always picks up the latest
        # obs whenever it's ready for the next chunk (no dedup needed here);
        # feed it every tick so that obs is never staler than one control period.
        try:
            self._backend.notify_obs(self._build_sample())
        except Exception as exc:
            now = time.monotonic()
            if now - self._last_skip_log_time > 1.0:
                self.get_logger().error(f"Failed to build/send observation: {exc}")
                self._last_skip_log_time = now

        action = self._backend.get_action()
        if action is not None:
            try:
                timestamp_s = self.get_clock().now().nanoseconds * 1e-9
                chunk_id = self._backend.action_chunk_id
                if chunk_id is not None and chunk_id != self._last_rerun_chunk_id:
                    self._rerun.log_inference_marker(timestamp_s=timestamp_s)
                    self._last_rerun_chunk_id = chunk_id
                self._publish_joint_action(action, timestamp_s=timestamp_s)
            except Exception as exc:
                self.get_logger().error(
                    f"Command validation/publication failed; pausing: {exc}\n"
                    f"{traceback.format_exc()}"
                )
                self._set_inference_enabled(False)

    def _publish_joint_action(
        self, action: np.ndarray, *, timestamp_s: float
    ) -> None:
        command_positions = dict(self.joint_positions)
        command_positions.update(self._held_joint_positions)
        commands = self.build_joint_commands(action, command_positions)

        wbc_msg = Float64MultiArray()
        wbc_msg.data = commands.wbc.tolist()
        left_hand_msg = Float64MultiArray()
        left_hand_msg.data = commands.left_hand.tolist()
        right_hand_msg = Float64MultiArray()
        right_hand_msg.data = commands.right_hand.tolist()

        self.wbc_pub.publish(wbc_msg)
        self.left_hand_pub.publish(left_hand_msg)
        self.right_hand_pub.publish(right_hand_msg)
        self._rerun.log_command(
            commands,
            timestamp_s=timestamp_s,
        )

    def destroy_node(self) -> bool:
        if not self._shutdown_started:
            self._shutdown_started = True
            self._inference_enabled = False
            self.timer.cancel()
            try:
                self._backend.stop()
            except Exception as exc:
                self.get_logger().warn(f"Failed to stop inference backend: {exc}")
            self._rerun.close()
            self._restore_wbc_interpolation()
        return super().destroy_node()

    def _configure_wbc_interpolation(self) -> bool:
        """Set WBC interpolation to none for this process and remember its value."""
        self._wbc_param_client = AsyncParameterClient(self, self.wbc_controller_node)
        try:
            if not self._wbc_param_client.wait_for_services(timeout_sec=2.0):
                raise TimeoutError("WBC parameter service is unavailable")
            get_future = self._wbc_param_client.get_parameters(
                ["movej_interpolation_type"]
            )
            rclpy.spin_until_future_complete(self, get_future, timeout_sec=2.0)
            if not get_future.done():
                raise TimeoutError("timed out reading WBC interpolation")
            original = get_future.result().values[0].string_value.strip()
            if not original:
                raise RuntimeError("WBC returned an empty interpolation value")
            self._wbc_original_interpolation_type = original
            self.get_logger().info(
                f"[startup] WBC interpolation original value={original!r}"
            )
            if original.lower() == "none":
                return True

            set_future = self._wbc_param_client.set_parameters_atomically(
                [Parameter("movej_interpolation_type", value="none")]
            )
            rclpy.spin_until_future_complete(self, set_future, timeout_sec=2.0)
            if not set_future.done():
                raise TimeoutError("timed out setting WBC interpolation to 'none'")
            result = set_future.result().result
            if not result.successful:
                raise RuntimeError(result.reason or "controller rejected parameter")
            self._wbc_override_applied = True
            self.get_logger().info(
                "[startup] WBC interpolation temporarily set to 'none'"
            )
            return True
        except Exception as exc:
            self.get_logger().error(
                "WBC interpolation setup failed: "
                f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            )
            return False

    def _restore_wbc_interpolation(self) -> None:
        if not self._wbc_override_applied:
            return
        original = self._wbc_original_interpolation_type
        try:
            if original is None or not rclpy.ok():
                raise RuntimeError("ROS shutdown started before restore")
            if not self._wbc_param_client.wait_for_services(timeout_sec=2.0):
                raise TimeoutError("WBC parameter service is unavailable")
            future = self._wbc_param_client.set_parameters_atomically(
                [Parameter("movej_interpolation_type", value=original)]
            )
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            if not future.done():
                raise TimeoutError("timed out restoring original value")
            result = future.result().result
            if not result.successful:
                raise RuntimeError(result.reason or "controller rejected restore")
            self._wbc_override_applied = False
            self.get_logger().info(
                f"WBC interpolation restored to original value {original!r}"
            )
        except Exception as exc:
            self.get_logger().warn(
                "Could not restore WBC interpolation: "
                f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            )
