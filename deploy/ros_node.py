"""ROS2 sensor input, asynchronous inference, and guarded joint control."""

from __future__ import annotations

import io
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
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
    VALID_NORM_TYPES,
    VALID_RTC_METHODS,
    WorkerConfig,
    default_device,
)
from .utils import (
    MODEL_JOINT_DIM,
    MODEL_JOINT_NAMES,
    MODEL_TENSOR_DIM,
    TRAINING_HELD_JOINT_POSITIONS,
    WBC_JOINT_NAMES,
    JointCommands,
)
from .exec_engine import ChunkScheduler, ChunkSchedulerConfig, InferBackend
from .rerun_visualizer import RerunVisualizer


CTRL_START = 15
CTRL_STOP = 16
WBC_FSM_MOVEJ = 4


class JointInferenceNode(Node):
    """PI0.5 inference node with fail-closed command publication."""

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
        super().__init__("pistar06_inference_node")
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
        self._action_step_count = 0
        self._wbc_fsm_state: int | None = None
        self._held_joint_positions: dict[str, float] | None = None
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

        self._wbc_override_timer = (
            self.create_timer(1.0, self._request_wbc_interpolation_override)
            if self.manage_wbc_interpolation and self.require_direct_movej
            else None
        )
        self._backend.start()
        if hasattr(self._backend, "pause"):
            self._backend.pause()
        self.get_logger().info(
            f"[startup] inference backend thread started: "
            f"type={type(self._backend).__name__} paused=true"
        )
        self.timer = self.create_timer(self.step_interval_s, self._control_step)
        self._auto_start_timer = (
            self.create_timer(1.0, self._try_auto_start) if self.auto_start else None
        )

        start_mode = (
            "waiting for safety gates, then control_loop starts automatically"
            if self.auto_start
            else f"send {self.control_topic}={CTRL_START} to start control_loop"
        )
        self.get_logger().info(
            f"[startup] Inference node ready: "
            f"model_dim={MODEL_JOINT_DIM} wbc_dim={len(WBC_JOINT_NAMES)}; "
            f"{start_mode}; total={time.perf_counter() - startup_started:.2f}s"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("model_id", DEFAULT_MODEL_ID)
        self.declare_parameter("inference_config", DEFAULT_INFERENCE_CONFIG)
        self.declare_parameter("device", default_device())
        self.declare_parameter("dtype", "bf16")
        self.declare_parameter("task", DEFAULT_TASK)
        self.declare_parameter("num_inference_steps_override", 0)
        self.declare_parameter("norm_stats_path", "")
        self.declare_parameter("norm_type", "quantile")

        # fluxvla RTC guidance (see model.py's FluxVLAPolicy._build_rtc_kwargs
        # docstring; default 'none'). Fed by deploy/exec_engine's
        # ChunkScheduler, which supplies the real unconsumed queue tail
        # as prefix context on every call.
        self.declare_parameter("rtc_method", "none")
        self.declare_parameter("rtc_execution_horizon", 10)
        self.declare_parameter("rtc_max_guidance_weight", 10.0)
        self.declare_parameter("rtc_schedule", "linear")

        # CFG / advantage-conditioning (see model.py's apply_cfg_blend
        # docstring; inert until a checkpoint is trained on advantage-tagged
        # data - this is infrastructure for collecting that signal now).
        self.declare_parameter("advantage_enabled", False)
        self.declare_parameter("cfg_enabled", False)
        self.declare_parameter("cfg_scale", 2.0)
        self.declare_parameter("cfg_scale_joint", -1.0)
        self.declare_parameter("cfg_scale_gripper", -1.0)
        self.declare_parameter("cfg_cond_advantage_tag", "positive")
        self.declare_parameter("cfg_uncond_advantage_tag", "")
        self.declare_parameter("arm_dof", 7)

        self.declare_parameter("robot_exec_hz", 30.0)
        self.declare_parameter("auto_start", True)
        self.declare_parameter("require_direct_movej", True)
        self.declare_parameter("manage_wbc_interpolation", True)
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
        self.auto_start = bool(value("auto_start"))
        self.require_direct_movej = bool(value("require_direct_movej"))
        self.manage_wbc_interpolation = bool(value("manage_wbc_interpolation"))
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

        cfg_uncond_advantage_tag = str(value("cfg_uncond_advantage_tag")).strip()
        norm_type = str(value("norm_type")).strip().lower()
        if norm_type not in VALID_NORM_TYPES:
            self.get_logger().warn(
                f"Unknown norm_type={norm_type!r}; using 'quantile'"
            )
            norm_type = "quantile"
        rtc_method = str(value("rtc_method")).strip().lower()
        if rtc_method not in VALID_RTC_METHODS:
            self.get_logger().warn(
                f"Unknown rtc_method={rtc_method!r}; using 'none'"
            )
            rtc_method = "none"
        self.worker_cfg = WorkerConfig(
            model_id=self.model_id,
            inference_config=str(value("inference_config")),
            device=str(value("device")),
            dtype=str(value("dtype")),
            num_inference_steps_override=int(value("num_inference_steps_override")),
            norm_stats_path=str(value("norm_stats_path")).strip() or None,
            norm_type=norm_type,
            rtc_method=rtc_method,
            rtc_execution_horizon=int(value("rtc_execution_horizon")),
            rtc_max_guidance_weight=float(value("rtc_max_guidance_weight")),
            rtc_schedule=str(value("rtc_schedule")).strip().lower(),
            advantage_enabled=bool(value("advantage_enabled")),
            cfg_enabled=bool(value("cfg_enabled")),
            cfg_scale=float(value("cfg_scale")),
            cfg_scale_joint=float(value("cfg_scale_joint")),
            cfg_scale_gripper=float(value("cfg_scale_gripper")),
            cfg_cond_advantage_tag=str(value("cfg_cond_advantage_tag")).strip() or None,
            cfg_uncond_advantage_tag=cfg_uncond_advantage_tag or None,
            arm_dof=int(value("arm_dof")),
        )

    def _create_backend(self) -> InferBackend:
        # Keep ROS protocol inspection importable without a PyTorch installation.
        from .model import build_predict_fn

        debug = bool(self.get_parameter("debug").value)
        debug_dir = str(self.get_parameter("debug_dir").value)
        bundle = build_predict_fn(
            self.worker_cfg,
            log_info=self.get_logger().info,
        )
        return ChunkScheduler(
            bundle,
            ChunkSchedulerConfig(
                execution_horizon=self.worker_cfg.rtc_execution_horizon,
                debug=debug,
                debug_dir=debug_dir or "/tmp/rtc_debug",
            ),
            robot_exec_hz=self.robot_exec_hz,
        )

    def _request_wbc_interpolation_override(self) -> None:
        """Set WBC movej_interpolation_type='none' (required for a raw 30Hz
        joint-command stream); retries every second (see the timer in
        __init__) until it succeeds once, then self-cancels."""
        if not hasattr(self, "_wbc_param_client"):
            self._wbc_param_client = AsyncParameterClient(
                self, self.wbc_controller_node
            )
        if not self._wbc_param_client.wait_for_services(timeout_sec=0.0):
            return
        future = self._wbc_param_client.set_parameters_atomically(
            [Parameter("movej_interpolation_type", value="none")]
        )

        def done(result_future: Any) -> None:
            try:
                response = result_future.result()
                if not response.result.successful:
                    reason = response.result.reason or "controller rejected parameter"
                    self.get_logger().error(
                        f"Failed to set WBC interpolation to 'none': {reason}"
                    )
                    return
                self.get_logger().info("[startup] WBC interpolation set to 'none'")
                if self._wbc_override_timer is not None:
                    self._wbc_override_timer.cancel()
            except Exception as exc:
                self.get_logger().error(
                    f"Failed to set WBC interpolation to 'none': {exc}"
                )

        future.add_done_callback(done)

    def _try_auto_start(self) -> None:
        if self._shutdown_started:
            if self._auto_start_timer is not None:
                self._auto_start_timer.cancel()
            return
        if self._inference_enabled:
            if self._auto_start_timer is not None:
                self._auto_start_timer.cancel()
            return
        self._set_inference_enabled(True)

    def _on_controller_state(self, msg: Int32) -> None:
        if msg.data == CTRL_START:
            self._set_inference_enabled(True)
        elif msg.data == CTRL_STOP:
            self._set_inference_enabled(False)

    def _on_wbc_fsm_state(self, msg: Int32) -> None:
        self._wbc_fsm_state = int(msg.data)
        if self._inference_enabled and self._wbc_fsm_state != WBC_FSM_MOVEJ:
            self.get_logger().error(
                f"WBC left MOVEJ (fsm_state={self._wbc_fsm_state}); pausing publication"
            )
            self._set_inference_enabled(False)

    def _set_inference_enabled(self, enabled: bool) -> None:
        if enabled:
            if self._inference_enabled:
                return
            error = None
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
        except Exception as exc:
            self.get_logger().error(f"Failed to decode head image: {exc}")

    def _on_left_wrist_rgb(self, msg: CompressedImage) -> None:
        try:
            self.left_wrist_image, self._last_left_wrist_rgb_time = self._store_image(
                self.left_camera_topic, "left_wrist", msg
            )
        except Exception as exc:
            self.get_logger().error(f"Failed to decode left wrist image: {exc}")

    def _on_right_wrist_rgb(self, msg: CompressedImage) -> None:
        try:
            self.right_wrist_image, self._last_right_wrist_rgb_time = self._store_image(
                self.right_camera_topic, "right_wrist", msg
            )
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

    def _build_sample(self) -> dict[str, Any]:
        state = np.zeros(MODEL_TENSOR_DIM, dtype=np.float32)
        state[:MODEL_JOINT_DIM] = [
            self.joint_positions[name] for name in MODEL_JOINT_NAMES
        ]
        # PrivateInferenceDataset.__call__ (fluxvla/datasets/parquet_dataset.py)
        # hard-requires data['qpos'] and data.get('task_description', ...), and
        # calls data[img_key].transpose(2, 0, 1) -- a numpy HWC array, not PIL.
        return {
            "observation.images.base_0_rgb": np.asarray(self.head_image),
            "observation.images.left_wrist_0_rgb": np.asarray(self.left_wrist_image),
            "observation.images.right_wrist_0_rgb": np.asarray(self.right_wrist_image),
            "qpos": state,
            "task_description": self.task,
        }

    def _control_step(self) -> None:
        if not self._inference_enabled:
            return
        if self._backend.failed:
            self.get_logger().error("Inference backend failed; pausing publication")
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
                self._publish_joint_action(action)
            except Exception as exc:
                self.get_logger().error(
                    f"Command validation/publication failed; pausing: {exc}\n"
                    f"{traceback.format_exc()}"
                )
                self._set_inference_enabled(False)

    def _publish_joint_action(self, action: np.ndarray) -> None:
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
            timestamp_s=self.get_clock().now().nanoseconds * 1e-9,
        )

    def destroy_node(self) -> bool:
        if not self._shutdown_started:
            self._shutdown_started = True
            self._inference_enabled = False
            self.timer.cancel()
            if self._wbc_override_timer is not None:
                self._wbc_override_timer.cancel()
            if self._auto_start_timer is not None:
                self._auto_start_timer.cancel()
            try:
                self._backend.stop()
            except Exception as exc:
                self.get_logger().warn(f"Failed to stop inference backend: {exc}")
            self._rerun.close()
        return super().destroy_node()
