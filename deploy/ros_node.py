"""ROS2 sensor input, asynchronous inference, and guarded joint control."""

from __future__ import annotations

import io
import time
import traceback
from typing import Any

import numpy as np
from PIL import Image

import rclpy
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
    VALID_RTC_MODES,
    WorkerConfig,
    default_device,
)
from .pistar06_inference_runner import (
    ALL_JOINT_NAMES,
    HELD_JOINT_NAMES,
    LEFT_HAND_JOINT_NAMES,
    MODEL_JOINT_DIM,
    RIGHT_HAND_JOINT_NAMES,
    TRAINING_HELD_JOINT_POSITIONS,
    WBC_JOINT_NAMES,
    build_model_state,
    compose_joint_commands,
    require_joint_positions,
    validate_held_training_pose,
)
from .rtc_infer import OverlapInferEngine, PlainInferEngine, RTCInferenceEngine
from .rtc_infer.base import InferBackend
from .rtc_infer.config import OverlapConfig, PlainConfig, RTCConfig
from .rerun_visualizer import RerunVisualizer


CTRL_START = 15
CTRL_STOP = 16
WBC_FSM_MOVEJ = 4


class AsyncJointInferenceNode(Node):
    """PI0.5 inference node with fail-closed command publication."""

    def __init__(self, *, default_rtc_mode: str = "plain") -> None:
        super().__init__("pistar06_inference_node")
        startup_started = time.perf_counter()

        self._declare_parameters(default_rtc_mode)
        self._read_parameters(default_rtc_mode)

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
        self._last_log_time = time.monotonic()
        self._shutdown_started = False
        self._inference_enabled = False
        self._action_step_count = 0
        self._stat_steps = 0
        self._stat_missing_count = 0
        self._stat_publish_count = 0
        self._stat_queue_empty = 0
        self._wbc_interpolation_type: str | None = None
        self._wbc_joint_names: tuple[str, ...] | None = None
        self._wbc_param_query_pending = False
        self._wbc_original_interpolation_type: str | None = None
        self._hand_param_query_pending: set[str] = set()
        self._hand_joint_names: dict[str, tuple[str, ...]] = {}
        self._wbc_fsm_state: int | None = None
        self._held_joint_positions: dict[str, float] | None = None
        self._backend: InferBackend = self._create_backend()
        self.get_logger().info(
            f"[startup] inference backend created: mode={self.rtc_mode} "
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

        self._request_wbc_contract()
        self._safety_timer = self.create_timer(1.0, self._audit_controller_contract)
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
            f"[startup] Inference node ready: mode={self.rtc_mode} "
            f"model_dim={MODEL_JOINT_DIM} wbc_dim={len(WBC_JOINT_NAMES)}; "
            f"{start_mode}; total={time.perf_counter() - startup_started:.2f}s"
        )

    def _declare_parameters(self, default_rtc_mode: str) -> None:
        self.declare_parameter("model_id", DEFAULT_MODEL_ID)
        self.declare_parameter("inference_config", DEFAULT_INFERENCE_CONFIG)
        self.declare_parameter("device", default_device())
        self.declare_parameter("dtype", "bf16")
        self.declare_parameter("task", DEFAULT_TASK)
        self.declare_parameter("num_inference_steps_override", 0)
        self.declare_parameter("max_normalized_action_abs", 1.25)

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
        self.declare_parameter("log_interval_s", 2.0)
        self.declare_parameter("input_timeout_s", 0.25)
        self.declare_parameter("max_camera_skew_s", 0.10)
        self.declare_parameter("auto_start", True)
        self.declare_parameter("allow_shared_control", False)
        self.declare_parameter("require_direct_movej", True)
        self.declare_parameter("manage_wbc_interpolation", True)
        self.declare_parameter("max_held_pose_error_rad", 0.10)
        self.declare_parameter("wbc_controller_node", "/ocs2_wbc_controller")
        self.declare_parameter("left_hand_controller_node", "/left_hand_controller")
        self.declare_parameter("right_hand_controller_node", "/right_hand_controller")

        self.declare_parameter("rtc_mode", default_rtc_mode)
        self.declare_parameter("notify_every", 12)
        self.declare_parameter("plain_execution_fraction", 0.5)
        self.declare_parameter("overlap_steps", 15)
        self.declare_parameter("discard_latency_steps", True)
        self.declare_parameter("min_queue_keep_steps", 1)
        self.declare_parameter("action_blend_steps", 10)
        self.declare_parameter("state_fusion_alpha", 0.5)
        self.declare_parameter("rtc_infer_execution_horizon", 10)
        self.declare_parameter("rtc_infer_max_guidance_weight", 10.0)
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

    def _read_parameters(self, default_rtc_mode: str) -> None:
        def value(name: str) -> Any:
            return self.get_parameter(name).value

        self.model_id = str(value("model_id"))
        self.task = str(value("task"))
        self.robot_exec_hz = float(value("robot_exec_hz"))
        self.step_interval_s = 1.0 / max(self.robot_exec_hz, 1e-6)
        self.log_interval_s = float(value("log_interval_s"))
        self.input_timeout_s = float(value("input_timeout_s"))
        self.max_camera_skew_s = float(value("max_camera_skew_s"))
        self.auto_start = bool(value("auto_start"))
        self.allow_shared_control = bool(value("allow_shared_control"))
        self.require_direct_movej = bool(value("require_direct_movej"))
        self.manage_wbc_interpolation = bool(value("manage_wbc_interpolation"))
        self.max_held_pose_error_rad = float(value("max_held_pose_error_rad"))
        self.wbc_controller_node = str(value("wbc_controller_node"))
        self.left_hand_controller_node = str(value("left_hand_controller_node"))
        self.right_hand_controller_node = str(value("right_hand_controller_node"))

        self.rtc_mode = str(value("rtc_mode")).strip().lower()
        if self.rtc_mode not in VALID_RTC_MODES:
            self.get_logger().warn(
                f"Unknown rtc_mode={self.rtc_mode!r}; using {default_rtc_mode!r}"
            )
            self.rtc_mode = default_rtc_mode
        self.notify_every = max(1, int(value("notify_every")))
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
        self.worker_cfg = WorkerConfig(
            model_id=self.model_id,
            inference_config=str(value("inference_config")),
            device=str(value("device")),
            dtype=str(value("dtype")),
            num_inference_steps_override=int(value("num_inference_steps_override")),
            max_normalized_action_abs=float(value("max_normalized_action_abs")),
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
        from .model import build_predict_fn, build_rtc_predict_fn

        debug = bool(self.get_parameter("debug").value)
        debug_dir = str(self.get_parameter("debug_dir").value)
        if self.rtc_mode == "rtc_infer":
            rtc_cfg = RTCConfig(
                execution_horizon=int(
                    self.get_parameter("rtc_infer_execution_horizon").value
                ),
                max_guidance_weight=float(
                    self.get_parameter("rtc_infer_max_guidance_weight").value
                ),
                debug=debug,
                debug_dir=debug_dir or "/tmp/rtc_debug",
            )
            bundle = build_rtc_predict_fn(
                self.worker_cfg,
                rtc_cfg,
                log_info=self.get_logger().info,
            )
            return RTCInferenceEngine(bundle, rtc_cfg, self.robot_exec_hz)

        bundle = build_predict_fn(
            self.worker_cfg,
            log_info=self.get_logger().info,
        )
        if self.rtc_mode == "plain":
            return PlainInferEngine(
                bundle,
                PlainConfig(
                    execution_fraction=float(
                        self.get_parameter("plain_execution_fraction").value
                    ),
                    debug=debug,
                    debug_dir=debug_dir or "/tmp/plain_debug",
                ),
            )
        return OverlapInferEngine(
            bundle,
            OverlapConfig(
                overlap_steps=int(self.get_parameter("overlap_steps").value),
                action_blend_steps=int(
                    self.get_parameter("action_blend_steps").value
                ),
                state_fusion_alpha=float(
                    self.get_parameter("state_fusion_alpha").value
                ),
                discard_latency_steps=bool(
                    self.get_parameter("discard_latency_steps").value
                ),
                min_queue_keep_steps=max(
                    0, int(self.get_parameter("min_queue_keep_steps").value)
                ),
                debug=debug,
                debug_dir=debug_dir or "/tmp/overlap_debug",
            ),
            robot_exec_hz=self.robot_exec_hz,
        )

    def _request_wbc_contract(self) -> None:
        if not hasattr(self, "_wbc_param_client"):
            self._wbc_param_client = AsyncParameterClient(
                self, self.wbc_controller_node
            )
        if self._wbc_param_query_pending:
            return
        if not self._wbc_param_client.wait_for_services(timeout_sec=0.0):
            return
        self._wbc_param_query_pending = True
        future = self._wbc_param_client.get_parameters(
            ["movej_interpolation_type", "joints"]
        )

        def done(result_future: Any) -> None:
            self._wbc_param_query_pending = False
            try:
                result = result_future.result()
                previous_interpolation_type = self._wbc_interpolation_type
                self._wbc_interpolation_type = result.values[0].string_value.strip().lower()
                self._wbc_joint_names = tuple(result.values[1].string_array_value)
                if self._wbc_original_interpolation_type is None:
                    self._wbc_original_interpolation_type = self._wbc_interpolation_type
                    self.get_logger().info(
                        "[startup] WBC interpolation read: "
                        f"original={self._wbc_original_interpolation_type!r}"
                    )
                    if (
                        self.manage_wbc_interpolation
                        and self.require_direct_movej
                        and self._wbc_interpolation_type != "none"
                    ):
                        self._request_wbc_interpolation_override()
                elif (
                    self._wbc_interpolation_type == "none"
                    and previous_interpolation_type is None
                ):
                    self.get_logger().info(
                        "[startup] WBC interpolation override verified: value='none'"
                    )
                if self._inference_enabled:
                    parameter_error = None
                    if tuple(WBC_JOINT_NAMES) != self._wbc_joint_names:
                        parameter_error = "WBC joint order changed"
                    elif (
                        self.require_direct_movej
                        and self._wbc_interpolation_type != "none"
                    ):
                        parameter_error = "WBC interpolation mode changed"
                    if parameter_error is not None:
                        self.get_logger().error(
                            f"{parameter_error} while inference was active; pausing publication"
                        )
                        self._set_inference_enabled(False)
            except Exception as exc:
                self.get_logger().error(f"Could not query WBC interpolation mode: {exc}")

        future.add_done_callback(done)

    def _request_wbc_interpolation_override(self) -> None:
        future = self._wbc_param_client.set_parameters_atomically(
            [Parameter("movej_interpolation_type", value="none")]
        )

        def done(result_future: Any) -> None:
            try:
                response = result_future.result()
                successful = bool(response.result.successful)
                if not successful:
                    reason = response.result.reason or "controller rejected parameter"
                    self.get_logger().error(
                        f"Failed to set WBC interpolation to 'none': {reason}"
                    )
                    return
                self._wbc_interpolation_type = None
                self.get_logger().info(
                    "[startup] WBC interpolation temporarily set to 'none'; "
                    "waiting for read-back verification"
                )
                self._request_wbc_contract()
            except Exception as exc:
                self.get_logger().error(
                    f"Failed to set WBC interpolation to 'none': {exc}"
                )

        future.add_done_callback(done)

    def _audit_controller_contract(self) -> None:
        if self._shutdown_started:
            return
        self._request_wbc_contract()
        self._request_hand_contract(
            "left", self.left_hand_controller_node, LEFT_HAND_JOINT_NAMES
        )
        self._request_hand_contract(
            "right", self.right_hand_controller_node, RIGHT_HAND_JOINT_NAMES
        )
        if not self._inference_enabled:
            return
        error = self._controller_contract_error()
        if error is not None:
            self.get_logger().error(f"Controller safety contract changed: {error}")
            self._set_inference_enabled(False)

    def _request_hand_contract(
        self, side: str, controller_node: str, expected_joints: list[str]
    ) -> None:
        client_name = f"_{side}_hand_param_client"
        if not hasattr(self, client_name):
            setattr(self, client_name, AsyncParameterClient(self, controller_node))
        client = getattr(self, client_name)
        if side in self._hand_param_query_pending:
            return
        if not client.wait_for_services(timeout_sec=0.0):
            return
        self._hand_param_query_pending.add(side)
        future = client.get_parameters(["joints"])

        def done(result_future: Any) -> None:
            self._hand_param_query_pending.discard(side)
            try:
                result = result_future.result()
                actual = tuple(result.values[0].string_array_value)
                self._hand_joint_names[side] = actual
                if self._inference_enabled and actual != tuple(expected_joints):
                    self.get_logger().error(
                        f"{side} hand controller joint order changed; pausing publication"
                    )
                    self._set_inference_enabled(False)
            except Exception as exc:
                self.get_logger().error(
                    f"Could not query {side} hand controller joint order: {exc}"
                )

        future.add_done_callback(done)

    def _other_command_publishers(self) -> list[str]:
        publishers = []
        for topic in (
            self.wbc_command_topic,
            self.left_hand_topic,
            self.right_hand_topic,
        ):
            for endpoint in self.get_publishers_info_by_topic(topic):
                if endpoint.node_name in (self.get_name(), "rviz"):
                    continue
                node = f"{endpoint.node_namespace.rstrip('/')}/{endpoint.node_name}"
                publishers.append(f"{topic} <- {node}")
        return publishers

    def _controller_contract_error(self) -> str | None:
        if self._wbc_fsm_state != WBC_FSM_MOVEJ:
            return (
                f"WBC must be in MOVEJ (fsm_state={WBC_FSM_MOVEJ}); "
                f"current value is {self._wbc_fsm_state!r}"
            )
        if self._wbc_joint_names is None:
            return "WBC joint order has not been verified"
        expected_joints = tuple(WBC_JOINT_NAMES)
        if self._wbc_joint_names != expected_joints:
            return (
                "WBC joint order does not match deployment contract; "
                f"expected={expected_joints}, actual={self._wbc_joint_names}"
            )
        expected_hands = {
            "left": tuple(LEFT_HAND_JOINT_NAMES),
            "right": tuple(RIGHT_HAND_JOINT_NAMES),
        }
        for side, expected in expected_hands.items():
            actual = self._hand_joint_names.get(side)
            if actual is None:
                return f"{side} hand controller joint order has not been verified"
            if actual != expected:
                return (
                    f"{side} hand controller joint order mismatch; "
                    f"expected={expected}, actual={actual}"
                )
        if self.require_direct_movej:
            if self._wbc_interpolation_type is None:
                return "WBC movej_interpolation_type has not been verified"
            if self._wbc_interpolation_type != "none":
                return (
                    "30 Hz learned joint commands require WBC "
                    "movej_interpolation_type='none'; current value is "
                    f"{self._wbc_interpolation_type!r}"
                )
        conflicts = self._other_command_publishers()
        if conflicts and not self.allow_shared_control:
            return (
                f"other command publishers exist: {conflicts}; "
                "stop them or explicitly set allow_shared_control=true"
            )
        return None

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
            error = self._controller_contract_error()
            if error is None:
                try:
                    self._build_sample()
                    validate_held_training_pose(
                        self.joint_positions,
                        max_error_rad=self.max_held_pose_error_rad,
                    )
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

    def _assert_input_freshness(self) -> None:
        timestamps = {
            "joint_states": self._last_joint_state_time,
            "head_image": self._last_head_rgb_time,
            "left_wrist_image": self._last_left_wrist_rgb_time,
            "right_wrist_image": self._last_right_wrist_rgb_time,
        }
        missing = [name for name, timestamp in timestamps.items() if timestamp is None]
        if missing:
            raise RuntimeError(f"missing inputs: {missing}")
        now = time.monotonic()
        stale = {
            name: now - timestamp
            for name, timestamp in timestamps.items()
            if timestamp is not None and now - timestamp > self.input_timeout_s
        }
        if stale:
            raise RuntimeError(f"stale inputs: {stale}")
        camera_times = [
            self._last_head_rgb_time,
            self._last_left_wrist_rgb_time,
            self._last_right_wrist_rgb_time,
        ]
        finite_times = [timestamp for timestamp in camera_times if timestamp is not None]
        if max(finite_times) - min(finite_times) > self.max_camera_skew_s:
            raise RuntimeError("camera receipt-time skew exceeds max_camera_skew_s")

    def _build_sample(self) -> dict[str, Any]:
        self._assert_input_freshness()
        require_joint_positions(self.joint_positions, ALL_JOINT_NAMES)
        if self.head_image is None or self.left_wrist_image is None or self.right_wrist_image is None:
            raise RuntimeError("decoded camera images are unavailable")
        return {
            "observation.images.base_0_rgb": self.head_image.copy(),
            "observation.images.left_wrist_0_rgb": self.left_wrist_image.copy(),
            "observation.images.right_wrist_0_rgb": self.right_wrist_image.copy(),
            "observation.state": build_model_state(self.joint_positions),
            "task": self.task,
        }

    def _control_step(self) -> None:
        if not self._inference_enabled:
            return
        if self._backend.failed:
            self.get_logger().error("Inference backend failed; pausing publication")
            self._set_inference_enabled(False)
            return

        self._action_step_count += 1
        should_notify = (
            self._backend.qsize() == 0
            if self.rtc_mode == "plain"
            else self._action_step_count % self.notify_every == 1
        )
        if should_notify:
            try:
                self._backend.notify_obs(self._build_sample())
            except Exception as exc:
                self._log_waiting(exc)
                self._stat_missing_count += 1

        action = self._backend.get_action()
        if action is None:
            self._stat_queue_empty += 1
        else:
            try:
                self._publish_joint_action(action)
                self._stat_publish_count += 1
            except Exception as exc:
                self.get_logger().error(
                    f"Command validation/publication failed; pausing: {exc}\n"
                    f"{traceback.format_exc()}"
                )
                self._set_inference_enabled(False)
        self._stat_steps += 1
        self._maybe_log_stats()

    def _log_waiting(self, exc: Exception) -> None:
        now = time.monotonic()
        if now - self._last_skip_log_time > 1.0:
            self.get_logger().info(f"Waiting for synchronized inputs: {exc}")
            self._last_skip_log_time = now

    def _publish_joint_action(self, action: np.ndarray) -> None:
        self._assert_input_freshness()
        if self._wbc_fsm_state != WBC_FSM_MOVEJ:
            raise RuntimeError(
                f"WBC is not in MOVEJ (fsm_state={self._wbc_fsm_state!r})"
            )
        if self._held_joint_positions is None:
            raise RuntimeError("held joint targets were not latched at inference start")
        command_positions = dict(self.joint_positions)
        command_positions.update(self._held_joint_positions)
        commands = compose_joint_commands(action, command_positions)

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

    def _maybe_log_stats(self) -> None:
        now = time.monotonic()
        if now - self._last_log_time < self.log_interval_s:
            return
        steps = max(self._stat_steps, 1)
        self.get_logger().info(
            f"Loop stats: pub={self._stat_publish_count}/{steps} "
            f"miss={self._stat_missing_count}/{steps} "
            f"empty={self._stat_queue_empty}/{steps} buf={self._backend.qsize()}"
        )
        self._last_log_time = now
        self._stat_steps = 0
        self._stat_missing_count = 0
        self._stat_publish_count = 0
        self._stat_queue_empty = 0

    def destroy_node(self) -> bool:
        if not self._shutdown_started:
            self._shutdown_started = True
            self._inference_enabled = False
            self.timer.cancel()
            self._safety_timer.cancel()
            if self._auto_start_timer is not None:
                self._auto_start_timer.cancel()
            try:
                self._backend.stop()
            except Exception as exc:
                self.get_logger().warn(f"Failed to stop inference backend: {exc}")
            self._rerun.close()
            self._restore_wbc_interpolation()
        return super().destroy_node()

    def _restore_wbc_interpolation(self) -> None:
        original = self._wbc_original_interpolation_type
        if (
            not self.manage_wbc_interpolation
            or not self.require_direct_movej
            or original in (None, "none")
        ):
            return
        if not rclpy.ok() or not self._wbc_param_client.wait_for_services(timeout_sec=0.5):
            self.get_logger().warn(
                f"Could not restore WBC interpolation to {original!r}: "
                "parameter service is unavailable"
            )
            return
        try:
            set_future = self._wbc_param_client.set_parameters_atomically(
                [Parameter("movej_interpolation_type", value=original)]
            )
            rclpy.spin_until_future_complete(self, set_future, timeout_sec=2.0)
            if not set_future.done():
                raise TimeoutError("timed out restoring original value")
            result = set_future.result().result
            if not result.successful:
                raise RuntimeError(result.reason or "controller rejected restore")
            self.get_logger().info(
                f"WBC interpolation restored to original value {original!r}"
            )
        except Exception as exc:
            self.get_logger().warn(
                f"Could not restore WBC interpolation to {original!r}: {exc}"
            )
