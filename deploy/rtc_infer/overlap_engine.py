"""OverlapInferEngine — background-thread overlap inference."""
from __future__ import annotations

import logging
import math
import threading
import time
import traceback
from collections import deque
from typing import Callable

import numpy as np

from .config import OverlapConfig
from .latency_tracker import LatencyTracker
from .visualizer import RTCDebugVisualizer

logger = logging.getLogger(__name__)


class OverlapInferEngine:
    def __init__(
        self,
        predict_fn: Callable,
        cfg: OverlapConfig,
        *,
        robot_exec_hz: float = 30.0,
    ) -> None:
        self._predict_fn = predict_fn
        self._cfg = cfg
        self._robot_exec_hz = max(robot_exec_hz, 1e-6)
        self.n_action_steps: int = predict_fn.n_action_steps

        self._buffer: deque[np.ndarray] = deque()
        self._buffer_lock = threading.Lock()
        self._obs: dict | None = None
        self._obs_lock = threading.Lock()
        self._obs_event = threading.Event()
        self._policy_active = threading.Event()
        self._last_published: np.ndarray | None = None
        self._consumed_count = 0
        self._chunk_counter = 0
        self._thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()
        self._exc: BaseException | None = None
        self._latency_tracker = LatencyTracker()
        self._generation = 0
        self._lifecycle_lock = threading.Lock()

        debug_dir = cfg.debug_dir
        self._chunk_debugger: RTCDebugVisualizer | None = (
            RTCDebugVisualizer(debug_dir, execution_horizon=cfg.overlap_steps)
            if cfg.debug else None
        )

    def start(self) -> None:
        self._shutdown_event.clear()
        self._exc = None
        self._latency_tracker.reset()
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="overlap-infer")
        self._thread.start()
        logger.info("OverlapInferEngine started")

    def stop(self) -> None:
        self._shutdown_event.set()
        self._policy_active.clear()
        self._obs_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._chunk_debugger is not None:
            self._chunk_debugger.close()
        logger.info("OverlapInferEngine stopped")

    def pause(self) -> None:
        self._policy_active.clear()

    def resume(self) -> None:
        self._policy_active.set()

    def reset(self) -> None:
        with self._lifecycle_lock:
            with self._buffer_lock:
                self._buffer.clear()
            with self._obs_lock:
                self._obs = None
            self._generation += 1
        self._obs_event.clear()
        self._last_published = None

    def notify_obs(self, obs: dict) -> None:
        with self._obs_lock:
            self._obs = obs
        self._obs_event.set()

    def get_action(self) -> np.ndarray | None:
        with self._buffer_lock:
            if not self._buffer:
                return None
            action = self._buffer.popleft()
            self._consumed_count += 1
            self._last_published = action
        return action

    def qsize(self) -> int:
        with self._buffer_lock:
            return len(self._buffer)

    @property
    def failed(self) -> bool:
        return self._exc is not None

    def _merge_chunk(
        self,
        actions: np.ndarray,
        queued_at_send: int,
        consumed_since: int,
        delay: int,
    ) -> tuple[int, int]:
        cfg = self._cfg
        drop_steps = 0

        if queued_at_send > 0:
            drop_steps = min(
                max(consumed_since, delay),
                max(actions.shape[0] - 1, 0),
            )

        append_actions = actions[drop_steps:]
        keep_existing = min(len(self._buffer), cfg.min_queue_keep_steps)
        kept: deque = deque()
        for idx, item in enumerate(self._buffer):
            if idx >= keep_existing:
                break
            kept.append(item)
        self._buffer = kept

        reference = self._buffer[-1] if self._buffer else self._last_published
        n_blend = max(int(cfg.action_blend_steps), 0)
        for local_idx, action in enumerate(append_actions):
            a = action.copy()
            if n_blend > 0 and local_idx < n_blend and reference is not None:
                t = (local_idx + 1) / (n_blend + 1)
                ref = np.asarray(reference, dtype=np.float32).reshape(a.shape)
                a = ref + t * (a - ref)
            self._buffer.append(a.astype(np.float32))

        return drop_steps, keep_existing

    def _worker_loop(self) -> None:
        try:
            while not self._shutdown_event.is_set():
                if not self._policy_active.is_set():
                    self._policy_active.wait(timeout=0.1)
                    continue
                self._obs_event.wait(timeout=0.1)
                if self._shutdown_event.is_set():
                    break

                with self._lifecycle_lock:
                    with self._obs_lock:
                        obs = self._obs
                        generation = self._generation
                        self._obs = None
                    self._obs_event.clear()

                if obs is None:
                    continue

                # Estimate delay from p95 of previous inferences
                latency = self._latency_tracker.p95()
                time_per_step = 1.0 / self._robot_exec_hz
                delay = (
                    math.ceil(latency / time_per_step) if self._cfg.discard_latency_steps and latency else 0
                )

                with self._buffer_lock:
                    queued_at_send = len(self._buffer)
                    consumed_at_send = self._consumed_count
                    # state fusion: blend obs state with buffer tail when buffer non-empty
                    if queued_at_send > 0 and "observation.state" in obs:
                        alpha = self._cfg.state_fusion_alpha
                        tail = list(self._buffer)[-1]
                        real = np.asarray(obs["observation.state"], dtype=np.float32).reshape(-1)
                        # Fuse only on overlapping dims to avoid truncation on dimension mismatch
                        fused = real.copy()
                        n = min(len(tail), len(real))
                        fused[:n] = tail[:n] + alpha * (real[:n] - tail[:n])
                        obs = dict(obs)
                        obs["observation.state"] = fused

                t0 = time.perf_counter()
                actions_np = self._predict_fn(obs)
                infer_latency_s = time.perf_counter() - t0
                self._latency_tracker.add(infer_latency_s)

                with self._lifecycle_lock:
                    result_is_current = (
                        generation == self._generation
                        and self._policy_active.is_set()
                        and not self._shutdown_event.is_set()
                    )
                    if not result_is_current:
                        continue

                    with self._buffer_lock:
                        consumed_since = max(self._consumed_count - consumed_at_send, 0)
                        self._chunk_counter += 1
                        prev_buf_np = (
                            np.stack(list(self._buffer))
                            if self._chunk_debugger and self._buffer else None
                        )
                        drop, keep = self._merge_chunk(
                            actions_np, queued_at_send, consumed_since, delay,
                        )

                if self._chunk_debugger is not None:
                    chunk_prev = prev_buf_np if prev_buf_np is not None else actions_np[:0]
                    self._chunk_debugger.record(
                        chunk_prev=chunk_prev, chunk_rtc=actions_np, chunk_nortc=None,
                        delay=delay, chunk2_start=consumed_at_send,
                    )

                logger.info(
                    "OverlapInferEngine: infer_ms=%.1f delay=%d drop=%d keep=%d buf=%d",
                    infer_latency_s * 1000, delay, drop, keep, self.qsize(),
                )

        except Exception:
            self._exc = Exception(traceback.format_exc())
            logger.error("OverlapInferEngine thread died:\n%s", traceback.format_exc())
