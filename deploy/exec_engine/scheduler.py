"""ChunkScheduler — background thread that produces action chunks."""

from __future__ import annotations

import logging
import math
import time
from threading import Event, Lock, Thread
from typing import Callable

import numpy as np

from .action_queue import ActionQueue
from .config import ChunkSchedulerConfig
from .visualizer import RTCDebugVisualizer
from .latency_tracker import LatencyTracker

logger = logging.getLogger(__name__)


_IDLE_SLEEP_S: float = 0.01
_JOIN_TIMEOUT_S: float = 3.0


def aligned_guidance_window(
    previous_leftover: np.ndarray | None,
    delay: int,
    execution_horizon: int,
) -> np.ndarray | None:
    """Keep old/new action indices aligned across inference latency."""

    if previous_leftover is None or len(previous_leftover) == 0:
        return None
    if previous_leftover.ndim != 2:
        raise ValueError(
            f"Expected previous actions with shape (T, A), got {previous_leftover.shape}"
        )
    end = max(delay, 0) + max(execution_horizon, 0)
    return previous_leftover[:end].copy()


class ChunkScheduler:
    """Async producer/consumer scheduler for action chunks.

    The background thread continuously calls ``predict_fn`` and merges results
    into the ActionQueue.  The main thread calls ``get_action()`` each tick.

    predict_fn protocol:
        ``predict_fn(obs: dict, delay: int, prev_chunk: np.ndarray | None) -> dict``
        ``prev_chunk`` is the time-aligned leftover from the action queue
        (up to ``delay + execution_horizon`` rows, or ``None`` on the first call).
        The returned dict must contain:
          ``"chunk"``: np.ndarray (T, A) — raw action array for RTC guidance
        Optional keys:
          ``"_step_info"``: list[tuple[float, float, float]] — (time, weight, err_norm) per denoising step

    Args:
        predict_fn: Callable following the protocol above.
        cfg: ChunkSchedulerConfig.
        robot_exec_hz: Control-loop frequency (Hz).
    """

    def __init__(
        self,
        predict_fn: Callable,
        cfg: ChunkSchedulerConfig,
        robot_exec_hz: float,
    ) -> None:
        self._predict_fn = predict_fn
        self._cfg = cfg
        self._robot_exec_hz = robot_exec_hz

        self._action_queue: ActionQueue = ActionQueue()
        self._current_prev_chunk: np.ndarray | None = None
        self._prev_chunk_lock = Lock()
        self._obs: dict | None = None
        self._obs_lock = Lock()
        self._policy_active = Event()
        self._shutdown_event = Event()
        self._generation = 0
        self._lifecycle_lock = Lock()
        self._scheduler_thread: Thread | None = None
        self._next_chunk_id = 0
        self._action_chunk_id: int | None = None
        self.n_action_steps: int = int(getattr(predict_fn, "n_action_steps", 1) or 1)

        debug_dir = cfg.debug_dir
        self._chunk_debugger: RTCDebugVisualizer | None = (
            RTCDebugVisualizer(debug_dir, cfg.execution_horizon)
            if cfg.debug else None
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def predict_fn(self) -> Callable:
        """The predict function passed at construction (useful for JIT warmup)."""
        return self._predict_fn

    @property
    def ready(self) -> bool:
        return (
            self._scheduler_thread is not None
            and self._scheduler_thread.is_alive()
            and not self._shutdown_event.is_set()
        )

    @property
    def failed(self) -> bool:
        return (
            self._scheduler_thread is not None
            and not self._scheduler_thread.is_alive()
            and not self._shutdown_event.is_set()
        )

    @property
    def action_queue(self) -> ActionQueue:
        return self._action_queue

    @property
    def action_chunk_id(self) -> int | None:
        """ID of the chunk that produced the most recently popped action."""
        return self._action_chunk_id

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._action_queue.clear()
        self._shutdown_event.clear()
        self._policy_active.set()
        self._scheduler_thread = Thread(
            target=self._scheduler_loop, daemon=True, name="ChunkScheduler"
        )
        self._scheduler_thread.start()
        logger.info("Chunk scheduler thread started")

    def stop(self) -> None:
        logger.info("Stopping chunk scheduler thread")
        self._shutdown_event.set()
        self._policy_active.clear()
        if self._scheduler_thread is not None and self._scheduler_thread.is_alive():
            self._scheduler_thread.join(timeout=_JOIN_TIMEOUT_S)
            if self._scheduler_thread.is_alive():
                logger.warning("Scheduler thread did not join within %.1f s — it may still be finishing", _JOIN_TIMEOUT_S)
            else:
                logger.info("Chunk scheduler thread stopped")
        self._scheduler_thread = None
        if self._chunk_debugger is not None:
            self._chunk_debugger.close()

    def pause(self) -> None:
        logger.info("Chunk scheduler paused")
        self._policy_active.clear()

    def resume(self) -> None:
        logger.info("Chunk scheduler resumed")
        self._policy_active.set()

    def reset(self) -> None:
        with self._lifecycle_lock:
            self._action_queue.clear()
            with self._obs_lock:
                self._obs = None
            self._generation += 1
            with self._prev_chunk_lock:
                self._current_prev_chunk = None

    # ------------------------------------------------------------------
    # Main-thread API
    # ------------------------------------------------------------------

    def get_action(self) -> np.ndarray | None:
        """Pop the next item from the queue.  Non-blocking; returns ``None`` when empty."""
        item = self._action_queue.get()
        if item is None:
            return None
        chunk_id, action = item
        self._action_chunk_id = int(chunk_id)
        return np.asarray(action, dtype=np.float32)

    def qsize(self) -> int:
        return self._action_queue.qsize()

    def notify_obs(self, obs: dict) -> None:
        import torch
        obs_np = {
            k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
            for k, v in obs.items()
        }
        obs_copy = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in obs_np.items()}
        with self._obs_lock:
            self._obs = obs_copy

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def _scheduler_loop(self) -> None:
        latency_tracker = LatencyTracker()
        time_per_step = 1.0 / max(self._robot_exec_hz, 1e-6)
        while not self._shutdown_event.is_set():
            if not self._policy_active.is_set():
                time.sleep(_IDLE_SLEEP_S)
                continue

            queue = self._action_queue
            if not self._cfg.rtc_enabled and queue.qsize() > 0:
                time.sleep(_IDLE_SLEEP_S)
                continue

            with self._lifecycle_lock:
                with self._obs_lock:
                    obs = self._obs
                    generation = self._generation

            if obs is None:
                time.sleep(_IDLE_SLEEP_S)
                continue

            t0 = time.perf_counter()
            chunk2_start, prev_full, prev_left_over = queue.snapshot()

            latency = latency_tracker.p95() if self._cfg.rtc_enabled else 0.0
            delay = math.ceil(latency / time_per_step) if latency else 0
            guidance_prev = (
                aligned_guidance_window(
                    prev_left_over, delay, self._cfg.execution_horizon
                )
                if self._cfg.rtc_enabled else None
            )
            with self._prev_chunk_lock:
                self._current_prev_chunk = guidance_prev

            result = self._predict_fn(obs, delay, guidance_prev)
            chunk_np = np.asarray(result["chunk"], dtype=np.float32)
            if not np.isfinite(chunk_np).all():
                raise ValueError("predict_fn returned NaN/Inf in chunk")
            step_info = result.get("_step_info")
            chunk_nortc = (
                np.asarray(result["_chunk_nortc"], dtype=np.float32)
                if "_chunk_nortc" in result else None
            )
            with self._lifecycle_lock:
                result_is_current = (
                    generation == self._generation
                    and self._policy_active.is_set()
                    and not self._shutdown_event.is_set()
                )
                if not result_is_current:
                    continue

                elapsed = time.perf_counter() - t0
                new_delay = math.ceil(elapsed / time_per_step)
                latency_tracker.add(elapsed)
                self._next_chunk_id += 1
                chunk_id = self._next_chunk_id
                items = [(chunk_id, action) for action in chunk_np]
                queue.merge(chunk_np, items, delay if self._cfg.rtc_enabled else 0)

            if self._chunk_debugger is not None and prev_full is not None:
                self._chunk_debugger.record(
                    prev_full, chunk_np, chunk_nortc, delay,
                    chunk2_start=chunk2_start,
                    step_info=step_info,
                )

            logger.debug(
                "Chunk scheduled: latency=%.3fs measured_delay=%d applied_delay=%d qsize=%d",
                elapsed, new_delay, delay if self._cfg.rtc_enabled else 0,
                queue.qsize(),
            )
