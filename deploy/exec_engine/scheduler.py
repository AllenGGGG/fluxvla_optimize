"""ChunkScheduler — background thread that produces action chunks."""

from __future__ import annotations

import math
import time
from threading import Event, Lock, Thread
from typing import Callable

import numpy as np

from .action_queue import ActionQueue
from .config import ChunkSchedulerConfig
from .latency_tracker import LatencyTracker


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
        (up to ``delay + 2 * execution_horizon`` rows -- the locked handoff
        window plus a trailing horizon of real data for the RTC guidance
        decay taper -- or ``None`` on the first call).
        The returned dict must contain:
          ``"chunk"``: np.ndarray (T, A) — raw action array for RTC guidance
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
        target = (
            self._scheduler_loop_rtc
            if self._cfg.rtc_enabled
            else self._scheduler_loop_plain
        )
        self._scheduler_thread = Thread(
            target=target, daemon=True, name="ChunkScheduler"
        )
        self._scheduler_thread.start()

    def stop(self) -> None:
        self._shutdown_event.set()
        self._policy_active.clear()
        if self._scheduler_thread is not None and self._scheduler_thread.is_alive():
            self._scheduler_thread.join(timeout=_JOIN_TIMEOUT_S)
        self._scheduler_thread = None

    def pause(self) -> None:
        self._policy_active.clear()

    def resume(self) -> None:
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

    def _predict_and_merge(
        self,
        obs: dict,
        generation: int,
        queue: ActionQueue,
        delay: int,
        guidance_prev: np.ndarray | None,
        merge_delay: int,
        t0: float,
        latency_tracker: LatencyTracker,
    ) -> bool:
        """Call predict_fn and merge the result into ``queue``.

        Shared by both the RTC and plain scheduler loops so the staleness
        check under ``_lifecycle_lock`` (discarding results made obsolete by
        a concurrent ``reset()``) only lives in one place. Returns ``False``
        if the result was discarded as stale (caller should ``continue``).
        """
        with self._prev_chunk_lock:
            self._current_prev_chunk = guidance_prev

        result = self._predict_fn(obs, delay, guidance_prev)
        chunk_np = np.asarray(result["chunk"], dtype=np.float32)
        # RTC needs enough queued actions to cover both inference latency and
        # its guidance prefix.  A short plain-mode horizon can leave too few
        # actions for that alignment, causing discontinuities when the queue
        # is replaced.  In RTC mode, zero means keep the full model chunk.
        publish_horizon = (
            self._cfg.rtc_publish_horizon
            if self._cfg.rtc_enabled
            else self._cfg.publish_horizon
        )
        if publish_horizon > 0:
            chunk_np = chunk_np[:publish_horizon]
        if not np.isfinite(chunk_np).all():
            raise ValueError("predict_fn returned NaN/Inf in chunk")
        with self._lifecycle_lock:
            result_is_current = (
                generation == self._generation
                and self._policy_active.is_set()
                and not self._shutdown_event.is_set()
            )
            if not result_is_current:
                return False

            elapsed = time.perf_counter() - t0
            latency_tracker.add(elapsed)
            self._next_chunk_id += 1
            chunk_id = self._next_chunk_id
            items = [(chunk_id, action) for action in chunk_np]
            queue.merge(chunk_np, items, merge_delay)

        return True

    def _scheduler_loop_plain(self) -> None:
        """Non-RTC path: produce the next chunk only once the queue drains."""
        latency_tracker = LatencyTracker()
        while not self._shutdown_event.is_set():
            if not self._policy_active.is_set():
                time.sleep(_IDLE_SLEEP_S)
                continue

            queue = self._action_queue
            if queue.qsize() > 0:
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

            self._predict_and_merge(
                obs, generation, queue,
                delay=0, guidance_prev=None, merge_delay=0,
                t0=t0, latency_tracker=latency_tracker,
            )

    def _scheduler_loop_rtc(self) -> None:
        """RTC path: continuously produce chunks guided by the leftover queue."""
        latency_tracker = LatencyTracker()
        time_per_step = 1.0 / max(self._robot_exec_hz, 1e-6)
        while not self._shutdown_event.is_set():
            if not self._policy_active.is_set():
                time.sleep(_IDLE_SLEEP_S)
                continue

            queue = self._action_queue

            with self._lifecycle_lock:
                with self._obs_lock:
                    obs = self._obs
                    generation = self._generation

            if obs is None:
                time.sleep(_IDLE_SLEEP_S)
                continue

            latency = latency_tracker.p95()
            delay = math.ceil(latency / time_per_step) if latency else 0
            _, _, prev_left_over = queue.snapshot()

            # Keep enough old actions for the measured inference delay, the
            # RTC handoff window, and a trailing horizon of real data for
            # the guidance decay taper (see aligned_guidance_window below --
            # it can only return data that is still in the leftover queue,
            # so if we let the queue drain below delay + 2H before
            # replanning, the decay region has nothing to taper against and
            # collapses back to a hard cut). The explicit value is only a
            # minimum; latency always wins so the splice has valid context.
            required_remaining = delay + 2 * self._cfg.execution_horizon
            replan_remaining = max(
                self._cfg.replan_remaining,
                required_remaining,
            )
            if (
                prev_left_over is not None
                and len(prev_left_over) > replan_remaining
            ):
                time.sleep(_IDLE_SLEEP_S)
                continue

            t0 = time.perf_counter()
            # Fetch delay+2H of old queue so the locked prefix (delay+H) has
            # a full H-step tail left over for the RTC decay region. Without
            # the extra H, decay_end collapses onto prefix_len and guidance
            # weight drops from 1.0 straight to 0 with no taper.
            guidance_prev = aligned_guidance_window(
                prev_left_over, delay, 2 * self._cfg.execution_horizon
            )

            self._predict_and_merge(
                obs, generation, queue,
                delay=delay, guidance_prev=guidance_prev, merge_delay=delay,
                t0=t0, latency_tracker=latency_tracker,
            )
