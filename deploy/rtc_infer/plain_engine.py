"""Asynchronous plain chunk inference without overlap or RTC guidance."""

from __future__ import annotations

import logging
import threading
import traceback
from collections import deque
from typing import Callable

import numpy as np

from .config import PlainConfig
from .visualizer import RTCDebugVisualizer

logger = logging.getLogger(__name__)


class PlainInferEngine:
    def __init__(self, predict_fn: Callable, cfg: PlainConfig) -> None:
        self._predict_fn = predict_fn
        self.n_action_steps: int = predict_fn.n_action_steps
        self._fraction = max(0.0, min(1.0, cfg.execution_fraction))
        self._buffer: deque[np.ndarray] = deque()
        self._buffer_lock = threading.Lock()
        self._last_chunk: np.ndarray | None = None
        self._obs: dict | None = None
        self._obs_lock = threading.Lock()
        self._obs_event = threading.Event()
        self._inference_pending = threading.Event()
        self._policy_active = threading.Event()
        self._shutdown_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._exc: BaseException | None = None
        self._generation = 0
        self._lifecycle_lock = threading.Lock()
        debug_dir = cfg.debug_dir
        self._chunk_debugger: RTCDebugVisualizer | None = (
            RTCDebugVisualizer(debug_dir, execution_horizon=0) if cfg.debug else None
        )

    def start(self) -> None:
        self._shutdown_event.clear()
        self._exc = None
        self._thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="plain-infer"
        )
        self._thread.start()

    def stop(self) -> None:
        self._shutdown_event.set()
        self._policy_active.clear()
        self._obs_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("Plain inference thread did not stop within 5 seconds")
        self._thread = None
        if self._chunk_debugger is not None:
            self._chunk_debugger.close()

    def pause(self) -> None:
        self._policy_active.clear()

    def resume(self) -> None:
        self._policy_active.set()

    def reset(self) -> None:
        with self._lifecycle_lock:
            with self._buffer_lock:
                self._buffer.clear()
                self._last_chunk = None
            with self._obs_lock:
                self._obs = None
            self._generation += 1
        self._obs_event.clear()
        self._inference_pending.clear()

    def notify_obs(self, obs: dict) -> None:
        with self._buffer_lock:
            if self._buffer:
                return
        if self._inference_pending.is_set():
            return
        with self._obs_lock:
            if self._obs is None and not self._inference_pending.is_set():
                self._obs = obs
                self._inference_pending.set()
                self._obs_event.set()

    def get_action(self) -> np.ndarray | None:
        with self._buffer_lock:
            if not self._buffer:
                return None
            return self._buffer.popleft()

    def qsize(self) -> int:
        with self._buffer_lock:
            return len(self._buffer)

    @property
    def failed(self) -> bool:
        return self._exc is not None

    def _worker_loop(self) -> None:
        try:
            while not self._shutdown_event.is_set():
                self._obs_event.wait(timeout=0.1)
                if self._shutdown_event.is_set():
                    break
                if not self._policy_active.is_set():
                    self._policy_active.wait(timeout=0.1)
                    continue
                with self._lifecycle_lock:
                    with self._obs_lock:
                        obs = self._obs
                        generation = self._generation
                        self._obs = None
                        self._obs_event.clear()
                if obs is None:
                    continue

                chunk = np.asarray(self._predict_fn(obs), dtype=np.float32)
                if chunk.ndim != 2 or not np.isfinite(chunk).all():
                    raise ValueError(f"Invalid plain action chunk shape/content: {chunk.shape}")
                keep = max(1, int(round(chunk.shape[0] * self._fraction)))
                new_chunk = chunk[:keep]

                with self._lifecycle_lock:
                    result_is_current = (
                        generation == self._generation
                        and self._policy_active.is_set()
                        and not self._shutdown_event.is_set()
                    )
                    if not result_is_current:
                        self._inference_pending.clear()
                        continue
                    with self._buffer_lock:
                        previous = self._last_chunk
                        self._buffer.extend(new_chunk)
                        self._last_chunk = new_chunk
                    self._inference_pending.clear()

                if self._chunk_debugger is not None:
                    old = previous if previous is not None else new_chunk[:0]
                    self._chunk_debugger.record(
                        chunk_prev=old,
                        chunk_rtc=new_chunk,
                        chunk_nortc=None,
                        delay=0,
                        chunk2_start=0,
                    )
        except Exception:
            self._exc = Exception(traceback.format_exc())
            logger.error("PlainInferEngine thread died:\n%s", traceback.format_exc())
