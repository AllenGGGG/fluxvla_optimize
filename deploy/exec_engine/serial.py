"""Synchronous chunk inference backend.

Unlike :class:`ChunkScheduler`, prediction runs in the caller's ROS timer
callback when the previous action chunk has been fully consumed.  This is the
traditional serial baseline: infer one chunk, execute it, then infer the next.
RTC is intentionally unsupported because there is no unconsumed old prefix at
the serial chunk boundary.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .action_queue import ActionQueue


class SerialInferenceBackend:
    """Run chunk prediction synchronously when the action queue is empty."""

    def __init__(self, predict_fn: Callable, publish_horizon: int = 0) -> None:
        if publish_horizon < 0:
            raise ValueError("publish_horizon must be non-negative")
        self._predict_fn = predict_fn
        self._publish_horizon = int(publish_horizon)
        self._action_queue = ActionQueue()
        self._active = False
        self._failed = False
        self._next_chunk_id = 0
        self._action_chunk_id: int | None = None
        self.n_action_steps = int(
            getattr(predict_fn, "n_action_steps", 1) or 1
        )

    @property
    def failed(self) -> bool:
        return self._failed

    @property
    def action_chunk_id(self) -> int | None:
        return self._action_chunk_id

    def start(self) -> None:
        self._active = True
        self._failed = False

    def stop(self) -> None:
        self._active = False

    def pause(self) -> None:
        self._active = False

    def resume(self) -> None:
        self._active = True

    def reset(self) -> None:
        self._action_queue.clear()
        self._failed = False
        self._action_chunk_id = None

    def notify_obs(self, obs: dict) -> None:
        if not self._active or self._action_queue.qsize() > 0:
            return

        obs_copy = {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in obs.items()
        }
        try:
            result = self._predict_fn(obs_copy, 0, None)
            chunk = np.asarray(result["chunk"], dtype=np.float32)
            if chunk.ndim != 2 or len(chunk) == 0:
                raise ValueError(
                    f"predict_fn returned invalid chunk shape {chunk.shape}"
                )
            if not np.isfinite(chunk).all():
                raise ValueError("predict_fn returned NaN/Inf in chunk")
            if self._publish_horizon > 0:
                chunk = chunk[:self._publish_horizon]

            self._next_chunk_id += 1
            chunk_id = self._next_chunk_id
            items = [(chunk_id, action) for action in chunk]
            self._action_queue.merge(chunk, items, real_delay=0)
        except Exception:
            self._failed = True
            raise

    def get_action(self) -> np.ndarray | None:
        item = self._action_queue.get()
        if item is None:
            return None
        chunk_id, action = item
        self._action_chunk_id = int(chunk_id)
        return np.asarray(action, dtype=np.float32)

    def qsize(self) -> int:
        return self._action_queue.qsize()
