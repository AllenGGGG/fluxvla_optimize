"""Thread-safe action queue for Real-Time Chunking."""

import logging
from threading import Lock
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ActionQueue:
    """Thread-safe queue for managing action chunks.

    Maintains two parallel structures:
    - ``queue``: list of arbitrary items served to the robot via ``get()``.
    - ``original_queue``: raw numpy action array ``(T, A)`` used as RTC prefix
      input for VJP guidance.

    Incoming chunks replace the queue, skipping ``delay`` steps that elapsed
    during inference.
    """

    def __init__(self) -> None:
        self.queue: list[Any] | None = None
        self.original_queue: np.ndarray | None = None
        self._lock = Lock()
        self._last_index: int = 0

    # ------------------------------------------------------------------
    # Public read API (called from the main control thread)
    # ------------------------------------------------------------------

    def get(self) -> Any | None:
        """Pop the next item.  Returns ``None`` when empty."""
        with self._lock:
            if self.queue is None or self._last_index >= len(self.queue):
                return None
            item = self.queue[self._last_index]
            self._last_index += 1
            return item

    def qsize(self) -> int:
        """Number of unconsumed items."""
        with self._lock:
            if self.queue is None:
                return 0
            return max(0, len(self.queue) - self._last_index)

    def snapshot(self) -> tuple[int, np.ndarray | None, np.ndarray | None]:
        """Atomically return ``(action_index, full_original, left_over)``."""
        with self._lock:
            idx = self._last_index
            if self.original_queue is None:
                return idx, None, None
            full = self.original_queue.copy()
            tail = full[idx:]
            return idx, full, (tail if len(tail) else None)

    def clear(self) -> None:
        with self._lock:
            self.queue = None
            self.original_queue = None
            self._last_index = 0

    # ------------------------------------------------------------------
    # Write API (called from the background inference thread)
    # ------------------------------------------------------------------

    def merge(self, original_actions: np.ndarray, items: list[Any], real_delay: int) -> None:
        """Replace the queue with a new chunk, skipping ``real_delay`` steps."""
        with self._lock:
            delay = max(0, real_delay)
            clamped = max(0, min(delay, len(original_actions), len(items)))
            self.original_queue = original_actions[clamped:].copy()
            self.queue = items[clamped:]
            self._last_index = 0
            if not self.queue:
                logger.warning(
                    "Entire chunk consumed by delay (%d steps) — queue empty after merge", delay
                )
            logger.debug(
                "Queue replaced: delay=%d clamped=%d remaining=%d",
                delay, clamped, len(self.queue),
            )
