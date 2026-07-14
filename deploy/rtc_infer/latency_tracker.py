"""Latency tracking for Real-Time Chunking."""

from collections import deque

import numpy as np


class LatencyTracker:
    """Sliding-window latency statistics.

    Args:
        maxlen: Window size. Only the most recent ``maxlen`` samples are kept.
        skip_first: Discard the first N samples. Defaults to 0; set > 0 only if
            the caller cannot guarantee JIT warmup happens before the loop starts.
        max_latency: Hard clamp — samples above this value are dropped.
    """

    def __init__(self, maxlen: int = 100, skip_first: int = 0,
                 max_latency: float = 5.0) -> None:
        self._values: deque[float] = deque(maxlen=maxlen)
        self._skip_remaining = skip_first
        self._max_latency = max_latency

    def reset(self) -> None:
        """Clear all recorded samples."""
        self._values.clear()

    def add(self, latency: float) -> None:
        """Record a latency sample (seconds).

        The first ``skip_first`` calls and any sample above ``max_latency`` are
        silently dropped so JIT-compile / CUDA warm-up spikes never inflate p95.
        """
        if self._skip_remaining > 0:
            self._skip_remaining -= 1
            return
        val = float(latency)
        if val < 0 or val > self._max_latency:
            return
        self._values.append(val)

    def max(self) -> float:
        """Maximum latency within the current sliding window (0.0 if empty).

        Computed from the live deque so evicted samples do not inflate the estimate.
        """
        return max(self._values) if self._values else 0.0

    def percentile(self, q: float) -> float:
        """Return the q-quantile (q in [0, 1]) of recorded latencies."""
        if not self._values:
            return 0.0
        q = float(q)
        if q <= 0.0:
            return min(self._values)
        if q >= 1.0:
            return self.max()
        vals = np.array(list(self._values), dtype=np.float32)
        return float(np.quantile(vals, q))

    def p95(self) -> float:
        """95th-percentile latency."""
        return self.percentile(0.95)

    def __len__(self) -> int:
        return len(self._values)
