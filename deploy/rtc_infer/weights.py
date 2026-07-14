"""Weight schedules for RTC guidance.

Two independent weight dimensions:

* **prefix weights** (spatial, over action steps):
  Applied element-wise to ``(prev_chunk - x1_t)``.
  Always 1.0 in ``[0, start)``; fade in ``[start, end)`` per schedule; 0.0 after.

* **guidance weights** (temporal, over denoising steps):
  Scalar ``w(t)`` multiplying the correction velocity.
  Derived from the flow-matching SNR schedule.
"""

from __future__ import annotations

import numpy as np

from .config import AttentionSchedule


def prefix_weights(
    start: int,
    end: int,
    total: int,
    schedule: AttentionSchedule = AttentionSchedule.LINEAR,
) -> np.ndarray:
    """Prefix weight array over ``total`` action steps, shape ``(total,)`` float32.

    Always 1.0 in ``[0, start)``, 0.0 after ``end``.
    The fade in ``[start, end)`` depends on ``schedule``:

    * ``LINEAR``: linear 1→0
    * ``EXP``:    exponential decay (e^{-5x}, x in [0,1))
    * ``ONES``:   all 1.0 up to end, hard cutoff
    * ``ZEROS``:  hard cutoff at start, no fade

    Args:
        start:    First step of the fade zone (= delay).
        end:      Last step of the fade zone (= delay + execution_horizon).
        total:    Total action-chunk length T.
        schedule: Fade curve to use in the overlap window.
    """
    start = min(start, end)
    w = np.zeros(total, dtype=np.float32)
    n = end - start

    if schedule is AttentionSchedule.ZEROS:
        w[:start] = 1.0
    elif schedule is AttentionSchedule.ONES:
        w[:end] = 1.0
    elif schedule is AttentionSchedule.LINEAR:
        w[:start] = 1.0
        if n > 0:
            w[start:end] = np.linspace(1.0, 0.0, n + 2, dtype=np.float32)[1:-1]
    elif schedule is AttentionSchedule.EXP:
        w[:start] = 1.0
        if n > 0:
            x = np.linspace(0.0, 1.0, n + 2, dtype=np.float32)[1:-1]
            w[start:end] = np.exp(-5.0 * x).astype(np.float32)
    else:
        raise ValueError(f"Unknown AttentionSchedule: {schedule!r}")

    return w


def guidance_step_params(
    num_steps: int,
    max_weight: float,
) -> list[tuple[float, float]]:
    """Precompute ``(t_val, guidance_weight)`` for each denoising step.

    Follows the flow-matching SNR schedule:
        w(t) = (t / tau) * (t² + tau²) / t²,  clamped to max_weight
    where tau = 1 - t.

    Returned as Python constants so JAX can fold them into XLA literals.

    Args:
        num_steps:  Number of Euler denoising steps.
        max_weight: Upper clamp for the guidance weight.

    Returns:
        List of (t_val, w_val) tuples, length num_steps.
    """
    dt = -1.0 / num_steps
    params: list[tuple[float, float]] = []
    for i in range(num_steps):
        t_val = 1.0 + i * dt
        tau   = 1.0 - t_val
        t2    = t_val * t_val
        if tau < 1e-8:
            w_val = max_weight
        elif t2 < 1e-10:
            w_val = 0.0
        else:
            inv_r2 = (t2 + tau * tau) / t2
            w_val  = min((t_val / tau) * inv_r2, max_weight)
        params.append((t_val, w_val))
    return params
