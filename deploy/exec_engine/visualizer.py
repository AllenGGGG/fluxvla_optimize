"""Visualization helpers for RTC debug snapshots."""

from __future__ import annotations

import logging
import pathlib
import queue
import threading
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    import matplotlib.figure

logger = logging.getLogger(__name__)

_COLORS = {
    "chunk1": "#212121",
    "nortc":  "#E53935",
    "rtc":    "#43A047",
    "window": "#90A4AE",
}


def _hard_match_diff(
    chunk: np.ndarray,
    ref: np.ndarray,
    horizon: int,
    ref_offset: int = 0,
) -> float:
    """Mean absolute error in the fade window as % of reference magnitude.

    Compares ``chunk[0:horizon]`` vs ``ref[ref_offset : ref_offset+horizon]``.

    Args:
        chunk:      New chunk, x-axis starts at splice point.
        ref:        Previous chunk (full), x-axis starts at last merge point.
        horizon:    Guidance fade window length H.
        ref_offset: ``chunk2_start + delay`` — offset into ref where the splice
                    point falls, i.e. the index in ref that aligns with chunk[0].
    """
    if horizon <= 0:
        return float("nan")
    ref2d = ref if ref.ndim == 2 else ref[0]
    n = min(horizon, len(chunk), max(0, len(ref2d) - ref_offset))
    if n <= 0:
        return float("nan")
    ref_seg = ref2d[ref_offset : ref_offset + n]
    ref_mag = float(np.mean(np.abs(ref_seg))) + 1e-6
    return float(np.mean(np.abs(chunk[:n] - ref_seg))) / ref_mag * 100


class RTCDebugVisualizer:
    """RTC debug plots with a persistent worker thread for async saving.

    Instance methods: record(), used by ChunkScheduler.
    Static methods:   plot_chunk_comparison(), plot_waypoints(), usable standalone.
    """

    def __init__(
        self,
        debug_dir: str,
        execution_horizon: int,
    ) -> None:
        self._dir = pathlib.Path(debug_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._horizon = execution_horizon
        self._idx: int = 0
        self._queue: queue.Queue = queue.Queue()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True, name="RTCDebugSave")
        self._worker.start()

    # ------------------------------------------------------------------
    # Instance API — called from inference thread
    # ------------------------------------------------------------------

    def record(
        self,
        chunk_prev: np.ndarray,
        chunk_rtc: np.ndarray,
        chunk_nortc: np.ndarray | None,
        delay: int,
        *,
        chunk2_start: int = 0,
        step_info: list[tuple[float, float, float]] | None = None,
    ) -> None:
        """Queue a debug plot for async saving.

        Args:
            chunk_prev:   Previous chunk leftover after dropping the delay prefix,
                          shape ``(T - delay, A)``.  x-axis starts at 0 = splice point.
            chunk_rtc:    New chunk with RTC guidance, shape ``(T, A)``.  x-axis
                          also starts at 0 = splice point (same origin as chunk_prev).
            chunk_nortc:  New chunk without guidance, or None.
            delay:        Inference delay in action steps (info only, not used for
                          coordinate offsetting).
            chunk2_start: Steps consumed from the previous queue at snapshot time
                          (for title display only).
            step_info:    Per-denoising-step diagnostics.
        """
        idx = self._idx
        self._idx += 1
        self._queue.put(dict(
            chunk_prev=chunk_prev.copy(),
            chunk_rtc=chunk_rtc.copy(),
            chunk_nortc=chunk_nortc.copy() if chunk_nortc is not None else None,
            delay=delay,
            idx=idx,
            chunk2_start=chunk2_start,
            step_info=step_info,
        ))

    def close(self) -> None:
        """Drain the save queue and stop the worker thread."""
        self._queue.put(None)
        self._worker.join()

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            try:
                self._save(**item)
            except Exception as exc:
                logger.warning("Failed to save chunk debug plot: %s", exc)
            finally:
                self._queue.task_done()

    def _save(
        self,
        chunk_prev: np.ndarray,
        chunk_rtc: np.ndarray,
        chunk_nortc: np.ndarray | None,
        delay: int,
        idx: int,
        *,
        chunk2_start: int = 0,
        step_info: list[tuple[float, float, float]] | None = None,
    ) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        H = self._horizon
        ref_offset = chunk2_start + delay
        mse_rtc   = _hard_match_diff(chunk_rtc,   chunk_prev, H, ref_offset)
        mse_nortc = _hard_match_diff(chunk_nortc, chunk_prev, H, ref_offset) if chunk_nortc is not None else float("nan")
        logger.debug(
            "[chunk#%d] delay=%d  chunk2_start=%d  rtc_diff=%.2f%%  nortc_diff=%.2f%%",
            idx, delay, chunk2_start, mse_rtc, mse_nortc,
        )
        fig = RTCDebugVisualizer.plot_chunk_comparison(
            chunk_1=chunk_prev,
            chunk_2_rtc=chunk_rtc,
            chunk_2_nortc=chunk_nortc,
            step_info=step_info,
            delay=delay,
            horizon=H,
            chunk2_start=chunk2_start,
            mse_rtc=mse_rtc,
            mse_nortc=mse_nortc,
            title=f"chunk #{idx}  delay={delay}  q_pos={chunk2_start}",
        )
        path = self._dir / f"chunk_{idx:05d}.png"
        fig.savefig(path, dpi=80, bbox_inches="tight")
        plt.close(fig)
        logger.debug("Saved chunk debug plot: %s", path)

    # ------------------------------------------------------------------
    # Static plotting API — usable without instantiation
    # ------------------------------------------------------------------

    @staticmethod
    def plot_chunk_comparison(
        chunk_1: np.ndarray,
        chunk_2_rtc: np.ndarray | None,
        chunk_2_nortc: np.ndarray | None = None,
        delay: int = 0,
        horizon: int = 10,
        chunk2_start: int = 0,
        mse_rtc: float = float("nan"),
        mse_nortc: float = float("nan"),
        title: str = "",
        num_dims: int = 2,
        dims: list[int] | None = None,
        step_info: list[tuple[float, float, float]] | None = None,
    ) -> "matplotlib.figure.Figure":
        """Plot chunk overlap and denoising diagnostics.

        Coordinate convention:
          - chunk_1 (prev full):  x = [0, T1),  step 0 = last merge point.
          - chunk_2_*:            x = [a, a+T2), a = chunk2_start + delay
                                  (splice point, i.e. first step robot executes
                                  from the new chunk).
          - fade zone:            x = [a, a+horizon)  — guided overlap region.
          - splice line:          x = a

        Args:
            chunk_1:      Full previous chunk (from last merge, length T1).
            chunk_2_rtc:  New chunk with RTC guidance, length T2.
            chunk_2_nortc: New chunk without guidance (optional).
            delay:        Inference delay in steps.
            horizon:      Guidance fade window length H.
            chunk2_start: Steps consumed from prev chunk at snapshot time.
            dims:         Which action dims to plot. Default: left/right arm joints [7,10,15,18].
            step_info:    [(diffusion_time, guidance_weight, err_norm), ...].
        """
        # Default: base[0,1,2] + torso[3,4] + left_arm[7,10] + right_arm[15,18]
        if dims is None:
            dims = [0, 1, 2, 3, 4, 7, 10, 15, 18]
        num_dims = len(dims)
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        dim_labels = {
            0: "base_x", 1: "base_y", 2: "base_rot",
            3: "torso[0]", 4: "torso[1]", 5: "torso[2]", 6: "torso[3]",
            7: "left_arm[0]", 8: "left_arm[1]", 9: "left_arm[2]",
            10: "left_arm[3]", 11: "left_arm[4]", 12: "left_arm[5]", 13: "left_arm[6]",
            14: "left_gripper",
            15: "right_arm[0]", 16: "right_arm[1]", 17: "right_arm[2]",
            18: "right_arm[3]", 19: "right_arm[4]", 20: "right_arm[5]", 21: "right_arm[6]",
            22: "right_gripper",
        }

        T1 = chunk_1.shape[0]
        T2 = chunk_2_rtc.shape[0] if chunk_2_rtc is not None else (
             chunk_2_nortc.shape[0] if chunk_2_nortc is not None else T1)
        a        = chunk2_start + delay
        x_end    = max(T1, a + T2)
        fade_end = min(a + horizon, x_end)

        # Layout: dims arranged in a 2-column grid, guidance panels in last row
        n_dim_rows = (num_dims + 1) // 2          # ceil(num_dims / 2)
        n_rows_total = n_dim_rows + 1              # +1 row for guidance panels
        n_cols = 2

        fig = plt.figure(figsize=(16, 4 * n_rows_total))
        gs = gridspec.GridSpec(n_rows_total, n_cols, figure=fig, hspace=0.55, wspace=0.35)

        # One subplot per dim, filled left→right, top→bottom
        dim_axes = []
        for i in range(num_dims):
            row, col = divmod(i, n_cols)
            dim_axes.append(fig.add_subplot(gs[row, col]))

        # Slice to selected dims before plotting
        def _sel(arr):
            if arr is None:
                return None
            a2 = np.asarray(arr)
            if a2.ndim == 3:
                return a2[:, :, dims]
            return a2[:, dims]

        RTCDebugVisualizer.plot_waypoints(
            dim_axes, _sel(chunk_1),
            color=_COLORS["chunk1"],
            label=f"chunk prev  t=[0, {T1})",
            linewidth=2.0,
        )
        if chunk_2_nortc is not None:
            RTCDebugVisualizer.plot_waypoints(
                dim_axes, _sel(chunk_2_nortc), start_from=a,
                color=_COLORS["nortc"],
                label=f"no-rtc  t=[{a}, {a+T2})  diff={mse_nortc:.1f}%",
                linewidth=1.5, alpha=0.75, linestyle="--",
            )
        if chunk_2_rtc is not None:
            RTCDebugVisualizer.plot_waypoints(
                dim_axes, _sel(chunk_2_rtc), start_from=a,
                color=_COLORS["rtc"],
                label=f"rtc  t=[{a}, {a+T2})  diff={mse_rtc:.1f}%",
                linewidth=1.5, alpha=0.85,
            )

        for i, ax in enumerate(dim_axes):
            ax.axvspan(a, fade_end, alpha=0.10, color=_COLORS["window"],
                       label=f"fade  [{a}, {fade_end})")
            ax.axvline(a, color="black", linewidth=1.2, linestyle="--",
                       alpha=0.7, label=f"splice  t={a}")
            ax.set_xlim(0, x_end)
            d = dims[i]
            ax.set_title(f"dim {d}  {dim_labels.get(d, '')}")
            ax.set_ylabel("action value")
            ax.legend(fontsize=7.5)

        steps = step_info or []
        step_times   = [s[0] for s in steps]
        step_weights = [s[1] for s in steps]
        err_norms    = [s[2] for s in steps]

        ax_w = fig.add_subplot(gs[n_dim_rows, 0])
        if step_times:
            ax_w.plot(step_times, step_weights, marker="o",
                      color=_COLORS["rtc"], linewidth=1.5, markersize=5)
        ax_w.set_xlabel("diffusion time t  (1=noisy → 0=clean)")
        ax_w.set_ylabel("guidance weight  w(t)")
        ax_w.set_title("Guidance weight per denoising step")
        ax_w.invert_xaxis()
        ax_w.grid(alpha=0.25)

        ax_c = fig.add_subplot(gs[n_dim_rows, 1])
        if step_times:
            ax_c.plot(step_times, err_norms, marker="s",
                      color=_COLORS["chunk1"], linewidth=1.5, markersize=5)
        ax_c.set_xlabel("diffusion time t")
        ax_c.set_ylabel("mean |err|")
        ax_c.set_title("Chunk error norm per denoising step")
        ax_c.invert_xaxis()
        ax_c.grid(alpha=0.25)

        suptitle = (
            f"RTC guidance  |  T={T2}  delay={delay}  horizon={horizon}"
            f"  splice={a}  q_pos={chunk2_start}"
        )
        if title:
            suptitle += f"  |  {title}"
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")

        return fig

    @staticmethod
    def plot_waypoints(
        axes: Sequence,
        array: np.ndarray,
        start_from: int = 0,
        color: str = "blue",
        label: str = "",
        alpha: float = 0.7,
        linewidth: float = 2,
        marker: str | None = None,
        markersize: int = 4,
        linestyle: str = "-",
    ) -> None:
        if array is None:
            return

        arr = np.asarray(array) if not isinstance(array, np.ndarray) else array

        if arr.ndim == 3:
            arr = arr[0]
        elif arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        T, D = arr.shape
        x = np.arange(start_from, start_from + T)

        num_axes = len(axes) if hasattr(axes, "__len__") else 1
        plot_kw = dict(color=color, alpha=alpha, linewidth=linewidth, linestyle=linestyle)
        if marker:
            plot_kw.update(marker=marker, markersize=markersize)

        for dim in range(min(D, num_axes)):
            ax = axes[dim] if hasattr(axes, "__len__") else axes
            ax.plot(x, arr[:, dim], label=label if dim == 0 else "", **plot_kw)
            if not ax.xaxis.get_label().get_text():
                ax.set_xlabel("Step", fontsize=10)
            if not ax.yaxis.get_label().get_text():
                ax.set_ylabel(f"Dim {dim}", fontsize=10)
            ax.grid(True, alpha=0.3)
