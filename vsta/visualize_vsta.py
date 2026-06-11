"""
VSTA Augmentation Visualizer — multi-task, full episodes

One video per task (first episode of each task), saved to task_videos/.

Layout:
  - Top: global title bar
  - Middle: 3 camera columns (0.5x / 1x / 2x) — head + wrist feeds
  - Bottom: full-width action comparison panel — all 3 speeds overlaid per channel
"""

import sys
sys.path.insert(0, "/data/wqz/code/vsta")

import numpy as np
import pandas as pd
import av, cv2, imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from vsta import VSTA, VSTAConfig

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEMO_ROOT  = Path("/data/wqz/code/openpi/behavior-1k/2025-challenge-demos")
OUTPUT_DIR = Path("/data/wqz/code/vsta/task_videos")
OUTPUT_FPS = 30

SPEEDS     = [0.5, 1.0, 2.0]
SPD_COLORS = ["#60a5fa", "#4ade80", "#f87171"]
SPD_LABELS = ["0.5x", "1.0x", "2.0x"]

CAM_KEYS = {
    "head": "observation.images.rgb.head",
    "lw":   "observation.images.rgb.left_wrist",
    "rw":   "observation.images.rgb.right_wrist",
}

TEST_MODE      = True
TEST_MAX_STEPS = 200
N_TASKS        = 10

# Action channels to compare (index into 23-D action vector)
ACTION_PARTS: List[Tuple[str, int]] = [
    ("base-vx",  0),   # base velocity
    ("trunk j0", 3),   # waist
    ("arm-L j6", 13),  # left wrist
    ("arm-R j6", 21),  # right wrist
    ("grip-L",   14),  # left gripper
    ("grip-R",   22),  # right gripper
]

VSTA_CFG = VSTAConfig(
    velocity_indices  = [0, 1, 2],
    abs_joint_indices = [3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21],
    gripper_indices   = [14, 22],
    speed_set=SPEEDS,
)

# ---------------------------------------------------------------------------
# Canvas layout
# ---------------------------------------------------------------------------
OUT_W, OUT_H = 1280, 864
DPI = 100
NCOLS = len(SPEEDS)
COL_WIDTHS = [OUT_W // NCOLS + (1 if i < OUT_W % NCOLS else 0) for i in range(NCOLS)]

GTITLE_H = 22
STITLE_H = 20
HEAD_H   = 206
WRIST_H  = 82
CAM_H    = STITLE_H + HEAD_H + WRIST_H   # per-column camera section height
COMP_H   = OUT_H - GTITLE_H - CAM_H       # full-width comparison panel height

BG_BGR     = (42,  23,  15)
FG_BGR     = (240, 232, 226)
CURSOR_BGR = (36,  191, 251)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def hex_to_bgr(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def fit_bgr(img_rgb: np.ndarray, h: int, w: int) -> np.ndarray:
    ih, iw = img_rgb.shape[:2]
    scale = max(h / ih, w / iw)
    nh, nw = max(1, int(ih * scale)), max(1, int(iw * scale))
    bgr = cv2.resize(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), (nw, nh))
    y0, x0 = (nh - h) // 2, (nw - w) // 2
    return bgr[y0:y0+h, x0:x0+w]


def text_bar(text: str, w: int, h: int, fg_bgr, font_scale: float = 0.38) -> np.ndarray:
    bar = np.full((h, w, 3), BG_BGR, dtype=np.uint8)
    tw, th = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
    x = max(0, (w - tw) // 2)
    y = (h + th) // 2
    cv2.putText(bar, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, fg_bgr, 1, cv2.LINE_AA)
    return bar


# ---------------------------------------------------------------------------
# Streaming video decoder
# ---------------------------------------------------------------------------

class SequentialDecoder:
    def __init__(self, path: Path):
        self._c = av.open(str(path))
        s = self._c.streams.video[0]
        s.thread_type = "AUTO"
        self._gen = self._c.decode(s)
        self._cur_idx: int = -1
        self._cur_frame: Optional[np.ndarray] = None

    def advance_to(self, target: int) -> np.ndarray:
        while self._cur_idx < target:
            try:
                frame = next(self._gen)
                self._cur_frame = frame.to_ndarray(format="rgb24")
                self._cur_idx += 1
            except StopIteration:
                break
        return self._cur_frame

    def close(self):
        self._c.close()


# ---------------------------------------------------------------------------
# Action comparison panel (full-width, all speeds overlaid)
# ---------------------------------------------------------------------------

def _action_vals(aug_actions: np.ndarray, idx: int, abs_joint_indices: List[int]) -> np.ndarray:
    """Return per-step action signal for one channel."""
    return aug_actions[:, idx].astype(np.float64)


def prerender_comparison_bg(
    aug_actions_list: List[np.ndarray],   # one array per speed, shapes (T_i, D)
    T_out: int,
    total_w: int,
    comp_h: int,
    abs_joint_indices: List[int],
) -> Tuple[np.ndarray, list]:
    """
    Render all speeds overlaid on the same axes for each action channel.
    X-axis is normalized to [0, T_out-1] so shorter (2x) sequences appear compressed left.
    Returns (bgr image, cursor_infos list-of-list[speed_idx][chan_idx]).
    """
    n_chan  = len(ACTION_PARTS)
    abs_set = set(abs_joint_indices)

    # Pre-compute all signals and global y limits per channel
    all_vals: List[List[np.ndarray]] = []  # [speed_idx][chan_idx]
    for actions in aug_actions_list:
        speed_vals = [_action_vals(actions, idx, abs_joint_indices)
                      for _, idx in ACTION_PARTS]
        all_vals.append(speed_vals)

    ylims = []
    for ci in range(n_chan):
        combined = np.concatenate([all_vals[si][ci] for si in range(len(SPEEDS))])
        vmin, vmax = combined.min(), combined.max()
        margin = (vmax - vmin) * 0.10 if vmax > vmin else abs(vmax) * 0.1 + 1e-6
        ylims.append((vmin - margin, vmax + margin))

    fig = plt.figure(figsize=(total_w / DPI, comp_h / DPI), dpi=DPI,
                     facecolor="#0f172a")
    gs  = gridspec.GridSpec(n_chan, 1, figure=fig,
                            left=0.06, right=0.99, top=0.96, bottom=0.12,
                            hspace=0.15)

    # cursor_infos[speed_idx] = list of per-chan dicts
    cursor_infos: List[List[dict]] = [[] for _ in SPEEDS]

    axes = []
    for ci, (name, idx) in enumerate(ACTION_PARTS):
        ax = fig.add_subplot(gs[ci])
        ax.set_facecolor("#0d1b2e")
        for sp in ax.spines.values():
            sp.set_color("#1e3a5f")
        ax.tick_params(colors="#94a3b8", labelsize=5, length=2, pad=1,
                       bottom=(ci == n_chan - 1), left=True)

        label = name

        for si, (speed, color, actions) in enumerate(
                zip(SPEEDS, SPD_COLORS, aug_actions_list)):
            T_i  = len(actions)
            xs   = np.linspace(0, T_out - 1, T_i)   # map T_i steps into [0, T_out)
            vals = all_vals[si][ci]
            ax.plot(xs, vals, color=color, linewidth=0.9, alpha=0.85,
                    label=f"{speed}x" if ci == 0 else "_nolegend_")

        ax.set_xlim(0, max(T_out - 1, 1))
        ax.set_ylim(ylims[ci])
        ax.yaxis.set_major_locator(plt.MaxNLocator(2))
        if ci < n_chan - 1:
            ax.set_xticklabels([])

        ax.text(0.005, 0.80, label, transform=ax.transAxes, color="#cbd5e1",
                fontsize=5.5, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.1", fc="#0d1b2e", ec="none", alpha=0.6))
        axes.append(ax)

    if n_chan > 0:
        axes[0].legend(fontsize=5, loc="upper right",
                       facecolor="#0d1b2e", edgecolor="#1e3a5f",
                       labelcolor=SPD_COLORS)
        axes[-1].set_xlabel(f"output step  (T_out={T_out})", color="#94a3b8", fontsize=5)

    fig.canvas.draw()
    fig_h_px = fig.get_figheight() * DPI

    for si, actions in enumerate(aug_actions_list):
        T_i = len(actions)
        # xs_frac: fractional position of each data point in [0, 1] relative to x-axis span
        xs_frac = np.linspace(0, 1.0, T_i)
        for ci, ax in enumerate(axes):
            bb   = ax.get_window_extent()
            ymin, ymax = ax.get_ylim()
            vals = all_vals[si][ci]
            cursor_infos[si].append({
                "xs_frac": xs_frac,       # fractional x positions in [0, 1]
                "x0": bb.x0, "x1": bb.x1,
                "y_top": fig_h_px - bb.y1,
                "y_bot": fig_h_px - bb.y0,
                "ymin": ymin, "ymax": ymax,
                "vals": vals,
            })

    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    cw, ch = fig.canvas.get_width_height()
    rgb    = buf.reshape(ch, cw, 4)[:, :, :3].copy()
    plt.close(fig)

    if rgb.shape[:2] != (comp_h, total_w):
        sy, sx = comp_h / ch, total_w / cw
        rgb = cv2.resize(rgb, (total_w, comp_h), interpolation=cv2.INTER_AREA)
        for si in range(len(SPEEDS)):
            for info in cursor_infos[si]:
                info["x0"] *= sx; info["x1"] *= sx
                info["y_top"] *= sy; info["y_bot"] *= sy
                # xs_frac is in [0,1] — no scaling needed

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cursor_infos


def draw_comparison_cursors(
    comp_bg: np.ndarray,
    cursor_infos: List[List[dict]],   # [speed_idx][chan_idx]
    aug_ts: List[int],                # current aug_t for each speed
    T_augs: List[int],
    T_out: int,
) -> np.ndarray:
    """Draw one cursor per speed (colored) on the comparison panel."""
    img   = comp_bg.copy()
    font  = cv2.FONT_HERSHEY_SIMPLEX
    fscl  = 0.28
    thick = 1

    for si, (aug_t, T_aug, color_hex) in enumerate(zip(aug_ts, T_augs, SPD_COLORS)):
        c_bgr = hex_to_bgr(color_hex)
        xs_frac = cursor_infos[si][0]["xs_frac"]   # fractional positions in [0, 1]
        safe_t  = min(aug_t, len(xs_frac) - 1)
        frac    = xs_frac[safe_t]

        for ci, info in enumerate(cursor_infos[si]):
            # convert fraction → pixel using the axes bounding box
            x_px = int(round(info["x0"] + frac * (info["x1"] - info["x0"])))
            top  = int(round(info["y_top"]))
            bot  = int(round(info["y_bot"]))
            cv2.line(img, (x_px, top), (x_px, bot), c_bgr, 1, cv2.LINE_AA)

            idx  = min(aug_t, len(info["vals"]) - 1)
            val  = info["vals"][idx]
            yr   = info["ymax"] - info["ymin"]
            yn   = (val - info["ymin"]) / yr if yr > 1e-10 else 0.5
            yn   = max(0.0, min(1.0, yn))
            y_px = int(round(bot - yn * (bot - top)))
            cv2.circle(img, (x_px, y_px), 2, c_bgr, -1, cv2.LINE_AA)

    return img


# ---------------------------------------------------------------------------
# Per-frame composition
# ---------------------------------------------------------------------------

def compose_frame(
    t:            int,
    decoders:     Dict[str, List[SequentialDecoder]],
    vid_maps:     List[np.ndarray],
    T_augs:       List[int],
    ep_name:      str,
    T_out:        int,
    comp_bg:      np.ndarray,
    cursor_infos: List[List[dict]],
) -> np.ndarray:
    """Returns one output frame as RGB (OUT_H, OUT_W, 3)."""

    gtitle = text_bar(
        f"VSTA - {ep_name}   step {t}/{T_out-1}   "
        f"[blue=0.5x  green=1x  red=2x]",
        OUT_W, GTITLE_H, FG_BGR, font_scale=0.38,
    )

    # ── camera columns ──────────────────────────────────────────────────────
    cols = []
    aug_ts = []
    for si, (speed, color, label) in enumerate(zip(SPEEDS, SPD_COLORS, SPD_LABELS)):
        cw    = COL_WIDTHS[si]
        T_aug = T_augs[si]
        aug_t = min(t, T_aug - 1)
        aug_ts.append(aug_t)
        vid_t = int(vid_maps[si][aug_t])

        color_bgr  = hex_to_bgr(color)
        border_bgr = (74, 222, 128)
        is_frozen  = (t >= T_aug)

        sub = text_bar(
            f"{label}   f={vid_t}{'  [FROZEN]' if is_frozen else ''}",
            cw, STITLE_H, color_bgr, font_scale=0.33,
        )

        head_bgr = fit_bgr(decoders["head"][si].advance_to(vid_t), HEAD_H, cw)
        cv2.rectangle(head_bgr, (0, 0), (cw - 1, HEAD_H - 1), border_bgr, 2)

        ww     = cw // 2
        lw_bgr = fit_bgr(decoders["lw"][si].advance_to(vid_t), WRIST_H, ww)
        rw_bgr = fit_bgr(decoders["rw"][si].advance_to(vid_t), WRIST_H, cw - ww)
        wrist  = np.concatenate([lw_bgr, rw_bgr], axis=1)

        if is_frozen:
            head_bgr = (head_bgr * 0.45).astype(np.uint8)
            wrist    = (wrist    * 0.45).astype(np.uint8)

        col = np.concatenate([sub, head_bgr, wrist], axis=0)
        if col.shape[0] != CAM_H:
            col = cv2.resize(col, (cw, CAM_H))
        cols.append(col)

    cam_strip = np.concatenate(cols, axis=1)
    if cam_strip.shape[1] < OUT_W:
        pad = np.full((CAM_H, OUT_W - cam_strip.shape[1], 3), BG_BGR, dtype=np.uint8)
        cam_strip = np.concatenate([cam_strip, pad], axis=1)

    # ── comparison panel with cursors ────────────────────────────────────────
    comp = draw_comparison_cursors(comp_bg, cursor_infos, aug_ts, T_augs, T_out)
    if comp.shape[1] != OUT_W:
        comp = cv2.resize(comp, (OUT_W, COMP_H))

    frame_bgr = np.concatenate([gtitle, cam_strip, comp], axis=0)
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Per-task entry point
# ---------------------------------------------------------------------------

def process_task(task_name: str, vsta_obj: VSTA, rng: np.random.Generator) -> None:
    data_dir  = DEMO_ROOT / "data"   / task_name
    video_dir = DEMO_ROOT / "videos" / task_name

    episodes = sorted(data_dir.glob("*.parquet"))
    if not episodes:
        print(f"  [skip] {task_name}: no parquet files", flush=True)
        return

    ep_path  = episodes[int(rng.integers(len(episodes)))]
    ep_name  = ep_path.stem
    prefix   = "test_" if TEST_MODE else ""
    out_path = OUTPUT_DIR / f"{prefix}{task_name}.mp4"

    if out_path.exists():
        out_path.unlink()

    print(f"\n-- {task_name}  ep={ep_name}", flush=True)

    df           = pd.read_parquet(ep_path)
    actions_orig = np.stack(df["action"].values).astype(np.float32)
    states_orig  = np.stack(df["observation.state"].values).astype(np.float32)
    T_orig       = len(actions_orig)

    start_frame = 0
    if TEST_MODE and T_orig > TEST_MAX_STEPS:
        start_frame  = int(rng.integers(0, T_orig - TEST_MAX_STEPS + 1))
        actions_orig = actions_orig[start_frame:start_frame + TEST_MAX_STEPS]
        states_orig  = states_orig[start_frame:start_frame + TEST_MAX_STEPS]
        T_orig       = TEST_MAX_STEPS
        print(f"   [test] sampled {T_orig} steps at offset={start_frame}")

    print(f"   T_orig={T_orig}  action={actions_orig.shape[1]}D  state={states_orig.shape[1]}D")

    frame_idx_obs = np.arange(start_frame, start_frame + T_orig, dtype=np.float32).reshape(T_orig, 1)
    aug_actions, aug_vid_maps = [], []
    for s in SPEEDS:
        rs = vsta_obj({"action": actions_orig, "observation": states_orig}, speed=s, seed=42)
        ri = vsta_obj({"action": actions_orig, "observation": frame_idx_obs}, speed=s, seed=42)
        vm = rs["valid_mask"]
        aug_actions.append(rs["action"].numpy()[vm])
        aug_vid_maps.append(ri["observation"].numpy().flatten().astype(int)[vm])
        act_norm = np.linalg.norm(rs["action"].numpy()[vm], axis=1).mean()
        print(f"   {s}x  T={T_orig}->{vm.sum()}  act_norm_mean={act_norm:.4f}")

    T_augs = [len(a) for a in aug_actions]
    T_out  = max(T_augs)
    print(f"   T_augs={T_augs}  T_out={T_out}  duration={T_out/OUTPUT_FPS:.1f}s @ {OUTPUT_FPS} fps")

    # Diagnostics
    abs_set_diag = set(VSTA_CFG.abs_joint_indices)
    print("   Action diff diagnostics (mean |per-step change|):")
    for name, idx in ACTION_PARTS:
        parts = []
        for si, s in enumerate(SPEEDS):
            vals = aug_actions[si][:, idx]
            if idx in abs_set_diag:
                metric = float(np.abs(np.diff(vals)).mean())
            else:
                metric = float(np.abs(vals).mean())
            parts.append(f"{s}x={metric:.5f}")
        print(f"     {name:14s}: {' | '.join(parts)}")

    print("   pre-rendering comparison panel...", flush=True)
    comp_bg, cursor_infos = prerender_comparison_bg(
        aug_actions, T_out, OUT_W, COMP_H, VSTA_CFG.abs_joint_indices,
    )

    decoders: Dict[str, List[SequentialDecoder]] = {}
    for cam_key, cam_path in CAM_KEYS.items():
        vid_file = video_dir / cam_path / f"{ep_name}.mp4"
        decoders[cam_key] = [SequentialDecoder(vid_file) for _ in SPEEDS]

    print(f"   rendering -> {out_path.name}", flush=True)

    writer = imageio.get_writer(
        str(out_path), fps=OUTPUT_FPS,
        codec="libx264", quality=7, macro_block_size=16,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    for t in range(T_out):
        frame = compose_frame(
            t, decoders, aug_vid_maps, T_augs, ep_name, T_out,
            comp_bg, cursor_infos,
        )
        writer.append_data(frame)
        if t % 100 == 0:
            print(f"   {t}/{T_out}", flush=True)
    writer.close()

    for cam_decs in decoders.values():
        for dec in cam_decs:
            dec.close()

    mb = out_path.stat().st_size / 1e6
    print(f"   saved: {out_path.name}  {mb:.1f} MB", flush=True)


# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    vsta_obj = VSTA(VSTA_CFG)
    rng = np.random.default_rng(42)

    all_tasks = sorted(p for p in (DEMO_ROOT / "data").iterdir() if p.is_dir())
    n = min(N_TASKS, len(all_tasks))
    chosen_idx = rng.choice(len(all_tasks), size=n, replace=False)
    tasks = [all_tasks[i] for i in sorted(chosen_idx)]
    print(f"Tasks: {[t.name for t in tasks]}  TEST_MODE={TEST_MODE}  MAX_STEPS={TEST_MAX_STEPS if TEST_MODE else 'full'}")

    for task_dir in tasks:
        try:
            process_task(task_dir.name, vsta_obj, rng)
        except Exception as exc:
            import traceback
            print(f"  [ERROR] {task_dir.name}: {exc}", flush=True)
            traceback.print_exc()

    print("\nDone.")


if __name__ == "__main__":
    main()
