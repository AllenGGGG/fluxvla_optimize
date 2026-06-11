"""
Variable-Speed Trajectory Augmentation — Delta Action Space
TempoVLA (arXiv 2606.06491) Algorithm 1, exact implementation.

Implements the paper's accumulate-then-split for delta action spaces:
    Δ = Σ aᵢ  (chunk-level accumulation)
    output[j] = interp(cum, (j+1)·q/p) − interp(cum, j·q/p)

This is exact (no q/p post-scaling needed) because the telescoping sum
Σⱼ output[j] = cum[q] − cum[0] = Δ by construction.

Action format:
    7D  [Δx, Δy, Δz, Δax, Δay, Δaz, gripper_abs]
    delta_indices   = [0,1,2,3,4,5]  — delta pos + delta axis-angle (so(3))
    gripper_indices = [6]            — discrete, copy last frame

Observation format:
    8D  [x, y, z, ax, ay, az, g1, g2]  (absolute, axis-angle)
    Handled by copying the chunk-start frame — no interpolation needed.

Segmentation: gripper hard boundaries (paper §3).
Valid-mask:   chunk-start observations only (paper §3).

Composition property: adding delta axis-angle vectors equals composing the
corresponding rotations when the rotation axis stays approximately constant
within a segment (ensured by gripper-boundary segmentation for manipulation
tasks with small per-step rotations).
"""

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class VSTADeltaConfig:
    # Delta dims (Δpos + Δaxis-angle): accumulate-then-split per paper
    delta_indices:   List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])

    # Discrete gripper: copy last frame of each chunk
    gripper_indices: List[int] = field(default_factory=lambda: [6])

    speed_set: List[float] = field(
        default_factory=lambda: [0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    fixed_speed: Optional[float] = None
    speed_text_template: str = "Perform the task at {speed}× speed. "


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def speed_to_qp(s: float, max_denom: int = 20) -> Tuple[int, int]:
    """s = q/p with coprime positive integers q, p."""
    frac = Fraction(s).limit_denominator(max_denom)
    q, p = frac.numerator, frac.denominator
    assert q > 0 and p > 0, f"invalid speed s={s}"
    return q, p


def segment_gripper(actions: np.ndarray, cfg: VSTADeltaConfig) -> List[List[int]]:
    """
    Split a demo into segments at gripper state-change boundaries.
    A boundary is added whenever any gripper dim changes sign (open↔close).
    """
    T = len(actions)
    if T == 0:
        return []

    boundaries = {0}
    for t in range(1, T):
        g_prev = np.sign(actions[t - 1, cfg.gripper_indices])
        g_curr = np.sign(actions[t,     cfg.gripper_indices])
        if not np.array_equal(g_prev, g_curr):
            boundaries.add(t)

    sorted_b = sorted(boundaries) + [T]
    return [list(range(sorted_b[i], sorted_b[i + 1]))
            for i in range(len(sorted_b) - 1)
            if sorted_b[i + 1] > sorted_b[i]]


# ---------------------------------------------------------------------------
# Chunk-level: accumulate-then-split  (paper Algorithm 1, steps 8-9)
# ---------------------------------------------------------------------------

def _resample_chunk_delta(
    chunk: np.ndarray,       # (q, D)
    p:     int,
    cfg:   VSTADeltaConfig,
) -> np.ndarray:
    """
    Map q source frames → p output frames for one chunk.

    Delta dims (paper Algorithm 1):
        cum[0]=0, cum[k] = Σᵢ₌₀ᵏ⁻¹ aᵢ      (cumulative sum, length q+1)
        output[j] = interp(cum, (j+1)·q/p) − interp(cum, j·q/p)
        Sum of outputs = cum[q] − cum[0] = Δ  (exact, no q/p scaling needed)

    Gripper:
        Copy the last frame of the chunk to all p outputs.
    """
    q      = len(chunk)
    D      = chunk.shape[1]
    out    = np.zeros((p, D), dtype=np.float64)
    cum_ts = np.arange(q + 1, dtype=float)           # 0, 1, …, q

    t_hi = np.array([(j + 1) * q / p for j in range(p)], dtype=float)
    t_lo = np.array([j       * q / p for j in range(p)], dtype=float)

    for vi in cfg.delta_indices:
        col = chunk[:, vi].astype(np.float64)
        cum = np.concatenate([[0.0], np.cumsum(col)])  # shape (q+1,)
        out[:, vi] = np.interp(t_hi, cum_ts, cum) - np.interp(t_lo, cum_ts, cum)

    for gi in cfg.gripper_indices:
        out[:, gi] = chunk[-1, gi]

    return out.astype(chunk.dtype)


# ---------------------------------------------------------------------------
# Segment-level speed transform
# ---------------------------------------------------------------------------

def _resample_segment_delta(
    actions_seg: np.ndarray,            # (L, D)
    obs_seg:     Optional[np.ndarray],  # (L, ...) or None
    q:           int,
    p:           int,
    cfg:         VSTADeltaConfig,
    rng:         np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Speed-transform one segment via chunk-level accumulate-then-split.

    Chunk-start offset r ~ U{0, …, q-1} per segment (paper §3 online sampling).

    Valid-mask:
      - Leading passthrough [0, r): all valid
      - Each chunk: only j=0 (chunk-start observation) is valid
      - Trailing remainder: all valid
    """
    L = len(actions_seg)
    r = int(rng.integers(0, q)) if q > 1 else 0

    new_actions: List[np.ndarray] = []
    valid:        List[bool]       = []
    new_obs:      Optional[List]   = [] if obs_seg is not None else None

    def _append(a, v, o=None):
        new_actions.append(a)
        valid.append(v)
        if new_obs is not None:
            new_obs.append(o)

    # Leading passthrough [0, r) — verbatim, all valid
    for t in range(r):
        _append(actions_seg[t], True,
                obs_seg[t] if obs_seg is not None else None)

    # Non-overlapping chunks of size q
    chunk_start = r
    while chunk_start + q <= L:
        chunk     = actions_seg[chunk_start: chunk_start + q]
        new_steps = _resample_chunk_delta(chunk, p, cfg)
        obs_chunk = obs_seg[chunk_start] if obs_seg is not None else None
        for j in range(p):
            _append(new_steps[j], j == 0, obs_chunk)
        chunk_start += q

    # Trailing remainder — verbatim, all valid
    for t in range(chunk_start, L):
        _append(actions_seg[t], True,
                obs_seg[t] if obs_seg is not None else None)

    out_actions = np.stack(new_actions, axis=0)
    out_valid   = np.array(valid, dtype=bool)
    out_obs     = np.stack(new_obs, axis=0) if new_obs is not None else None

    return out_actions, out_valid, out_obs


# ---------------------------------------------------------------------------
# VSTADelta — main class
# ---------------------------------------------------------------------------

class VSTADelta:
    """
    Full VSTA transform for delta action spaces.

    Input episode keys:
      "action":      Tensor / ndarray  (T, action_dim)   required
      "observation": Tensor / ndarray  (T, ...)           optional

    Output adds:
      "valid_mask":  ndarray (T',) bool
      "speed":       float
      "speed_text":  str
    """

    def __init__(self, cfg: Optional[VSTADeltaConfig] = None, **kwargs):
        self.cfg  = cfg or VSTADeltaConfig(**kwargs)
        self._rng = np.random.default_rng()

    def __call__(
        self,
        episode: Dict,
        speed:   Optional[float] = None,
        seed:    Optional[int]   = None,
    ) -> Dict:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        actions = episode["action"]
        if isinstance(actions, torch.Tensor):
            actions = actions.numpy()
        actions = np.asarray(actions, dtype=np.float32)

        obs = episode.get("observation", None)
        if obs is not None and isinstance(obs, torch.Tensor):
            obs = obs.numpy()

        if speed is None:
            speed = self.cfg.fixed_speed
        if speed is None:
            speed = float(self._rng.choice(self.cfg.speed_set))

        q, p     = speed_to_qp(speed)
        segments = segment_gripper(actions, self.cfg)

        all_actions: List[np.ndarray] = []
        all_valid:   List[np.ndarray] = []
        all_obs:     Optional[List]   = [] if obs is not None else None

        for seg_idx in segments:
            seg_act = actions[seg_idx]
            seg_obs = obs[seg_idx] if obs is not None else None
            new_act, valid_mask, new_obs = _resample_segment_delta(
                seg_act, seg_obs, q, p, self.cfg, self._rng
            )
            all_actions.append(new_act)
            all_valid.append(valid_mask)
            if all_obs is not None:
                all_obs.append(new_obs)

        new_actions = np.concatenate(all_actions, axis=0)
        valid_mask  = np.concatenate(all_valid,   axis=0)

        out = dict(episode)
        out["action"]     = torch.from_numpy(new_actions)
        out["valid_mask"] = valid_mask
        out["speed"]      = speed
        out["speed_text"] = self.cfg.speed_text_template.format(speed=speed)

        if all_obs is not None:
            out["observation"] = torch.from_numpy(
                np.concatenate(all_obs, axis=0))

        return out

    def apply_valid_mask(self, episode: Dict) -> Dict:
        """Keep only valid frames (chunk-start observations)."""
        mask = episode.get("valid_mask")
        if mask is None:
            return episode
        return {
            k: (v[mask] if isinstance(v, (np.ndarray, torch.Tensor))
                and len(v) == len(mask) else v)
            for k, v in episode.items()
            if k != "valid_mask"
        }


# ---------------------------------------------------------------------------
# LeRobot transform wrapper
# ---------------------------------------------------------------------------

class VSTADeltaTransform:
    """Drop-in for VSTATransform, delta action spaces."""

    def __init__(self, cfg: Optional[VSTADeltaConfig] = None, **kwargs):
        self.vsta = VSTADelta(cfg, **kwargs)

    def __call__(self, sample: Dict) -> Dict:
        episode = {"action": sample["action"]}
        if "observation.state" in sample:
            episode["observation"] = sample["observation.state"]

        aug = self.vsta(episode)
        out = dict(sample)
        out["action"]     = aug["action"]
        out["speed"]      = aug["speed"]
        out["valid_mask"] = torch.from_numpy(aug["valid_mask"])
        if "task" in out:
            out["task"] = aug["speed_text"] + out["task"]
        if "observation.state" in out and "observation" in aug:
            out["observation.state"] = aug["observation"]
        return out


# ---------------------------------------------------------------------------
# Tests + Visual
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 65)
    print("VSTADelta validation")
    print("=" * 65)

    cfg  = VSTADeltaConfig()
    vsta = VSTADelta(cfg)
    D    = 7

    # ── Test 1: 1x identity ───────────────────────────────────────────────
    print("\n[Test 1] 1x identity")
    T1   = 60
    act1 = np.random.default_rng(0).random((T1, D)).astype(np.float32) * 0.05
    act1[:, 6] = 0.0
    ep1  = {"action": torch.from_numpy(act1)}
    r1   = vsta(ep1, speed=1.0, seed=0)
    assert r1["valid_mask"].all(), "all frames must be valid at 1x"
    assert np.allclose(r1["action"].numpy(), act1, atol=1e-5), "1x must be identity"
    print(f"  T={T1}→{len(r1['action'])}  valid={r1['valid_mask'].mean():.0%}  ✓")

    # ── Test 2: exact displacement preservation (non-constant delta) ──────
    # vsta.py (segment-level interp) fails this; vsta_delta.py must pass.
    print("\n[Test 2] Exact displacement preservation (non-constant delta)")
    T2   = 40
    act2 = np.zeros((T2, D), dtype=np.float32)
    act2[:, 6] = 0.0
    act2[:, 0] = np.tile([0.1, 0.3], T2 // 2)   # alternating magnitudes
    act2[:, 3] = np.tile([0.02, 0.06], T2 // 2)
    ep2  = {"action": torch.from_numpy(act2)}
    for s in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        res = vsta(ep2, speed=s, seed=0)
        aug = res["action"].numpy()
        for di in [0, 3]:
            orig = float(act2[:, di].sum())
            augd = float(aug[:, di].sum())
            err  = abs(augd - orig) / (abs(orig) + 1e-8)
            assert err < 1e-5, \
                f"dim={di} s={s}: orig={orig:.5f} aug={augd:.5f} err={err:.2e}"
        print(f"  s={s:.2f}x  T:{T2}→{len(aug):3d}  displacement exact ✓")

    # ── Test 3: frame counts ──────────────────────────────────────────────
    print("\n[Test 3] Speed control (frame counts)")
    for s in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        res   = vsta(ep1, speed=s, seed=0)
        T_new = len(res["action"])
        q_, p_ = speed_to_qp(s)
        if s > 1.0:
            assert T_new < T1, f"T should decrease at {s}x"
        elif s < 1.0:
            assert T_new > T1, f"T should increase at {s}x"
        print(f"  s={s:.2f}x (q={q_},p={p_})  T:{T1}→{T_new:3d}"
              f"  valid={res['valid_mask'].mean():.0%}"
              f"  \"{res['speed_text']}\" ✓")

    # ── Test 4: gripper stays binary ──────────────────────────────────────
    print("\n[Test 4] Gripper binary")
    T4   = 30
    act4 = np.zeros((T4, D), dtype=np.float32)
    act4[15:, 6] = 1.0
    ep4  = {"action": torch.from_numpy(act4)}
    for s in [0.5, 0.75, 1.25, 2.0]:
        res  = vsta(ep4, speed=s, seed=0)
        grip = res["action"].numpy()[:, 6]
        vals = set(np.unique(np.round(grip, 4)))
        assert vals.issubset({0.0, 1.0}), f"gripper non-binary at {s}x: {vals}"
    print("  gripper ∈ {0,1} at all speeds ✓")

    # ── Test 5: obs aligns to chunk-start originals ───────────────────────
    print("\n[Test 5] Observation chunk-start alignment")
    T5   = 20
    act5 = np.zeros((T5, D), dtype=np.float32)
    obs5 = np.arange(T5, dtype=np.float32).reshape(T5, 1)
    ep5  = {"action": torch.from_numpy(act5),
            "observation": torch.from_numpy(obs5)}
    orig_vals = set(float(v) for v in obs5[:, 0])
    for s in [0.5, 2.0]:
        res      = vsta(ep5, speed=s, seed=42)
        aug_obs  = res["observation"].numpy()
        valid    = res["valid_mask"]
        for t in range(len(aug_obs)):
            if valid[t]:
                assert float(aug_obs[t, 0]) in orig_vals, \
                    f"s={s}x t={t}: obs={aug_obs[t,0]} not in original frames"
    print("  valid-frame obs ∈ original frames ✓")

    print("\nAll tests passed ✓")

    # ── Visual ────────────────────────────────────────────────────────────
    print("\nGenerating visual…")

    T_vis = 60
    act_v = np.zeros((T_vis, D), dtype=np.float32)
    act_v[:, 6] = 0.0
    t_lin = np.linspace(0, 4 * np.pi, T_vis)
    act_v[:, 0] = (0.02 * np.sin(t_lin)).astype(np.float32)
    act_v[:, 3] = (0.01 * np.sin(t_lin * 1.3 + 1.0)).astype(np.float32)
    ep_v  = {"action": torch.from_numpy(act_v)}

    speeds = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    colors = ["#2196F3", "#4CAF50", "#9C27B0", "#FF9800", "#F44336", "#00BCD4"]
    dims   = [(0, "Δx (pos)"), (3, "Δax (axis-angle)")]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("VSTADelta — Accumulate-then-Split\n"
                 "Left: per-step delta  |  Right: cumulative trajectory",
                 fontsize=12)

    for row, (di, dim_name) in enumerate(dims):
        ax_act = axes[row, 0]
        ax_cum = axes[row, 1]

        orig_cum = np.cumsum(act_v[:, di])
        ax_cum.plot(np.linspace(0, T_vis - 1, T_vis), orig_cum,
                    color="black", lw=2.5, label="original", zorder=5)

        for s, c in zip(speeds, colors):
            res    = vsta(ep_v, speed=s, seed=0)
            aug    = res["action"].numpy()
            valid  = res["valid_mask"]
            T_new  = len(aug)
            t_out  = np.linspace(0, T_vis - 1, T_new)
            lw     = 2.2 if s == 1.0 else 1.1

            ax_act.plot(t_out, aug[:, di], color=c, lw=lw, alpha=0.85,
                        label=f"{s}×")
            ax_act.scatter(t_out[valid], aug[valid, di],
                           color=c, s=12, zorder=4, alpha=0.5)

            aug_cum = np.cumsum(aug[:, di])
            ax_cum.plot(t_out, aug_cum, color=c, lw=lw, alpha=0.85,
                        label=f"{s}×")

            orig_s = float(act_v[:, di].sum())
            aug_s  = float(aug[:, di].sum())
            err    = abs(aug_s - orig_s) / (abs(orig_s) + 1e-8)
            print(f"  visual dim={di} s={s}  "
                  f"orig={orig_s:.5f}  aug={aug_s:.5f}  err={err:.2e}")

        ax_act.set_title(f"{dim_name} — per-step")
        ax_act.set_xlabel("time (normalised)")
        ax_act.legend(fontsize=7, ncol=3)
        ax_act.grid(alpha=0.3)

        ax_cum.set_title(f"{dim_name} — cumulative")
        ax_cum.set_xlabel("time (normalised)")
        ax_cum.legend(fontsize=7, ncol=3)
        ax_cum.grid(alpha=0.3)

    plt.tight_layout()
    out_path = "/data/wqz/code/vsta/vsta_delta_visual.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved → {out_path}")
