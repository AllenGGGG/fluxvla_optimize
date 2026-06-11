"""
Variable-Speed Trajectory Augmentation (VSTA)
TempoVLA (arXiv 2606.06491) Algorithm 1 — simplified for absolute action spaces.

All continuous dims use direct segment-level interpolation (no cumsum).
  - abs_joint_indices: absolute joint positions — interp only.
  - velocity_indices:  velocity commands (base) — interp then scale by q/p
                       to preserve integrated displacement.
  - gripper_indices:   discrete — copy last frame per chunk.

Segmentation: gripper hard boundaries only.
Valid-mask:   chunk-start observations only (paper §3).
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
class VSTAConfig:
    # Velocity commands (e.g. base vx/vy/ω): direct interp + scale by q/p
    velocity_indices:  List[int] = field(default_factory=lambda: [0, 1, 2])

    # Absolute joint-angle commands: direct interp, no scaling
    abs_joint_indices: List[int] = field(default_factory=lambda: [3, 4, 5])

    # Discrete gripper commands: copy last frame of each chunk
    gripper_indices:   List[int] = field(default_factory=lambda: [6])

    # Speed set sampled during training
    speed_set: List[float] = field(default_factory=lambda: [0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    fixed_speed: Optional[float] = None
    speed_text_template: str = "Perform the task at {speed}× speed. "


# ---------------------------------------------------------------------------
# SEGMENT — gripper hard boundaries only
# ---------------------------------------------------------------------------

def segment(actions: np.ndarray, cfg: VSTAConfig) -> List[List[int]]:
    """
    Split a demo into segments at gripper state-change boundaries.

    A boundary is added whenever any gripper dim changes sign
    (open ↔ close), ensuring the resampling never interpolates
    across a grasp/release event.
    """
    T = len(actions)
    if T == 0:
        return []

    boundaries = {0}
    if cfg.gripper_indices:
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
# Helper: s → coprime integers (q, p)
# ---------------------------------------------------------------------------

def speed_to_qp(s: float, max_denom: int = 20) -> Tuple[int, int]:
    """s = q/p with coprime positive integers q, p."""
    frac = Fraction(s).limit_denominator(max_denom)
    q, p = frac.numerator, frac.denominator
    assert q > 0 and p > 0, f"invalid speed s={s}"
    return q, p


# ---------------------------------------------------------------------------
# CHUNK-LEVEL — gripper only
# ---------------------------------------------------------------------------

def _resample_chunk(actions_chunk: np.ndarray, p: int, cfg: VSTAConfig) -> np.ndarray:
    """
    Produce p output steps from a chunk of q input steps.
    Continuous dims are left as zeros — overwritten by segment-level interp.
    Gripper: copy the last frame of the chunk to all p output steps.
    """
    D   = actions_chunk.shape[1]
    out = np.zeros((p, D), dtype=np.float64)
    for gi in cfg.gripper_indices:
        out[:, gi] = actions_chunk[-1, gi]
    return out.astype(actions_chunk.dtype)


# ---------------------------------------------------------------------------
# SEGMENT-LEVEL SPEED TRANSFORM
# ---------------------------------------------------------------------------

def _resample_segment(
    actions_seg: np.ndarray,           # (L, D)
    obs_seg:     Optional[np.ndarray], # (L, ...) or None
    q: int,
    p: int,
    cfg: VSTAConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Speed-transform one segment.

    Chunk structure determines valid_mask and observation alignment (paper §3).
    All continuous action dims are filled by segment-level np.interp at the
    chunk-aligned original times (abs_jt), then velocity dims are scaled by q/p.

    abs_jt for output step j in chunk [chunk_start, chunk_start+q):
        abs_jt = clamp(chunk_start + (j+1)*q/p - 1, 0, L-1)

    This places abs_joint targets at the END of each output sub-step's
    covered interval, keeping velocity and abs_joint dims time-aligned.

    Valid-mask:
      - Leading passthrough [0, r): all valid
      - Each chunk: only j=0 (chunk-start observation) is valid
      - Trailing remainder: all valid
    """
    L = len(actions_seg)
    r = int(rng.integers(0, q)) if q > 1 else 0

    new_actions: List[np.ndarray] = []
    valid:       List[bool]       = []
    new_obs:     Optional[List]   = [] if obs_seg is not None else None
    abs_jt:      List[float]      = []

    def _append(a, v, jt, o=None):
        new_actions.append(a)
        valid.append(v)
        abs_jt.append(jt)
        if new_obs is not None:
            new_obs.append(o)

    # ── Leading passthrough [0, r) ────────────────────────────────────────
    for t in range(r):
        _append(actions_seg[t], True, float(t),
                obs_seg[t] if obs_seg is not None else None)

    # ── Non-overlapping chunks of size q ──────────────────────────────────
    chunk_start = r
    while chunk_start + q <= L:
        chunk     = actions_seg[chunk_start: chunk_start + q]
        new_steps = _resample_chunk(chunk, p, cfg)
        for j in range(p):
            jt = max(0.0, min(float(L - 1),
                              chunk_start + (j + 1) * q / p - 1.0))
            _append(new_steps[j], j == 0, jt,
                    obs_seg[chunk_start] if obs_seg is not None else None)
        chunk_start += q

    # ── Trailing remainder ────────────────────────────────────────────────
    for t in range(chunk_start, L):
        _append(actions_seg[t], True, float(t),
                obs_seg[t] if obs_seg is not None else None)

    out_actions = np.stack(new_actions, axis=0)   # (T_new, D)

    # ── Continuous dims: direct segment-level interpolation ───────────────
    all_cont = cfg.velocity_indices + cfg.abs_joint_indices
    if all_cont and L > 0:
        jt_arr  = np.array(abs_jt, dtype=float)
        orig_ts = np.arange(L, dtype=float)
        for ai in all_cont:
            out_actions[:, ai] = np.interp(
                jt_arr, orig_ts, actions_seg[:, ai].astype(float)
            ).astype(out_actions.dtype)

    # ── Velocity dims: scale by q/p to preserve integrated displacement ───
    # At 2x (q=2,p=1): T_new≈T/2 steps × 2× velocity = same total displacement.
    # At 0.5x (q=1,p=2): T_new≈2T steps × 0.5× velocity = same total displacement.
    if cfg.velocity_indices and p != q:
        scale = float(q) / float(p)
        out_actions[:, cfg.velocity_indices] = (
            out_actions[:, cfg.velocity_indices].astype(np.float64) * scale
        ).astype(out_actions.dtype)

    out_valid = np.array(valid, dtype=bool)
    out_obs   = np.stack(new_obs, axis=0) if new_obs is not None else None

    return out_actions, out_valid, out_obs


# ---------------------------------------------------------------------------
# VSTA main class
# ---------------------------------------------------------------------------

class VSTA:
    """
    Full VSTA transform; input/output follow the LeRobot episode dict format.

    Input episode keys:
      "action":      Tensor / ndarray  (T, action_dim)   required
      "observation": Tensor / ndarray  (T, ...)           optional

    Output adds:
      "valid_mask":  ndarray (T',) bool
      "speed":       float
      "speed_text":  str
    """

    def __init__(self, cfg: Optional[VSTAConfig] = None, **kwargs):
        self.cfg = cfg or VSTAConfig(**kwargs)
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
        segments = segment(actions, self.cfg)

        all_actions: List[np.ndarray] = []
        all_valid:   List[np.ndarray] = []
        all_obs:     Optional[List]   = [] if obs is not None else None

        for seg_idx in segments:
            seg_act = actions[seg_idx]
            seg_obs = obs[seg_idx] if obs is not None else None
            new_act, valid_mask, new_obs = _resample_segment(
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
            out["observation"] = torch.from_numpy(np.concatenate(all_obs, axis=0))

        return out

    def apply_valid_mask(self, episode: Dict) -> Dict:
        """Keep only valid frames (for variable-length sequence training)."""
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

class VSTATransform:
    def __init__(self, cfg: Optional[VSTAConfig] = None, **kwargs):
        self.vsta = VSTA(cfg, **kwargs)

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
# Validation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("VSTA validation")
    print("=" * 65)

    rng = np.random.default_rng(42)
    cfg = VSTAConfig(speed_set=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0])

    # ── Test 1: Gripper-only segmentation ────────────────────────────────
    print("\n[Test 1] Gripper-only segmentation")
    T, D = 100, 7
    a = np.zeros((T, D), dtype=np.float32)
    a[:50, 0]  = 0.01   # base velocity (no segment boundary)
    a[50:, 6]  = 1.0    # gripper closes at t=50
    segs = segment(a, cfg)
    assert len(segs) == 2, f"expected 2 segments, got {len(segs)}"
    assert segs[0] == list(range(50)), "first segment wrong"
    assert segs[1] == list(range(50, 100)), "second segment wrong"
    print(f"  2 segments at gripper boundary ✓  lengths={[len(s) for s in segs]}")

    # no gripper change → single segment
    a3 = np.zeros((20, D), dtype=np.float32)
    a3[:, 0] = 0.01
    segs3 = segment(a3, cfg)
    assert len(segs3) == 1, "no gripper change should give 1 segment"
    print(f"  no gripper change → 1 segment ✓")

    # ── Test 2: 1x identity ───────────────────────────────────────────────
    print("\n[Test 2] 1x identity")
    T2 = 60
    act2 = np.random.default_rng(0).random((T2, D)).astype(np.float32) * 0.1
    act2[:, 6] = 0.0   # gripper open throughout
    ep2  = {"action": torch.from_numpy(act2)}
    vsta = VSTA(cfg)
    r1x  = vsta(ep2, speed=1.0, seed=0)
    assert r1x["valid_mask"].all(), "all frames valid at 1x"
    assert np.allclose(r1x["action"].numpy(), act2, atol=1e-5), "1x should be identity"
    print(f"  1x identity ✓  T={T2}→{len(r1x['action'])}")

    # ── Test 3: Speed control (frame counts) ─────────────────────────────
    print("\n[Test 3] Speed control (frame counts)")
    for s in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        res  = vsta(ep2, speed=s, seed=0)
        T_new = len(res["action"])
        q_, p_ = speed_to_qp(s)
        if s > 1.0:
            assert T_new < T2, f"T should decrease at {s}x"
        elif s < 1.0:
            assert T_new > T2, f"T should increase at {s}x"
        print(f"  s={s:4.2f}x (q={q_},p={p_}) T:{T2}→{T_new:3d}  "
              f"valid:{res['valid_mask'].mean():.1%}  "
              f"text:\"{res['speed_text']}\" ✓")

    # ── Test 4: Velocity scaling preserves displacement ───────────────────
    print("\n[Test 4] Velocity scaling")
    T4  = 40
    act4 = np.zeros((T4, D), dtype=np.float32)
    act4[:, 0] = 0.05   # constant base-vx
    act4[:, 6] = 0.0
    ep4  = {"action": torch.from_numpy(act4)}
    for s in [0.5, 1.0, 2.0]:
        res   = vsta(ep4, speed=s, seed=0)
        v_aug = res["action"].numpy()[:, 0]
        disp_orig = float(act4[:, 0].sum())
        disp_aug  = float(v_aug.sum())
        err = abs(disp_aug - disp_orig) / (abs(disp_orig) + 1e-8)
        print(f"  {s}x  disp_orig={disp_orig:.4f}  disp_aug={disp_aug:.4f}  "
              f"rel_err={err:.4f} {'✓' if err < 0.05 else '✗'}")

    # ── Test 5: abs_joint step-size ratio ────────────────────────────────
    print("\n[Test 5] abs_joint step-size ratio")
    T5  = 20
    act5 = np.zeros((T5, D), dtype=np.float32)
    act5[:, 3] = np.linspace(0.0, 1.9, T5)   # abs joint linearly increasing
    act5[:, 6] = 0.0
    ep5  = {"action": torch.from_numpy(act5)}
    results = {}
    for s in [0.5, 1.0, 2.0]:
        res = vsta(ep5, speed=s, seed=0)
        vals = res["action"].numpy()[:, 3]
        results[s] = float(np.abs(np.diff(vals)).mean())
    print(f"  mean |Δ abs_joint|: 0.5x={results[0.5]:.4f}  "
          f"1x={results[1.0]:.4f}  2x={results[2.0]:.4f}")
    assert results[0.5] < results[1.0] * 0.8, "0.5x steps should be smaller"
    assert results[2.0] > results[1.0] * 1.2, "2x steps should be larger"
    print("  0.5x < 1x < 2x ✓")

    # ── Test 6: Gripper not interpolated ─────────────────────────────────
    print("\n[Test 6] Gripper values stay 0 or 1")
    T6   = 30
    act6 = np.zeros((T6, D), dtype=np.float32)
    act6[15:, 6] = 1.0   # gripper closes at t=15
    ep6  = {"action": torch.from_numpy(act6)}
    for s in [0.5, 2.0]:
        res  = vsta(ep6, speed=s, seed=0)
        grip = res["action"].numpy()[:, 6]
        assert set(np.unique(np.round(grip, 3))).issubset({0.0, 1.0}), \
            f"gripper has non-binary values at {s}x"
    print("  gripper binary at 0.5x and 2x ✓")

    print("\nAll tests passed ✓")
