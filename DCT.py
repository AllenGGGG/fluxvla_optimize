# ============================================================
# DCT Action Compression — delta vs absolute, three lines
# Colab Ready
# ============================================================

import numpy as np
from scipy.fft import dct
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import Image, display

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


# ── Core functions ────────────────────────────────────────────────────────────

def _dct_analytic(signal_2d, k, T_new):
    T      = len(signal_2d)
    coeffs = dct(signal_2d.astype(np.float64), axis=0, norm='ortho')
    ks     = np.arange(k, dtype=np.float64)
    n_new  = np.linspace(0, T - 1, T_new)
    basis  = np.cos(np.pi * (2*n_new[:,None]+1) * ks[None,:] / (2*T))
    norms  = np.where(ks == 0, 1.0/np.sqrt(T), np.sqrt(2.0/T))
    return basis @ (coeffs[:k] * norms[:, None])


def dct_compress(chunk, speed=1.5, keep_ratio=0.4,
                 action_space='delta', cont_idx=None, gripper_idx=None):
    """
    Compress action chunk to fewer timesteps via DCT.

    action_space='absolute': DCT applied directly (EEF absolute pose).
    action_space='delta':    cumsum -> DCT -> endpoint correction -> diff
                             (EEF delta, joint velocity).

    Returns (T_new, D) float32, T_new = max(2, int(T / speed)).
    """
    T, D = chunk.shape
    if cont_idx is None:    cont_idx    = list(range(D - 1))
    if gripper_idx is None: gripper_idx = [D - 1]

    T_new = max(2, int(T / speed))
    k     = max(2, int(T * keep_ratio))
    out   = np.zeros((T_new, D), dtype=chunk.dtype)
    cont  = chunk[:, cont_idx].astype(np.float64)

    if action_space == 'absolute':
        recon = _dct_analytic(cont, k, T_new)

    elif action_space == 'delta':
        abs_traj      = np.cumsum(cont, axis=0)
        abs_comp      = _dct_analytic(abs_traj, k, T_new)
        drift         = abs_traj[-1] - abs_comp[-1]
        alpha         = np.linspace(0, 1, T_new)[:, None]
        abs_corrected = abs_comp + drift * alpha
        recon         = np.empty_like(abs_corrected)
        recon[0]      = abs_corrected[0]
        recon[1:]     = np.diff(abs_corrected, axis=0)

    else:
        raise ValueError(f"action_space must be 'delta' or 'absolute'")

    for i, d in enumerate(cont_idx):
        out[:, d] = recon[:, i].astype(chunk.dtype)

    n_new   = np.linspace(0, T - 1, T_new)
    nearest = np.round(n_new).astype(int).clip(0, T - 1)
    for gi in gripper_idx:
        out[:, gi] = chunk[nearest, gi]

    return out


# ── Synthetic data ────────────────────────────────────────────────────────────

T = 60
t = np.linspace(0, 2 * np.pi, T)

# Delta action chunk (EEF delta, incremental)
chunk_delta = np.zeros((T, 7), dtype=np.float32)
chunk_delta[:, 0] = np.sin(t) * 0.002 + np.sin(5*t) * 0.0003
chunk_delta[:, 1] = 0.001 + np.sin(2*t) * 0.0005
chunk_delta[:, 2] = np.cos(t) * 0.0015 + np.cos(3*t) * 0.0002
chunk_delta[:, 3] = np.sin(t * 0.8) * 0.005
chunk_delta[:, 4] = np.cos(t * 1.2) * 0.003
chunk_delta[:, 5] = np.sin(t * 0.5) * 0.002
chunk_delta[35:, 6] = 1.0

# Absolute action chunk (EEF absolute pose)
chunk_abs = np.zeros((T, 7), dtype=np.float32)
chunk_abs[:, 0] = 0.3 + np.sin(t) * 0.15
chunk_abs[:, 1] = 0.2 + np.linspace(0, 0.2, T)
chunk_abs[:, 2] = 0.4 + np.cos(t) * 0.1
chunk_abs[:, 3] = np.sin(t * 0.8) * 0.5
chunk_abs[:, 4] = np.cos(t * 1.2) * 0.3
chunk_abs[:, 5] = np.sin(t * 0.5) * 0.2
chunk_abs[35:, 6] = 1.0

cont_idx    = [0, 1, 2, 3, 4, 5]
gripper_idx = [6]


# ── Plot ──────────────────────────────────────────────────────────────────────

speeds    = [1.5, 2.0, 3.0]
show_dims = [0, 1, 2, 6]
orig_idx  = np.arange(T)

configs = [
    (chunk_delta, 'delta',    1000.0,
     ['dx (mm/step)', 'dy (mm/step)', 'dz (mm/step)', 'gripper']),
    (chunk_abs,   'absolute', 1.0,
     ['x pos (m)',    'y pos (m)',    'z pos (m)',    'gripper']),
]

for chunk_plot, space, scale, dim_labels in configs:

    fig, axes = plt.subplots(
        len(show_dims), len(speeds),
        figsize=(5 * len(speeds), 3 * len(show_dims))
    )
    fig.patch.set_facecolor('#F5F5F5')
    fig.suptitle(
        f'DCT Compression  action_space="{space}"  keep_ratio=0.4\n'
        f'Gray  : original {T}f  (x = 0~{T-1})\n'
        f'Blue  : compressed raw  (x = 0~T_new-1, shorter)\n'
        f'Orange: interp back to {T}f  (x = 0~{T-1})',
        fontsize=10, fontweight='bold', y=1.01
    )

    for col, speed in enumerate(speeds):
        T_new    = max(2, int(T / speed))
        comp_idx = np.arange(T_new)          # own frame index 0..T_new-1

        out = dct_compress(chunk_plot, speed=speed, keep_ratio=0.4,
                           action_space=space,
                           cont_idx=cont_idx, gripper_idx=gripper_idx)

        # positions of compressed frames in original index space
        comp_pos = np.linspace(0, T - 1, T_new)

        # interpolate compressed back to T frames
        interp_back = {}
        for d in show_dims:
            if d in gripper_idx:
                nn = np.round(
                    np.linspace(0, T_new-1, T)
                ).astype(int).clip(0, T_new-1)
                interp_back[d] = out[nn, d]
            else:
                interp_back[d] = np.interp(orig_idx, comp_pos, out[:, d])

        for row, (d, dlabel) in enumerate(zip(show_dims, dim_labels)):
            ax = axes[row, col]
            ax.set_facecolor('white')
            ax.set_xlim(-1, T)

            # 1. Original — gray, full x range
            ax.step(orig_idx, chunk_plot[:, d] * scale,
                    color='#9E9E9E', lw=1.8, where='mid', alpha=0.9,
                    label=f'original ({T}f, x:0~{T-1})', zorder=1)

            # 2. Compressed raw — blue, own shorter x range 0..T_new-1
            ax.plot(comp_idx, out[:, d] * scale,
                    color='#1565C0', lw=1.3, alpha=0.7, zorder=3)
            ax.scatter(comp_idx, out[:, d] * scale,
                       color='#1565C0', s=30, zorder=4,
                       label=f'compressed ({T_new}f, x:0~{T_new-1})')

            # vertical line marking where compressed ends
            ax.axvline(T_new - 1, color='#1565C0', lw=0.8,
                       linestyle=':', alpha=0.5)

            # 3. Interp back — orange dashed, full x range
            ax.step(orig_idx, interp_back[d] * scale,
                    color='#E65100', lw=1.5, linestyle='--',
                    alpha=0.9, where='mid',
                    label=f'interp back ({T}f, x:0~{T-1})', zorder=2)

            if row == 0:
                ax.set_title(f'speed={speed}x  ({T}->{T_new}f)',
                             fontsize=10, fontweight='bold')
            if col == 0:
                ax.set_ylabel(dlabel, fontsize=9)
            if row == len(show_dims) - 1:
                ax.set_xlabel('Frame index', fontsize=8)

            ax.legend(fontsize=6, loc='upper right')
            ax.grid(True, alpha=0.3, lw=0.5)
            ax.tick_params(labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fname = f'dct_three_{space}.png'
    plt.savefig(fname, dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    display(Image(fname))
    print(f'Saved: {fname}')