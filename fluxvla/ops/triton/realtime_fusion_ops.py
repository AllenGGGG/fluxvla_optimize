"""Enhanced fusion kernel from RealTimeVLA for aggressive optimization.

Only the verified `matmul_res_gate_fused` kernel lives here. Two earlier
hand-written kernels were removed after profiling/verification:

- ``adarms_norm_gate_fused``: had wrong normalization semantics (dropped the
  ``(1 + scale)`` term and the shift, read the gate at the wrong style offset).
  The ``adarms_norm_gate_optimized`` wrapper in ``atomic_ops`` now delegates to
  the proven ``adarms_norm_kernel`` instead.
- ``time_mlp_speed_fused``: had a broken matmul tiling (shape-broadcast crash,
  truncated reduction dimension). The time MLP is a tiny per-step GEMM whose
  cost is negligible, so ``time_mlp_with_speed_optimized`` now uses the verified
  ``matmul_small_bias_silu`` kernels.

``matmul_res_gate_fused`` fuses matmul + residual + gate into a single kernel
and matches the reference ``matmul_small_res_gate`` semantics. It is on the hot
path (2x per decoder layer x num_steps), so the fusion is worth keeping.
"""

import triton
import triton.language as tl


@triton.jit
def matmul_res_gate_fused_kernel(
    inp_ptr,
    weight_ptr,
    out_ptr,
    res_ptr,
    gate_ptr,
    seq_len: tl.constexpr,
    features: tl.constexpr,
    hidden: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    """Fused matmul + residual + gate in a single kernel.

    Computes: out = res + (inp @ weight) * gate

    Matches the reference matmul_small_res_gate semantics. This eliminates
    separate add and multiply operations after matmul.
    """
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)

    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M

        # Load residual
        acc = tl.load(
            res_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other=0.0
        ).to(tl.float32)

        # Matmul accumulation
        matmul_acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        for k in range(0, features, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * features + (k + tl.arange(0, BLOCK_SIZE_K))[None, :],
                mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((k + tl.arange(0, BLOCK_SIZE_K))[None, :] < features),
                other=0.0
            )
            w = tl.load(
                weight_ptr + (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
                mask=((k + tl.arange(0, BLOCK_SIZE_K))[:, None] < features) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
                other=0.0
            )
            matmul_acc = tl.dot(x, w, matmul_acc)

        # Load gate
        gate = tl.load(
            gate_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other=0.0
        ).to(tl.float32)

        # Fused: res + matmul * gate
        acc += matmul_acc * gate

        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            acc.to(tl.bfloat16),
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden)
        )


def matmul_res_gate_fused(inp, weight, out, res, gate, in_features, out_features,
                          BLOCK_SIZE_N=32, BLOCK_SIZE_M=32, BLOCK_SIZE_K=128):
    """Wrapper for fused matmul + residual + gate kernel.

    Args:
        inp: Input [seq_len, in_features]
        weight: Weight matrix [in_features, out_features]
        out: Output buffer [seq_len, out_features]
        res: Residual [seq_len, out_features]
        gate: Gate values [seq_len, out_features]
        in_features: Input feature dimension
        out_features: Output feature dimension
        BLOCK_SIZE_N/M/K: Tile sizes
    """
    seq_len = inp.shape[0]
    grid = ((seq_len + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N) * \
           ((out_features + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M)

    matmul_res_gate_fused_kernel[grid,](
        inp, weight, out, res, gate,
        seq_len=seq_len,
        features=in_features,
        hidden=out_features,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
