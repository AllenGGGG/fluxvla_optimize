"""Ultra-fusion v2: autotuned variants of the decoder hot-path matmuls.

The standard decoder kernels (`matmul_small_res_gate`, `matmul_small_gate` in
`matmul_triton_ops.py`) use hand-pinned block sizes that were tuned for other
shapes. In the flow-matching action decoder the GEMMs are extremely tall-skinny
(M = chunk_size ~= 10 action tokens, K up to 4096), so those fixed tilings are
almost certainly suboptimal.

These v2 kernels implement the same math as the standard ones, but autotune is
free to pick a different reduction tile. That should be numerically equivalent
within bf16 tolerance, but it still needs the end-to-end correctness check in
`scripts/verify_ultra_v2.py` before use.

IMPORTANT: this is a *candidate* optimization. The whole point of the
microbench (scripts/microbench_ultra_v2.py) is to confirm whether the autotuned
tiling actually beats the standard kernel end-to-end inside the CUDA graph. If
it does not, this path should not be used — see REALTIME_OPTIMIZATION_REPORT.md
for why microbench wins get diluted in the full graph.
"""

import triton
import triton.language as tl


def _res_gate_configs():
    """Autotune space for tall-skinny matmul + residual + gate.

    seq_len (M) is tiny (~10) so BLOCK_SIZE_N stays small; we sweep the
    output-tile (M) and K-block plus warps/stages, which is where the real
    occupancy/latency tradeoff lives for these shapes.
    """
    configs = []
    for bn in (16, 32):
        for bm in (32, 64, 128):
            for bk in (64, 128, 256):
                for warps in (4, 8):
                    for stages in (2, 3, 4):
                        configs.append(
                            triton.Config(
                                {
                                    'BLOCK_SIZE_N': bn,
                                    'BLOCK_SIZE_M': bm,
                                    'BLOCK_SIZE_K': bk,
                                },
                                num_warps=warps,
                                num_stages=stages))
    return configs


@triton.autotune(configs=_res_gate_configs(), key=['seq_len', 'features', 'hidden'])
@triton.jit
def matmul_res_gate_autotuned(inp_ptr, weight_ptr, out_ptr, res_ptr, gate_ptr,
                              seq_len: tl.constexpr, features: tl.constexpr,
                              hidden: tl.constexpr,
                              BLOCK_SIZE_N: tl.constexpr,
                              BLOCK_SIZE_M: tl.constexpr,
                              BLOCK_SIZE_K: tl.constexpr):
    """out = res + (inp @ weight) * gate. Same math as matmul_small_res_gate."""
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M
        acc = tl.load(
            res_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden +
            (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) &
            ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other=0.0).to(tl.float32)
        matmul_acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        for k in range(0, features, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr +
                (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * features +
                (k + tl.arange(0, BLOCK_SIZE_K))[None, :],
                mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) &
                ((k + tl.arange(0, BLOCK_SIZE_K))[None, :] < features),
                other=0.0)
            w = tl.load(
                weight_ptr +
                (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * hidden +
                (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
                mask=((k + tl.arange(0, BLOCK_SIZE_K))[:, None] < features) &
                ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
                other=0.0)
            matmul_acc = tl.dot(x, w, matmul_acc)
        gate = tl.load(
            gate_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden +
            (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) &
            ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other=0.0).to(tl.float32)
        acc += matmul_acc * gate
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden +
            (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            acc.to(tl.bfloat16),
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) &
            ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden))


def _gate_configs():
    """Autotune space for the gate+up fused GEMM (silu(gate) * up)."""
    configs = []
    for bn in (16, 32):
        for bm in (32, 64, 128):
            for bk in (32, 64, 128):
                for warps in (4, 8):
                    for stages in (2, 3, 4):
                        configs.append(
                            triton.Config(
                                {
                                    'BLOCK_SIZE_N': bn,
                                    'BLOCK_SIZE_M': bm,
                                    'BLOCK_SIZE_K': bk,
                                },
                                num_warps=warps,
                                num_stages=stages))
    return configs


@triton.autotune(configs=_gate_configs(), key=['seq_len', 'features', 'hidden'])
@triton.jit
def matmul_gate_autotuned(inp_ptr, weight1_ptr, weight2_ptr, out_ptr,
                          seq_len: tl.constexpr, features: tl.constexpr,
                          hidden: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                          BLOCK_SIZE_M: tl.constexpr,
                          BLOCK_SIZE_K: tl.constexpr):
    """out = gelu(inp @ w1) * (inp @ w2). Same math as matmul_small_gate.

    Uses a 1-D persistent grid (vs the 2-D grid in matmul_small_gate) so the
    autotuner can pick block sizes freely without a fixed launch grid.
    """
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M
        acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        for k in range(0, features, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * features +
                (k + tl.arange(0, BLOCK_SIZE_K))[None, :],
                mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) &
                ((k + tl.arange(0, BLOCK_SIZE_K))[None, :] < features),
                other=0.0)
            w = tl.load(
                weight1_ptr + (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * hidden +
                (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
                mask=((k + tl.arange(0, BLOCK_SIZE_K))[:, None] < features) &
                ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
                other=0.0)
            acc = tl.dot(x, w, acc)
            w2 = tl.load(
                weight2_ptr + (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * hidden +
                (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
                mask=((k + tl.arange(0, BLOCK_SIZE_K))[:, None] < features) &
                ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
                other=0.0)
            acc2 = tl.dot(x, w2, acc2)
        acc = acc * tl.sigmoid(1.5957691216057308 * acc *
                               (1 + 0.044715 * acc * acc))
        acc = (acc * acc2).to(tl.bfloat16)
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden +
            (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            acc,
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) &
            ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden))
