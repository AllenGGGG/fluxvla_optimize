"""Enhanced fusion kernels from RealTimeVLA for aggressive optimization.

These kernels provide more aggressive fusion than the standard ops, targeting
sub-50ms inference latency for PI0.5 models.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def adarms_norm_gate_fused_kernel(
    x_ptr,
    style_ptr,
    normed_x_ptr,
    gate_ptr,
    seq_len: tl.constexpr,
    features: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """Fused AdaRMSNorm + gate computation.

    This is more aggressive than standard adarms_norm_kernel because it
    computes both the normalized output AND the gate values in a single pass,
    avoiding a separate style projection kernel call.

    Args:
        x_ptr: Input tensor [seq_len, features]
        style_ptr: Style embedding [1, style_dim] where style_dim >= 2*features
        normed_x_ptr: Output normalized tensor [seq_len, features]
        gate_ptr: Output gate values [seq_len, features]
        seq_len: Sequence length
        features: Hidden dimension
        BLOCK_SIZE: Tile size for reduction
    """
    pid = tl.program_id(0)
    psize = tl.num_programs(0)

    for i in range(pid, seq_len, psize):
        row_x_offset = i * features

        # Compute RMS
        sum_sq = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for j in range(0, features, BLOCK_SIZE):
            cols = j + tl.arange(0, BLOCK_SIZE)
            mask = cols < features
            x_val = tl.load(x_ptr + row_x_offset + cols, mask=mask, other=0.0).to(tl.float32)
            sum_sq += x_val * x_val

        rms_factor = tl.rsqrt(tl.sum(sum_sq) / features + 1e-6)

        # Load style (scale at [0:features], gate at [features:2*features])
        for j in range(0, features, BLOCK_SIZE):
            cols = j + tl.arange(0, BLOCK_SIZE)
            mask = cols < features

            # Load input
            x_val = tl.load(x_ptr + row_x_offset + cols, mask=mask, other=0.0).to(tl.float32)

            # Load scale and gate from style
            scale = tl.load(style_ptr + cols, mask=mask, other=1.0).to(tl.float32)
            gate = tl.load(style_ptr + features + cols, mask=mask, other=1.0).to(tl.float32)

            # Normalize with style scale
            normed = x_val * rms_factor * scale

            # Store normalized output and gate
            tl.store(normed_x_ptr + row_x_offset + cols, normed.to(tl.bfloat16), mask=mask)
            tl.store(gate_ptr + row_x_offset + cols, gate.to(tl.bfloat16), mask=mask)


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

    This eliminates separate add and multiply operations after matmul.
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


def adarms_norm_gate_fused(x, style, normed_x, gate, hidden_dim):
    """Wrapper for fused AdaRMSNorm + gate kernel.

    Args:
        x: Input [seq_len, hidden_dim]
        style: Style embedding [1, style_dim] where style_dim >= 2*hidden_dim
        normed_x: Output buffer for normalized x
        gate: Output buffer for gate values
        hidden_dim: Hidden dimension
    """
    seq_len = x.shape[0]
    BLOCK_SIZE = 128

    adarms_norm_gate_fused_kernel[seq_len,](
        x, style, normed_x, gate,
        seq_len=seq_len,
        features=hidden_dim,
        BLOCK_SIZE=BLOCK_SIZE
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


@triton.jit
def time_mlp_speed_fused_kernel(
    time_embed_ptr,
    time_mlp_in_w_ptr,
    time_mlp_in_b_ptr,
    time_mlp_out_w_ptr,
    time_mlp_out_b_ptr,
    speed_emb_ptr,
    out_ptr,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """Fused time MLP + speed addition in one kernel.

    Computes: out = time_mlp_out(silu(time_mlp_in(time_embed))) + speed_emb

    This eliminates multiple kernel launches for time conditioning.
    """
    # Single-row processing (batch size 1 for time embedding)
    tid = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = tid < hidden_dim

    # Load time embedding
    time_val = tl.load(time_embed_ptr + tid, mask=mask, other=0.0).to(tl.float32)

    # First MLP layer with SiLU
    mlp_hidden = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for k in range(0, hidden_dim, BLOCK_SIZE):
        k_idx = k + tl.arange(0, BLOCK_SIZE)
        k_mask = k_idx < hidden_dim
        w = tl.load(time_mlp_in_w_ptr + k_idx * hidden_dim + tid[None, :],
                   mask=k_mask[:, None] & mask[None, :], other=0.0)
        mlp_hidden += tl.sum(time_val[None, :] * w, axis=0)

    bias = tl.load(time_mlp_in_b_ptr + tid, mask=mask, other=0.0).to(tl.float32)
    mlp_hidden = mlp_hidden + bias
    mlp_hidden = mlp_hidden * tl.sigmoid(mlp_hidden)  # SiLU

    # Second MLP layer
    mlp_out = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for k in range(0, hidden_dim, BLOCK_SIZE):
        k_idx = k + tl.arange(0, BLOCK_SIZE)
        k_mask = k_idx < hidden_dim
        w = tl.load(time_mlp_out_w_ptr + k_idx * hidden_dim + tid[None, :],
                   mask=k_mask[:, None] & mask[None, :], other=0.0)
        mlp_out += tl.sum(mlp_hidden[None, :] * w, axis=0)

    bias = tl.load(time_mlp_out_b_ptr + tid, mask=mask, other=0.0).to(tl.float32)
    mlp_out = mlp_out + bias

    # Add speed embedding
    speed = tl.load(speed_emb_ptr + tid, mask=mask, other=0.0).to(tl.float32)
    mlp_out = mlp_out + speed

    tl.store(out_ptr + tid, mlp_out.to(tl.bfloat16), mask=mask)


def time_mlp_speed_fused(time_embed, time_mlp_in_w, time_mlp_in_b,
                        time_mlp_out_w, time_mlp_out_b, speed_emb, out, hidden_dim):
    """Wrapper for fused time MLP + speed kernel.

    This replaces the sequence:
      1. matmul_bias_silu(time_embed, time_mlp_in_w, time_mlp_in_b, buf)
      2. matmul_bias_silu(buf, time_mlp_out_w, time_mlp_out_b, time_emb)
      3. time_emb.add_(speed_emb)

    With a single kernel call.
    """
    BLOCK_SIZE = 128
    grid = (hidden_dim + BLOCK_SIZE - 1) // BLOCK_SIZE

    time_mlp_speed_fused_kernel[grid,](
        time_embed, time_mlp_in_w, time_mlp_in_b,
        time_mlp_out_w, time_mlp_out_b, speed_emb, out,
        hidden_dim=hidden_dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
