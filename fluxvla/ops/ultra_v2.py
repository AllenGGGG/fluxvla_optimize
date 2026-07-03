"""Ultra-v2 atomic ops — wrappers for the autotuned v2 kernels.

Only exports the kernel(s) that actually beat the standard path in microbench:
- matmul_gate_v2: autotuned gated MLP (ffn_gate), 1.46x faster at M=10

Kernels that did NOT beat standard (attn_o res_gate, ffn_down res_gate) are
intentionally NOT wrapped here — the standard path stays for those.
"""

from fluxvla.ops.triton.ultra_fusion_v2_ops import matmul_gate_autotuned


def matmul_gate_v2(x, gate_w, up_w, out, in_features, intermediate_dim):
    """Drop-in replacement for matmul_gate (atomic_ops.py) using autotuned kernel.

    Same signature, same semantics: out = gelu(x @ gate_w) * (x @ up_w)
    """
    seq_len = x.shape[0]
    grid = lambda meta: (
        ((seq_len + meta['BLOCK_SIZE_N'] - 1) // meta['BLOCK_SIZE_N']) *
        ((intermediate_dim + meta['BLOCK_SIZE_M'] - 1) // meta['BLOCK_SIZE_M']),
    )
    matmul_gate_autotuned[grid](
        x, gate_w, up_w, out,
        seq_len=seq_len, features=in_features, hidden=intermediate_dim)
