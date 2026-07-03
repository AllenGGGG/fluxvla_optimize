#!/usr/bin/env python3
"""验证 matmul_split_k_res_gate 数值正确性 (对照 matmul_res_gate)。

计算: out = res + (inp @ weight) * gate
分别用 split-K 路径和现有 matmul_res_gate 路径, 比较输出。

用法: CUDA_VISIBLE_DEVICES=1 python scripts/verify_split_k_correctness.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from fluxvla.ops.atomic_ops import matmul_split_k_res_gate
from fluxvla.ops.triton.matmul_triton_ops import matmul_small_res_gate


def ref_matmul_res_gate(inp, weight, res, gate):
    """PyTorch 参考实现: res + (inp @ weight) * gate (fp32 精算)。"""
    mm = (inp.float() @ weight.float())
    return res.float() + mm * gate.float()


def check(M, K, N, split_k, device):
    torch.manual_seed(0)
    inp = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    weight = torch.randn(K, N, dtype=torch.bfloat16, device=device) * 0.02
    res = torch.randn(M, N, dtype=torch.bfloat16, device=device)
    gate = torch.randn(M, N, dtype=torch.bfloat16, device=device)

    ref = ref_matmul_res_gate(inp, weight, res, gate)

    # split-K 路径
    partial = torch.zeros(split_k, M, N, dtype=torch.float32, device=device)
    out_sk = torch.zeros(M, N, dtype=torch.bfloat16, device=device)
    matmul_split_k_res_gate(inp, weight, out_sk, res, gate, partial,
                            in_features=K, out_features=N, split_k=split_k)

    # 现有 matmul_res_gate 路径 (out 原地: res 传 out, 需先拷)
    out_cur = res.clone()
    BLOCK_N, BLOCK_M, BLOCK_K = 16, 32, 256
    grid = ((((M + BLOCK_N - 1) // BLOCK_N) * ((N + BLOCK_M - 1) // BLOCK_M)),)
    matmul_small_res_gate[grid](
        inp, weight, out_cur, out_cur, gate,
        seq_len=M, features=K, hidden=N,
        BLOCK_SIZE_N=BLOCK_N, BLOCK_SIZE_M=BLOCK_M, BLOCK_SIZE_K=BLOCK_K)

    sk_err = (out_sk.float() - ref).abs().max().item()
    cur_err = (out_cur.float() - ref).abs().max().item()
    sk_vs_cur = (out_sk.float() - out_cur.float()).abs().max().item()
    ref_scale = ref.abs().max().item()

    print(f"\nM={M} K={K} N={N} split_k={split_k}:")
    print(f"  参考幅度:           {ref_scale:.4f}")
    print(f"  split-K vs 参考:    {sk_err:.6f}  ({sk_err/ref_scale*100:.3f}%)")
    print(f"  current vs 参考:    {cur_err:.6f}  ({cur_err/ref_scale*100:.3f}%)")
    print(f"  split-K vs current: {sk_vs_cur:.6f}  ({sk_vs_cur/ref_scale*100:.3f}%)")
    # bf16 容差: split-K 误差应和 current 同量级
    ok = sk_err < max(cur_err * 2, ref_scale * 0.02)
    print(f"  {'✅ 通过' if ok else '❌ 误差过大'}")
    return ok


def main():
    device = torch.device('cuda:0')
    print("=" * 60)
    print("验证 split-K res_gate 数值正确性")
    print("=" * 60)
    all_ok = True
    # ffn_down 形状 (要集成的)
    for sk in [2, 4, 8]:
        all_ok &= check(10, 4096, 1024, sk, device)
    # o_proj 形状 (对照)
    all_ok &= check(10, 2048, 1024, 8, device)
    print("\n" + "=" * 60)
    print("✅ 全部通过, split-K 数值正确" if all_ok else "❌ 有失败")
    print("=" * 60)
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
