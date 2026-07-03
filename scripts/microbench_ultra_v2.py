#!/usr/bin/env python3
"""Microbench: autotuned v2 decoder kernels vs the standard pinned-block ones.

De-risks the ultra-v2 idea BEFORE any integration. For the exact shapes the
flow-matching decoder hits (M = chunk_size ~= 10, K up to 4096), we compare:

  ffn_down : matmul_res_gate  (K=4096 -> 1024)   std vs v2-autotuned
  attn_o   : matmul_res_gate  (K=2048 -> 1024)   std vs v2-autotuned
  ffn_gate : matmul_gate      (K=1024 -> 4096)   std vs v2-autotuned

Each kernel is timed inside its own CUDA graph (replay), matching how it runs
in production, and checked for numerical equality vs the standard kernel.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/microbench_ultra_v2.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from fluxvla.ops.triton.matmul_triton_ops import (matmul_small_gate,
                                                  matmul_small_res_gate)
from fluxvla.ops.triton.ultra_fusion_v2_ops import (matmul_gate_autotuned,
                                                    matmul_res_gate_autotuned)


def time_graph(fn, runs=50):
    """Build a CUDA graph around fn and time replay."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()

    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(runs):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return np.array(ts)


def bench_res_gate(seq_len, K, N, label):
    dev = 'cuda'
    inp = torch.randn(seq_len, K, dtype=torch.bfloat16, device=dev)
    weight = torch.randn(K, N, dtype=torch.bfloat16, device=dev) * 0.02
    res = torch.randn(seq_len, N, dtype=torch.bfloat16, device=dev)
    gate = torch.randn(seq_len, N, dtype=torch.bfloat16, device=dev)
    out_std = torch.zeros(seq_len, N, dtype=torch.bfloat16, device=dev)
    out_v2 = torch.zeros(seq_len, N, dtype=torch.bfloat16, device=dev)

    # Standard pinned-block config used in transformer_decoder:
    # ffn_down (K=4096): BN=16, BM=32, BK=256 ; attn_o (K=2048): BN=32,BM=32,BK=128
    if K >= 4096:
        BN, BM, BK = 16, 32, 256
    else:
        BN, BM, BK = 32, 32, 128

    def std():
        grid = ((seq_len + BN - 1) // BN) * ((N + BM - 1) // BM)
        matmul_small_res_gate[(grid, )](
            inp, weight, out_std, res, gate,
            seq_len=seq_len, features=K, hidden=N,
            BLOCK_SIZE_N=BN, BLOCK_SIZE_M=BM, BLOCK_SIZE_K=BK)

    def v2():
        grid = lambda meta: (
            ((seq_len + meta['BLOCK_SIZE_N'] - 1) // meta['BLOCK_SIZE_N']) *
            ((N + meta['BLOCK_SIZE_M'] - 1) // meta['BLOCK_SIZE_M']), )
        matmul_res_gate_autotuned[grid](
            inp, weight, out_v2, res, gate,
            seq_len=seq_len, features=K, hidden=N)

    # correctness (run eager once each, outside graph)
    std(); v2()
    torch.cuda.synchronize()
    diff = (out_std.float() - out_v2.float()).abs()
    rel = diff.max().item() / (out_std.float().abs().max().item() + 1e-6)

    t_std = time_graph(std)
    t_v2 = time_graph(v2)
    print(f"\n[{label}] M={seq_len} K={K} N={N}")
    print(f"  std  (BN={BN},BM={BM},BK={BK}): {t_std.mean()*1000:7.2f} us (±{t_std.std()*1000:.2f})")
    print(f"  v2   (autotuned):              {t_v2.mean()*1000:7.2f} us (±{t_v2.std()*1000:.2f})")
    print(f"  speedup: {t_std.mean()/t_v2.mean():.3f}x   rel_err: {rel*100:.3f}%")
    return t_std.mean(), t_v2.mean(), rel


def bench_gate(seq_len, K, N, label):
    dev = 'cuda'
    inp = torch.randn(seq_len, K, dtype=torch.bfloat16, device=dev)
    w1 = torch.randn(K, N, dtype=torch.bfloat16, device=dev) * 0.02
    w2 = torch.randn(K, N, dtype=torch.bfloat16, device=dev) * 0.02
    out_std = torch.zeros(seq_len, N, dtype=torch.bfloat16, device=dev)
    out_v2 = torch.zeros(seq_len, N, dtype=torch.bfloat16, device=dev)

    def std():
        matmul_small_gate[((seq_len + 127) // 128, (N + 63) // 64)](
            inp, w1, w2, out_std, seq_len, K, N)

    def v2():
        grid = lambda meta: (
            ((seq_len + meta['BLOCK_SIZE_N'] - 1) // meta['BLOCK_SIZE_N']) *
            ((N + meta['BLOCK_SIZE_M'] - 1) // meta['BLOCK_SIZE_M']), )
        matmul_gate_autotuned[grid](
            inp, w1, w2, out_v2, seq_len=seq_len, features=K, hidden=N)

    std(); v2()
    torch.cuda.synchronize()
    diff = (out_std.float() - out_v2.float()).abs()
    rel = diff.max().item() / (out_std.float().abs().max().item() + 1e-6)

    t_std = time_graph(std)
    t_v2 = time_graph(v2)
    print(f"\n[{label}] M={seq_len} K={K} N={N}")
    print(f"  std  (BN=128,BM=64,BK=32): {t_std.mean()*1000:7.2f} us (±{t_std.std()*1000:.2f})")
    print(f"  v2   (autotuned):          {t_v2.mean()*1000:7.2f} us (±{t_v2.std()*1000:.2f})")
    print(f"  speedup: {t_std.mean()/t_v2.mean():.3f}x   rel_err: {rel*100:.3f}%")
    return t_std.mean(), t_v2.mean(), rel


def main():
    torch.manual_seed(0)
    M = 10  # chunk_size / decoder seq len
    print("=" * 70)
    print(f"Ultra-v2 microbench (M={M}, CUDA-graph timed, A100)")
    print("=" * 70)

    results = []
    results.append(('ffn_down', *bench_res_gate(M, 4096, 1024, 'ffn_down res_gate')))
    results.append(('attn_o',   *bench_res_gate(M, 2048, 1024, 'attn_o res_gate')))
    results.append(('ffn_gate', *bench_gate(M, 1024, 4096, 'ffn_gate')))

    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    per_layer_std = sum(r[1] for r in results)
    per_layer_v2 = sum(r[2] for r in results)
    max_rel = max(r[3] for r in results)
    print(f"  单层这三个 kernel 之和:  std {per_layer_std*1000:.2f} us → v2 {per_layer_v2*1000:.2f} us")
    print(f"  最大相对误差: {max_rel*100:.3f}% (应 < 5%)")
    # 18 层 x 10 步的粗略外推（仅参考，真实收益以端到端为准）
    scale = 18 * 10
    print(f"  粗略外推 (×{scale} = 18层×10步): "
          f"std {per_layer_std*scale:.2f} ms → v2 {per_layer_v2*scale:.2f} ms "
          f"(省 {(per_layer_std-per_layer_v2)*scale:.2f} ms)")
    print("\n  注意: 这是孤立微基准的外推。真实收益必然小于此 (整图里 kernel 抢 SM、"
          "且这三个不是全部)。以端到端 verify_ultra_fusion 为准。")


if __name__ == '__main__':
    main()
