#!/usr/bin/env python3
"""微基准: split-K vs 现有 matmul_res_gate, 在 AE 真实形状下, CUDA Graph 捕获条件。

目的: 在写任何集成代码之前, 先证明 split-K 在 AE 的形状 (M=10, K 大) 上
确实比现有 matmul 快。如果微基准不赢, 就不值得集成 (吸取 ultra fusion 教训)。

AE 里两个候选:
  o_proj:   M=10, K=2048, N=1024
  ffn_down: M=10, K=4096, N=1024

对比:
  1. 现有 matmul_res_gate (单 block 串行 K)
  2. split-K (SPLIT_K=2/4/8) + merge

全部在 CUDA Graph 下计时 (真实部署条件)。

用法:
    CUDA_VISIBLE_DEVICES=1 python scripts/microbench_split_k.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import triton

sys.path.insert(0, str(Path(__file__).parent.parent))

from fluxvla.ops.triton.matmul_triton_ops import (
    matmul_small_res_gate, matmul_split_k, merge_split_k_bias_res)


def time_graph(build_fn, runs=50):
    """在 CUDA Graph 下计时一个 callable。"""
    # warmup on side stream
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            build_fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        build_fn()
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


def run_current_matmul_res_gate(inp, weight, out, res, gate, M, K, N):
    """现有路径: matmul + residual + gate, 单 block 串行 K。"""
    BLOCK_N, BLOCK_M, BLOCK_K = 16, 32, 256
    grid = (((M + BLOCK_N - 1) // BLOCK_N) * ((N + BLOCK_M - 1) // BLOCK_M),)

    def fn():
        matmul_small_res_gate[grid](
            inp, weight, out, res, gate,
            seq_len=M, features=K, hidden=N,
            BLOCK_SIZE_N=BLOCK_N, BLOCK_SIZE_M=BLOCK_M, BLOCK_SIZE_K=BLOCK_K)
    return fn


def run_split_k(inp, weight, partial_buf, out_final, res, gate_dummy, M, K, N, split_k):
    """split-K 路径: 分 K 并行算部分和 -> merge 归约。

    注意: split_k kernel 不带 gate/res, 需要单独 merge。这里为了和现有路径
    公平对比 (现有是 matmul+res+gate 融合), split-K 后用 merge 加 res,
    gate 暂不计入 (gate 是 elementwise, 两条路径都要做, 不影响相对比较)。
    """
    BLOCK_N, BLOCK_M, BLOCK_K = 16, 64, 64
    grid_i = (M + BLOCK_N - 1) // BLOCK_N
    grid_j = (N + BLOCK_M - 1) // BLOCK_M
    grid = (grid_i * grid_j * split_k,)

    # partial_buf: [split_k, M, N] fp32
    zero_bias = torch.zeros(N, dtype=torch.float32, device=inp.device)
    merge_grid = (min(256, (M * N + 511) // 512),)

    def fn():
        matmul_split_k[grid](
            inp, weight, partial_buf,
            seq_len=M, features=K, hidden=N,
            BLOCK_SIZE_N=BLOCK_N, BLOCK_SIZE_M=BLOCK_M, BLOCK_SIZE_K=BLOCK_K,
            SPLIT_K=split_k)
        merge_split_k_bias_res[merge_grid](
            partial_buf, zero_bias, res, out_final,
            seq_len=M, hidden=N, SPLIT_K=split_k)
    return fn


def bench_shape(name, M, K, N, device, runs=50):
    print(f"\n{'='*64}")
    print(f"{name}: M={M}, K={K}, N={N}")
    print(f"{'='*64}")

    inp = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    weight = torch.randn(K, N, dtype=torch.bfloat16, device=device)
    res = torch.randn(M, N, dtype=torch.bfloat16, device=device)
    gate = torch.randn(M, N, dtype=torch.bfloat16, device=device)
    out = torch.zeros(M, N, dtype=torch.bfloat16, device=device)

    results = {}

    # 现有路径
    fn_cur = run_current_matmul_res_gate(inp, weight, out, res, gate, M, K, N)
    t = time_graph(fn_cur, runs)
    results['current (no split)'] = t.mean()
    print(f"  current matmul_res_gate:  {t.mean()*1000:7.2f} us  (±{t.std()*1000:.2f})")

    # split-K 各档
    for sk in [2, 4, 8]:
        partial = torch.zeros(sk, M, N, dtype=torch.float32, device=device)
        out_sk = torch.zeros(M, N, dtype=torch.bfloat16, device=device)
        try:
            fn_sk = run_split_k(inp, weight, partial, out_sk, res, gate, M, K, N, sk)
            t = time_graph(fn_sk, runs)
            results[f'split-K={sk}'] = t.mean()
            speedup = results['current (no split)'] / t.mean()
            print(f"  split-K={sk} (+merge):      {t.mean()*1000:7.2f} us  "
                  f"(±{t.std()*1000:.2f})  speedup {speedup:.2f}x")
        except Exception as e:
            print(f"  split-K={sk}: FAILED - {type(e).__name__}: {e}")

    # 最优
    best = min(results.items(), key=lambda kv: kv[1])
    cur = results['current (no split)']
    print(f"\n  最快: {best[0]} ({best[1]*1000:.2f} us)")
    if best[0] != 'current (no split)':
        print(f"  → split-K 赢 {cur/best[1]:.2f}x, 值得集成")
    else:
        print(f"  → 现有路径已最优, split-K 在此形状无收益")
    return results


def main():
    device = torch.device('cuda:0')
    torch.manual_seed(0)

    print("="*64)
    print("微基准: split-K vs matmul_res_gate (CUDA Graph 下, AE 真实形状)")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*64)

    bench_shape("o_proj   (attention output)", 10, 2048, 1024, device)
    bench_shape("ffn_down (FFN output)",       10, 4096, 1024, device)

    print(f"\n{'='*64}")
    print("结论: 若 split-K 在两个形状都赢 >1.1x, 则集成到 AE decoder;")
    print("否则放弃 (避免重蹈 ultra fusion 覆辙)。")
    print("="*64)
    return 0


if __name__ == '__main__':
    sys.exit(main())
