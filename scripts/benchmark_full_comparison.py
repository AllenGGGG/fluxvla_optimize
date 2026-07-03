#!/usr/bin/env python3
"""FluxVLA 推理综合对比测试 (单卡, GPU1)。

输出一张完整对比表，包含:
  - 配置 (ultra_fusion 开/关, num_steps)
  - 是否成功启用 (不崩溃)
  - 显存占用 (模型常驻 + 推理峰值 + 占总显存比)
  - 推理频率 (完整推理 ms / Hz)
  - 分段耗时 (VLM / Encoder / AE, 每段独立 CUDA Graph)
  - AE 带宽利用率 (实测 vs A100 理论下限 → 判断 split-K 是否有空间)

用法:
    CUDA_VISIBLE_DEVICES=1 python scripts/benchmark_full_comparison.py \
        --ckpt work_dirs/libero10_full_tempovla_speed_modulated/checkpoints/latest-checkpoint.safetensors \
        --runs 20
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from mmengine import Config
from fluxvla.engines.utils.builder import build_vla_from_cfg
from safetensors.torch import load_file

A100_BANDWIDTH_GBPS = 2039.0  # A100-SXM4-80GB HBM2e 理论带宽 GB/s


def make_fixed_obs(device, n_action_steps, seed=0):
    g = torch.Generator(device='cpu').manual_seed(seed)
    nv, sz, ll, sd = 2, 224, 20, 7
    return {
        'images': torch.randn(1, nv * 3, sz, sz, generator=g, dtype=torch.float32).to(device).bfloat16(),
        'lang_tokens': torch.randint(0, 32000, (1, ll), generator=g).to(device),
        'robot_state': torch.randn(1, sd, generator=g, dtype=torch.float32).to(device).bfloat16(),
        'noise': torch.randn(1, n_action_steps, 32, generator=g, dtype=torch.float32).to(device).bfloat16(),
    }


def run_once(model, obs):
    with torch.no_grad():
        return model.predict_action(
            images=obs['images'], lang_tokens=obs['lang_tokens'],
            states=obs['robot_state'], noise=obs['noise'])


def bench_full(model, obs, runs):
    for _ in range(3):
        run_once(model, obs)
    torch.cuda.synchronize()
    ts = []
    for _ in range(runs):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); run_once(model, obs); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return np.array(ts)


def bench_components(model, runs):
    """每段独立建 CUDA Graph 计时。返回 (vlm, enc, ae) ms。"""
    from fluxvla.models.vlas.pi05_flowmatching_inference import (
        vision_encoder, transformer_encoder, transformer_decoder)
    m, w, b = model, model._triton_weights, model._triton_bufs
    fns = {
        'vlm': lambda: vision_encoder(w, b, m.num_views, m._num_vit_layers),
        'enc': lambda: transformer_encoder(w, b, m._encoder_seq_len, m._num_encoder_layers,
                                           m._visual_tokens_per_view, m._visual_grid_size,
                                           m._visual_token_downscale_factor),
        'ae':  lambda: transformer_decoder(w, b, m._encoder_seq_len, m._num_decoder_layers,
                                           m._num_steps, use_ultra_fusion=getattr(m, 'use_ultra_fusion', False)),
    }
    out = {}
    for name, fn in fns.items():
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        ts = []
        for _ in range(runs):
            a = torch.cuda.Event(enable_timing=True); z = torch.cuda.Event(enable_timing=True)
            a.record(); g.replay(); z.record(); torch.cuda.synchronize()
            ts.append(a.elapsed_time(z))
        out[name] = float(np.mean(ts))
    return out


def estimate_ae_weight_bytes(model):
    """估算 AE 单次去噪一步要读的权重字节数 (bf16=2 bytes)。"""
    n_layers = model._num_decoder_layers
    # 每层主要权重: qkv(1024x2560) + o(2048x1024) + ffn_gate(1024x4096)
    # + ffn_up(1024x4096) + ffn_down(4096x1024)
    per_layer = (1024*2560 + 2048*1024 + 1024*4096 + 1024*4096 + 4096*1024)
    total_params = per_layer * n_layers
    return total_params * 2  # bf16


def build_model(cfg, ckpt, device, use_ultra_fusion, num_steps):
    model = build_vla_from_cfg(cfg.inference_model).to(device).eval()
    sd = load_file(ckpt)
    model.load_state_dict(sd, strict=False)
    model.use_ultra_fusion = use_ultra_fusion
    model.num_steps = num_steps
    model.prepare_triton_inference(
        num_views=2, max_prompt_len=cfg.get('triton_max_prompt_len', 64),
        chunk_size=cfg.inference_model.get('n_action_steps', 50),
        num_steps=num_steps, tempo_speed=1.0)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/pi05/pi05_libero10_task0_tempovla_speed_modulated_inference.py')
    ap.add_argument('--ckpt', default='work_dirs/libero10_full_tempovla_speed_modulated/checkpoints/latest-checkpoint.safetensors')
    ap.add_argument('--runs', type=int, default=20)
    args = ap.parse_args()

    device = torch.device('cuda:0')
    cfg = Config.fromfile(args.config)
    n_action = cfg.inference_model.get('n_action_steps', 10)
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3

    # 测试矩阵: (ultra_fusion, num_steps)
    scenarios = [
        ('baseline (ultra=off, steps=10)', False, 10),
        ('ultra=on,  steps=10',            True,  10),
        ('ultra=off, steps=5',             False, 5),
    ]

    rows = []
    ref_act = None  # 用 baseline 输出做正确性基准

    for name, uf, ns in scenarios:
        print(f"\n{'='*72}\n测试: {name}\n{'='*72}")
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            model = build_model(cfg, args.ckpt, device, uf, ns)
            mem_resident = torch.cuda.memory_allocated() / 1024**3
            obs = make_fixed_obs(device, n_action, seed=0)
            act = run_once(model, obs).float().cpu().numpy()
            full = bench_full(model, obs, args.runs)
            comp = bench_components(model, args.runs)
            mem_peak = torch.cuda.max_memory_allocated() / 1024**3

            # 正确性 (相对 baseline)
            if ref_act is None and not uf and ns == 10:
                ref_act = act
                rel_err = 0.0
            elif ref_act is not None and act.shape == ref_act.shape:
                rel_err = np.abs(act - ref_act).max() / (np.abs(ref_act).max() + 1e-6)
            else:
                rel_err = None  # steps 不同, 形状/数值本就不同, 不比

            # AE 带宽利用率
            ae_bytes = estimate_ae_weight_bytes(model) * ns
            ae_ideal_ms = ae_bytes / (A100_BANDWIDTH_GBPS * 1e9) * 1000
            bw_util = ae_ideal_ms / comp['ae'] * 100

            rows.append({
                'name': name, 'ok': True,
                'mem_resident': mem_resident, 'mem_peak': mem_peak,
                'mem_pct': mem_peak / total_mem * 100,
                'full_ms': full.mean(), 'full_std': full.std(),
                'hz': 1000 / full.mean(),
                'vlm': comp['vlm'], 'enc': comp['enc'], 'ae': comp['ae'],
                'ae_ideal': ae_ideal_ms, 'bw_util': bw_util,
                'rel_err': rel_err,
            })
            print(f"  ✅ 运行成功 | 完整 {full.mean():.2f}ms ({1000/full.mean():.1f}Hz) | "
                  f"AE {comp['ae']:.2f}ms | 峰值显存 {mem_peak:.2f}GB")
            del model; torch.cuda.empty_cache()
        except Exception as ex:
            import traceback; traceback.print_exc()
            rows.append({'name': name, 'ok': False, 'err': str(ex)})

    # ===== 汇总表 =====
    print("\n\n" + "="*72)
    print(f"最终对比结果 (A100-80GB, 总显存 {total_mem:.0f}GB, 每项 {args.runs} 次)")
    print("="*72)

    hdr = f"{'配置':<32} {'启用':<5} {'峰值显存':<12} {'推理频率':<14} {'AE耗时':<10} {'正确性':<8}"
    print("\n" + hdr)
    print("-" * 92)
    for r in rows:
        if not r['ok']:
            print(f"{r['name']:<30} {'❌崩溃':<5}")
            continue
        mem_s = f"{r['mem_peak']:.1f}GB ({r['mem_pct']:.0f}%)"
        hz_s = f"{r['full_ms']:.1f}ms/{r['hz']:.0f}Hz"
        err_s = ('基准' if r['rel_err'] == 0.0 else
                 f"{r['rel_err']*100:.1f}%" if r['rel_err'] is not None else 'N/A')
        print(f"{r['name']:<30} {'✅':<5} {mem_s:<12} {hz_s:<14} {r['ae']:.1f}ms{'':<4} {err_s:<8}")

    # 分段 + 带宽分析
    print("\n" + "="*72)
    print("分段耗时 & AE 带宽利用率 (判断 split-K 是否有空间)")
    print("="*72)
    print(f"\n{'配置':<32} {'VLM':<8} {'Enc':<8} {'AE实测':<9} {'AE理论下限':<11} {'带宽利用':<8}")
    print("-" * 80)
    for r in rows:
        if not r['ok']:
            continue
        print(f"{r['name']:<30} {r['vlm']:.1f}ms{'':<2} {r['enc']:.1f}ms{'':<2} "
              f"{r['ae']:.1f}ms{'':<3} {r['ae_ideal']:.1f}ms{'':<6} {r['bw_util']:.0f}%")

    # 结论
    base = next((r for r in rows if r['ok'] and 'baseline' in r['name']), None)
    if base:
        print("\n" + "="*72)
        print("结论")
        print("="*72)
        print(f"\n1. AE 带宽利用率约 {base['bw_util']:.0f}% "
              f"(实测 {base['ae']:.1f}ms vs 理论下限 {base['ae_ideal']:.1f}ms)")
        if base['bw_util'] < 50:
            print(f"   → 带宽利用 < 50%, M=10 的 GEMV 喂不满带宽, "
                  f"split-K 理论上有 ~{base['ae']/base['ae_ideal']:.1f}x 上限空间, 值得尝试")
        else:
            print(f"   → 带宽利用已较高, split-K 收益有限, 不建议投入")
        steps5 = next((r for r in rows if r['ok'] and 'steps=5' in r['name']), None)
        if steps5:
            print(f"\n2. num_steps 10→5: {base['hz']:.0f}Hz → {steps5['hz']:.0f}Hz "
                  f"(加速 {base['full_ms']/steps5['full_ms']:.2f}x), 零代码风险")
            print(f"   注意: 步数减半会改变输出, 需另测 LIBERO 成功率确认质量")
        ultra = next((r for r in rows if r['ok'] and 'ultra=on' in r['name']), None)
        if ultra:
            print(f"\n3. ultra fusion: {base['full_ms']:.1f}ms → {ultra['full_ms']:.1f}ms "
                  f"({'无收益/更慢' if ultra['full_ms'] >= base['full_ms'] else '有收益'}), "
                  f"输出误差 {ultra['rel_err']*100:.2f}% (数值正确)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
