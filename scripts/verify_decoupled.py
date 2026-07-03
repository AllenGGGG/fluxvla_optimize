#!/usr/bin/env python3
"""验证 RTC-like prefix/suffix decoupled 推理: 接口 + 速度。

当前最终版本不是 exact-match decoder-only:
1. Full inference 生成 A0:A10, 执行/固定 A0:A5
2. Prefix-clamped decoder-only 复用上一帧 encoder K/V, 以 A0:A5 为 prefix
   条件生成 10 步, caller 执行 suffix A5:A10
3. 因为 prefix clamp 改变了 decoder 输入轨迹,suffix 不应再要求和 full
   inference 逐位一致;这里只验证 prefix 固定正确、接口可跑、速度可测

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/verify_decoupled.py
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


def build_model(cfg_path, ckpt, device):
    """构建 decoupled 推理模型。"""
    cfg = Config.fromfile(cfg_path)
    cfg.inference_model['type'] = 'PI05FlowMatchingDecoupledInference'
    if 'exec_chunk_size' not in cfg.inference_model:
        cfg.inference_model['exec_chunk_size'] = 5
    model = build_vla_from_cfg(cfg.inference_model).to(device).eval()
    sd = load_file(ckpt)
    model.load_state_dict(sd, strict=False)

    # 从模型属性取参数,不 hardcode
    model.prepare_triton_inference(
        num_views=model.num_views,
        max_prompt_len=model.triton_max_prompt_len,
        chunk_size=model.n_action_steps,
        num_steps=model.num_steps,
    )
    return model, cfg


def make_fixed_obs(device, seed=0):
    g = torch.Generator(device='cpu').manual_seed(seed)
    num_views, img_size, lang_len, state_dim = 2, 224, 20, 7
    images = torch.randn(1, num_views * 3, img_size, img_size,
                         generator=g, dtype=torch.float32).to(device).bfloat16()
    lang = torch.randint(0, 32000, (1, lang_len), generator=g).to(device)
    state = torch.randn(1, state_dim, generator=g,
                        dtype=torch.float32).to(device).bfloat16()
    return {'images': images, 'lang_tokens': lang, 'robot_state': state}


def bench_full(model, obs, runs=30):
    """Benchmark full inference (VLM + Encoder + Decoder)."""
    kw = dict(images=obs['images'], lang_tokens=obs['lang_tokens'],
              states=obs['robot_state'])
    for _ in range(5):
        model.predict_action(**kw)
    torch.cuda.synchronize()
    ts = []
    for _ in range(runs):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        model.predict_action(**kw)
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return np.array(ts)


def bench_decoder_prefix(model, prefix_actions, prefix_len, runs=30):
    """Benchmark prefix-clamped decoder-only inference."""
    for _ in range(5):
        model.predict_action_decoder_only_prefix(prefix_actions, prefix_len)
    torch.cuda.synchronize()
    ts = []
    for _ in range(runs):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        model.predict_action_decoder_only_prefix(prefix_actions, prefix_len)
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return np.array(ts)


def verify_prefix_mode(model, obs, device):
    """验证 prefix-clamped decoder-only 接口和 prefix 固定行为。"""
    print("\n[接口验证] full A0:A10 → clamp A0:A5 → decoder-only 生成 suffix A5:A10")

    kw = dict(images=obs['images'], lang_tokens=obs['lang_tokens'],
              states=obs['robot_state'])

    # 固定噪声
    g = torch.Generator(device='cpu').manual_seed(123)
    n_action_steps = model.n_action_steps
    prefix_len = model.exec_chunk_size
    noise = torch.randn(1, n_action_steps, 32, generator=g,
                        dtype=torch.float32).to(device).bfloat16()

    # Full inference with fixed noise populates encoder K/V and gives prefix.
    with torch.no_grad():
        act_full = model.predict_action(**kw, noise=noise).float().cpu().numpy()

    prefix_actions = torch.from_numpy(act_full).to(device)

    # Decoder-only with same suffix noise. Suffix is expected to differ from
    # full because prefix clamp changes the decoder trajectory.
    with torch.no_grad():
        act_dec = model.predict_action_decoder_only_prefix(
            prefix_actions, prefix_len, noise=noise).float().cpu().numpy()

    assert act_full.shape == act_dec.shape, (
        f"Shape mismatch: full {act_full.shape} vs dec {act_dec.shape}")

    prefix_diff = np.abs(
        act_full[:, :prefix_len] - act_dec[:, :prefix_len])
    suffix_delta = act_dec[:, prefix_len:] - act_full[:, prefix_len:]
    suffix_diff = np.abs(suffix_delta)
    suffix_full = act_full[:, prefix_len:].reshape(-1)
    suffix_dec = act_dec[:, prefix_len:].reshape(-1)
    suffix_delta_flat = suffix_delta.reshape(-1)
    suffix_mae = suffix_diff.mean()
    suffix_rmse = np.sqrt(np.mean(suffix_delta_flat ** 2))
    suffix_max = suffix_diff.max()
    suffix_full_norm = np.linalg.norm(suffix_full)
    suffix_delta_norm = np.linalg.norm(suffix_delta_flat)
    suffix_rel_l2 = suffix_delta_norm / (suffix_full_norm + 1e-12)
    suffix_cos = np.dot(suffix_full, suffix_dec) / (
        np.linalg.norm(suffix_full) * np.linalg.norm(suffix_dec) + 1e-12)
    per_step_l2 = np.linalg.norm(suffix_delta[0], axis=-1)
    finite = np.isfinite(act_dec).all()

    print(f"  action shape: {act_dec.shape}")
    print(f"  prefix 最大误差: {prefix_diff.max():.10f}")
    print("  suffix B5:B10 vs full A5:A10 差异:")
    print(f"    MAE:        {suffix_mae:.10f}")
    print(f"    RMSE:       {suffix_rmse:.10f}")
    print(f"    Max abs:    {suffix_max:.10f}")
    print(f"    Relative L2:{suffix_rel_l2:.10f}")
    print(f"    Cosine:     {suffix_cos:.10f}")
    print("    Per-step L2:",
          " ".join(f"{x:.6f}" for x in per_step_l2))
    print(f"  输出有限值: {finite}")

    prefix_ok = prefix_diff.max() == 0.0
    if prefix_ok and finite:
        print("  ✅ prefix clamp 正常,decoder-only-prefix 可运行")
    else:
        print("  ❌ prefix clamp 或输出存在异常")

    return prefix_ok and finite, prefix_actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        default='configs/pi05/pi05_libero10_task0_tempovla_decoupled_inference.py')
    parser.add_argument('--ckpt',
                        default='work_dirs/mlp_tempovla/checkpoints/latest-checkpoint.safetensors')
    parser.add_argument('--runs', type=int, default=30)
    args = parser.parse_args()

    device = torch.device('cuda:0')

    print("=" * 70)
    print("RTC-like prefix/suffix 解耦推理验证")
    print("=" * 70)

    model, cfg = build_model(args.config, args.ckpt, device)
    n_action_steps = model.n_action_steps
    obs = make_fixed_obs(device, seed=42)

    print(f"\n模型参数:")
    print(f"  n_action_steps: {n_action_steps}")
    print(f"  num_steps (denoising): {model.num_steps}")
    print(f"  exec_chunk_size: {model.exec_chunk_size}")
    print(f"  encoder_seq_len: {model._encoder_seq_len}")
    print(f"  triton_max_prompt_len: {model.triton_max_prompt_len}")

    # 1. 接口和 prefix clamp 验证
    prefix_ok, prefix_actions = verify_prefix_mode(model, obs, device)

    # 2. 速度对比
    print("\n" + "=" * 70)
    print("[速度测试]")
    print("=" * 70)
    t_full = bench_full(model, obs, args.runs)
    t_dec = bench_decoder_prefix(model, prefix_actions, model.exec_chunk_size,
                                 args.runs)

    print(f"\n  Full inference (VLM+Enc+Dec): {t_full.mean():.2f} ms "
          f"(±{t_full.std():.2f})  {1000/t_full.mean():.1f} Hz")
    print(f"  Decoder-only-prefix:          {t_dec.mean():.2f} ms "
          f"(±{t_dec.std():.2f})  {1000/t_dec.mean():.1f} Hz")
    print(f"  Decoder-only-prefix 单次省:   {t_full.mean()-t_dec.mean():.2f} ms")

    # 3. 正确的速度结论: 按"每 10 步 action 的总推理开销"对比
    print(f"\n  --- 按「每 {n_action_steps} 步 action 的推理开销」对比 ---")
    print(f"  Baseline (1x full, 执行 {n_action_steps} 步):")
    print(f"    {t_full.mean():.2f} ms / {n_action_steps} 步")

    # Pattern A: 5+5 (两次推理,每次执行 5 步 = 共 10 步)
    exec_5 = model.exec_chunk_size
    cost_5_5 = t_full.mean() + t_dec.mean()
    print(f"\n  Pattern A: {exec_5}+{exec_5} 交替 (full + decoder-only-prefix):")
    print(f"    {cost_5_5:.2f} ms / {n_action_steps} 步  "
          f"(每 {exec_5} 步决策一次, 决策延迟 ≤{t_full.mean():.0f}ms)")
    print(f"    vs baseline: 总开销 +{(cost_5_5/t_full.mean()-1)*100:.0f}%, "
          f"但决策频率 2x")

    print(f"\n  Pattern B 不适用于当前 prefix/suffix 版本:")
    print("    当前版本第二次只应该执行 suffix,不是执行完整 10 步。")

    # 4. 显存
    mem = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\n  峰值显存: {mem:.2f} GB")

    # Summary
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print(f"\n  prefix clamp: {'✅ ok' if prefix_ok else '❌ failed'}")
    print(f"  Decoder-only-prefix 延迟: {t_dec.mean():.2f} ms "
          f"(省 {t_full.mean()-t_dec.mean():.2f} ms)")
    print(f"\n  适用场景:")
    print(f"    Pattern A (5+5): RTC-like prefix/suffix 实验")
    print(f"      代价: 总推理开销翻倍, 但第二次只要 {t_dec.mean():.0f}ms")

    return 0


if __name__ == '__main__':
    sys.exit(main())
