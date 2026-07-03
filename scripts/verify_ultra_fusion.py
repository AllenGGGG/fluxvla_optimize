#!/usr/bin/env python3
"""验证 ultra fusion 修复：正确性 + 速度对比。

对同一份权重、同一份输入，分别用 use_ultra_fusion=True / False 跑推理，
比较：
1. 两条路径是否都能正常运行（不崩溃）
2. 两条路径输出动作是否一致（数值正确性）
3. 两条路径的速度差异

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/verify_ultra_fusion.py
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from mmengine import Config
from fluxvla.engines.utils.builder import build_vla_from_cfg
from safetensors.torch import load_file


def build_model(cfg, ckpt, device, use_ultra_fusion):
    """构建模型并设置 ultra fusion 开关。"""
    model = build_vla_from_cfg(cfg.inference_model).to(device).eval()
    sd = load_file(ckpt)
    model.load_state_dict(sd, strict=False)
    model.use_ultra_fusion = use_ultra_fusion
    model.prepare_triton_inference(
        num_views=2,
        max_prompt_len=cfg.get('triton_max_prompt_len', 64),
        chunk_size=cfg.inference_model.get('n_action_steps', 50),
        num_steps=cfg.inference_model.get('num_steps', 10),
        tempo_speed=1.0,
    )
    return model


def make_fixed_obs(device, seed=0):
    """固定随机种子，生成可复现的输入。"""
    g = torch.Generator(device='cpu').manual_seed(seed)
    num_views, img_size, lang_len, state_dim = 2, 224, 20, 7
    images = torch.randn(1, num_views * 3, img_size, img_size,
                         generator=g, dtype=torch.float32).to(device).bfloat16()
    lang = torch.randint(0, 32000, (1, lang_len), generator=g).to(device)
    state = torch.randn(1, state_dim, generator=g, dtype=torch.float32).to(device).bfloat16()
    # 固定 diffusion 噪声，保证两条路径起点一致
    noise = torch.randn(1, cfg_n_action_steps, 32, generator=g, dtype=torch.float32).to(device).bfloat16()
    return {'images': images, 'lang_tokens': lang, 'robot_state': state, 'noise': noise}


def run_once(model, obs):
    with torch.no_grad():
        return model.predict_action(
            images=obs['images'],
            lang_tokens=obs['lang_tokens'],
            states=obs['robot_state'],
            noise=obs['noise'],
        )


def bench(model, obs, runs=20):
    for _ in range(3):
        run_once(model, obs)
    torch.cuda.synchronize()
    ts = []
    for _ in range(runs):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        run_once(model, obs)
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return np.array(ts)


cfg_n_action_steps = 10  # 会在 main 里覆盖


def main():
    global cfg_n_action_steps
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/pi05/pi05_libero10_task0_tempovla_speed_modulated_inference.py')
    parser.add_argument('--ckpt', default='work_dirs/libero10_full_tempovla_speed_modulated/checkpoints/latest-checkpoint.safetensors')
    parser.add_argument('--runs', type=int, default=20)
    args = parser.parse_args()

    device = torch.device('cuda:0')
    cfg = Config.fromfile(args.config)
    cfg_n_action_steps = cfg.inference_model.get('n_action_steps', 10)

    print("=" * 70)
    print("Ultra Fusion 修复验证：正确性 + 速度")
    print("=" * 70)

    obs = make_fixed_obs(device, seed=0)

    # --- 标准路径 ---
    print("\n[1/2] 标准路径 (use_ultra_fusion=False)...")
    model_std = build_model(cfg, args.ckpt, device, use_ultra_fusion=False)
    act_std = run_once(model_std, obs).float().cpu().numpy()
    t_std = bench(model_std, obs, args.runs)
    del model_std
    torch.cuda.empty_cache()

    # --- Ultra fusion 路径 ---
    print("[2/2] Ultra Fusion 路径 (use_ultra_fusion=True)...")
    try:
        model_ultra = build_model(cfg, args.ckpt, device, use_ultra_fusion=True)
        act_ultra = run_once(model_ultra, obs).float().cpu().numpy()
        t_ultra = bench(model_ultra, obs, args.runs)
        ultra_ok = True
    except Exception as ex:
        print(f"\n❌ Ultra fusion 路径崩溃: {type(ex).__name__}: {ex}")
        ultra_ok = False

    # --- 结果 ---
    print("\n" + "=" * 70)
    print("结果")
    print("=" * 70)
    print(f"\n标准路径速度:     {t_std.mean():6.2f} ms (±{t_std.std():.2f})  {1000/t_std.mean():.1f} Hz")

    if not ultra_ok:
        print("\nUltra fusion 仍有问题，需进一步排查。")
        return 1

    print(f"Ultra fusion 速度: {t_ultra.mean():6.2f} ms (±{t_ultra.std():.2f})  {1000/t_ultra.mean():.1f} Hz")
    print(f"加速比:            {t_std.mean()/t_ultra.mean():.3f}x")

    # 正确性
    diff = np.abs(act_std - act_ultra)
    print(f"\n正确性对比 (两条路径输出动作):")
    print(f"  最大绝对误差:   {diff.max():.6f}")
    print(f"  平均绝对误差:   {diff.mean():.6f}")
    print(f"  标准动作幅度:   {np.abs(act_std).mean():.6f}")

    # bf16 容差：相对误差 < 5% 视为一致
    rel = diff.max() / (np.abs(act_std).max() + 1e-6)
    print(f"  相对最大误差:   {rel*100:.2f}%")

    if rel < 0.05:
        print("\n✅ 通过：两条路径输出一致（bf16 容差内），ultra fusion 数值正确。")
        return 0
    else:
        print("\n⚠️  两条路径输出差异较大，ultra fusion 可能仍有语义问题。")
        return 1


if __name__ == '__main__':
    sys.exit(main())
