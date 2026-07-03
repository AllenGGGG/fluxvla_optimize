#!/usr/bin/env python3
"""端到端验证: Ultra-V2 (autotuned ffn_gate) vs 标准路径。

对同一份权重、同一份输入(固定种子),比较:
1. 两条路径输出是否一致(数值正确性)
2. 两条路径的速度差异(CUDA Graph replay)

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/verify_ultra_v2.py
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


def cfg_model_value(cfg, name, default):
    return cfg.inference_model.get(name, getattr(cfg, name, default))


def build_model(cfg_path, ckpt, device, model_type_override=None):
    """构建模型。可选覆盖 inference_model.type。"""
    cfg = Config.fromfile(cfg_path)
    if model_type_override:
        cfg.inference_model['type'] = model_type_override
    model = build_vla_from_cfg(cfg.inference_model).to(device).eval()
    sd = load_file(ckpt)
    model.load_state_dict(sd, strict=False)
    model.prepare_triton_inference(
        num_views=cfg_model_value(cfg, 'num_views', model.num_views),
        max_prompt_len=cfg_model_value(
            cfg, 'triton_max_prompt_len', model.triton_max_prompt_len),
        chunk_size=cfg_model_value(cfg, 'n_action_steps', model.n_action_steps),
        num_steps=cfg_model_value(cfg, 'num_steps', model.num_steps),
        tempo_speed=1.0,
    )
    return model, cfg


def make_fixed_obs(device, n_action_steps, seed=0):
    """固定随机种子,生成可复现的输入。"""
    g = torch.Generator(device='cpu').manual_seed(seed)
    num_views, img_size, lang_len, state_dim = 2, 224, 20, 7
    images = torch.randn(1, num_views * 3, img_size, img_size,
                         generator=g, dtype=torch.float32).to(device).bfloat16()
    lang = torch.randint(0, 32000, (1, lang_len), generator=g).to(device)
    state = torch.randn(1, state_dim, generator=g,
                        dtype=torch.float32).to(device).bfloat16()
    noise = torch.randn(1, n_action_steps, 32, generator=g,
                        dtype=torch.float32).to(device).bfloat16()
    return {'images': images, 'lang_tokens': lang,
            'robot_state': state, 'noise': noise}


def run_once(model, obs):
    with torch.no_grad():
        return model.predict_action(
            images=obs['images'],
            lang_tokens=obs['lang_tokens'],
            states=obs['robot_state'],
            noise=obs['noise'],
        )


def bench(model, obs, runs=30):
    for _ in range(5):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        default='configs/pi05/pi05_libero10_task0_tempovla_speed_modulated_inference.py')
    parser.add_argument('--ckpt',
                        default='work_dirs/libero10_full_tempovla_speed_modulated/checkpoints/latest-checkpoint.safetensors')
    parser.add_argument('--runs', type=int, default=30)
    args = parser.parse_args()

    device = torch.device('cuda:0')

    print("=" * 70)
    print("Ultra-V2 端到端验证: autotuned ffn_gate vs 标准路径")
    print("=" * 70)

    # 读 config 拿 n_action_steps
    cfg = Config.fromfile(args.config)
    n_action_steps = cfg.inference_model.get('n_action_steps', 10)
    obs = make_fixed_obs(device, n_action_steps, seed=42)

    # --- 标准路径 ---
    print("\n[1/2] 标准路径 (PI05FlowMatchingSpeedModulatedInference)...")
    model_std, _ = build_model(args.config, args.ckpt, device,
                               model_type_override='PI05FlowMatchingSpeedModulatedInference')
    act_std = run_once(model_std, obs).float().cpu().numpy()
    t_std = bench(model_std, obs, args.runs)
    del model_std
    torch.cuda.empty_cache()

    # --- Ultra-V2 路径 ---
    print("[2/2] Ultra-V2 路径 (PI05FlowMatchingUltraV2Inference)...")
    try:
        model_v2, _ = build_model(args.config, args.ckpt, device,
                                  model_type_override='PI05FlowMatchingUltraV2Inference')
        act_v2 = run_once(model_v2, obs).float().cpu().numpy()
        t_v2 = bench(model_v2, obs, args.runs)
        v2_ok = True
    except Exception as ex:
        import traceback
        traceback.print_exc()
        print(f"\n❌ Ultra-V2 路径崩溃: {type(ex).__name__}: {ex}")
        v2_ok = False

    # --- 结果 ---
    print("\n" + "=" * 70)
    print("结果")
    print("=" * 70)
    print(f"\n标准路径:   {t_std.mean():.2f} ms (±{t_std.std():.2f})  "
          f"{1000/t_std.mean():.1f} Hz")

    if not v2_ok:
        print("\nUltra-V2 失败,需排查。")
        return 1

    print(f"Ultra-V2:   {t_v2.mean():.2f} ms (±{t_v2.std():.2f})  "
          f"{1000/t_v2.mean():.1f} Hz")
    speedup = t_std.mean() / t_v2.mean()
    saved = t_std.mean() - t_v2.mean()
    print(f"加速比:     {speedup:.3f}x  (省 {saved:.2f} ms)")

    # 正确性
    diff = np.abs(act_std - act_v2)
    print(f"\n正确性对比:")
    print(f"  最大绝对误差: {diff.max():.6f}")
    print(f"  平均绝对误差: {diff.mean():.6f}")
    print(f"  标准动作幅度: {np.abs(act_std).mean():.6f}")
    rel = diff.max() / (np.abs(act_std).max() + 1e-6)
    print(f"  相对最大误差: {rel*100:.4f}%")

    if rel < 0.05:
        print("\n✅ 数值正确 (bf16 容差内)")
    else:
        print("\n⚠️ 数值偏差较大,需检查")

    if speedup > 1.0:
        print(f"\n✅ Ultra-V2 确实更快: 省 {saved:.2f} ms ({(speedup-1)*100:.1f}%)")
    else:
        print(f"\n❌ Ultra-V2 没有加速 (反而慢 {-saved:.2f} ms)。"
              f"微基准收益被整图稀释,不建议使用。")

    return 0


if __name__ == '__main__':
    sys.exit(main())
