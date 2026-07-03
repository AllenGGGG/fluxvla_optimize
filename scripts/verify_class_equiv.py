#!/usr/bin/env python3
"""验证两条推理类在 "10步 speed=1.0" 下是否数值等价。

对比:
  A) PI05FlowMatchingSpeedModulatedInference (num_steps=10)           -> 80% 那条
  B) PI05FlowMatchingSpeedModulatedTimeScheduleInference
       time_schedule=[1,0.9,...,0.1], time_deltas=[-0.1]*10           -> 69% 那条

理论上 A 和 B 应该 bit-level 接近 (同样的 sinusoidal embedding + 同样的 -0.1 Euler 步长)。
如果输出差异很大 -> time-schedule 类有 bug, 是它而不是 split-k 拉低了成功率。
如果输出几乎一致 -> 80% vs 69% 是统计噪声 / eval harness 差异, 跟推理类无关。

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/verify_class_equiv.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from mmengine import Config
from fluxvla.engines.utils.builder import build_vla_from_cfg
from safetensors.torch import load_file

CONFIG = 'configs/pi05/pi05_libero10_task0_tempovla_speed_modulated_inference.py'
CKPT = ('work_dirs/libero10_full_tempovla_speed_modulated/checkpoints/'
        'latest-checkpoint.safetensors')
N_STEPS = 10
N_ACTION = 10


def build(cfg, ckpt, device, cls_type, extra):
    cfg.inference_model.type = cls_type
    cfg.inference_model.use_ultra_fusion = False  # 隔离推理类, 不掺 split-k
    for k, v in extra.items():
        cfg.inference_model[k] = v
    model = build_vla_from_cfg(cfg.inference_model).to(device).eval()
    sd = load_file(ckpt)
    model.load_state_dict(sd, strict=False)
    model.prepare_triton_inference(
        num_views=2,
        max_prompt_len=cfg.get('triton_max_prompt_len', 64),
        chunk_size=N_ACTION,
        num_steps=N_STEPS,
        tempo_speed=1.0,
    )
    return model


def fixed_obs(device, seed=0):
    g = torch.Generator(device='cpu').manual_seed(seed)
    images = torch.randn(1, 6, 224, 224, generator=g).to(device).bfloat16()
    lang = torch.randint(0, 32000, (1, 20), generator=g).to(device)
    state = torch.randn(1, 7, generator=g).to(device).bfloat16()
    noise = torch.randn(1, N_ACTION, 32, generator=g).to(device).bfloat16()
    return dict(images=images, lang_tokens=lang, robot_state=state, noise=noise)


def run(model, obs):
    with torch.no_grad():
        return model.predict_action(
            images=obs['images'], lang_tokens=obs['lang_tokens'],
            states=obs['robot_state'], noise=obs['noise']).float().cpu().numpy()


def main():
    device = torch.device('cuda:0')
    cfg = Config.fromfile(CONFIG)
    obs = fixed_obs(device, seed=0)

    print("=" * 70)
    print("A: PI05FlowMatchingSpeedModulatedInference (10步, 80%那条)")
    mA = build(cfg, CKPT, device, 'PI05FlowMatchingSpeedModulatedInference', {})
    actA = run(mA, obs)
    del mA; torch.cuda.empty_cache()

    print("B: ...TimeScheduleInference (schedule=[1..0.1], 69%那条)")
    sched = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    deltas = [-0.1] * 10
    cfg2 = Config.fromfile(CONFIG)
    mB = build(cfg2, CKPT, device,
               'PI05FlowMatchingSpeedModulatedTimeScheduleInference',
               dict(time_schedule=sched, time_deltas=deltas))
    actB = run(mB, obs)

    print("\n" + "=" * 70)
    print("结果")
    print("=" * 70)
    diff = np.abs(actA - actB)
    print(f"  A 动作幅度均值:   {np.abs(actA).mean():.6f}")
    print(f"  最大绝对误差:     {diff.max():.6f}")
    print(f"  平均绝对误差:     {diff.mean():.6f}")
    rel = diff.max() / (np.abs(actA).max() + 1e-6)
    print(f"  相对最大误差:     {rel*100:.3f}%")
    if rel < 0.02:
        print("\n✅ 两类数值等价 -> 80% vs 69% 是噪声/harness 差异, 不是推理类的锅")
    else:
        print("\n❌ 两类输出明显不同 -> time-schedule 类本身改变了动作, "
              "成功率下降来自这里")
    return 0


if __name__ == '__main__':
    sys.exit(main())
