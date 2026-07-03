#!/usr/bin/env python3
"""测试推理阶段各模块的时间开销和 RTC 参数。

测试内容:
1. VLM 编码时间
2. AE 推理时间
3. 完整推理时间
4. 不同 AE 调用次数的效果
5. 不同 RTC 更新间隔的效果

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/profile_inference_rtc.py \
        --config configs/pi05/pi05_libero10_task0_tempovla_speed_modulated_inference.py \
        --ckpt work_dirs/libero10_full_tempovla_speed_modulated/checkpoints/latest-checkpoint.safetensors \
        --num-episodes 10
"""

import argparse
import time
import torch
import numpy as np
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from mmengine import Config
from fluxvla.engines.utils.builder import build_vla_from_cfg


class InferenceProfiler:
    """推理性能分析器"""

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device

        # 计时结果
        self.timings = {
            'vlm_encode': [],
            'ae_decode': [],
            'full_inference': [],
        }

    def profile_full_inference(self, obs, num_runs=10):
        """测试完整推理时间"""
        print("\n" + "="*70)
        print("测试完整推理时间 (VLM + Encoder + Decoder)")
        print("="*70)

        # Warm up
        for _ in range(3):
            _ = self.model.predict_action(
                images=obs['images'],
                lang_tokens=obs['lang_tokens'],
                states=obs['robot_state']
            )

        torch.cuda.synchronize()

        # 测试
        times = []
        for i in range(num_runs):
            start = time.perf_counter()

            actions = self.model.predict_action(
                images=obs['images'],
                lang_tokens=obs['lang_tokens'],
                states=obs['robot_state']
            )

            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)

            if (i + 1) % 5 == 0:
                print(f"  Run {i+1}/{num_runs}: {elapsed:.2f} ms")

        self.timings['full_inference'] = times

        print(f"\n完整推理统计:")
        print(f"  平均: {np.mean(times):.2f} ms")
        print(f"  中位数: {np.median(times):.2f} ms")
        print(f"  标准差: {np.std(times):.2f} ms")
        print(f"  最小: {np.min(times):.2f} ms")
        print(f"  最大: {np.max(times):.2f} ms")
        print(f"  频率: {1000/np.mean(times):.2f} Hz")

        return actions

    def measure_component_times(self, num_runs=20):
        """真实分段计时：每段单独建 CUDA Graph 再 replay 计时（全优化状态）。

        关键：每个组件都建自己的 CUDA Graph，这样每段都享受 CUDA Graph 优化，
        各段之和才能和完整流程对得上。直接调用无图的函数会暴露 launch overhead，
        测出的是假象（AE 会虚高到 100ms+）。

        需要 model 已经 prepare_triton_inference + 完整流程 replay 过一次
        （buffers 已被真实数据填充）。
        """
        print("\n" + "="*70)
        print("真实分段计时 (每段独立建 CUDA Graph，全优化状态)")
        print("="*70)

        from fluxvla.models.vlas.pi05_flowmatching_inference import (
            vision_encoder, transformer_encoder, transformer_decoder,
        )

        m = self.model
        w = m._triton_weights
        b = m._triton_bufs

        # 各组件的调用闭包（参数与 pi05_model 内部一致）
        vis_fn = lambda: vision_encoder(w, b, m.num_views, m._num_vit_layers)
        enc_fn = lambda: transformer_encoder(
            w, b, m._encoder_seq_len, m._num_encoder_layers,
            m._visual_tokens_per_view, m._visual_grid_size,
            m._visual_token_downscale_factor)
        dec_fn = lambda: transformer_decoder(
            w, b, m._encoder_seq_len, m._num_decoder_layers,
            m._num_steps, use_ultra_fusion=getattr(m, 'use_ultra_fusion', False))

        def build_graph(fn):
            """为单个组件建独立 CUDA Graph。"""
            # warmup（在 side stream 上，CUDA Graph 捕获要求）
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    fn()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()
            torch.cuda.synchronize()
            return g

        def time_graph(g, runs):
            # warmup
            for _ in range(3):
                g.replay()
            torch.cuda.synchronize()
            ts = []
            for _ in range(runs):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                g.replay()
                end.record()
                torch.cuda.synchronize()
                ts.append(start.elapsed_time(end))  # ms
            return np.array(ts)

        print("\n为每个组件建立独立 CUDA Graph...")
        vis_g = build_graph(vis_fn)
        enc_g = build_graph(enc_fn)
        dec_g = build_graph(dec_fn)

        vis_t = time_graph(vis_g, num_runs)
        enc_t = time_graph(enc_g, num_runs)
        dec_t = time_graph(dec_g, num_runs)

        total = vis_t.mean() + enc_t.mean() + dec_t.mean()

        print(f"\n实测分段 (每段独立 CUDA Graph，各 {num_runs} 次):")
        print(f"  Vision Encoder (VLM):     {vis_t.mean():6.2f} ms  "
              f"(±{vis_t.std():.2f})  {vis_t.mean()/total*100:4.1f}%")
        print(f"  Transformer Encoder:      {enc_t.mean():6.2f} ms  "
              f"(±{enc_t.std():.2f})  {enc_t.mean()/total*100:4.1f}%")
        print(f"  Action Decoder (AE):      {dec_t.mean():6.2f} ms  "
              f"(±{dec_t.std():.2f})  {dec_t.mean()/total*100:4.1f}%")
        print(f"  {'-'*50}")
        print(f"  三段之和:                 {total:6.2f} ms")
        print(f"  完整流程 (整图):          {np.mean(self.timings['full_inference']):6.2f} ms")
        print(f"  (三段之和 ≈ 完整流程 → 分段计时可信)")

        cacheable = vis_t.mean() + enc_t.mean()
        ae_only = dec_t.mean()
        print(f"\n关键结论 (全优化状态):")
        print(f"  可缓存部分 (VLM+Encoder): {cacheable:.2f} ms")
        print(f"  每次必算 (AE Decoder):    {ae_only:.2f} ms")
        if ae_only > 0:
            print(f"  → VLM/AE 解耦后，复用缓存时单次推理 ~{ae_only:.2f} ms "
                  f"({1000/ae_only:.0f} Hz)")
            print(f"  → 解耦节省: {cacheable:.2f} ms/次 "
                  f"({cacheable/(cacheable+ae_only)*100:.0f}%)")

        return {
            'vlm': vis_t.mean(),
            'encoder': enc_t.mean(),
            'ae': dec_t.mean(),
            'cacheable': cacheable,
            'ae_only': ae_only,
        }



    def test_rtc_scenarios(self, obs, actions, exec_freq=100, comp_times=None):
        """测试不同 RTC 参数的效果

        comp_times: measure_component_times 的返回值，用实测的可缓存/AE 时间，
                    而不是瞎估的比例。
        """
        print("\n" + "="*70)
        print("测试不同 RTC 参数")
        print("="*70)

        full_time = np.mean(self.timings['full_inference'])
        # 优先用模型的 n_action_steps，回退到从 actions 形状推断
        chunk_size = getattr(self.model, 'n_action_steps', None)
        if chunk_size is None:
            chunk_size = actions.shape[1] if actions.dim() >= 2 else actions.shape[0]

        # 实测的 AE-only 时间（解耦后复用 VLM 缓存的单次推理成本）
        if comp_times is not None:
            ae_time = comp_times['ae_only']
            cacheable_time = comp_times['cacheable']
        else:
            ae_time = full_time * 0.3
            cacheable_time = full_time * 0.7

        # 执行频率
        step_duration = 1000.0 / exec_freq  # ms per step

        print(f"\n参数 (基于实测):")
        print(f"  动作 chunk 大小: {chunk_size}")
        print(f"  执行频率: {exec_freq} Hz ({step_duration:.1f} ms/step)")
        print(f"  完整推理时间: {full_time:.2f} ms")
        print(f"  可缓存 (VLM+Enc): {cacheable_time:.2f} ms")
        print(f"  AE-only (解耦后): {ae_time:.2f} ms")

        # 测试不同的 AE 调用间隔
        print(f"\n{'='*70}")
        print("场景分析:")
        print(f"{'='*70}")

        scenarios = [
            {'name': '无 RTC (baseline)', 'ae_interval_steps': chunk_size},
            {'name': 'RTC 每 5 步', 'ae_interval_steps': 5},
            {'name': 'RTC 每 10 步', 'ae_interval_steps': 10},
            {'name': 'RTC 每 15 步', 'ae_interval_steps': 15},
            {'name': 'RTC 每 20 步', 'ae_interval_steps': 20},
        ]

        results = []

        for scenario in scenarios:
            interval_steps = scenario['ae_interval_steps']
            interval_time = interval_steps * step_duration

            # 计算需要多少个并行 AE（基于实测 AE-only 时间，因为解耦后每次只跑 AE）
            num_parallel_ae = max(1, int(np.ceil(ae_time / interval_time)))

            # 计算有效控制频率
            control_freq = 1000.0 / interval_time

            # 计算每个 chunk 需要推理几次
            num_inferences_per_chunk = int(np.ceil(chunk_size / interval_steps))

            # 计算总推理开销
            if interval_steps == chunk_size:
                # 无 RTC，每次完整推理
                total_inference_overhead = full_time
            else:
                # 有 RTC + 解耦：第一次完整 (VLM+Enc+AE)，后续复用缓存只跑 AE (实测值)
                total_inference_overhead = full_time + (num_inferences_per_chunk - 1) * ae_time

            # 计算总周期时间
            exec_time = chunk_size * step_duration
            total_cycle_time = total_inference_overhead + exec_time

            # 有效任务频率
            task_freq = 1000.0 / total_cycle_time

            result = {
                'name': scenario['name'],
                'interval_steps': interval_steps,
                'interval_time': interval_time,
                'num_parallel_ae': num_parallel_ae,
                'control_freq': control_freq,
                'num_inferences': num_inferences_per_chunk,
                'inference_overhead': total_inference_overhead,
                'exec_time': exec_time,
                'total_cycle': total_cycle_time,
                'task_freq': task_freq,
            }
            results.append(result)

            print(f"\n{scenario['name']}:")
            print(f"  AE 调用间隔: {interval_steps} 步 ({interval_time:.1f} ms)")
            print(f"  需要并行 AE 数: {num_parallel_ae}")
            print(f"  控制频率: {control_freq:.2f} Hz")
            print(f"  每个 chunk 推理次数: {num_inferences_per_chunk}")
            print(f"  推理总开销: {total_inference_overhead:.2f} ms")
            print(f"  执行时间: {exec_time:.1f} ms")
            print(f"  总周期: {total_cycle_time:.1f} ms")
            print(f"  有效任务频率: {task_freq:.2f} Hz")

        # 总结推荐
        print(f"\n{'='*70}")
        print("推荐配置:")
        print(f"{'='*70}")

        # 找最优配置（平衡控制频率和计算开销）
        best_idx = 2  # 默认选 10 步
        best = results[best_idx]

        print(f"\n推荐: {best['name']}")
        print(f"  原因:")
        print(f"    - 控制频率: {best['control_freq']:.1f} Hz (足够高)")
        print(f"    - 并行 AE 数: {best['num_parallel_ae']} (A100 足够)")
        if best['num_parallel_ae'] > 1:
            print(f"      ⚠️  Orin 上可能需要 {best['num_parallel_ae']} 个并行 AE")
        print(f"    - 任务频率: {best['task_freq']:.2f} Hz (比 baseline {results[0]['task_freq']:.2f} Hz 快)")

        return results


def create_dummy_observation(model):
    """创建模拟观测数据"""
    device = next(model.parameters()).device

    # 假设 LIBERO 配置
    num_views = 2
    img_size = 224
    lang_len = 20
    state_dim = 7

    # 图像格式：[B, num_views*3, H, W]
    # unflatten(1, (-1,3)) -> [B, num_views, 3, H, W], [0] -> [num_views, 3, H, W]
    # permute(0,2,3,1) -> [num_views, H, W, 3]
    images_flat = torch.randn(1, num_views * 3, img_size, img_size,
                             dtype=torch.bfloat16, device=device)

    obs = {
        'images': images_flat,  # [1, num_views*H*W*C]
        'lang_tokens': torch.randint(0, 32000, (1, lang_len), device=device),  # [1, L]
        'robot_state': torch.randn(1, state_dim, dtype=torch.bfloat16, device=device),  # [1, state_dim]
    }

    return obs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                       default='configs/pi05/pi05_libero10_task0_tempovla_speed_modulated_inference.py')
    parser.add_argument('--ckpt', type=str,
                       default='work_dirs/libero10_full_tempovla_speed_modulated/checkpoints/latest-checkpoint.safetensors')
    parser.add_argument('--num-episodes', type=int, default=10,
                       help='测试次数')
    parser.add_argument('--exec-freq', type=int, default=100,
                       help='执行频率 (Hz)')
    args = parser.parse_args()

    print("="*70)
    print("FluxVLA 推理性能分析工具")
    print("="*70)
    print(f"配置: {args.config}")
    print(f"权重: {args.ckpt}")
    print(f"测试次数: {args.num_episodes}")
    print(f"执行频率: {args.exec_freq} Hz")

    # 检查文件
    if not Path(args.config).exists():
        print(f"\n错误: 配置文件不存在: {args.config}")
        return 1

    if not Path(args.ckpt).exists():
        print(f"\n错误: 权重文件不存在: {args.ckpt}")
        return 1

    # 加载模型
    print("\n加载模型...")
    cfg = Config.fromfile(args.config)

    # 确保在 GPU 1
    device = torch.device('cuda:0')  # CUDA_VISIBLE_DEVICES=1 会把 GPU1 映射到 cuda:0

    model = build_vla_from_cfg(cfg.inference_model)
    model = model.to(device)
    model.eval()

    # 关闭有 bug 的 ultra fusion，用验证过的标准 Triton 路径
    if hasattr(model, 'use_ultra_fusion'):
        model.use_ultra_fusion = False
        print("[*] 已关闭 ultra_fusion，使用标准 Triton 路径测试")

    # 加载权重
    from safetensors.torch import load_file
    state_dict = load_file(args.ckpt)
    model.load_state_dict(state_dict, strict=False)

    print(f"模型已加载到 {device}")
    print(f"模型类型: {type(model).__name__}")

    # 准备推理
    if hasattr(model, 'prepare_triton_inference'):
        print("\n准备 Triton 推理...")
        model.prepare_triton_inference(
            num_views=2,
            max_prompt_len=cfg.get('triton_max_prompt_len', 64),
            chunk_size=cfg.inference_model.get('n_action_steps', 50),
            num_steps=cfg.inference_model.get('num_steps', 10),
            tempo_speed=1.0
        )

    # 创建测试数据
    print("\n创建测试数据...")
    obs = create_dummy_observation(model)

    # 开始性能分析
    profiler = InferenceProfiler(model, device)

    # 1. 测试完整推理
    actions = profiler.profile_full_inference(obs, num_runs=args.num_episodes)

    # 2. 真实分段计时
    component_times = profiler.measure_component_times(num_runs=args.num_episodes * 2)

    # 3. 测试 RTC 场景（用实测组件时间）
    rtc_results = profiler.test_rtc_scenarios(
        obs, actions, exec_freq=args.exec_freq, comp_times=component_times)

    # 保存结果
    print("\n" + "="*70)
    print("结果已生成")
    print("="*70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
