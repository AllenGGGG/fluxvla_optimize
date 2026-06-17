#!/usr/bin/env python3
"""快速对比标准版 vs 优化版的推理延迟。

对比：
- 标准版：Triton + CUDA Graph + Speed 缓存（无 ultra fusion）
- 优化版：标准版 + Ultra Fusion Kernels

Usage:
    python scripts/benchmark_standard_vs_ultra.py \\
        --ckpt work_dirs/libero10_full_tempovla_speed_modulated/checkpoints/latest-checkpoint.safetensors \\
        --num-iters 100
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_eval(config, checkpoint, num_trials=5, task_id=4):
    """运行评估并返回延迟。"""
    cmd = [
        sys.executable, '-m', 'torch.distributed.run',
        '--nnodes', '1',
        '--nproc-per-node', '1',
        '--master-port', '29500',
        'scripts/eval.py',
        '--config', config,
        '--ckpt-path', checkpoint,
        '--cfg-options',
        f'eval.num_trials_per_task={num_trials}',
        f'eval.eval_task_ids=[{task_id}]',
        'eval.measure_predict_latency=True',
    ]

    print(f"\n运行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    # 提取延迟
    for line in result.stdout.split('\n'):
        if 'mean ms:' in line:
            try:
                latency = float(line.split('mean ms:')[1].strip())
                return latency
            except:
                pass

    print("警告: 无法从输出中提取延迟")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True, help='Checkpoint path')
    parser.add_argument('--num-trials', type=int, default=5, help='Trials per variant')
    parser.add_argument('--task-id', type=int, default=4, help='Task ID to test')
    args = parser.parse_args()

    if not Path(args.ckpt).exists():
        print(f"错误: Checkpoint 不存在: {args.ckpt}")
        return 1

    print("=" * 70)
    print("  标准版 vs 优化版 推理延迟对比")
    print("=" * 70)

    configs = {
        '标准版 (无 ultra fusion)': 'configs/pi05/pi05_libero10_task0_tempovla_standard_inference.py',
        '优化版 (有 ultra fusion)': 'configs/pi05/pi05_libero10_task0_tempovla_speed_modulated_inference.py',
    }

    results = {}

    for name, config in configs.items():
        print(f"\n{'='*70}")
        print(f"测试: {name}")
        print(f"配置: {config}")
        print(f"{'='*70}")

        latency = run_eval(config, args.ckpt, args.num_trials, args.task_id)
        if latency is not None:
            results[name] = latency
            print(f"✅ {name}: {latency:.2f} ms")
        else:
            print(f"❌ {name}: 测试失败")

    # 总结
    print("\n" + "=" * 70)
    print("  测试结果汇总")
    print("=" * 70)

    if len(results) == 2:
        standard = results.get('标准版 (无 ultra fusion)')
        ultra = results.get('优化版 (有 ultra fusion)')

        print(f"\n标准版延迟: {standard:.2f} ms")
        print(f"优化版延迟: {ultra:.2f} ms")
        print(f"\n加速比: {standard/ultra:.2f}x")
        print(f"延迟降低: {standard - ultra:.2f} ms ({(standard-ultra)/standard*100:.1f}%)")
    else:
        print("\n未能完成所有测试")

    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
