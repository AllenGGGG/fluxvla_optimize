#!/usr/bin/env python3
"""快速测试 RealTimeVLA 优化集成是否正确。"""

import sys


def test_imports():
    """测试所有模块能否正常导入。"""
    print("=" * 70)
    print("测试模块导入...")
    print("=" * 70)

    try:
        # 只有 matmul_res_gate_fused 保留（已验证正确）。另两个坏 kernel
        # 已移除，详见 REALTIME_OPTIMIZATION_REPORT.md。
        from fluxvla.ops.triton.realtime_fusion_ops import matmul_res_gate_fused
        print("✅ realtime_fusion_ops 导入成功")
    except Exception as e:
        print(f"❌ realtime_fusion_ops 导入失败: {e}")
        return False

    try:
        from fluxvla.models.vlas import PI05FlowMatchingSpeedModulatedInference
        print("✅ PI05FlowMatchingSpeedModulatedInference 导入成功")
    except Exception as e:
        print(f"❌ PI05FlowMatchingSpeedModulatedInference 导入失败: {e}")
        return False

    try:
        from fluxvla.ops.atomic_ops import (
            adarms_norm_gate_optimized,
            matmul_res_gate_optimized,
            time_mlp_with_speed_optimized,
        )
        print("✅ atomic_ops 优化函数导入成功")
    except Exception as e:
        print(f"❌ atomic_ops 优化函数导入失败: {e}")
        return False

    return True


def test_config():
    """测试配置文件是否正确。"""
    print("\n" + "=" * 70)
    print("测试配置文件...")
    print("=" * 70)

    try:
        from mmengine import Config

        cfg = Config.fromfile(
            'configs/pi05/pi05_libero10_task0_tempovla_speed_modulated_inference.py'
        )

        print(f"✅ 配置文件加载成功")
        print(f"   模型类型: {cfg.inference_model.type}")
        print(f"   默认 speed: {cfg.inference_model.default_tempo_speed}")
        print(f"   Ultra fusion: {cfg.inference_model.use_ultra_fusion}")

        if cfg.inference_model.type != 'PI05FlowMatchingSpeedModulatedInference':
            print(f"❌ 错误的模型类型: {cfg.inference_model.type}")
            return False

        # Ultra fusion 默认关闭（实测在 CUDA Graph 下无加速，详见
        # REALTIME_OPTIMIZATION_REPORT.md）。这里只验证字段存在且为布尔。
        if not isinstance(cfg.inference_model.use_ultra_fusion, bool):
            print(f"❌ use_ultra_fusion 应为布尔: {cfg.inference_model.use_ultra_fusion}")
            return False
        print(f"   (ultra fusion 默认 {cfg.inference_model.use_ultra_fusion}，"
              f"实测无加速，仅供 A/B 对比)")

        return True

    except Exception as e:
        print(f"❌ 配置文件测试失败: {e}")
        return False


def main():
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 20 + "RealTimeVLA 优化集成测试" + " " * 24 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    all_passed = True

    all_passed &= test_imports()
    all_passed &= test_config()

    print("\n" + "=" * 70)
    if all_passed:
        print("✅ 所有测试通过！")
        print("\n下一步:")
        print("1. 运行推理测试:")
        print("   python scripts/eval.py \\")
        print("     --config configs/pi05/pi05_libero10_task0_tempovla_speed_modulated_inference.py \\")
        print("     --ckpt-path work_dirs/libero10_full_tempovla_speed_modulated/checkpoints/latest-checkpoint.safetensors")
        print("\n2. 对比 ultra fusion 开关:")
        print("   --cfg-options inference_model.use_ultra_fusion=False")
        print("=" * 70)
        return 0
    else:
        print("❌ 部分测试失败")
        print("请检查错误信息并修复。")
        print("=" * 70)
        return 1


if __name__ == '__main__':
    sys.exit(main())
