# 🚀 快速上手指南

## 优化已完成 ✅

你现在有**两个推理配置**可以对比测试：

### 1. 标准版（Baseline）
配置: `configs/pi05/pi05_libero10_task0_tempovla_standard_inference.py`

包含优化:
- ✅ Triton + CUDA Graph + Speed 缓存
- ❌ 无 Ultra Fusion

预期延迟: **~40-50ms** (A100)

### 2. 优化版（Ultra）
配置: `configs/pi05/pi05_libero10_task0_tempovla_speed_modulated_inference.py`

包含优化:
- ✅ Triton + CUDA Graph + Speed 缓存
- ✅ **Ultra Fusion Kernels**

预期延迟: **~30-40ms** (A100)

---

## 使用方法

### 标准版推理
```bash
python scripts/eval.py \
  --config configs/pi05/pi05_libero10_task0_tempovla_standard_inference.py \
  --ckpt-path work_dirs/libero10_full_tempovla_speed_modulated/checkpoints/latest-checkpoint.safetensors
```

### 优化版推理（推荐）
```bash
python scripts/eval.py \
  --config configs/pi05/pi05_libero10_task0_tempovla_speed_modulated_inference.py \
  --ckpt-path work_dirs/libero10_full_tempovla_speed_modulated/checkpoints/latest-checkpoint.safetensors
```

---

## 对比测试

### 自动对比脚本
```bash
python scripts/benchmark_standard_vs_ultra.py \
  --ckpt work_dirs/libero10_full_tempovla_speed_modulated/checkpoints/latest-checkpoint.safetensors \
  --num-trials 50 \
  --task-id 4
```

### 手动对比
分别运行标准版和优化版，加上 `--cfg-options eval.measure_predict_latency=True` 参数。

---

## 文档

- **INFERENCE_VARIANTS.md** - 两个版本详细对比
- **REALTIME_OPTIMIZATIONS.md** - 完整技术说明
- **CODE_VERIFICATION.md** - 验证和问题解答
- **FINAL_SUMMARY.md** - 最终总结

---

## 推荐工作流

1. **先测试标准版** - 确保基础功能正常
2. **再测试优化版** - 验证 Ultra Fusion 效果
3. **对比结果** - 看看加速了多少
4. **生产部署** - 使用优化版

详见 `INFERENCE_VARIANTS.md` 📖
