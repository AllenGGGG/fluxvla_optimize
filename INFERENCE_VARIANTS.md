# 推理版本对比说明

## 两个推理配置

为了方便对比测试，提供了两个推理配置：

### 1. 标准版（Baseline）
**配置文件**: `configs/pi05/pi05_libero10_task0_tempovla_standard_inference.py`

**包含的优化**:
- ✅ Triton fused kernels
- ✅ CUDA Graph
- ✅ Speed 预编码缓存
- ❌ **无** Ultra Fusion Kernels

**预期延迟**: ~40-50ms (A100)

**使用场景**: 对比基准测试

---

### 2. 优化版（Ultra）
**配置文件**: `configs/pi05/pi05_libero10_task0_tempovla_speed_modulated_inference.py`

**包含的优化**:
- ✅ Triton fused kernels
- ✅ CUDA Graph
- ✅ Speed 预编码缓存
- ✅ **Ultra Fusion Kernels** (RealTimeVLA)

**预期延迟**: ~30-40ms (A100)

**使用场景**: 生产部署，最快推理

---

## 对比测试

### 方法 1: 手动对比

**标准版**:
```bash
python scripts/eval.py \
  --config configs/pi05/pi05_libero10_task0_tempovla_standard_inference.py \
  --ckpt-path work_dirs/libero10_full_tempovla_speed_modulated/checkpoints/latest-checkpoint.safetensors \
  --cfg-options eval.measure_predict_latency=True
```

**优化版**:
```bash
python scripts/eval.py \
  --config configs/pi05/pi05_libero10_task0_tempovla_speed_modulated_inference.py \
  --ckpt-path work_dirs/libero10_full_tempovla_speed_modulated/checkpoints/latest-checkpoint.safetensors \
  --cfg-options eval.measure_predict_latency=True
```

---

### 方法 2: 自动对比脚本

```bash
python scripts/benchmark_standard_vs_ultra.py \
  --ckpt work_dirs/libero10_full_tempovla_speed_modulated/checkpoints/latest-checkpoint.safetensors \
  --num-trials 50 \
  --task-id 4
```

**输出示例**:
```
标准版延迟: 45.2 ms
优化版延迟: 32.8 ms

加速比: 1.38x
延迟降低: 12.4 ms (27.4%)
```

---

### 方法 3: 使用现有的对比脚本

```bash
python scripts/compare_pi05_task0_with_l_pixshuffle.py \
  --tag standard_vs_ultra \
  --variant standard \
  configs/pi05/pi05_libero10_task0_tempovla_standard_inference.py \
  work_dirs/libero10_full_tempovla_speed_modulated/checkpoints/latest-checkpoint.safetensors \
  --variant ultra \
  configs/pi05/pi05_libero10_task0_tempovla_speed_modulated_inference.py \
  work_dirs/libero10_full_tempovla_speed_modulated/checkpoints/latest-checkpoint.safetensors \
  --eval-speeds 1.0 \
  --task-id 4 \
  --speed-warmup-iters 10 \
  --speed-bench-iters 50
```

---

## 技术细节对比

| 特性 | 标准版 | 优化版 |
|------|--------|--------|
| Vision Encoder | Triton | Triton |
| Transformer Encoder | Triton | Triton |
| Transformer Decoder | Triton | Triton + Ultra Fusion |
| Time MLP + Speed | 3 ops | **1 fused op** |
| Matmul + Res + Gate | 3 ops | **1 fused op** |
| Kernel Launches (per layer) | ~20 | ~15 (-25%) |
| Speed 缓存 | ✅ | ✅ |
| CUDA Graph | ✅ | ✅ |

---

## 何时使用哪个版本？

### 使用标准版的场景:
1. **首次测试** - 确保基础推理正常工作
2. **调试问题** - 如果优化版有问题，回退到标准版
3. **对比基准** - 测量 Ultra Fusion 的实际收益
4. **保守部署** - 不确定硬件兼容性时

### 使用优化版的场景:
1. **生产部署** - 需要最低延迟
2. **实时控制** - 机器人实时反应
3. **高频推理** - 大规模评估
4. **资源受限** - GPU 利用率已经很高

---

## 实现细节

两个版本使用**同一个推理模型**：`PI05FlowMatchingSpeedModulatedInference`

唯一区别是 `use_ultra_fusion` 参数：
```python
# 标准版
inference_model = dict(
    type='PI05FlowMatchingSpeedModulatedInference',
    use_ultra_fusion=False,  # 标准路径
)

# 优化版
inference_model = dict(
    type='PI05FlowMatchingSpeedModulatedInference',
    use_ultra_fusion=True,   # Ultra fusion
)
```

这意味着：
- ✅ 共享同一套代码，易于维护
- ✅ 可以随时切换，无需重新编译
- ✅ 出问题时可以快速回退
- ✅ 对比测试公平（其他条件完全相同）

---

## 总结

| 版本 | 配置文件 | use_ultra_fusion | 延迟 | 使用场景 |
|------|----------|------------------|------|----------|
| 标准版 | `*_standard_inference.py` | False | ~40-50ms | 基准测试 |
| 优化版 | `*_speed_modulated_inference.py` | True | ~30-40ms | 生产部署 |

**推荐流程**:
1. 先用标准版验证功能
2. 再用优化版测试性能
3. 对比两者差异
4. 生产环境使用优化版
