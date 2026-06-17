# 对比测试命令更新说明

## ⚠️ 重要变更

你之前的测试命令用的配置文件：
```
configs/pi05/pi05_libero10_task0_tempovla_realtime_inference.py  ❌ 已删除
```

**这个文件已经删除了**，因为它只是一个占位符，功能和 `speed_modulated_inference.py` 完全一样。

---

## ✅ 新的测试命令

### 方法 1: 使用提供的脚本（推荐）

```bash
./run_standard_vs_ultra_comparison.sh
```

这个脚本会对比：
- **标准版**: `pi05_libero10_task0_tempovla_standard_inference.py` (无 Ultra Fusion)
- **优化版**: `pi05_libero10_task0_tempovla_speed_modulated_inference.py` (有 Ultra Fusion)

---

### 方法 2: 手动运行

```bash
setsid /home/guohao/miniconda3/envs/fluxvla/bin/python \
scripts/compare_pi05_task0_with_l_pixshuffle.py \
--tag standard_vs_ultra \
--variant standard_baseline \
configs/pi05/pi05_libero10_task0_tempovla_standard_inference.py \
work_dirs/libero10_full_tempovla_speed_modulated/checkpoints/latest-checkpoint.safetensors \
--variant ultra_optimized \
configs/pi05/pi05_libero10_task0_tempovla_speed_modulated_inference.py \
work_dirs/libero10_full_tempovla_speed_modulated/checkpoints/latest-checkpoint.safetensors \
--base-weights work_dirs/libero10_full_tempovla_speed_modulated/checkpoints/latest-checkpoint.safetensors \
--eval-speeds 0.5,0.75,1.0,1.25,1.5,1.75,2.0 \
--task-id 4 \
--success-trials-per-task 50 \
--success-seeds 7 \
--success-gpus 2,3,4,5,6,7 \
--success-nproc-per-node 6 \
--speed-single-gpus 2 \
--speed-multi-gpus 2,3,4,5,6,7 \
--speed-warmup-iters 10 \
--speed-bench-iters 50 \
> logs/standard_vs_ultra_compare.log 2>&1 < /dev/null &
```

---

## 对比内容

| Variant | 配置文件 | 优化内容 |
|---------|---------|---------|
| `standard_baseline` | `*_standard_inference.py` | Triton + CUDA Graph + Speed 缓存 |
| `ultra_optimized` | `*_speed_modulated_inference.py` | 标准版 + Ultra Fusion Kernels |

---

## 预期结果

### Success Rate（成功率）
两个版本应该**完全相同**，因为：
- 使用同一个训练权重
- 使用相同的模型架构
- 只是推理优化不同，不影响输出

### Speed Benchmark（推理速度）
- **标准版**: ~40-50ms
- **优化版**: ~30-40ms
- **加速比**: 1.2-1.4x

---

## 如果你想对比其他配置

### 对比 PixShuffle vs 非 PixShuffle
```bash
--variant pixshuffle \
configs/pi05/pi05_libero10_task0_tempovla_speed_modulated_inference.py \
work_dirs/libero10_full_tempovla_speed_modulated/checkpoints/latest-checkpoint.safetensors \
--variant no_pixshuffle \
configs/pi05/pi05_libero10_task0_with_l_inference.py \
work_dirs/libero10_full/checkpoints/latest-checkpoint.safetensors
```

### 对比不同的训练权重
```bash
--variant checkpoint_A \
configs/pi05/pi05_libero10_task0_tempovla_speed_modulated_inference.py \
work_dirs/run_A/checkpoints/latest-checkpoint.safetensors \
--variant checkpoint_B \
configs/pi05/pi05_libero10_task0_tempovla_speed_modulated_inference.py \
work_dirs/run_B/checkpoints/latest-checkpoint.safetensors
```

---

## 查看测试结果

```bash
# 实时查看日志
tail -f logs/standard_vs_ultra_compare.log

# 查看完整日志
cat logs/standard_vs_ultra_compare.log

# 查看最终结果
grep -A 20 "Final Results" logs/standard_vs_ultra_compare.log
```

---

## 总结

❌ **旧命令**（不能用了）:
```
--variant A100_100epoch_tempo_realtime \
configs/pi05/pi05_libero10_task0_tempovla_realtime_inference.py  # 已删除
```

✅ **新命令**（用这个）:
```
--variant ultra_optimized \
configs/pi05/pi05_libero10_task0_tempovla_speed_modulated_inference.py  # 优化版

--variant standard_baseline \
configs/pi05/pi05_libero10_task0_tempovla_standard_inference.py  # 标准版
```

或者直接运行：
```bash
./run_standard_vs_ultra_comparison.sh
```
