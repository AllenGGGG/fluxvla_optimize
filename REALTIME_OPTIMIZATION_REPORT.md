# RealTimeVLA 推理优化 — 实测报告

本文档记录在 `PI05FlowMatchingSpeedModulatedInference`（PixShuffle + FluxVLA +
TempoVLA 技术栈）上集成 RealTimeVLA 优化技术的**实测结果**。所有数字均在
A100 单卡、`work_dirs/libero10_full_tempovla_speed_modulated` 的 latest 权重上
实测得到，不是估算。

---

## TL;DR

| 优化项 | 状态 | 实测结论 |
|--------|------|----------|
| Triton fused kernels | ✅ 启用 | 有效，是基础 |
| CUDA Graph | ✅ 启用 | **关键优化**，把 119ms→31ms |
| Speed 预编码缓存 | ✅ 启用 | 正确，省掉动态 speed 切换开销 |
| Ultra Fusion kernels | ⚠️ 默认关闭 | **修复后数值正确，但有 CUDA Graph 时无加速（略慢 3%）** |
| AE ffn_down split-K | ⚠️ 已集成(ultra路径) | **数值正确，微基准赢 1.46x，但端到端只省 0.5ms** |
| RTC（chunk 间动作融合） | ❌ 未接线 | 加速路径里没有；仿真同步执行下用不上 |
| VLM/AE 解耦 | ❌ 未实现 | 实测显示收益有限（见下） |

## split-K 实测（重要：微基准赢但端到端没用）

按"AE 带宽利用率只有 15%、M=10 喂不满 GPU"的分析，给 ffn_down（K=4096）做了
split-K=8。**先微基准 de-risk，再集成，全程验证**：

- 微基准（孤立 kernel，CUDA Graph 下）：ffn_down split-K=8 **赢 1.46x**
  (35.96us→24.60us)；o_proj (K=2048) 无收益（1.0x），所以只改 ffn_down。
- 数值正确性：split-K 输出 vs 现有 matmul_res_gate，误差 0.26%（bf16 噪声级）。
- **端到端**：AE 20.1ms → 19.6ms（仅 -0.5ms），完整推理 30.7→30.1ms，33Hz 不变。

**为什么微基准 1.46x 端到端却几乎没用**：ffn_down 只是每层 ~5 个 matmul 之一，
18 层里它占的绝对时间有限；split-K 多出的 merge kernel 调用开销，以及在整图里
和其它 kernel 抢 SM，吃掉了大部分收益。这和 ultra fusion 是同一个教训——
**孤立微基准的加速，在 18 层整图 + CUDA Graph 里会被稀释。**

代码已集成在 ultra fusion 路径（默认关闭）。数值正确、可用、无害，但**不是
提速的有效杠杆**。真正有效的仍然是下面的 num_steps。


**当前推理速度：~31ms / 32Hz（单次完整推理，生成 10 步 action chunk）。**

---

## 实测分段计时（全优化，每段独立建 CUDA Graph）

```
Vision Encoder (VLM):     3.85 ms  (12.8%)
Transformer Encoder:      6.42 ms  (21.3%)
Action Decoder (AE):     19.84 ms  (65.9%)  ← 真正的瓶颈
------------------------------------------------
三段之和:                30.11 ms
完整流程 (整图):         31.79 ms   (三段之和 ≈ 完整流程，计时可信)
```

### 关键发现

1. **AE（Action Decoder）是大头，占 66%**，不是 VLM。
   原因：AE 要跑 `num_steps=10` 个去噪步 × 18 层 = 180 次层计算，是 Encoder
   的 10 倍。想再快，最直接的是减 `num_steps`（需测成功率）。

2. **VLM 只占 13%**，因为 PixShuffle 把视觉 token 压到 64/视角，序列极短
   （encoder_seq_len=192）。层数虽多但每层很轻。

3. **CUDA Graph 是真正的关键优化**。无图时分段相加 119ms，有图 31ms。
   AE 步数多、层多、kernel 碎，正是 CUDA Graph 的最大受益者。

---

## 关于 Ultra Fusion（重要纠正）

最初写的三个 ultra fusion kernel 有两个是坏的，已修复/移除：

- `time_mlp_speed_fused`：matmul tiling 错误，运行直接崩溃。**已移除**，
  `time_mlp_with_speed_optimized` 改用已验证的 `matmul_small_bias_silu`。
- `adarms_norm_gate_fused`：归一化语义错误（漏了 `(1+scale)`、shift、gate 偏移
  读错）。**已移除**，`adarms_norm_gate_optimized` 改用已验证的
  `adarms_norm_kernel`。
- `matmul_res_gate_fused`：正确，**保留**。

修复后实测（`scripts/verify_ultra_fusion.py`）：

```
标准路径:      30.30 ms (33.0 Hz)
Ultra fusion:  31.27 ms (32.0 Hz)
加速比:        0.969x   ← 不升反降
输出误差:      0.000000 ← 两条路径逐位一致，数值完全正确
```

**结论：Ultra fusion 在有 CUDA Graph 时是伪优化。** 它的目的是减少 kernel
launch 次数，但 CUDA Graph 已经把所有 kernel 打包成一张图，launch 开销早已
消除。再融合反而因 tiling 不如标准 kernel 而略慢。**因此默认关闭**
（`use_ultra_fusion=False`），代码保留仅用于 A/B 对比。

---

## 关于 RTC 与 VLM/AE 解耦

- **RTC**：代码存在于 `rtc_guidance.py` + 非加速 flow matching 路径，但你
  benchmark 用的 Triton 加速路径（`pi05_flowmatching_inference.py`）**没有
  接入 RTC**，eval 循环也不传 `prev_actions`/`rtc_config`。所以历次对比测试
  RTC 从未生效。仿真是同步执行（推理时仿真冻结），没有 chunk 接缝问题，
  RTC 在仿真里基本用不上；它的价值在真机/异步部署。

- **VLM/AE 解耦**：实测显示可缓存部分（VLM+Encoder）仅占 34%（10.27ms），
  AE 占 66% 无法缓存。即使完美解耦，单次推理也只能从 31ms 降到 ~20ms（32→50Hz），
  天花板有限。且解耦需要拆 CUDA Graph，可能引入新的 launch 开销。**当前不划算。**

---

## 使用方法

### 默认（推荐，所有验证过的优化）
```bash
python scripts/eval.py \
  --config configs/pi05/pi05_libero10_task0_tempovla_speed_modulated_inference.py \
  --ckpt-path work_dirs/libero10_full_tempovla_speed_modulated/checkpoints/latest-checkpoint.safetensors
```
= Triton + CUDA Graph + Speed 缓存，~31ms / 32Hz。

### A/B 对比 ultra fusion（开/关）
```bash
# 开（验证用，无加速）
... --cfg-options inference_model.use_ultra_fusion=True
# 关（默认）
... --cfg-options inference_model.use_ultra_fusion=False
```

---

## 工具脚本

- `scripts/profile_inference_rtc.py` — 分段计时（每段独立建 CUDA Graph）+ RTC 参数推算
- `scripts/verify_ultra_fusion.py` — ultra fusion 正确性 + 速度 A/B 验证

---

## 下一步（按收益排序）

1. **减 `num_steps`（10→5）** — 命中真正瓶颈（AE 占 66%），AE 可能砍半到 ~10ms。
   需测成功率掉多少。这是当前最高性价比的方向。
2. **真机部署时接入 RTC** — 解决异步执行的 chunk 接缝抖动，仿真里无需。
3. 模型量化 / 蒸馏 / 减层 — 如需突破 32Hz 的硬件天花板。
