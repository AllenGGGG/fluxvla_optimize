# FSDP 优化报告：pi05_parcel_sort bs=32 训练

**目标**：单卡 batch_size=32，显存稳定，速度约 5s/iter，参考 mozbrain 的 FSDP 实践。
**最终结果**：8 卡、单卡 bs=32、显存 ~71GB/80GB 稳定无 OOM、稳态 iter 时间 ~4.3-4.7s/iter，达成目标。

---

## 1. 单卡 bs=32 物理放不下，需要多卡真分片

**现象**：单卡（A100-80GB）跑 bs=32 在第一个 forward 阶段就 OOM，且随后即便降到 bs=16/28 依然 OOM。

**原因**：单卡场景下 `world_size=1`，PyTorch FSDP 会自动把 `FULL_SHARD` 退化成 `NO_SHARD`——即完全不分片。用 `torch.cuda.memory` 做过实测：权重 ~7.2GB + 梯度 ~7.2GB + AdamW 优化器状态 ~14.4GB，固定开销就有 ~29GB，还没算随 batch 增长的激活值。这是硬件物理限制，不是配置问题。

**效果**：确认了单卡 bs=32 在当前模型规模下不可行，必须走多卡真分片路线。这条判断是后续所有工作的前提。

---

## 2. `paligemma.py` FSDP wrap 粒度过细

**位置**：`fluxvla/models/backbones/vlms/paligemma.py:139`

**现象/原因**：`get_fsdp_wrapping_policy()` 原来把 `GemmaAttention`、`GemmaMLP`、`GemmaRMSNorm` 分别注册为独立 wrap 单元，导致每个 decoder layer 被拆成 3 个 FSDP unit，而不是 1 个整层。

**修复**：改成整个 `GemmaDecoderLayer` 一个 wrap 单元（`transformer_layer_cls={self.transformer_layer_cls}`）。

**效果**：这条路径（`vlm_backbone` 直接走标准 HuggingFace forward，非自定义 cross-attention）下，FSDP unit 数量减少、gather/reshard 次数减少，降低单卡场景下的固定开销。此改动对 `pi05_parcel_sort` 这个具体 config 不直接生效（它不用 `vlm_backbone`），但对其它用 `PaliGemma` 作为 vlm_backbone 的 config（如 `configs/pi0/pi0_paligemma_*`）是有效修复。

---

## 3. 显存主权重单卡场景下仍是 fp32（而非声明的 bf16）

**位置**：`fluxvla/engines/runners/fsdp_train_runner.py:276`（`is_effectively_unsharded`）

**现象**：config 里设了 `mixed_precision_dtype='bf16'`，但单卡跑起来显存占用接近纯 fp32 训练的水平。

**原因**：原代码用 `is_no_shard`（`sharding_strategy == NO_SHARD`）判断是否用 bf16 存主权重，但 `pi05_parcel_sort.py` 配置的是 `'full-shard'`，所以走的是 `target_dtype = torch.float32` 分支——即使 world_size=1、FSDP 内部已经自动退化成 NO_SHARD，代码逻辑上还是按"多卡分片"对待，把主权重维持在 fp32（4 字节/参数），而不是 bf16（2 字节/参数）。

**修复**：新增 `is_effectively_unsharded = is_no_shard or overwatch.world_size() == 1`，单卡时也用 bf16 存主权重。

**效果**：单卡场景下参数显存直接减半，这是让训练能往更大 batch 推进的关键一步（虽然最终发现即便如此单卡也放不下 bs=32，但这个修复本身是正确且必要的，多卡场景下同样受益于更准确的判断逻辑）。

---

## 4. 嵌套 activation checkpointing 导致内存峰值 + 计算浪费

**位置**：`fluxvla/engines/runners/fsdp_train_runner.py:334-347`

**现象**：OOM 报错栈显示崩在 `GemmaMLP.forward()` 内部的反向重计算阶段，分配失败的尺寸从最初的 968MB 逐步变化，说明这里有额外的、非必要的内存峰值。

**原因**：原代码同时把 `GemmaMLP` 和 `GemmaDecoderLayer`（外层，包含 MLP）都注册为 checkpoint 边界。PyTorch 的 checkpoint 语义是：外层边界的重计算过程中，如果内部还有一个独立的 checkpoint 边界，会再触发一次嵌套的"保存-丢弃-重算"，导致同一段计算被多算一遍，并在重算瞬间产生双缓冲的内存尖峰。

**修复**：移除嵌套，只保留 `GemmaMLP` + `GemmaRMSNorm`（详见第 6 点，这两个才是真正生效的 checkpoint 粒度）。

**效果**：单次实测中，同一处失败分配的申请量从 968MB 降到 242MB——直接量化验证了嵌套 checkpoint 确实在制造额外的内存压力。

---

## 5. FSDP wrap 完成后有 ~7GB 显存被 caching allocator 无谓占用

**位置**：`fluxvla/engines/runners/fsdp_train_runner.py:384-388`

**现象**：用 `torch.cuda.memory_allocated()` / `memory_reserved()` 分段打点发现，FSDP wrap 刚完成时：`allocated=7.23GB`（跟模型规模吻合，正常），但 `reserved=13.97GB`——几乎两倍，多出来的 ~6.7GB 是 wrap 过程中 flatten/copy 参数产生的临时分配，被 PyTorch 缓存分配器保留但一直没归还。

**修复**：FSDP wrap 后立即 `torch.cuda.empty_cache()`。

**效果**：实测 reserved 从 13.97GB 降到 7.30GB，精确吻合。这是本次排查中发现的单笔最大的"白白浪费"的显存。

---

## 6. checkpoint_layer_classes 里大部分设置从未真正生效（隐藏 bug）

**位置**：`fluxvla/engines/runners/fsdp_train_runner.py:337-346, 362-372`

**现象/原因**：这是本次排查里最深的一处发现。`pi0_flowmatching.py` 的 `_forward_transformer_layers`（PI0 的 prefix/suffix cross-attention 核心逻辑）会**直接**调用 `layer.self_attn.q_proj(...)`、`layer.mlp(...)`、`layer.input_layernorm(...)` 等子模块，**从不**调用 `layer(...)`（即 `GemmaDecoderLayer.forward()`）或 `layer.self_attn(...)`（即 `GemmaAttention.forward()`）本身。

而 PyTorch 的 activation checkpoint 机制是"只对被 `check_fn` 命中、且其自身 `forward()` 真正被调用的模块生效"。原代码把 `GemmaDecoderLayer`（通过 `self.llm_transformer_layer_cls`）和 `GemmaAttention`（通过 `llm_expert.transformer_layer_cls`）都注册为 checkpoint 目标——但这两个类的 `forward()` 在这套自定义代码路径里从未被调用过，所以对它们的 checkpoint 包装其实完全是**惰性/无效代码**，一直没有真正省过显存。

**修复**：改成注册 `GemmaMLP`（对应 `layer.mlp(...)` 调用）和 `GemmaRMSNorm`（对应 `layer.input_layernorm(...)`/`layer.post_attention_layernorm(...)` 调用）——这两个才是真正被直接调用、checkpoint 会生效的粒度。

**效果**：让 llm_backbone/llm_expert 的 activation checkpointing 从"名义上开启、实际不生效"变成真正生效，这是解决内存问题的核心修复之一。

---

## 7. 多卡真分片下崩溃：FSDP wrap 粒度必须匹配"直接子模块调用"模式（架构级修复）

这是本次投入最大、也最关键的一处修复。

**现象**：切到 2 卡或 8 卡真分片后，每个 rank 都报 `RuntimeError: size mismatch, got input (30976), mat (30976x2048), vec (...)`，且各 rank 报的 `vec` 尺寸都不一样（0、655360、1179648）。

**根因排查过程**：
1. 一开始误判为"FSDP wrap 切得太细"，把 `pi0_flowmatching.py` 里逐 `nn.Linear`/`nn.LayerNorm` 的 wrap 匹配（`match_module`）当作冗余移除了，这是**错误的**判断。
2. 对比 mozbrain 的参考实现（`mozbrain/lerobot/common/policies/pi0/paligemma_with_expert.py`）后发现：mozbrain 用的是**几乎一模一样**的自定义 cross-attention 写法（`layer.self_attn.q_proj(...)` 直接调用），而它的 FSDP wrap policy 恰恰是**逐 Linear** 包装（`transformer_layer_cls=[nn.Linear, nn.Embedding]`）——这不是效率选择，而是**正确性要求**。
3. 原理：FSDP 只在被 wrap 模块自己的 `forward()`/`__call__` 触发时才会把该 rank 持有的分片 all-gather 成完整参数。如果只 wrap 了外层（比如整个 `GemmaAttention` 或 `GemmaDecoderLayer`），但代码从不调用这个外层的 `forward()`，只是直接伸手进去调它的子模块（如 `q_proj`），那么这次调用完全绕过了 FSDP 的 gather 钩子——被访问到的子模块参数依然是"只有本 rank 那一份分片"，尺寸自然对不上，直接崩溃。
4. 逐 Linear 单独 wrap，则每次 `layer.self_attn.q_proj(hidden_states)` 调用的正是 `q_proj` 自己（作为一个独立 FSDP unit）的 `forward()`，会正确触发它自己的 gather。

**修复**：
   - `pi0_flowmatching.py:865-877`：恢复 `match_module` 对 `nn.Linear`/`nn.LayerNorm` 的匹配（连同一段详细注释，解释这是正确性要求而非效率选项）。
   - `condition_gemma.py:997-998`：`transformer_layer_cls` 属性从错误改动的 `GemmaDecoderLayer` 撤回到原来的 `GemmaAttention`——因为如果外层（`GemmaDecoderLayer`）被 wrap 成一个单元，会把从未被单独 wrap 过的 `input_layernorm`/`post_attention_layernorm`（RMSNorm）参数"困"在这个从未被真正调用过 `forward()` 的外层单元里，导致这些参数在多卡场景下也拿不到完整分片——是同一类 bug 的另一种表现形式。

**效果**：这是让多卡真分片从"必崩"变成"能跑"的决定性修复。修复后 8 卡训练全程无崩溃，loss 正常下降。

---

## 8. Dataloader 视频解码：单帧解码耗时过长导致 GPU 间歇性空闲

**位置**：`fluxvla/datasets/packed_parquet_dataset_v3.py:_decode_video_frames`

**现象**：训练稳定后仍观察到周期性的 GPU 利用率下降（个别卡瞬间掉到 0%-70%），但没有拖累整体平均 iter 时间。

**根因排查**：
1. 实测发现单帧视频解码平均耗时 **178ms**，每次要扫描 ~32-33 帧才能命中目标帧——跟代码里固定的"提前 1.0 秒 seek"（`first_ts - 1.0`）严格对应（1秒×30fps≈30帧）。
2. 拉了 mozbrain/lerobot 自己的参考实现（`video_utils.py:decode_video_frames_torchvision`）对比，发现它**根本没有**这个额外的 1 秒 margin，直接 `reader.seek(first_ts, keyframes_only=True)`——因为 `keyframes_only=True` 本身就会自动往回吸附最近关键帧，不需要手动多减。
3. 实测验证：把 margin 从 1.0s 降到 0.15s，80 个随机样本全部零失败（原有的"匹配失败则整段重扫"兜底逻辑保证了安全性），解码耗时从 182ms 降到 51ms，提速 **3.6 倍**。

**修复**：`_load(first_ts - 1.0)` → `_load(first_ts - 0.15)`，并同时把 `per_device_num_workers` 从 4 提到 8（`configs/pi05/pi05_parcel_sort.py`）以进一步缓解并发压力。

**效果**：单帧解码耗时降低 3.6 倍。但深挖后发现，剩余的间歇性 GPU 利用率抖动，根源是 **8 卡 × 8 worker = 64 个 dataloader 进程同时做 CPU 密集的视频解码**，在高并发下产生的尾延迟——这是机器整体负载/并发架构层面的问题，不是单帧解码耗时能完全解决的。已验证不影响目标达成（iter 时间稳定在 4.3-4.7s/iter），暂未继续深挖。

---

## 9. （已定位方案，未实施）b1k IterableDataset 的 buffer 架构可根治抖动

参考了 `wqingzex/openpi` 仓库 `behavior1k-uniform-task-sampling` 分支的 `b1k_iterable_dataset.py`：它用**按 GOP 批量解码 + 内存 buffer + 后台异步重填**的架构，彻底把"视频解码"和"训练取样"解耦——取样永远是纯内存操作，解码在后台线程池异步、批量进行。这是能从架构上根治第 8 点残留抖动的方案，但需要重写 dataset 层（map-style → buffer-style IterableDataset），改动量和风险明显超出本次修复范围，留作后续单独任务。

---

## 最终验证结果

| 指标 | 数值 |
|---|---|
| GPU 数量 | 8（真实 FSDP full-shard） |
| 单卡 batch size | 32 |
| 单卡显存占用 | ~71GB / 80GB，稳定无 OOM |
| 稳态 iter 时间 | ~4.3-4.7s/iter |
| Loss | 正常收敛下降 |

## 改动文件清单

- `fluxvla/engines/runners/fsdp_train_runner.py` —— bf16 主权重判断、checkpoint 粒度修复、FSDP wrap 后 empty_cache、forward_prefetch
- `fluxvla/models/backbones/vlms/paligemma.py` —— wrap 粒度改为整层
- `fluxvla/models/vlas/pi0_flowmatching.py` —— 恢复逐 Linear wrap 匹配（含详细正确性说明注释）
- `fluxvla/models/backbones/llms/condition_gemma.py` —— 撤销错误改动，恢复原状（与 HEAD 无 diff）
- `fluxvla/datasets/packed_parquet_dataset_v3.py` —— 视频解码 pre-seek margin 优化
- `configs/pi05/pi05_parcel_sort.py` —— `per_device_num_workers` 4→8
