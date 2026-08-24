# FluxVLA PI0.5 Parcel Sort

本分支同时包含 PI0.5 快递分拣策略的 A100/PPU 训练代码和 4090 真机 ROS 2
推理部署代码。A100 与 PPU 使用完全独立的启动脚本和配置，不提供 Libero、
RoboCasa 或 FluxBiSim 仿真配置。

## 环境安装

训练和推理的 Python、PyTorch、CUDA/PPU SDK 和 pip 依赖版本不同，必须使用
独立 conda 环境，不能在同一个环境里来回安装。

| 用途 | 目标设备 | 默认环境 | Python | PyTorch / CUDA | pip 依赖文件 |
|---|---|---|---|---|---|
| 训练 | A100 | `fluxvla_train` | 3.10 | 2.6.0 / cu124 | `environment/pip-requirements-train.txt` |
| 训练 | PPU | `fluxvla` | 3.12 | PPU PyTorch 2.8.0 / PPU SDK | `environment/pip-requirements-train.txt` |
| 推理 | RTX 4090 | `fluxvla_infer` | 3.12.13 | 2.8.0 / cu128 | `environment/pip-requirements.txt` |

新设备需要先安装 NVIDIA 驱动、与目标环境匹配的 CUDA Toolkit/NVCC、
Conda/Miniforge、`g++`、`cmake` 和 `ninja`。只有显卡驱动而没有 NVCC 时，
项目 CUDA 扩展无法编译。PPU 环境必须运行在带
`/usr/local/PPU_SDK/envsetup.sh` 和平台内部 Python 包源的 PPU 容器中。4090
真机推理还需要 ROS 2 Jazzy 和机器人控制工作区。

从项目根目录直接运行安装脚本，在菜单中输入 `1` 安装 A100 训练环境、输入 `2`
安装 4090 推理环境、输入 `3` 安装 PPU 训练环境：

```bash
./scripts/install_env.sh
```

菜单的 PPU 选项会委托给独立的 `install_env_ppu.sh`，不会进入 A100/4090 的 CUDA
安装流程。脚本会创建并在安装过程中激活目标环境，然后安装对应版本的 PyTorch、
PyAV、pip 依赖，并执行 editable 项目安装以编译设备扩展：

```bash
python -m pip install --no-build-isolation -e .
```

安装脚本结束后，环境激活状态不会传回当前 shell，需要手动执行一次
`conda activate`。非交互安装命令如下：

```bash
# A100 训练：Python 3.10、PyTorch 2.6.0 + CUDA 12.4
bash scripts/install_env.sh train
conda activate fluxvla_train

# 4090 推理：Python 3.12.13、PyTorch 2.8.0 + CUDA 12.8
bash scripts/install_env.sh inference
conda activate fluxvla_infer

# PPU 训练：在 PPU 容器中运行；也可直接运行 install_env_ppu.sh
bash scripts/install_env.sh ppu
conda activate fluxvla
```

PPU 安装脚本固定从 `/usr/local/PPU_SDK/envsetup.sh` 加载 SDK，并使用 SDK 提供的
`PIP_INDEX_URL` 安装 PPU 版 PyTorch 2.8.0、torchvision 0.23.0 和 Triton 3.3.0。
该内部地址可能携带临时凭据，脚本不会打印或写死完整地址。PPU 使用 PyTorch
原生 SDPA，不依赖 FlashAttention，因此 PPU 安装脚本不会安装 FlashAttention。

需要自定义环境名或仅检查将要执行的命令时：

```bash
bash scripts/install_env.sh train --env-name my_train_env
bash scripts/install_env.sh inference --env-name my_infer_env
bash scripts/install_env.sh train --dry-run
bash scripts/install_env.sh inference --dry-run
bash scripts/install_env_ppu.sh --dry-run
```

完整参数说明：

```bash
./scripts/install_env.sh --help
bash scripts/install_env_ppu.sh --help
```

## pip 依赖

环境文件集中放在 `environment/`：

- `pip-requirements-train.txt`：训练公共依赖，由 A100 和 PPU 训练环境分别安装。
- `pip-requirements.txt`：从 4090 推理环境记录整理的 ROS 2 真机推理依赖。
- `conda-packages.txt`：4090 推理环境的 `conda list --export` 快照，仅用于核对
  版本，不是跨设备直接安装文件。

PyTorch、PyAV 和 FlashAttention 由环境安装脚本单独安装，不写入上述 pip 依赖
文件。PPU 不安装 FlashAttention。ROS 2 的 `rclpy` 由系统 ROS 2 环境提供，也不
通过 pip 安装。

已有环境只需要补装普通 pip 依赖时，先激活正确环境，再执行对应命令：

```bash
# A100 训练环境
conda activate fluxvla_train
python -m pip install -r environment/pip-requirements-train.txt

# PPU 训练环境（先在 PPU 容器中加载 SDK）
source /usr/local/PPU_SDK/envsetup.sh ""
conda activate fluxvla
python -m pip install -r environment/pip-requirements-train.txt

# 4090 推理环境
conda activate fluxvla_infer
python -m pip install -r environment/pip-requirements.txt
```

更换设备、Python、PyTorch 或 CUDA 后必须重新编译项目 CUDA 扩展，不要复制其他
环境生成的 `.so` 文件：

```bash
python -m pip install --no-build-isolation -e .
```

## 安装验证

A100、PPU 和推理环境都可以用下面的命令验证 PyTorch、CUDA 兼容接口和项目扩展：

```bash
python - <<'PY'
import torch
import fluxvla
from fluxvla.ops.cuda.matmul_bias import matmul_bias_cuda

print("fluxvla:", fluxvla.__file__)
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("CUDA extension: OK")
PY
```

训练环境还应确认分布式启动器可用；PPU 环境中的 NCCL 接口实际由 PCCL 提供：

```bash
torchrun --help >/dev/null && echo "torchrun: OK"
python -c "import torch; assert torch.distributed.is_nccl_available(); print('NCCL/PCCL: OK')"
```

## 启动训练

### A100 训练

`scripts/train.sh` 中的训练环境固定默认路径是
`/home/guohao/miniconda3/envs/fluxvla_train`，原 A100 机器直接激活该环境：

```bash
conda activate fluxvla_train
```

如果明确把训练环境安装到了其他路径，先激活它，再显式覆盖启动脚本使用的路径：

```bash
conda activate fluxvla_train
export FLUXVLA_ENV_PREFIX="$CONDA_PREFIX"
```

`scripts/train.sh` 默认单机 8 卡、后台运行、WandB offline。脚本里的数据、基础模型、
checkpoint 和日志默认路径来自原 A100 训练机器；在新机器上启动时应通过参数明确
指定实际路径：

```bash
bash scripts/train.sh \
  --dataset /path/to/lerobot_dataset \
  --base-model-dir /path/to/pi05_base \
  --base-weights /path/to/model.safetensors \
  --work-dir /path/to/checkpoints/run_name \
  --log-dir /path/to/logs/run_name \
  --nproc-per-node 8
```

后台启动成功后，终端会打印 PID、`console.log` 和 PID 文件路径。前台调试时增加
`--foreground`：

```bash
bash scripts/train.sh \
  --dataset /path/to/lerobot_dataset \
  --base-model-dir /path/to/pi05_base \
  --base-weights /path/to/model.safetensors \
  --work-dir /path/to/checkpoints/debug \
  --log-dir /path/to/logs/debug \
  --nproc-per-node 8 \
  --foreground
```

从 `.pt` checkpoint 断点续训：

```bash
bash scripts/train.sh \
  --resume-from /path/to/checkpoint.pt \
  --dataset /path/to/lerobot_dataset \
  --base-model-dir /path/to/pi05_base \
  --base-weights /path/to/model.safetensors \
  --work-dir /path/to/checkpoints/run_name \
  --log-dir /path/to/logs/run_name
```

数据集目录必须包含 `meta/`。基础模型目录负责提供 tokenizer/config，
`--base-weights` 指向模型 `.safetensors` 文件。查看 batch size、epoch、学习率、
GPU 数量和其他覆盖参数：

```bash
bash scripts/train.sh --help
```

默认始终保存模型 `.safetensors`。只有需要完整训练状态用于断点续训时才增加
`--save-pt`，让脚本额外保存 `.pt`；该规则在 A100 和 PPU 脚本中一致。

### PPU 训练

PPU 使用独立的 `scripts/train_ppu.sh` 和
`configs/pi05/pi05_parcel_sort_ppu.py`，不会修改或复用 A100 脚本里的机器路径。
脚本按原 PPU 分支写死以下默认值：

- PPU SDK：`/usr/local/PPU_SDK/envsetup.sh`
- conda 环境：`/opt/conda/envs/fluxvla`
- 数据集：`/mnt/cpfs_data/allen/dataset/2026_07_23`、`2026_07_24`、
  `2026_07_25`、`2026_07_27`、`2026_07_29`
- 初始权重：`/mnt/cpfs_data/allen/checkpoints/2026_07_29/checkpoints/step-020088-epoch-04-loss=0.0041.safetensors`
- checkpoint：`/mnt/cpfs_data/allen/checkpoints/2026_07_30`
- 日志：`/mnt/cpfs_data/allen/logs/2026_07_30`
- 训练参数：16 进程、每设备 batch size 20、7 epochs、学习率 `3e-5`、BF16 混合精度、SDPA

基础模型目录仍是当前 checkout 下的 `checkpoints/pi05_base`。确认这些固定路径已经
准备好后直接启动，默认后台运行：

```bash
bash scripts/train_ppu.sh
```

前台调试或额外保存 `.pt`：

```bash
bash scripts/train_ppu.sh --foreground
bash scripts/train_ppu.sh --save-pt
```

PPU 脚本启动前会检查 PyAV、16 个 PPU 设备、NCCL/PCCL、BF16 和 SDPA，并通过
CUDA 兼容接口使用 PPU。路径确实需要变化时仍可用 `--dataset`、`--base-model-dir`、
`--base-weights`、`--work-dir` 和 `--log-dir` 覆盖；完整参数见：

```bash
bash scripts/train_ppu.sh --help
```

混合精度和 attention backend 不是同一个设置：BF16/FP32 决定张量计算精度，
SDPA/eager 决定 attention 的计算实现。当前 A100 和 PPU 配置都使用 BF16 混合
精度与 SDPA；PPU 仍使用自己的独立配置，后续可以单独调节。SDPA 会调用当前
PyTorch/PPU 后端提供的 scaled dot-product attention 优化实现，通常比逐步计算的
eager attention 更节省显存并具有更高吞吐。这些设置不改变 PI0.5 模型结构或
checkpoint 参数命名。

当前 PPU 集成只包含普通 `PackedParquetDatasetV3` 训练，不包含旧分支的
`PackedVSTATempoPromptParquetDatasetV3`、速度重采样或速度 prompt。

## 启动推理

先激活 4090 推理环境并加载 ROS 2 Jazzy 与机器人工作区；机器人工作区路径按
实际安装位置修改：

```bash
conda activate fluxvla_infer
source /opt/ros/jazzy/setup.bash
source /path/to/robot_ws/install/setup.bash
python -c "import rclpy; print('ROS 2 Python: OK')"
```

从项目根目录运行启动器：

```bash
./deploy/launch.sh
```

启动器会交互选择 checkpoint、异步或串行执行、RTC 模式及 PyTorch/Triton
后端。默认组合是异步执行、guidance RTC 和 Triton 加速，详细参数与安全检查见
[`deploy/README.md`](deploy/README.md)。

也可以通过环境变量进行非交互配置：

```bash
PISTAR_CHECKPOINT_ROOT=/path/to/checkpoint \
PISTAR_EXECUTION_MODE=async \
PISTAR_RTC_MODE=guidance \
PISTAR_ACCELERATION=triton \
./deploy/launch.sh
```

## 配置结构

```text
configs/pi05/
├── pi05_parcel_sort.py
├── pi05_parcel_sort_ppu.py
├── none/
│   ├── pytorch_inference.py
│   └── triton_inference.py
├── guidance/
│   ├── pytorch_inference.py
│   └── triton_inference.py
└── prefix/
    ├── pytorch_inference.py
    └── triton_inference.py
```

## 保留内容

`fluxvla/` 的整体模型与算子框架保持不变；A100 训练、PPU 训练和真机推理分别使用
独立环境与启动入口，避免不同设备版本的依赖和固定路径互相覆盖。
