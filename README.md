# FluxVLA PI0.5 Parcel Sort

本分支同时包含 PI0.5 快递分拣策略的 A100 训练代码和 4090 真机 ROS 2
推理部署代码，不提供 Libero、RoboCasa 或 FluxBiSim 仿真配置。

## 环境安装

训练和推理的 Python、PyTorch、CUDA 和 pip 依赖版本不同，必须使用两个独立
conda 环境，不能在同一个环境里来回安装。

| 用途 | 目标设备 | 默认环境 | Python | PyTorch / CUDA | pip 依赖文件 |
|---|---|---|---|---|---|
| 训练 | A100 | `fluxvla_train` | 3.10 | 2.6.0 / cu124 | `environment/pip-requirements-train.txt` |
| 推理 | RTX 4090 | `fluxvla_infer` | 3.12.13 | 2.8.0 / cu128 | `environment/pip-requirements.txt` |

新设备需要先安装 NVIDIA 驱动、与目标环境匹配的 CUDA Toolkit/NVCC、
Conda/Miniforge、`g++`、`cmake` 和 `ninja`。只有显卡驱动而没有 NVCC 时，
项目 CUDA 扩展无法编译。4090 真机推理还需要 ROS 2 Jazzy 和机器人控制工作区。

从项目根目录直接运行安装脚本，在菜单中输入 `1` 安装训练环境，输入 `2` 安装
推理环境：

```bash
./scripts/install_env.sh
```

脚本会创建并在安装过程中激活目标环境，然后依次安装对应版本的 PyTorch、PyAV、
pip 依赖、FlashAttention，并执行 editable 项目安装以编译 CUDA 扩展：

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
```

需要自定义环境名或仅检查将要执行的命令时：

```bash
bash scripts/install_env.sh train --env-name my_train_env
bash scripts/install_env.sh inference --env-name my_infer_env
bash scripts/install_env.sh train --dry-run
bash scripts/install_env.sh inference --dry-run
```

完整参数说明：

```bash
./scripts/install_env.sh --help
```

## pip 依赖

环境文件集中放在 `environment/`：

- `pip-requirements-train.txt`：从 A100 训练环境记录整理的训练依赖。
- `pip-requirements.txt`：从 4090 推理环境记录整理的 ROS 2 真机推理依赖。
- `conda-packages.txt`：4090 推理环境的 `conda list --export` 快照，仅用于核对
  版本，不是跨设备直接安装文件。

PyTorch、PyAV 和 FlashAttention 由 `install_env.sh` 单独安装，不写入上述 pip
依赖文件。ROS 2 的 `rclpy` 由系统 ROS 2 环境提供，也不通过 pip 安装。

已有环境只需要补装普通 pip 依赖时，先激活正确环境，再执行对应命令：

```bash
# A100 训练环境
conda activate fluxvla_train
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

训练和推理环境都可以用下面的命令验证 PyTorch、CUDA 和项目扩展：

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

训练环境还应确认分布式启动器可用：

```bash
torchrun --help >/dev/null && echo "torchrun: OK"
```

## 启动训练

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

`fluxvla/` 的整体模型与算子框架保持不变；训练和真机推理分别使用独立环境，
避免两端不同版本的依赖互相覆盖。
