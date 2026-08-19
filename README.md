# FluxVLA PI0.5 Parcel Sort

本分支用于 PI0.5 快递分拣策略的真机 ROS 2 推理部署，不提供 Libero、
RoboCasa 或 FluxBiSim 仿真配置。

## 配置结构

```text
configs/pi05/
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

## 新设备装机

当前验证过的环境是 Python 3.12.13、PyTorch 2.8.0 + CUDA 12.8，路径为：

```text
/home/fiveages/runtime/miniforge3/envs/fluxvla_infer
```

新设备需要先准备：

- NVIDIA 驱动以及与 PyTorch 匹配的 CUDA Toolkit/NVCC；仅有显卡驱动不能编译项目 CUDA 扩展。
- Conda/Miniforge、`g++`、`cmake` 和 `ninja`。
- ROS 2 Jazzy 及真机控制工作区；启动前必须 source ROS 2 环境，并确保 `rclpy` 可导入。

在项目根目录执行自动安装：

```bash
bash scripts/install_env.sh real-only \
  --profile cu128 \
  --env-name fluxvla_infer \
  --python-version 3.12.13
conda activate fluxvla_infer
```

安装脚本会安装匹配的 PyTorch、FlashAttention、真机推理依赖，并执行：

```bash
python -m pip install --no-build-isolation -e .
```

这一步会自动调用根目录的 `setup.py`，编译 PI0.5 推理使用的 CUDA 扩展。
不要直接运行 `python setup.py install`。更换设备、Python、PyTorch 或 CUDA 后都要
重新执行上述 editable install；不要复用其他设备生成的 `.so` 文件。

安装完成后验证：

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

随后加载 ROS 2 和机器人工作区；实际路径按新设备安装位置修改：

```bash
source /opt/ros/jazzy/setup.bash
source /path/to/robot_ws/install/setup.bash
python -c "import rclpy; print('ROS 2 Python: OK')"
```

## 环境文件

环境文件集中放在 `environment/`：

- `pip-requirements.txt`：pip 安装使用的真机推理依赖。
- `conda-packages.txt`：当前已验证环境的 `conda list --export` 快照，用于核对版本，
  不是跨设备直接安装文件。

需要手动补装或重编译项目时执行：

```bash
python -m pip install -r environment/pip-requirements.txt
python -m pip install --no-build-isolation -e .
```

PyTorch、FlashAttention 和 ROS 2 不包含在 `pip-requirements.txt` 中：前两者由安装
脚本根据 CUDA 环境处理，ROS 2 Python 包由已安装并 source 的 ROS 2 环境提供。

## 启动

```bash
./deploy/launch.sh
```

启动器会交互选择 checkpoint、异步或串行执行、RTC 模式及 PyTorch/Triton
后端。详细参数与安全检查见 `deploy/README.md`。

默认组合是异步执行、guidance RTC 和 Triton 加速。也可通过环境变量进行
非交互配置，例如：

```bash
PISTAR_CHECKPOINT_ROOT=/path/to/checkpoint \
PISTAR_EXECUTION_MODE=async \
PISTAR_RTC_MODE=guidance \
PISTAR_ACCELERATION=triton \
./deploy/launch.sh
```

## 保留内容

`fluxvla/` 的整体模型与算子框架保持不变；当前清理只针对仿真配置、仿真
依赖和重复文档，避免影响真机推理加载及后续维护。
