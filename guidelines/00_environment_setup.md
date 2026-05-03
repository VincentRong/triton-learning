# 00 环境安装与验证

目标：准备一个能运行 PyTorch CUDA 和 Triton kernel 的本地环境，并确认 GPU、驱动、Python 包都可用。

## 1. 检查系统和 GPU

先确认机器能看到 NVIDIA GPU：

```bash
nvidia-smi
```

你需要记录：

- GPU 型号
- Driver Version
- CUDA Version
- 显存大小

如果 `nvidia-smi` 不存在，说明当前环境可能没有 NVIDIA 驱动，或者你不在 GPU 节点上。Triton 的多数实战需要 GPU。

## 2. 创建 Python 虚拟环境

建议在项目目录下建独立环境：

```bash
cd /home/rwenxiao/dev-learning/Triton
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

之后每次进入项目：

```bash
cd /home/rwenxiao/dev-learning/Triton
source .venv/bin/activate
```

## 3. 安装 PyTorch

去 PyTorch 官方安装页选择与你机器匹配的 CUDA wheel：

https://docs.pytorch.org/get-started/locally/

常见 Linux pip 形式如下，具体 CUDA 版本以官网 selector 为准：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

如果你的机器 CUDA runtime 对应别的版本，用官网生成的命令替换。不要凭感觉混装多个 CUDA wheel。

验证：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
```

期望：

- `cuda available: True`
- 能打印出 GPU 名称

## 4. 安装 Triton

官方稳定版安装：

```bash
pip install triton
```

验证：

```bash
python - <<'PY'
import triton
import triton.language as tl
print("triton:", triton.__version__)
print("tl loaded:", tl)
PY
```

Triton 官方文档说明稳定版可通过 `pip install triton` 安装；不同 Python 版本 wheel 支持会变化，以官方安装页为准。

## 5. 创建 smoke test

创建后续会用到的目录：

```bash
mkdir -p triton-kernels-lab/{kernels,tests,benchmarks,notes}
touch triton-kernels-lab/kernels/__init__.py
```

写一个最小 CUDA 测试文件 `triton-kernels-lab/tests/test_env.py`：

```python
import torch
import triton


def test_cuda_available():
    assert torch.cuda.is_available()


def test_basic_tensor_on_gpu():
    x = torch.randn((1024,), device="cuda")
    y = x + 1
    assert y.is_cuda
    assert y.numel() == 1024


def test_triton_import():
    assert triton.__version__ is not None
```

安装 pytest：

```bash
pip install pytest
```

运行：

```bash
cd triton-kernels-lab
pytest -q
```

## 6. 常见问题

### `torch.cuda.is_available()` 是 False

优先检查：

- `nvidia-smi` 是否可用。
- PyTorch 是否安装了 CUDA wheel，而不是 CPU wheel。
- 当前 shell 是否进入了正确的 `.venv`。

### `pip install triton` 失败

优先检查：

- Python 版本是否被当前 Triton wheel 支持。
- pip 是否太旧，先执行 `python -m pip install -U pip`。
- 当前系统是否为 Linux x86_64 或官方支持平台。

### CUDA toolkit 是否必须安装

用 PyTorch wheel 跑大多数 Python 训练和 Triton 学习时，通常依赖 wheel 自带的 CUDA runtime。你仍然需要 NVIDIA driver。需要源码编译或 Nsight 工具时，才更可能涉及系统 CUDA toolkit。

## 7. 完成标准

- [ ] `nvidia-smi` 正常。
- [ ] `torch.cuda.is_available()` 返回 True。
- [ ] `import triton` 成功。
- [ ] `pytest -q` 能通过最小环境测试。
- [ ] 在 `notes/env.md` 记录 GPU 型号、torch 版本、triton 版本。
