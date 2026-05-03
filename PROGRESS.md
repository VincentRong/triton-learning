# Triton 学习进度

这个文件用于记录真实学习进度。`PLAND.md` 保持为课程计划和总 checklist；这里记录实际完成内容、环境结果、benchmark 数据、遇到的问题、修复方式和下一步入口。

## 当前状态

- 开始时间：2026-05-02
- 工作目录：`/home/rwenxiao/dev-learning/Triton`
- 学习计划：`PLAND.md`
- 展开材料：`guidelines/`
- 当前阶段：环境检查与项目结构搭建

## 记录规则

- `PLAND.md`：记录计划、目标能力和高层 checklist。
- `PROGRESS.md`：记录真实进度、命令结果、benchmark 表格、错误、修复方式和下次继续的位置。
- `triton-kernels-lab/notes/`：记录更深入的专题笔记，例如环境、GPU 概念、面试复盘和优化观察。

## 进度日志

### 2026-05-02

- 进入 Triton 学习目录。
- 阅读并确认已有学习材料：
  - `PLAND.md`
  - `guidelines/README.md`
  - `guidelines/00_environment_setup.md`
  - `guidelines/01_project_structure.md`
- 决定：使用 `PROGRESS.md` 作为独立进度记录文件，不把真实进度混写进 `PLAND.md`。
- 初步环境检查：
  - GPU：NVIDIA GeForce RTX 3070，8GB 显存。
  - `nvidia-smi` 可用。
  - 系统 `python3` 是 3.12.3，但没有安装 `torch` 和 `triton`。
  - `/home/rwenxiao/pytorch-cuda/bin/python` 可用，包含 PyTorch 和 Triton。
  - PyTorch：2.11.0+cu128。
  - Triton：3.6.0。
  - PyTorch CUDA：12.8。
  - `torch.cuda.is_available()` 在直接探测命令中为 True。
- 已创建 `triton-kernels-lab/` 基础目录结构：
  - `README.md`
  - `kernels/__init__.py`
  - `tests/test_env.py`
  - `benchmarks/`
  - `notes/env.md`
- 已安装 `pytest` 到 `/home/rwenxiao/pytorch-cuda` 环境。
- 注意：在普通沙盒中运行 `pytest` 时看不到 NVIDIA driver，GPU smoke test 失败；需要在允许访问 GPU 的环境中重跑。

## 环境记录

待完成：

- [x] 记录 `nvidia-smi`。
- [x] 记录 Python 版本。
- [x] 记录 PyTorch 版本和 CUDA 可用性。
- [x] 记录 Triton 版本。
- [x] 创建或确认 `triton-kernels-lab/`。
- [ ] 在可访问 GPU driver 的环境中运行最小环境测试。

## 下一步

1. 在可访问 GPU 的环境中重跑：

   ```bash
   cd /home/rwenxiao/dev-learning/Triton/triton-kernels-lab
   /home/rwenxiao/pytorch-cuda/bin/python -m pytest -q
   ```

2. 如果 smoke test 通过，开始 `guidelines/02_triton_basics.md`。
3. 实现第一个 kernel：`vector_add`。
4. 每完成一个 kernel，同步更新：
   - `PROGRESS.md`
   - `triton-kernels-lab/README.md`
   - 对应 `notes/*.md`

## 阻塞 / 问题

- 普通沙盒下 `pytest` 无法访问 NVIDIA driver；直接 Python 探测 GPU 正常，说明更像是运行环境权限差异，不是 PyTorch/Triton 安装问题。
