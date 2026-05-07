# Triton 学习进度

这个文件用于记录真实学习进度。`PLAND.md` 保持为课程计划和总 checklist；这里记录实际完成内容、环境结果、benchmark 数据、遇到的问题、修复方式和下一步入口。

## 当前状态

- 开始时间：2026-05-02
- 工作目录：`/home/rwenxiao/dev-learning/Triton`
- 学习计划：`PLAND.md`
- 展开材料：`guidelines/`
- 当前阶段：matmul baseline 与 fused matmul 入门

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

### 2026-05-03

- 已将本地 Triton 学习目录初始化为 git 仓库并推送到 GitHub：
  - `https://github.com/VincentRong/triton-learning`
- 环境 smoke test 已在可访问 GPU 的环境中通过：

  ```text
  3 passed in 3.83s
  ```

- 完成第一组 elementwise kernels：
  - `vector_add(a, b)`
  - `vector_mul(a, b)`
  - `vector_affine(x, scale, bias)`
- 新增 correctness 测试：
  - shape 覆盖 `(1024,)`、`(1000,)`、`(1_000_000,)`、`(1024, 1024)`。
  - 测试结果：

  ```text
  15 passed in 4.25s
  ```

- 完成 `vector_add` benchmark，并与 PyTorch `a + b` 对比。shape `(1024, 1024)`，dtype `float32`：

  | block size | Triton ms | PyTorch ms | speedup | GB/s |
  | --- | ---: | ---: | ---: | ---: |
  | 256 | 0.0375 | 0.0362 | 0.96 | 335.66 |
  | 512 | 0.0367 | 0.0354 | 0.96 | 343.07 |
  | 1024 | 0.0357 | 0.0354 | 0.99 | 352.07 |
  | 2048 | 0.0362 | 0.0352 | 0.97 | 347.17 |

- 学习要点：
  - 一个 Triton program 处理一块 offsets。
  - `grid = (ceil(n / BLOCK_SIZE),)` 决定 program 数量。
  - `mask = offsets < n` 保护最后一个 block 的越界访问。
  - `tl.arange` 生成的是 program 内的向量化逻辑位置，不是简单等同于 CUDA thread id。
  - elementwise 通常是 memory-bound，因为每个元素计算很少，但要读写 global memory。
  - 单个简单 elementwise kernel 不一定明显快过 PyTorch；Triton 的常见优势来自 fusion，把多个读写内存的步骤合成一次 kernel launch 和一次内存往返。

- 完成二维 memory / stride 练习：
  - `copy_2d(x)`
  - `transpose(x)`
  - `rowwise_add(x, bias)`
- `copy_2d` 支持非连续 2D view，通过显式传入 `x.stride(0)` 和 `x.stride(1)` 做地址计算。
- correctness 测试增加到：

  ```text
  31 passed in 9.85s
  ```

- 完成 2D op benchmark，shape `(1024, 1024)`，dtype `float32`：

  | op | block | Triton ms | PyTorch ms | speedup | GB/s |
  | --- | --- | ---: | ---: | ---: | ---: |
  | copy_2d | 16x16 | 0.0278 | 0.0261 | 0.94 | 302.13 |
  | transpose | 16x16 | 0.0271 | 0.0363 | 1.34 | 309.42 |
  | rowwise_add | 16x16 | 0.0273 | 0.0272 | 1.00 | 461.43 |
  | copy_2d | 16x32 | 0.0276 | 0.0260 | 0.94 | 304.18 |
  | transpose | 16x32 | 0.0264 | 0.0334 | 1.26 | 317.46 |
  | rowwise_add | 16x32 | 0.0260 | 0.0273 | 1.05 | 483.52 |
  | copy_2d | 32x32 | 0.0257 | 0.0262 | 1.02 | 326.26 |
  | transpose | 32x32 | 0.0250 | 0.0337 | 1.35 | 335.52 |
  | rowwise_add | 32x32 | 0.0259 | 0.0274 | 1.06 | 486.22 |

- 学习要点：
  - 二维 Tensor 地址一般是 `base + row * stride_0 + col * stride_1`。
  - `offs_m[:, None]` 和 `offs_n[None, :]` 会广播成一个 `BLOCK_M x BLOCK_N` tile。
  - shape 只描述逻辑维度，stride 才描述内存布局。
  - contiguous row-major tensor 的 `stride` 通常是 `(N, 1)`；转置 view 的 stride 可能变成 `(1, M)`。
  - transpose 改变读写方向，容易导致 load 或 store 其中一侧访存不连续。

- 完成 reduction / softmax / LayerNorm 练习：
  - `row_sum(x)`
  - `row_max(x)`
  - `row_mean(x)`
  - `softmax(x)`
  - `layernorm(x, gamma, beta)`
- correctness 测试增加到：

  ```text
  47 passed in 3.25s
  ```

- 修复一个 LayerNorm padding bug：
  - 问题：`N=768` 时 `BLOCK_N=1024`，越界 lane load 成 `0.0`。
  - mean 阶段没问题，因为 sum 后除以真实 `N`。
  - var 阶段如果直接 `diff = x - mean`，越界 lane 会变成 `0 - mean`，污染方差。
  - 修复：`diff = tl.where(mask, x - mean, 0.0)`。

- reduction benchmark，shape `(1024, 1024)`，dtype `float32`：

  | op | Triton ms | PyTorch ms | speedup | GB/s |
  | --- | ---: | ---: | ---: | ---: |
  | row_sum | 0.0196 | 0.0192 | 0.98 | 214.74 |
  | row_max | 0.0179 | 0.0201 | 1.13 | 234.82 |
  | row_mean | 0.0179 | 0.0194 | 1.08 | 234.51 |

- softmax / LayerNorm benchmark，dtype `float32`：

  | op | shape | Triton ms | PyTorch ms | speedup |
  | --- | --- | ---: | ---: | ---: |
  | softmax | 512x2048 | 0.0241 | 0.0252 | 1.04 |
  | layernorm | 32x1024 | 0.0060 | 0.0078 | 1.32 |

- 学习要点：
  - reduction 比 elementwise 多了“行内聚合”步骤，一个 program 先 load 一整行，再用 `tl.sum` 或 `tl.max` 压成标量。
  - `sum/mean` 的越界 lane 用 `other=0.0`。
  - `max` 的越界 lane 必须用 `other=-float("inf")`，否则全负数输入会错。
  - softmax 先减 row max，避免 `exp` 溢出。
  - LayerNorm 的 mean/var 建议 fp32 计算。

### 2026-05-07

- 完成 matmul / fused matmul 练习：
  - `matmul(a, b)`
  - `matmul_bias_relu(a, b, bias)`
- 设计选择：
  - 输入 `a` / `b` 使用 `float16`。
  - accumulator 使用 `float32`。
  - 输出使用 `float32`，便于观察 fp32 accumulate。
  - 一个 Triton program 计算一个 `BLOCK_M x BLOCK_N` 的 C tile。
  - 沿 K 维用 `BLOCK_K` 分块循环累加。
- correctness 测试增加到：

  ```text
  53 passed in 7.82s
  ```

- matmul benchmark，输入 dtype `float16`，输出 dtype `float32`：

  | shape | config | Triton ms | PyTorch ms | speedup | TFLOPS |
  | --- | --- | ---: | ---: | ---: | ---: |
  | 256x256x256 | 32x64x32/w4 | 0.0093 | 0.0092 | 1.00 | 3.61 |
  | 256x256x256 | 64x64x32/w4 | 0.0091 | 0.0092 | 1.01 | 3.67 |
  | 256x256x256 | 64x128x32/w4 | 0.0105 | 0.0092 | 0.88 | 3.20 |
  | 512x1024x2048 | 32x64x32/w4 | 0.0648 | 0.1158 | 1.79 | 33.12 |
  | 512x1024x2048 | 64x64x32/w4 | 0.0648 | 0.1158 | 1.79 | 33.12 |
  | 512x1024x2048 | 64x128x32/w4 | 0.0839 | 0.1158 | 1.38 | 25.60 |
  | 1024x1024x1024 | 32x64x32/w4 | 0.0666 | 0.0860 | 1.29 | 32.22 |
  | 1024x1024x1024 | 64x64x32/w4 | 0.0612 | 0.0860 | 1.41 | 35.11 |
  | 1024x1024x1024 | 64x128x32/w4 | 0.0614 | 0.0860 | 1.40 | 34.95 |
  | 1024x4096x4096 | 32x64x32/w4 | 1.8313 | 0.9575 | 0.52 | 18.76 |
  | 1024x4096x4096 | 64x64x32/w4 | 1.2644 | 0.9575 | 0.76 | 27.18 |
  | 1024x4096x4096 | 64x128x32/w4 | 1.0383 | 0.9575 | 0.92 | 33.09 |

- fused matmul + bias + ReLU benchmark：

  | shape | Triton ms | PyTorch separate ms | speedup | TFLOPS |
  | --- | ---: | ---: | ---: | ---: |
  | 1024x1024x1024 | 0.0658 | 0.1426 | 2.17 | 32.64 |

- 学习要点：
  - matmul 是 compute-bound 代表，优化重点从“少读写”转向“tile 复用”和 tensor core 利用率。
  - 一个 program 负责 C 的一个 tile，K 维循环加载 A/B tile。
  - `tl.dot(a, b)` 做 tile 级矩阵乘加，fp16 输入时通常走高效矩阵指令并 fp32 accumulate。
  - K 尾块必须 mask，否则任意 K 会错。
  - fused bias + ReLU 快很多，因为省掉了额外 kernel launch 和中间 C 的读写。
  - 大宽矩阵仍落后 PyTorch，下一步需要 grouped ordering 和 autotune。

## 环境记录

待完成：

- [x] 记录 `nvidia-smi`。
- [x] 记录 Python 版本。
- [x] 记录 PyTorch 版本和 CUDA 可用性。
- [x] 记录 Triton 版本。
- [x] 创建或确认 `triton-kernels-lab/`。
- [x] 在可访问 GPU driver 的环境中运行最小环境测试。

## 下一步

1. 提交并推送 matmul / fused matmul baseline 代码与记录。
2. 给 matmul 加 grouped ordering。
3. 添加 `@triton.autotune` 配置。
4. 对比 grouped / autotune 后的 benchmark。
4. 每完成一个 kernel，同步更新：
   - `PROGRESS.md`
   - `triton-kernels-lab/README.md`
   - 对应 `notes/*.md`

## 阻塞 / 问题

- 普通沙盒下 `pytest` 无法访问 NVIDIA driver；直接 Python 探测 GPU 正常，说明更像是运行环境权限差异，不是 PyTorch/Triton 安装问题。
