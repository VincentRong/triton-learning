# 01 项目结构与开发规则

目标：搭建一个适合学习、测试、benchmark 和面试展示的 Triton 项目结构。

## 1. 推荐目录

在 `/home/rwenxiao/dev-learning/Triton` 下创建：

```text
triton-kernels-lab/
  README.md
  kernels/
    __init__.py
    vector_add.py
    transpose.py
    reductions.py
    softmax.py
    layernorm.py
    matmul.py
    fused_matmul.py
    rmsnorm.py
    quantize.py
  tests/
    test_correctness.py
  benchmarks/
    bench_all.py
  notes/
    env.md
    gpu_concepts.md
    triton_interview.md
```

创建命令：

```bash
mkdir -p triton-kernels-lab/{kernels,tests,benchmarks,notes}
touch triton-kernels-lab/kernels/__init__.py
touch triton-kernels-lab/README.md
```

## 2. 每个 kernel 文件的固定结构

建议每个文件都用相同布局：

```python
import torch
import triton
import triton.language as tl


@triton.jit
def _kernel(...):
    ...


def public_api(...):
    ...
    _kernel[grid](...)
    return out


def reference(...):
    ...


if __name__ == "__main__":
    ...
```

规则：

- `_kernel` 是 Triton JIT 函数。
- `public_api` 是 Python 包装函数，负责分配输出 tensor、计算 grid、传参数。
- `reference` 是 PyTorch 对照实现。
- `__main__` 可以放临时 sanity check，但正式测试放到 `tests/`。

## 3. 测试规则

每个 kernel 至少测：

- shape 正常情况。
- shape 不是 block size 整数倍。
- dtype 至少覆盖 fp32，适合的算子再覆盖 fp16。
- 输入 tensor 在 CUDA 上。
- 输出和 PyTorch reference 对齐。

常用断言：

```python
torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
```

fp16 matmul / norm 的容忍度可以放宽：

```python
torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
```

## 4. Benchmark 规则

每个算子至少跑 3 个 shape：

- 小 shape：看 launch overhead。
- 中 shape：看常规延迟。
- 大 shape：看带宽或算力上限。

记录：

- shape
- dtype
- Triton latency
- PyTorch latency
- speedup
- GB/s 或 TFLOPS

## 5. README 写法

README 不要只写“我实现了什么”，要写“怎么验证、怎么优化、结果如何”。

推荐结构：

```markdown
# Triton Kernels Lab

## Environment

## Kernels

## Correctness

## Benchmarks

## Optimization Notes

## Interview Summary
```

## 6. Git 提交节奏

建议每完成一个阶段提交一次：

```bash
git add .
git commit -m "add vector add triton kernel"
```

提交粒度：

- 环境和项目结构一次。
- elementwise 一次。
- reduction / softmax 一次。
- matmul 一次。
- profiling / README 一次。

## 7. 完成标准

- [ ] 目录结构创建完毕。
- [ ] 所有 kernel 都有固定文件位置。
- [ ] `tests/` 能统一运行 correctness。
- [ ] `benchmarks/` 能统一跑性能。
- [ ] `README.md` 可以作为面试项目说明。
