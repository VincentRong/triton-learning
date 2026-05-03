# 06 Matmul 与 Fusion

目标：实现基础 matmul，理解 tiling、`tl.dot`、fp32 accumulation、grouped ordering、autotune，并扩展到 fused matmul + bias + activation。

## 1. 要实现的文件

```text
triton-kernels-lab/kernels/matmul.py
triton-kernels-lab/kernels/fused_matmul.py
```

## 2. Matmul 目标

输入：

```python
A: [M, K]
B: [K, N]
C: [M, N]
```

计算：

```python
C = A @ B
```

## 3. Tile 设计

一个 program 计算一个 `BLOCK_M x BLOCK_N` 的 C tile。

每次沿 K 方向加载：

- A tile：`BLOCK_M x BLOCK_K`
- B tile：`BLOCK_K x BLOCK_N`
- 累加到 C tile：`BLOCK_M x BLOCK_N`

核心形状：

```python
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
```

## 4. 基础 Kernel 结构

```python
pid = tl.program_id(0)
num_pid_n = tl.cdiv(N, BLOCK_N)
pid_m = pid // num_pid_n
pid_n = pid % num_pid_n

offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
offs_k = tl.arange(0, BLOCK_K)

a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

for k in range(0, K, BLOCK_K):
    k_mask = offs_k + k < K
    a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
    b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
    acc += tl.dot(a, b)
    a_ptrs += BLOCK_K * stride_ak
    b_ptrs += BLOCK_K * stride_bk
```

store：

```python
c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
tl.store(c_ptrs, acc, mask=c_mask)
```

## 5. 推荐参数

先试：

- `BLOCK_M=32, BLOCK_N=32, BLOCK_K=32, num_warps=4`
- `BLOCK_M=32, BLOCK_N=64, BLOCK_K=32, num_warps=4`
- `BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_warps=4`
- `BLOCK_M=64, BLOCK_N=128, BLOCK_K=32, num_warps=4/8`

初学阶段不要一开始追求极致性能。先正确，再逐步调参。

## 6. Grouped Ordering

普通 pid 顺序可能导致 tile 复用差。grouped ordering 让相邻 program 更集中地访问相近的 A/B tile，提高 L2 cache locality。

面试表达：

> grouped ordering 改变 program 到 C tile 的映射顺序，让一组 program 在 M 方向或 N 方向上聚集执行，从而提高 A 或 B tile 在 L2 cache 中的复用概率。它不改变数学结果，只改变调度访问顺序。

实现时可以参考 Triton 官方 matmul tutorial 的 grouped ordering 写法。

## 7. Autotune

用 `@triton.autotune` 管理多个 config：

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_kernel(...):
    ...
```

key 一般选择会影响最佳配置的 shape 参数，例如 `M`、`N`、`K`。

## 8. Fused Matmul + Bias + Activation

目标：

```python
C = activation(A @ B + bias)
```

bias shape：

```python
bias: [N]
```

融合位置：

```python
bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
acc = acc + bias[None, :]
acc = tl.maximum(acc, 0.0)  # relu
```

GELU 近似可以作为进阶：

```python
0.5 * x * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

Triton 中如果 tanh 支持受限，可先实现 ReLU，再实现 SiLU。

## 9. Benchmark

Matmul 用 TFLOPS：

```python
flops = 2 * M * N * K
tflops = flops / (ms * 1e-3) / 1e12
```

测试 shape：

```python
[
    (256, 256, 256),
    (1024, 1024, 1024),
    (4096, 4096, 4096),
    (1024, 4096, 4096),
]
```

对比：

- Triton matmul
- `torch.matmul`
- fused Triton vs PyTorch 分开执行
- fused Triton vs `torch.compile` 后的 PyTorch

## 10. 常见错误

- K loop 中忘记移动 `a_ptrs` / `b_ptrs`。
- mask 只处理 M/N，忘记处理 K 尾块。
- `acc` 用 fp16，误差过大。
- store 时没有 C mask。
- B 的 stride 写反，结果完全不对。

## 11. 面试表达

> Matmul 是 compute-bound 的代表算子，优化核心是 tiling 和数据复用。一个 program 计算 C 的一个 tile，沿 K 方向分块加载 A 和 B，用 `tl.dot` 触发高效矩阵乘累加，通常 fp16 输入、fp32 accumulate。调优时会看 tile size、num_warps、num_stages、L2 locality、occupancy 和 tensor core 利用率。

## 12. 完成标准

- [ ] 基础 matmul 正确。
- [ ] 支持任意 M / N / K。
- [ ] fp16 输入 fp32 累加。
- [ ] benchmark 至少 4 组 shape。
- [ ] 实现 fused bias + ReLU。
- [ ] 能解释 tiling、`tl.dot`、grouped ordering、autotune。
