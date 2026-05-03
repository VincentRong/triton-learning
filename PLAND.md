# Triton 2 天高强度学习计划

目标：通过 2 天集中训练，达到中等 Triton 算子开发工程师水平。你应该能写中等复杂度 kernel，理解常见 GPU 算子概念，能参与加速运算开发，并具备足够的面试表达和完整实操经验。

配套展开材料见：[guidelines/README.md](guidelines/README.md)。

默认基础：

- 熟悉 Python / PyTorch。
- 了解 Tensor、shape、stride、dtype。
- 对 CUDA/GPU 有基础概念更好；如果完全陌生，建议先补半天 GPU 架构基础。

最终能力：

- 理解 Triton program、block、mask、stride、`tl.load`、`tl.store`。
- 能实现 elementwise、2D copy、transpose、reduction、softmax、LayerNorm、matmul、fused matmul、RMSNorm。
- 能做 correctness check、benchmark、基础 profiling。
- 能讲清 memory-bound / compute-bound、coalescing、tiling、occupancy、register pressure、fusion、autotune。

---

## Day 1：Triton 入门到常见算子

### 09:00 - 10:00：GPU 算子开发基础

学习目标：

- GPU 是 SIMT 执行模型。
- 一个 kernel 会被拆成很多 program / block。
- 每个 program 处理一块数据。
- Triton 的 `program_id` 类似 CUDA block id。
- `tl.arange` 用来构造 block 内 offsets。
- `mask` 用来处理越界。
- `stride` 用来处理非连续内存。

核心模式：

```python
pid = tl.program_id(0)
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n
x = tl.load(x_ptr + offsets, mask=mask)
tl.store(y_ptr + offsets, x, mask=mask)
```

实操：

- [x] 写 `vector_add`
- [x] 改成 `vector_mul`
- [x] 改成 `y = a * x + b`
- [x] 对比 PyTorch 速度

你需要能回答：

- 为什么要有 `mask`？
- `BLOCK_SIZE` 是什么？
- `tl.arange` 生成的是线程吗？
- Triton 的 program 和 CUDA block 有什么类似之处？

### 10:00 - 12:00：Memory、Stride、Coalescing

学习目标：

- 理解连续内存访问为什么快。
- 理解 Tensor stride。
- 理解 row-major / column-major。
- 会写处理二维 Tensor 的 kernel。

实操：

- [ ] 写 2D copy kernel
- [ ] 写 transpose kernel
- [ ] 写 row-wise add kernel

核心模式：

```python
pid_m = tl.program_id(0)
pid_n = tl.program_id(1)

offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

ptrs = x_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

x = tl.load(ptrs, mask=mask, other=0.0)
```

面试重点：

- 什么是 memory coalescing？
- stride 对性能有什么影响？
- transpose 为什么容易慢？
- 非连续 tensor 怎么处理？

### 13:30 - 15:00：Reduction 算子

学习目标：

- 掌握 block 内 reduction。
- 会用 `tl.sum`、`tl.max`。
- 理解 reduce axis。
- 会写 row-wise sum / max / mean。

实操：

- [ ] `row_sum`
- [ ] `row_max`
- [ ] `row_mean`
- [ ] 对比 `torch.sum(x, dim=1)`

核心模式：

```python
offs = tl.arange(0, BLOCK_N)
x = tl.load(
    x_ptr + row * stride_m + offs * stride_n,
    mask=offs < N,
    other=0.0,
)
s = tl.sum(x, axis=0)
tl.store(out_ptr + row, s)
```

注意：

- `BLOCK_N` 通常要是 2 的幂。
- 对 `max`，越界位置应该填 `-inf`。
- 对 `sum`，越界位置填 `0`。

你需要能回答：

- reduction 为什么比 elementwise 难？
- 为什么 softmax 需要先 max 再 exp？
- `other=0.0` 和 `other=-float("inf")` 分别适合什么场景？

### 15:00 - 17:30：Softmax 和 LayerNorm

学习目标：

- 理解 softmax 数值稳定性。
- 理解 LayerNorm 的 mean / variance。
- 学会融合多个步骤，减少 memory IO。

实操：

- [ ] 写 row-wise softmax
- [ ] 写 LayerNorm forward
- [ ] 对比 PyTorch 结果和速度
- [ ] 尝试不同 `BLOCK_SIZE`

Softmax 核心逻辑：

```python
x = tl.load(...)
x = x - tl.max(x, axis=0)
num = tl.exp(x)
den = tl.sum(num, axis=0)
y = num / den
tl.store(...)
```

LayerNorm 核心逻辑：

```python
mean = tl.sum(x, axis=0) / N
var = tl.sum((x - mean) * (x - mean), axis=0) / N
rstd = tl.rsqrt(var + eps)
y = (x - mean) * rstd * gamma + beta
```

面试重点：

- softmax 为什么要减 max？
- fusion 为什么能提升性能？
- LayerNorm 是 memory-bound 还是 compute-bound？
- `BLOCK_SIZE` 太大或太小会怎样？

### 19:30 - 21:30：Benchmark 与 Debug

学习目标：

- 会验证正确性。
- 会写 benchmark。
- 会看 latency / bandwidth。
- 了解 profiling 工具。

实操：

- [x] 用 `torch.testing.assert_close`
- [x] 用 `triton.testing.do_bench`
- [x] 计算 GB/s
- [x] 跑不同 shape
- [x] 记录性能表

示例：

```python
ms = triton.testing.do_bench(lambda: fn(x, y))
gbps = x.numel() * x.element_size() * 2 / ms / 1e6
```

Day 1 最终作品：

- [ ] `vector_add.py`
- [ ] `transpose.py`
- [ ] `row_reduction.py`
- [ ] `softmax.py`
- [ ] `layernorm.py`
- [ ] `benchmark.py`

---

## Day 2：Matmul、Fusion、优化与面试准备

### 09:00 - 11:30：Matmul Kernel

学习目标：

- 掌握 tiling。
- 理解 `BLOCK_M` / `BLOCK_N` / `BLOCK_K`。
- 理解 accumulator。
- 理解 fp32 accumulation。
- 会写基础 matmul。

核心模式：

```python
pid = tl.program_id(0)

pid_m = pid // num_pid_n
pid_n = pid % num_pid_n

offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
offs_k = tl.arange(0, BLOCK_K)

a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

for k in range(0, K, BLOCK_K):
    a = tl.load(a_ptrs, mask=...)
    b = tl.load(b_ptrs, mask=...)
    acc += tl.dot(a, b)
```

实操：

- [ ] 写 naive matmul
- [ ] 支持任意 M / N / K
- [ ] 支持 fp16 输入、fp32 累加
- [ ] 对比 `torch.matmul`
- [ ] 尝试 `16x16x32`
- [ ] 尝试 `32x32x32`
- [ ] 尝试 `32x64x32`
- [ ] 尝试 `64x64x32`

面试重点：

- matmul 为什么要 tiling？
- `tl.dot` 做了什么？
- fp16 为什么要 fp32 accumulate？
- `BLOCK_M` / `BLOCK_N` / `BLOCK_K` 怎么选？

### 11:30 - 12:30：优化 Matmul

学习目标：

- 理解 grouped ordering。
- 理解 L2 cache locality。
- 理解 `num_warps` / `num_stages`。
- 初步掌握 autotune。

重点概念：

- grouped program ordering 可以提高 A/B tile 复用。
- `num_warps` 影响并行度和调度。
- `num_stages` 影响 pipeline。
- autotune 根据 shape 选择配置。

实操：

- [ ] 给 matmul 加 grouped ordering
- [ ] 使用 `@triton.autotune`
- [ ] benchmark 不同配置

你需要能讲清：

- 为什么 grouped ordering 可以提高 cache 命中？
- `num_warps=4` 和 `num_warps=8` 怎么选？
- autotune 的 key 应该选哪些维度？

### 13:30 - 15:00：Fused Operators

学习目标：

- 理解为什么算子融合重要。
- 能写 fused bias + activation。
- 能写 fused matmul + bias + ReLU/GELU。

实操：

- [ ] `matmul + bias`
- [ ] `matmul + bias + relu`
- [ ] `matmul + bias + gelu`
- [ ] 对比 PyTorch 分开执行
- [ ] 对比 PyTorch `torch.compile`
- [ ] 对比 Triton fused kernel

核心思路：

```python
acc = tl.dot(a, b)
bias = tl.load(bias_ptr + offs_n)
acc = acc + bias
acc = tl.maximum(acc, 0.0)
tl.store(c_ptrs, acc, mask=mask)
```

面试重点：

- fusion 省了什么？
- fusion 会不会总是更快？
- 为什么复杂 fusion 可能降低 occupancy？
- activation 放在 matmul 后面有什么性能优势？

### 15:00 - 16:30：真实场景算子

选择 2 个重点做。

#### 任务 A：RMSNorm

适合 LLM 面试。

公式：

```python
rms = sqrt(mean(x^2) + eps)
y = x / rms * weight
```

需要掌握：

- reduction
- broadcast
- fp32 accumulate
- memory-bound 优化

#### 任务 B：Embedding Lookup

适合推荐系统 / NLP。

需要掌握：

- gather
- 不规则访存
- cache locality
- index dtype

#### 任务 C：TopK / Argmax

适合算法和推理优化。

需要掌握：

- block 内比较
- index tracking
- mask
- tie-breaking

#### 任务 D：Quantization

适合推理优化。

做 int8 quant / dequant：

```python
q = clamp(round(x / scale), -128, 127)
x = q * scale
```

### 16:30 - 18:00：Profiling 与性能分析

需要会的工具：

- `triton.testing.do_bench`
- `torch.cuda.Event`
- Nsight Systems：看 kernel launch 和 timeline
- Nsight Compute：看 memory throughput、occupancy、SM utilization

面试回答模板：

> 我会先判断这个算子是 memory-bound 还是 compute-bound。
> 如果是 memory-bound，我会关注访存是否连续、是否可以 fusion、是否减少读写次数。
> 如果是 compute-bound，我会关注 tiling、tensor core 使用、occupancy、register pressure、并行粒度和数据复用。

整理性能分析表：

| 算子 | 类型 | 主要瓶颈 | 优化方向 |
| --- | --- | --- | --- |
| Vector Add | memory-bound | global memory bandwidth | coalescing / fusion |
| Softmax | memory-bound + reduction | 多次读写、exp | fusion / block reduction |
| LayerNorm | memory-bound | reduce + broadcast | one-pass / vectorized |
| Matmul | compute-bound | tensor core utilization | tiling / dot / cache |
| TopK | memory + control | compare + irregular access | block select |

### 19:30 - 22:00：面试项目整理

最终产出一个 GitHub repo：

```text
triton-kernels-lab/
  README.md
  kernels/
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
    gpu_concepts.md
    triton_interview.md
```

README 需要包含：

- 每个 kernel 的功能。
- 正确性验证。
- benchmark 结果。
- 和 PyTorch 对比。
- 优化思路。
- 遇到的问题。

面试项目介绍模板：

> 我系统实现了 elementwise、reduction、normalization、matmul 和 fused operator。
> 对每个算子我都做了 PyTorch correctness check 和 latency benchmark。
> 在优化上，我重点关注 memory coalescing、block tiling、mask 边界处理、fp32 accumulation、fusion 减少 global memory IO，以及 matmul 中的 grouped ordering 和 autotune。

---

## 必背面试问题

### Triton 基础

- Triton 和 CUDA 的区别是什么？
- `tl.program_id` 是什么？
- `BLOCK_SIZE` 是线程数吗？
- 为什么需要 mask？
- `tl.load` 的 `other` 参数有什么用？
- 什么是 stride？
- Triton 如何处理非连续 tensor？

### GPU 性能

- 什么是 memory coalescing？
- 什么是 occupancy？
- 什么是 register pressure？
- shared memory / SRAM 的作用是什么？
- memory-bound 和 compute-bound 怎么判断？
- 如何估算 bandwidth？
- kernel fusion 为什么有效？

### 算子开发

- softmax 为什么需要数值稳定？
- LayerNorm 和 RMSNorm 的区别？
- reduction kernel 怎么写？
- matmul 为什么要 tiling？
- Tensor Core 如何被使用？
- fp16 输入为什么经常 fp32 累加？
- grouped ordering 为什么能优化 matmul？
- autotune 怎么设计 config？

---

## 推荐学习资料

按优先级：

1. Triton 官方 tutorial：重点看 vector add、softmax、matmul、LayerNorm。
2. OpenAI Triton language docs：重点看 `tl.load`、`tl.store`、`tl.arange`、`tl.dot`、`tl.sum`、`@triton.jit`、`@triton.autotune`。
3. PyTorch 自定义算子和 `torch.compile`：理解 Triton 在 PyTorch 生态里的位置。
4. CUDA 编程基础：先掌握 thread block、warp、shared memory、global memory、occupancy。

---

## 最小实操顺序

如果时间很紧，按这个顺序做：

- [ ] vector add
- [ ] 2D copy / transpose
- [ ] row sum
- [ ] softmax
- [ ] LayerNorm
- [ ] matmul
- [ ] fused matmul bias relu
- [ ] RMSNorm
- [ ] benchmark + README

这条路线已经覆盖大部分 Triton 面试核心能力。
