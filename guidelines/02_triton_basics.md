# 02 Triton 编程模型基础

目标：理解 Triton kernel 的基本单位、指针计算、mask、block、grid，以及它和 CUDA block/thread 的关系。

## 1. Triton 的核心心智模型

Triton 不是让你直接写每个 thread 做什么，而是让你写“一个 program 处理一块数据”。

可以这样理解：

- 一个 kernel launch 启动很多 program。
- 每个 program 有自己的 `program_id`。
- 每个 program 内部用向量化方式处理一组 offsets。
- `tl.arange(0, BLOCK_SIZE)` 生成 block 内的一组逻辑位置。
- `tl.load` 和 `tl.store` 对一组地址执行加载和写回。

最小模式：

```python
@triton.jit
def _kernel(x_ptr, y_ptr, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(y_ptr + offsets, x, mask=mask)
```

## 2. `tl.constexpr`

`tl.constexpr` 表示编译期常量。常见用途：

- `BLOCK_SIZE`
- `BLOCK_M`
- `BLOCK_N`
- `BLOCK_K`
- 布尔开关，例如是否使用某个 activation

为什么 block size 要做编译期常量：

- 编译器需要知道向量大小。
- 编译器才能做展开和优化。
- 不同 block size 会编译成不同 kernel 版本。

## 3. Grid

Python wrapper 中这样启动 kernel：

```python
grid = (triton.cdiv(n, BLOCK_SIZE),)
_kernel[grid](x, y, n, BLOCK_SIZE=BLOCK_SIZE)
```

`triton.cdiv(a, b)` 是向上取整除法：

```python
ceil(a / b)
```

二维 grid：

```python
grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
```

kernel 内：

```python
pid_m = tl.program_id(0)
pid_n = tl.program_id(1)
```

## 4. Mask

几乎所有 Triton kernel 都要处理边界。

例如 `n = 1000`，`BLOCK_SIZE = 1024`，最后一个 program 会访问 1000 到 1023 的越界位置。mask 用来保护这些位置：

```python
mask = offsets < n
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
tl.store(y_ptr + offsets, x, mask=mask)
```

`other` 的选择：

- sum：越界填 `0.0`
- max：越界填 `-float("inf")`
- min：越界填 `float("inf")`
- softmax：max 阶段填 `-inf`，exp/sum 阶段通常基于 masked 后结果

## 5. Stride

Tensor 不一定是连续内存。二维 Tensor 元素地址一般是：

```python
base + row * stride_0 + col * stride_1
```

PyTorch 里查看：

```python
x.shape
x.stride()
x.is_contiguous()
```

Triton 里不要假设输入永远 contiguous。学习阶段可以先写 contiguous 版本，再升级支持 stride。

## 6. Triton 和 CUDA 的关系

面试表达：

> CUDA 通常让开发者直接控制 thread、block、shared memory 等细节。Triton 提供更高层的 block-level 向量化编程模型，我描述一个 program 如何处理一个 tile，Triton 编译器负责映射到底层 GPU 执行。它更适合快速写深度学习算子，尤其是 elementwise、reduction、matmul 和 fused operator。

## 7. 必须掌握的 API

- `tl.program_id`
- `tl.arange`
- `tl.load`
- `tl.store`
- `tl.sum`
- `tl.max`
- `tl.dot`
- `tl.exp`
- `tl.sqrt`
- `tl.rsqrt`
- `tl.maximum`
- `tl.minimum`

## 8. 完成标准

- [ ] 能写出一个 copy kernel。
- [ ] 能解释 `program_id` 和 `grid`。
- [ ] 能解释 `tl.arange` 不是直接等于 CUDA thread id。
- [ ] 能解释为什么 mask 必须存在。
- [ ] 能根据 shape 和 block size 算出 program 数量。
