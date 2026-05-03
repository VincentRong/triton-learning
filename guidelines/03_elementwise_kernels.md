# 03 Elementwise Kernels

目标：实现最基础的 Triton elementwise 算子，并建立“写 kernel、封装 API、验证正确性、benchmark”的完整闭环。

## 1. 要实现的文件

创建：

```text
triton-kernels-lab/kernels/vector_add.py
triton-kernels-lab/tests/test_correctness.py
triton-kernels-lab/benchmarks/bench_all.py
```

## 2. Vector Add

目标：

```python
y = a + b
```

Triton kernel 结构：

```python
@triton.jit
def _vector_add_kernel(a_ptr, b_ptr, y_ptr, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    y = a + b

    tl.store(y_ptr + offsets, y, mask=mask)
```

Python wrapper 要做：

- 检查输入都在 CUDA。
- 检查 shape 一致。
- 分配输出。
- 计算 grid。
- launch kernel。

wrapper 结构：

```python
def vector_add(a: torch.Tensor, b: torch.Tensor, block_size: int = 1024):
    assert a.is_cuda and b.is_cuda
    assert a.shape == b.shape
    y = torch.empty_like(a)
    n = a.numel()
    grid = (triton.cdiv(n, block_size),)
    _vector_add_kernel[grid](a, b, y, n, BLOCK_SIZE=block_size)
    return y
```

## 3. 扩展练习

按顺序实现：

- `vector_mul(a, b) -> a * b`
- `vector_affine(x, scale, bias) -> x * scale + bias`
- `silu(x) -> x * sigmoid(x)`
- `relu(x) -> max(x, 0)`

每个练习只改一小部分，这样你会非常熟悉 elementwise 模式。

## 4. Correctness

测试 shape：

```python
shapes = [
    (1024,),
    (1000,),
    (1_000_000,),
    (1024, 1024),
]
```

注意：

- `numel()` 会把多维 tensor 当成一维线性内存处理。
- 这个阶段先要求 contiguous 输入。

测试模板：

```python
def test_vector_add():
    a = torch.randn((1000,), device="cuda")
    b = torch.randn((1000,), device="cuda")
    y = vector_add(a, b)
    torch.testing.assert_close(y, a + b)
```

## 5. Benchmark

使用 `triton.testing.do_bench`：

```python
ms = triton.testing.do_bench(lambda: vector_add(a, b))
```

带宽估算：

```python
bytes_moved = a.numel() * a.element_size() * 3
gbps = bytes_moved / (ms * 1e-3) / 1e9
```

vector add 读两个 tensor，写一个 tensor，所以是 3 倍元素大小。

## 6. 观察点

尝试不同 `BLOCK_SIZE`：

- 256
- 512
- 1024
- 2048

记录：

- latency 是否明显变化
- GB/s 是否稳定
- 小 shape 是否被 kernel launch overhead 主导

## 7. 面试表达

> Vector add 是典型 memory-bound 算子。每个元素只做一次加法，但需要从 global memory 读两个值并写一个值，计算强度很低。优化重点不是提高算术吞吐，而是保证连续访存、减少不必要读写、通过 fusion 把多个 elementwise 合成一个 kernel。

## 8. 完成标准

- [ ] `vector_add` 正确。
- [ ] `vector_mul` 正确。
- [ ] `vector_affine` 正确。
- [ ] 至少记录 4 个 block size 的 benchmark。
- [ ] 能解释为什么 elementwise 通常是 memory-bound。
