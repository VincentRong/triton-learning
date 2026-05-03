# 04 Memory、Stride 与二维 Tensor

目标：从一维 elementwise 过渡到二维 tile，掌握 pointer arithmetic、stride、mask 和 transpose 访存特点。

## 1. 要实现的文件

```text
triton-kernels-lab/kernels/transpose.py
```

可以在同一个文件中放：

- `copy_2d`
- `transpose`
- `rowwise_add`

## 2. 二维地址计算

二维 Tensor 的通用地址：

```python
ptr + row * stride_0 + col * stride_1
```

Triton tile：

```python
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
ptrs = x_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
```

这里 `offs_m[:, None]` 和 `offs_n[None, :]` 会广播成一个 `BLOCK_M x BLOCK_N` 的地址矩阵。

## 3. 2D Copy

目标：

```python
y = x.clone()
```

步骤：

1. Python wrapper 接收 `x`。
2. 创建 `y = torch.empty_like(x)`。
3. 从 `x.stride()` 和 `y.stride()` 取 stride。
4. kernel 中 load `x` tile。
5. store 到 `y` tile。

测试：

- contiguous tensor
- sliced 或 transposed tensor 可以作为进阶测试

## 4. Transpose

目标：

```python
y = x.T
```

输入 shape：`M x N`

输出 shape：`N x M`

load 地址：

```python
x_ptr + m * stride_xm + n * stride_xn
```

store 地址：

```python
y_ptr + n * stride_yn + m * stride_ym
```

关键点：

- load 和 store 至少有一侧访问模式不够理想。
- transpose 的性能经常受访存模式影响。
- 后续可以比较 `BLOCK_M=16, BLOCK_N=16` 和 `32x32`。

## 5. Row-wise Add

目标：

```python
y[m, n] = x[m, n] + bias[n]
```

这是后面 `matmul + bias` 的前置练习。

kernel：

```python
x = tl.load(x_ptrs, mask=mask, other=0.0)
b = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
y = x + b[None, :]
tl.store(y_ptrs, y, mask=mask)
```

## 6. 测试 shape

```python
shapes = [
    (32, 32),
    (128, 256),
    (1000, 1024),
    (1023, 777),
]
```

重点测非整除 shape，例如 `(1023, 777)`。

## 7. 性能观察

记录：

- 2D copy 的 GB/s
- transpose 的 GB/s
- row-wise add 的 GB/s

思考：

- 为什么 copy 通常比 transpose 快？
- row-wise add 比纯 copy 多读一个 bias，但 bias 会被多行复用，这对 cache 有什么影响？

## 8. 面试表达

> 二维算子最重要的是地址计算和访存连续性。Triton 里我会用 `offs_m[:, None]` 和 `offs_n[None, :]` 构造 tile 地址，再用 mask 处理边界。对于非连续 tensor，必须传入 stride，不能假设数据是 contiguous。transpose 的难点在于读写方向会改变，容易导致一侧访存不连续。

## 9. 完成标准

- [ ] `copy_2d` 正确。
- [ ] `transpose` 正确。
- [ ] `rowwise_add` 正确。
- [ ] 支持非整除 shape。
- [ ] 能解释 stride 和 coalescing。
