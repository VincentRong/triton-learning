# 05 Reduction、Softmax 与 LayerNorm

目标：掌握 row-wise reduction，并用它实现 softmax 和 LayerNorm。这是 Triton 面试中最关键的一组基础算子。

## 1. 要实现的文件

```text
triton-kernels-lab/kernels/reductions.py
triton-kernels-lab/kernels/softmax.py
triton-kernels-lab/kernels/layernorm.py
```

## 2. Row-wise Sum

目标：

```python
out[m] = sum(x[m, :])
```

适合输入：`M x N`

一个 program 处理一行：

```python
row = tl.program_id(0)
offs = tl.arange(0, BLOCK_N)
mask = offs < N
x = tl.load(x_ptr + row * stride_m + offs * stride_n, mask=mask, other=0.0)
s = tl.sum(x, axis=0)
tl.store(out_ptr + row, s)
```

注意：

- `BLOCK_N` 应该大于等于 `N`，并且常用 2 的幂。
- 这是简单版本，适合 `N` 不太大的一行一个 program。
- 如果 `N` 很大，需要多 program 分块 reduction，这是进阶主题。

## 3. Row-wise Max

目标：

```python
out[m] = max(x[m, :])
```

越界位置必须填 `-inf`：

```python
x = tl.load(ptrs, mask=mask, other=-float("inf"))
m = tl.max(x, axis=0)
```

## 4. Row-wise Mean

目标：

```python
out[m] = mean(x[m, :])
```

实现：

```python
s = tl.sum(x, axis=0)
mean = s / N
```

如果输入是 fp16，建议转换成 fp32 累加：

```python
x = x.to(tl.float32)
```

## 5. Softmax

目标：

```python
y = torch.softmax(x, dim=-1)
```

数值稳定版本：

```python
x = tl.load(ptrs, mask=mask, other=-float("inf"))
x = x - tl.max(x, axis=0)
num = tl.exp(x)
den = tl.sum(num, axis=0)
y = num / den
tl.store(out_ptrs, y, mask=mask)
```

为什么要减 max：

- 防止 `exp(x)` 溢出。
- softmax 对整体平移不变。
- `softmax(x) == softmax(x - max(x))`。

## 6. LayerNorm

目标：

```python
y = (x - mean) / sqrt(var + eps) * gamma + beta
```

一行一个 program：

```python
x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
mean = tl.sum(x, axis=0) / N
diff = x - mean
var = tl.sum(diff * diff, axis=0) / N
rstd = tl.rsqrt(var + eps)
gamma = tl.load(gamma_ptr + offs, mask=mask, other=0.0)
beta = tl.load(beta_ptr + offs, mask=mask, other=0.0)
y = diff * rstd * gamma + beta
tl.store(out_ptrs, y, mask=mask)
```

注意：

- `mean` 和 `var` 建议 fp32 计算。
- `eps` 通常是 `1e-5`。
- 输出 dtype 可以跟输入一致。

## 7. 测试 shape

Reduction：

```python
[(4, 16), (128, 1024), (1023, 777)]
```

Softmax：

```python
[(16, 128), (128, 1024), (512, 2048)]
```

LayerNorm：

```python
[(32, 768), (32, 1024), (16, 4096)]
```

## 8. Benchmark 观察

记录：

- softmax vs `torch.softmax`
- layernorm vs `torch.nn.functional.layer_norm`
- fp32 和 fp16 的差异
- 不同 `BLOCK_N` 的延迟

## 9. 常见错误

- max reduction 越界位置用了 `0.0`，导致负数输入结果错误。
- 忘记 mask store，越界写内存。
- `BLOCK_N < N`，导致只处理一部分列。
- LayerNorm 使用 fp16 直接累加，误差变大。
- softmax 没有减 max，大值输入出现 `inf` 或 `nan`。

## 10. 面试表达

> Softmax 和 LayerNorm 都是典型 reduction + broadcast 的算子。优化重点是把一行数据尽量放在一个 program 内完成，减少 global memory 往返。softmax 需要 max、exp、sum、除法，LayerNorm 需要 mean 和 variance，二者通常偏 memory-bound，但也会受到 exp、sqrt 等特殊函数开销影响。

## 11. 完成标准

- [ ] row sum 正确。
- [ ] row max 正确。
- [ ] row mean 正确。
- [ ] softmax 正确且数值稳定。
- [ ] LayerNorm 正确。
- [ ] 能讲清 `other=0.0` 和 `other=-inf` 的区别。
