# 07 真实场景算子

目标：选择 2 到 3 个真实算子做专项练习，让项目从 tutorial 水平升级到面试可讲的工程作品。

## 1. RMSNorm

适合 LLM 面试。

公式：

```python
rms = sqrt(mean(x^2) + eps)
y = x / rms * weight
```

输入：

```python
x: [M, N]
weight: [N]
y: [M, N]
```

实现步骤：

1. 一行一个 program。
2. load `x` 一行。
3. 转 fp32。
4. 计算 `mean(x * x)`。
5. 计算 `rstd = rsqrt(mean_square + eps)`。
6. load `weight`。
7. store `x * rstd * weight`。

重点：

- 和 LayerNorm 不同，RMSNorm 不减 mean。
- LLM 中非常常见。
- 通常是 memory-bound。

测试对照：

```python
ref = x * torch.rsqrt(torch.mean(x.float() * x.float(), dim=-1, keepdim=True) + eps) * weight
```

## 2. Quantization

适合推理优化面试。

Per-tensor int8 quant：

```python
q = clamp(round(x / scale), -128, 127).to(int8)
```

Dequant：

```python
y = q.float() * scale
```

实现方向：

- `quantize.py`
- 支持 fp32/fp16 输入。
- 输出 int8。
- 支持单个 scale。

进一步扩展：

- per-row scale
- per-channel scale
- symmetric quant
- dequant + matmul fusion

面试表达：

> quantization 的核心是用更低 bit 表示数值，减少显存带宽和存储，并利用低精度计算加速推理。Triton 中常见做法是把 quant/dequant 和后续算子融合，避免中间结果反复读写 global memory。

## 3. Argmax / TopK

适合 reduction 和索引类面试。

Argmax 目标：

```python
values, indices = torch.max(x, dim=-1)
```

实现步骤：

1. 一行一个 program。
2. load 一行。
3. 越界填 `-inf`。
4. 用 `tl.max` 得到 max value。
5. 比较 `x == max_value` 得到位置。
6. 处理 tie-breaking，选择最小 index。

TopK 比 Argmax 难很多。初学阶段先做 Argmax，再做 TopK 的概念设计即可。

## 4. Embedding Lookup

适合推荐系统 / NLP。

目标：

```python
out[i, :] = table[indices[i], :]
```

特点：

- gather 访存。
- indices 不规则。
- cache locality 影响很大。
- 内存带宽和随机访问是重点。

实现建议：

- 一个 program 处理一个或多个 token 的 embedding vector。
- `BLOCK_D` 覆盖 embedding dimension。
- 支持 int64 / int32 indices。

## 5. 推荐选择

如果只有 2 小时：

- RMSNorm
- Quantization

如果有 4 小时：

- RMSNorm
- Quantization
- Argmax

如果做项目展示：

- RMSNorm 作为 LLM 算子。
- fused matmul 作为高性能算子。
- quantization 作为推理优化方向。

## 6. 完成标准

- [ ] 至少完成 RMSNorm。
- [ ] 至少完成 Quantization 或 Argmax 之一。
- [ ] 每个算子都有 PyTorch reference。
- [ ] 每个算子都有 benchmark。
- [ ] README 里能说明真实场景用途。
