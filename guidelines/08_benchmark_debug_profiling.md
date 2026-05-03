# 08 Benchmark、Debug 与 Profiling

目标：建立算子开发的验证闭环，能判断正确性、速度、瓶颈，并能用标准语言描述优化方向。

## 1. Correctness 第一

每个 kernel 写完先测 correctness，不要先追性能。

基础模板：

```python
torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
```

常见容忍度：

- fp32 elementwise：`rtol=1e-5, atol=1e-5`
- fp16 elementwise：`rtol=1e-3, atol=1e-3`
- fp16 matmul：`rtol=1e-2, atol=1e-2`
- norm 类算子：根据 shape 和 dtype 调整

## 2. Benchmark 模板

使用 `triton.testing.do_bench`：

```python
import triton

ms = triton.testing.do_bench(lambda: fn(*args))
```

对比 PyTorch：

```python
torch_ms = triton.testing.do_bench(lambda: torch_fn(*args))
triton_ms = triton.testing.do_bench(lambda: triton_fn(*args))
speedup = torch_ms / triton_ms
```

## 3. Bandwidth 估算

适合 memory-bound 算子。

Vector add：

```python
bytes_moved = n * element_size * 3
gbps = bytes_moved / (ms * 1e-3) / 1e9
```

LayerNorm 粗略估算：

- 读 x
- 读 gamma
- 读 beta
- 写 y
- 如果中间没有写回 global memory，就只算必要 global memory

注意：复杂算子的 bytes moved 是估算，不要把估算说成绝对真实。

## 4. TFLOPS 估算

适合 matmul：

```python
flops = 2 * M * N * K
tflops = flops / (ms * 1e-3) / 1e12
```

如果是 fused matmul + activation，activation 的 FLOPs 可忽略或单独说明，因为 matmul 占主导。

## 5. Debug 方法

优先级：

1. 先用很小 shape，例如 `(4, 8)`。
2. 打印 PyTorch reference 和 Triton output。
3. 测非整除 shape，暴露 mask 问题。
4. 测全负数，暴露 max 的 `other=0.0` 错误。
5. 测非 contiguous tensor，暴露 stride 假设。
6. 把 BLOCK_SIZE 缩小，方便定位。

## 6. 常见 bug 对照

| 现象 | 可能原因 |
| --- | --- |
| 最后一段结果错误 | mask 错了 |
| 只有整除 shape 正确 | 边界处理缺失 |
| softmax 出现 nan | 没有减 max 或 exp 溢出 |
| max 对负数输入错误 | 越界填了 0 |
| matmul 结果转置 | B stride 或 store 地址写反 |
| 小 shape Triton 比 PyTorch 慢 | launch overhead 主导 |
| benchmark 波动大 | 没 warmup、机器有其他任务、shape 太小 |

## 7. Nsight Systems

看整体 timeline：

```bash
nsys profile -o reports/run python benchmarks/bench_all.py
```

关注：

- kernel launch 数量
- kernel 时间占比
- 是否有多余 CPU/GPU 同步
- PyTorch 分开执行是否产生多个 kernel
- fused Triton 是否减少 launch

## 8. Nsight Compute

看单个 kernel 的深度性能指标：

```bash
ncu python benchmarks/bench_all.py
```

关注：

- memory throughput
- achieved occupancy
- register usage
- SM utilization
- tensor core utilization
- L2 hit rate

## 9. 判断瓶颈的回答模板

> 我会先估算这个算子的 arithmetic intensity，也就是每读取多少字节做多少计算。如果计算很少、读写很多，通常是 memory-bound，比如 vector add、LayerNorm、RMSNorm。如果是 matmul，FLOPs 很高，通常更接近 compute-bound，需要关注 tiling、tensor core、pipeline、occupancy 和数据复用。

## 10. 完成标准

- [ ] 每个 kernel 都有 correctness test。
- [ ] 每个 kernel 至少 benchmark 3 个 shape。
- [ ] memory-bound 算子能估算 GB/s。
- [ ] matmul 能估算 TFLOPS。
- [ ] 能用 Nsight Systems 看 timeline。
- [ ] 知道 Nsight Compute 里要看哪些指标。
