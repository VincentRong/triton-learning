# 09 面试复盘与项目讲法

目标：把两天实操沉淀成能在面试中讲清楚的知识结构和项目表达。

## 1. 一分钟项目介绍

模板：

> 我做了一个 Triton kernels lab，系统实现了 elementwise、2D memory、reduction、softmax、LayerNorm、matmul、fused matmul、RMSNorm 和 quantization 等算子。每个算子都有 PyTorch reference 做 correctness check，并用 `triton.testing.do_bench` 做 latency benchmark。优化上我重点关注连续访存、mask 边界处理、stride 支持、fp32 accumulation、fusion 减少 global memory IO、matmul tiling、grouped ordering 和 autotune。

## 2. 面试官追问路线

### 基础路线

- Triton 和 CUDA 有什么区别？
- `program_id` 是什么？
- `BLOCK_SIZE` 是线程数吗？
- 为什么需要 mask？
- `tl.load(..., other=...)` 是做什么的？

回答重点：

- Triton 是 block-level 向量化模型。
- 一个 program 处理一个 tile。
- `tl.arange` 构造 tile 内 offsets。
- mask 处理边界，防止越界读写。

### 性能路线

- 什么是 memory-bound？
- 什么是 compute-bound？
- 怎么估算 bandwidth？
- kernel fusion 为什么能提速？
- fusion 会不会总是更快？

回答重点：

- memory-bound 看 global memory 读写。
- compute-bound 看 FLOPs 和硬件算力利用。
- fusion 减少中间 tensor 的读写和 launch overhead。
- fusion 过度可能增加 register pressure，降低 occupancy。

### Matmul 路线

- matmul 为什么要 tiling？
- `tl.dot` 做了什么？
- fp16 为什么 fp32 accumulate？
- grouped ordering 为什么有用？
- autotune key 怎么选？

回答重点：

- tiling 提高数据复用。
- `tl.dot` 映射到底层高效矩阵乘累加能力。
- fp32 accumulate 提高数值稳定性。
- grouped ordering 改善 L2 cache locality。
- autotune key 选择影响最佳配置的 shape，例如 M/N/K。

## 3. 必背概念卡片

### Mask

> mask 用来处理 tile 边界。当 shape 不是 block size 整数倍时，最后一个 program 会生成越界 offsets。load mask 配合 `other` 可以给越界位置填安全值，store mask 防止越界写。

### Stride

> stride 描述 tensor 每个维度移动一格时底层 storage 的偏移。支持 stride 可以让 kernel 处理非 contiguous tensor。二维地址一般是 `base + row * stride_0 + col * stride_1`。

### Coalescing

> coalescing 指相邻执行单元访问连续或规则内存，使 global memory transaction 更高效。连续访问通常更快，transpose、gather 等不规则访问容易性能下降。

### Occupancy

> occupancy 描述一个 SM 上活跃 warp 数量相对于最大可支持数量的比例。它受 register、shared memory、block size 等影响。高 occupancy 有助于隐藏 latency，但不是越高越一定快。

### Register Pressure

> 一个 kernel 中每个 program 或 thread 使用太多寄存器，会限制并发度，降低 occupancy，甚至导致 spill 到 local memory。复杂 fusion 可能因为 register pressure 变慢。

### Fusion

> fusion 把多个小算子合成一个 kernel，减少 kernel launch 和中间结果读写。它对 memory-bound 算子特别有效，但如果融合后计算太复杂、寄存器太多，也可能降低性能。

## 4. 算子分类表

| 算子 | 类型 | 主要瓶颈 | 优化方向 |
| --- | --- | --- | --- |
| Vector Add | memory-bound | global memory bandwidth | coalescing、fusion |
| 2D Copy | memory-bound | memory bandwidth | contiguous access |
| Transpose | memory-bound | 非连续读写 | tile、访问模式 |
| Softmax | reduction + memory | max/sum/exp、多次访问 | row-wise fusion、mask |
| LayerNorm | reduction + memory | mean/var、读写多 | fp32 accumulate、fusion |
| Matmul | compute-bound | tensor core 利用 | tiling、dot、autotune |
| RMSNorm | memory-bound | reduction + broadcast | fp32 accumulate、融合 |
| Quantize | memory-bound | 读写和类型转换 | fusion、vectorized load |

## 5. 项目 README 必写内容

环境：

- GPU 型号
- torch 版本
- triton 版本
- CUDA 是否可用

正确性：

- 每个 kernel 的 reference。
- 测试 shape。
- dtype。
- 容忍度。

性能：

- latency 表。
- speedup 表。
- GB/s 或 TFLOPS。

复盘：

- 哪些算子 Triton 更快。
- 哪些算子 PyTorch 更快。
- 原因分析。
- 下一步优化方向。

## 6. 高频问题短答

### `BLOCK_SIZE` 是线程数吗？

不是。它是一个 program 内处理的元素数量或 tile 大小。Triton 编译器会把这个向量化计算映射到底层 GPU 执行。

### softmax 为什么要减 max？

为了数值稳定，避免 `exp(x)` 溢出。softmax 对输入整体平移不变，所以减去每行最大值不改变结果。

### LayerNorm 是 memory-bound 还是 compute-bound？

通常偏 memory-bound。它对每个元素的计算量不高，但需要读 x、读 gamma/beta、写 y，并做 reduction 和 broadcast。

### matmul 为什么通常是 compute-bound？

因为 `2*M*N*K` 的 FLOPs 很高，数据复用好时计算量相对于内存读写更大。优化重点是 tile、tensor core 利用率、pipeline 和 cache locality。

### fusion 为什么不总是更快？

fusion 会减少 global memory IO 和 launch overhead，但也可能增加寄存器使用、降低 occupancy、减少编译器优化空间。因此需要 benchmark 验证。

## 7. 最终检查清单

- [ ] 能从零解释 Triton program 模型。
- [ ] 能手写 elementwise kernel。
- [ ] 能手写二维 stride 地址计算。
- [ ] 能手写 row-wise reduction。
- [ ] 能讲 softmax 数值稳定。
- [ ] 能讲 LayerNorm / RMSNorm 区别。
- [ ] 能讲 matmul tiling。
- [ ] 能解释 fusion 的收益和风险。
- [ ] 能读 benchmark 表并判断瓶颈。
- [ ] 能用项目经历回答“你优化过什么 GPU 算子”。
