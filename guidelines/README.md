# Triton Guidelines

这组材料是 `PLAND.md` 的展开版，目标是让你从环境安装开始，逐步完成一套可面试、可展示、可复盘的 Triton 算子开发项目。

建议顺序：

1. [00_environment_setup.md](00_environment_setup.md)：安装 Python、PyTorch、Triton，验证 GPU 环境。
2. [01_project_structure.md](01_project_structure.md)：创建项目目录、测试目录、benchmark 目录。
3. [02_triton_basics.md](02_triton_basics.md)：理解 Triton 编程模型。
4. [03_elementwise_kernels.md](03_elementwise_kernels.md)：实现 vector add / mul / affine。
5. [04_memory_stride_2d.md](04_memory_stride_2d.md)：实现 2D copy、transpose、row-wise add。
6. [05_reduction_softmax_layernorm.md](05_reduction_softmax_layernorm.md)：实现 reduction、softmax、LayerNorm。
7. [06_matmul_and_fusion.md](06_matmul_and_fusion.md)：实现 matmul、grouped ordering、autotune、fused matmul。
8. [07_real_world_ops.md](07_real_world_ops.md)：实现 RMSNorm、quantization、TopK / Argmax 方向练习。
9. [08_benchmark_debug_profiling.md](08_benchmark_debug_profiling.md)：正确性、性能测试、profiling。
10. [09_interview_review.md](09_interview_review.md)：面试复盘、回答模板、项目讲法。

每个 guideline 的使用方法：

- 先读目标和产出。
- 再照着步骤创建或修改代码。
- 每完成一个 kernel，必须做 correctness check。
- 每完成一个模块，必须 benchmark 至少 3 个 shape。
- 每天结束后，把结果写进 `README.md` 或 `notes/`。

最终你应该得到：

```text
triton-kernels-lab/
  README.md
  kernels/
  tests/
  benchmarks/
  notes/
```

官方资料参考：

- Triton installation: https://triton-lang.org/main/getting-started/installation.html
- Triton tutorials: https://triton-lang.org/main/getting-started/tutorials.html
- Triton language API: https://triton-lang.org/main/python-api/triton.language.html
- PyTorch local install: https://docs.pytorch.org/get-started/locally/
