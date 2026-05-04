# Triton Kernels Lab

Learning project for implementing, testing, benchmarking, and explaining Triton kernels.

## Environment

See `notes/env.md`.

## Kernels

- `kernels/vector_add.py`
  - `vector_add(a, b) -> a + b`
  - `vector_mul(a, b) -> a * b`
  - `vector_affine(x, scale, bias) -> x * scale + bias`
- `kernels/transpose.py`
  - `copy_2d(x) -> x.clone()`
  - `transpose(x) -> x.T.contiguous()`
  - `rowwise_add(x, bias) -> x + bias`
- `kernels/reductions.py`
  - `row_sum(x)`
  - `row_max(x)`
  - `row_mean(x)`
- `kernels/softmax.py`
  - `softmax(x)`
- `kernels/layernorm.py`
  - `layernorm(x, gamma, beta)`

## Correctness

Run tests with:

```bash
/home/rwenxiao/pytorch-cuda/bin/python -m pytest -q
```

Current result:

```text
47 passed in 3.25s
```

## Benchmarks

Run benchmarks with:

```bash
/home/rwenxiao/pytorch-cuda/bin/python -m benchmarks.bench_all
```

Current `vector_add` result on RTX 3070, shape `(1024, 1024)`, dtype `float32`:

| block size | Triton ms | PyTorch ms | speedup | GB/s |
| --- | ---: | ---: | ---: | ---: |
| 256 | 0.0371 | 0.0355 | 0.96 | 339.21 |
| 512 | 0.0364 | 0.0360 | 0.99 | 345.74 |
| 1024 | 0.0358 | 0.0358 | 1.00 | 351.23 |
| 2048 | 0.0356 | 0.0358 | 1.00 | 353.16 |

Current 2D op results on RTX 3070, shape `(1024, 1024)`, dtype `float32`:

| op | block | Triton ms | PyTorch ms | speedup | GB/s |
| --- | --- | ---: | ---: | ---: | ---: |
| copy_2d | 16x16 | 0.0268 | 0.0255 | 0.95 | 312.65 |
| transpose | 16x16 | 0.0264 | 0.0327 | 1.24 | 317.87 |
| rowwise_add | 16x16 | 0.0269 | 0.0273 | 1.02 | 467.90 |
| copy_2d | 16x32 | 0.0262 | 0.0261 | 0.99 | 319.65 |
| transpose | 16x32 | 0.0263 | 0.0325 | 1.23 | 318.52 |
| rowwise_add | 16x32 | 0.0258 | 0.0272 | 1.05 | 487.96 |
| copy_2d | 32x32 | 0.0257 | 0.0260 | 1.01 | 326.50 |
| transpose | 32x32 | 0.0255 | 0.0324 | 1.27 | 329.54 |
| rowwise_add | 32x32 | 0.0258 | 0.0296 | 1.15 | 488.43 |

Current reduction results on RTX 3070, shape `(1024, 1024)`, dtype `float32`:

| op | Triton ms | PyTorch ms | speedup | GB/s |
| --- | ---: | ---: | ---: | ---: |
| row_sum | 0.0196 | 0.0192 | 0.98 | 214.74 |
| row_max | 0.0179 | 0.0201 | 1.13 | 234.82 |
| row_mean | 0.0179 | 0.0194 | 1.08 | 234.51 |

Current softmax and LayerNorm results on RTX 3070, dtype `float32`:

| op | shape | Triton ms | PyTorch ms | speedup |
| --- | --- | ---: | ---: | ---: |
| softmax | 512x2048 | 0.0241 | 0.0252 | 1.04 |
| layernorm | 32x1024 | 0.0060 | 0.0078 | 1.32 |

## Optimization Notes

`vector_add` is memory-bound: it reads two tensors and writes one tensor for only one arithmetic operation per element. In this test, Triton is roughly tied with PyTorch because PyTorch's simple elementwise kernel is already highly optimized. Triton's larger advantage usually appears when multiple elementwise operations are fused into one kernel.

For 2D kernels, the key pattern is computing addresses with `row * stride_0 + col * stride_1`. This supports non-contiguous views and makes shape separate from memory layout. `32x32` tiles performed best in this run.

For row-wise reductions, this project uses the simple one-program-per-row pattern. `BLOCK_N` must cover the full row. Use `other=0.0` for sum/mean padding and `other=-inf` for max padding. LayerNorm also needs masked `diff` during variance, otherwise padded lanes pollute the variance when `N` is not a power of two.

## Interview Summary

For elementwise kernels, describe one Triton program processing one contiguous block of elements. `tl.arange` builds offsets inside that block, `grid` controls how many programs are launched, and `mask` protects the tail when `numel` is not divisible by `BLOCK_SIZE`.

For 2D kernels, explain that `offs_m[:, None]` and `offs_n[None, :]` broadcast into a tile of addresses. Strides must be passed explicitly for non-contiguous tensors. Transpose changes the access direction, so at least one side of load/store can become less coalesced.

For softmax and LayerNorm, describe them as reduction plus broadcast. Softmax subtracts row max before `exp` for numerical stability. LayerNorm computes mean and variance in fp32, then broadcasts `rstd`, `gamma`, and `beta` across the row.
