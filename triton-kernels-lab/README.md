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

## Correctness

Run tests with:

```bash
/home/rwenxiao/pytorch-cuda/bin/python -m pytest -q
```

Current result:

```text
31 passed in 9.85s
```

## Benchmarks

Run benchmarks with:

```bash
/home/rwenxiao/pytorch-cuda/bin/python -m benchmarks.bench_all
```

Current `vector_add` result on RTX 3070, shape `(1024, 1024)`, dtype `float32`:

| block size | Triton ms | PyTorch ms | speedup | GB/s |
| --- | ---: | ---: | ---: | ---: |
| 256 | 0.0375 | 0.0362 | 0.96 | 335.66 |
| 512 | 0.0367 | 0.0354 | 0.96 | 343.07 |
| 1024 | 0.0357 | 0.0354 | 0.99 | 352.07 |
| 2048 | 0.0362 | 0.0352 | 0.97 | 347.17 |

Current 2D op results on RTX 3070, shape `(1024, 1024)`, dtype `float32`:

| op | block | Triton ms | PyTorch ms | speedup | GB/s |
| --- | --- | ---: | ---: | ---: | ---: |
| copy_2d | 16x16 | 0.0278 | 0.0261 | 0.94 | 302.13 |
| transpose | 16x16 | 0.0271 | 0.0363 | 1.34 | 309.42 |
| rowwise_add | 16x16 | 0.0273 | 0.0272 | 1.00 | 461.43 |
| copy_2d | 16x32 | 0.0276 | 0.0260 | 0.94 | 304.18 |
| transpose | 16x32 | 0.0264 | 0.0334 | 1.26 | 317.46 |
| rowwise_add | 16x32 | 0.0260 | 0.0273 | 1.05 | 483.52 |
| copy_2d | 32x32 | 0.0257 | 0.0262 | 1.02 | 326.26 |
| transpose | 32x32 | 0.0250 | 0.0337 | 1.35 | 335.52 |
| rowwise_add | 32x32 | 0.0259 | 0.0274 | 1.06 | 486.22 |

## Optimization Notes

`vector_add` is memory-bound: it reads two tensors and writes one tensor for only one arithmetic operation per element. In this test, Triton is roughly tied with PyTorch because PyTorch's simple elementwise kernel is already highly optimized. Triton's larger advantage usually appears when multiple elementwise operations are fused into one kernel.

For 2D kernels, the key pattern is computing addresses with `row * stride_0 + col * stride_1`. This supports non-contiguous views and makes shape separate from memory layout. `32x32` tiles performed best in this run.

## Interview Summary

For elementwise kernels, describe one Triton program processing one contiguous block of elements. `tl.arange` builds offsets inside that block, `grid` controls how many programs are launched, and `mask` protects the tail when `numel` is not divisible by `BLOCK_SIZE`.

For 2D kernels, explain that `offs_m[:, None]` and `offs_n[None, :]` broadcast into a tile of addresses. Strides must be passed explicitly for non-contiguous tensors. Transpose changes the access direction, so at least one side of load/store can become less coalesced.
