# Triton Kernels Lab

Learning project for implementing, testing, benchmarking, and explaining Triton kernels.

## Environment

See `notes/env.md`.

## Kernels

- `kernels/vector_add.py`
  - `vector_add(a, b) -> a + b`
  - `vector_mul(a, b) -> a * b`
  - `vector_affine(x, scale, bias) -> x * scale + bias`

## Correctness

Run tests with:

```bash
/home/rwenxiao/pytorch-cuda/bin/python -m pytest -q
```

Current result:

```text
15 passed in 4.25s
```

## Benchmarks

Run benchmarks with:

```bash
/home/rwenxiao/pytorch-cuda/bin/python -m benchmarks.bench_all
```

Current `vector_add` result on RTX 3070, shape `(1024, 1024)`, dtype `float32`:

| block size | Triton ms | PyTorch ms | speedup | GB/s |
| --- | ---: | ---: | ---: | ---: |
| 256 | 0.0453 | 0.0361 | 0.80 | 277.85 |
| 512 | 0.0362 | 0.0384 | 1.06 | 347.36 |
| 1024 | 0.0376 | 0.0382 | 1.02 | 334.83 |
| 2048 | 0.0358 | 0.0355 | 0.99 | 351.58 |

## Optimization Notes

`vector_add` is memory-bound: it reads two tensors and writes one tensor for only one arithmetic operation per element. In this test, Triton is roughly tied with PyTorch because PyTorch's simple elementwise kernel is already highly optimized. Triton's larger advantage usually appears when multiple elementwise operations are fused into one kernel.

## Interview Summary

For elementwise kernels, describe one Triton program processing one contiguous block of elements. `tl.arange` builds offsets inside that block, `grid` controls how many programs are launched, and `mask` protects the tail when `numel` is not divisible by `BLOCK_SIZE`.
