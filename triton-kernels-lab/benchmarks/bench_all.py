import torch
import triton

from kernels.transpose import copy_2d, rowwise_add, transpose
from kernels.vector_add import vector_add


def benchmark_vector_add() -> None:
    shape = (1024, 1024)
    a = torch.randn(shape, device="cuda")
    b = torch.randn(shape, device="cuda")

    print("vector_add")
    print("shape,dtype,block_size,triton_ms,pytorch_ms,speedup,gbps")
    for block_size in [256, 512, 1024, 2048]:
        triton_ms = triton.testing.do_bench(lambda: vector_add(a, b, block_size=block_size))
        pytorch_ms = triton.testing.do_bench(lambda: a + b)
        bytes_moved = a.numel() * a.element_size() * 3
        gbps = bytes_moved / (triton_ms * 1e-3) / 1e9
        speedup = pytorch_ms / triton_ms
        print(
            f"{shape},{a.dtype},{block_size},"
            f"{triton_ms:.4f},{pytorch_ms:.4f},{speedup:.2f},{gbps:.2f}"
        )


def benchmark_2d_ops() -> None:
    shape = (1024, 1024)
    x = torch.randn(shape, device="cuda")
    bias = torch.randn((shape[1],), device="cuda")

    print()
    print("2d_ops")
    print("op,shape,dtype,block_m,block_n,triton_ms,pytorch_ms,speedup,gbps")
    for block_m, block_n in [(16, 16), (16, 32), (32, 32)]:
        benches = [
            ("copy_2d", lambda: copy_2d(x, block_m, block_n), lambda: x.clone(), 2),
            ("transpose", lambda: transpose(x, block_m, block_n), lambda: x.T.contiguous(), 2),
            ("rowwise_add", lambda: rowwise_add(x, bias, block_m, block_n), lambda: x + bias, 3),
        ]
        for name, triton_fn, torch_fn, tensor_factor in benches:
            triton_ms = triton.testing.do_bench(triton_fn)
            pytorch_ms = triton.testing.do_bench(torch_fn)
            bytes_moved = x.numel() * x.element_size() * tensor_factor
            gbps = bytes_moved / (triton_ms * 1e-3) / 1e9
            speedup = pytorch_ms / triton_ms
            print(
                f"{name},{shape},{x.dtype},{block_m},{block_n},"
                f"{triton_ms:.4f},{pytorch_ms:.4f},{speedup:.2f},{gbps:.2f}"
            )


if __name__ == "__main__":
    benchmark_vector_add()
    benchmark_2d_ops()
