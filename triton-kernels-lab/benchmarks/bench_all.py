import torch
import triton

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


if __name__ == "__main__":
    benchmark_vector_add()
