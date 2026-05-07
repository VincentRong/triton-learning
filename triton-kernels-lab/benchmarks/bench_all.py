import torch
import triton

from kernels.layernorm import layernorm
from kernels.fused_matmul import matmul_bias_relu
from kernels.matmul import matmul
from kernels.reductions import row_max, row_mean, row_sum
from kernels.softmax import softmax
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


def benchmark_reductions() -> None:
    shape = (1024, 1024)
    x = torch.randn(shape, device="cuda")

    print()
    print("reductions")
    print("op,shape,dtype,triton_ms,pytorch_ms,speedup,gbps")
    benches = [
        ("row_sum", lambda: row_sum(x), lambda: torch.sum(x, dim=1), 1),
        ("row_max", lambda: row_max(x), lambda: torch.max(x, dim=1).values, 1),
        ("row_mean", lambda: row_mean(x), lambda: torch.mean(x, dim=1), 1),
    ]
    for name, triton_fn, torch_fn, read_factor in benches:
        triton_ms = triton.testing.do_bench(triton_fn)
        pytorch_ms = triton.testing.do_bench(torch_fn)
        bytes_moved = x.numel() * x.element_size() * read_factor + x.shape[0] * x.element_size()
        gbps = bytes_moved / (triton_ms * 1e-3) / 1e9
        speedup = pytorch_ms / triton_ms
        print(f"{name},{shape},{x.dtype},{triton_ms:.4f},{pytorch_ms:.4f},{speedup:.2f},{gbps:.2f}")


def benchmark_softmax_layernorm() -> None:
    softmax_shape = (512, 2048)
    x_softmax = torch.randn(softmax_shape, device="cuda")

    print()
    print("softmax")
    print("shape,dtype,triton_ms,pytorch_ms,speedup")
    triton_ms = triton.testing.do_bench(lambda: softmax(x_softmax))
    pytorch_ms = triton.testing.do_bench(lambda: torch.softmax(x_softmax, dim=-1))
    print(f"{softmax_shape},{x_softmax.dtype},{triton_ms:.4f},{pytorch_ms:.4f},{pytorch_ms / triton_ms:.2f}")

    layernorm_shape = (32, 1024)
    x_layernorm = torch.randn(layernorm_shape, device="cuda")
    gamma = torch.randn((layernorm_shape[1],), device="cuda")
    beta = torch.randn((layernorm_shape[1],), device="cuda")

    print()
    print("layernorm")
    print("shape,dtype,triton_ms,pytorch_ms,speedup")
    triton_ms = triton.testing.do_bench(lambda: layernorm(x_layernorm, gamma, beta))
    pytorch_ms = triton.testing.do_bench(
        lambda: torch.nn.functional.layer_norm(x_layernorm, (layernorm_shape[1],), gamma, beta)
    )
    print(f"{layernorm_shape},{x_layernorm.dtype},{triton_ms:.4f},{pytorch_ms:.4f},{pytorch_ms / triton_ms:.2f}")


def benchmark_matmul() -> None:
    shapes = [
        (256, 256, 256),
        (512, 1024, 2048),
        (1024, 1024, 1024),
        (1024, 4096, 4096),
    ]
    configs = [
        (32, 64, 32, 4),
        (64, 64, 32, 4),
        (64, 128, 32, 4),
    ]

    print()
    print("matmul")
    print("shape,config,triton_ms,pytorch_ms,speedup,tflops")
    for M, N, K in shapes:
        a = torch.randn((M, K), device="cuda", dtype=torch.float16)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16)
        pytorch_ms = triton.testing.do_bench(lambda: a @ b)
        for block_m, block_n, block_k, num_warps in configs:
            triton_ms = triton.testing.do_bench(
                lambda: matmul(a, b, block_m, block_n, block_k, num_warps)
            )
            flops = 2 * M * N * K
            tflops = flops / (triton_ms * 1e-3) / 1e12
            speedup = pytorch_ms / triton_ms
            config = f"{block_m}x{block_n}x{block_k}/w{num_warps}"
            print(f"{M}x{N}x{K},{config},{triton_ms:.4f},{pytorch_ms:.4f},{speedup:.2f},{tflops:.2f}")


def benchmark_fused_matmul() -> None:
    M, N, K = 1024, 1024, 1024
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    bias = torch.randn((N,), device="cuda", dtype=torch.float32)

    print()
    print("fused_matmul_bias_relu")
    print("shape,triton_ms,pytorch_ms,speedup,tflops")
    triton_ms = triton.testing.do_bench(lambda: matmul_bias_relu(a, b, bias))
    pytorch_ms = triton.testing.do_bench(lambda: torch.relu((a @ b).float() + bias))
    flops = 2 * M * N * K
    tflops = flops / (triton_ms * 1e-3) / 1e12
    print(f"{M}x{N}x{K},{triton_ms:.4f},{pytorch_ms:.4f},{pytorch_ms / triton_ms:.2f},{tflops:.2f}")


if __name__ == "__main__":
    benchmark_vector_add()
    benchmark_2d_ops()
    benchmark_reductions()
    benchmark_softmax_layernorm()
    benchmark_matmul()
    benchmark_fused_matmul()
