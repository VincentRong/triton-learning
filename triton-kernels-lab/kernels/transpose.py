import torch
import triton
import triton.language as tl


@triton.jit
def _copy_2d_kernel(
    x_ptr,
    y_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xn: tl.constexpr,
    stride_ym: tl.constexpr,
    stride_yn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn

    x = tl.load(x_ptrs, mask=mask, other=0.0)
    tl.store(y_ptrs, x, mask=mask)


@triton.jit
def _transpose_kernel(
    x_ptr,
    y_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xn: tl.constexpr,
    stride_ym: tl.constexpr,
    stride_yn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    y_ptrs = y_ptr + offs_n[None, :] * stride_ym + offs_m[:, None] * stride_yn

    x = tl.load(x_ptrs, mask=mask, other=0.0)
    tl.store(y_ptrs, x, mask=mask)


@triton.jit
def _rowwise_add_kernel(
    x_ptr,
    bias_ptr,
    y_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xn: tl.constexpr,
    stride_ym: tl.constexpr,
    stride_yn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn

    x = tl.load(x_ptrs, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    y = x + bias[None, :]

    tl.store(y_ptrs, y, mask=mask)


def _check_2d_cuda(x: torch.Tensor) -> None:
    assert x.is_cuda, "input must be a CUDA tensor"
    assert x.ndim == 2, "input must be a 2D tensor"


def copy_2d(
    x: torch.Tensor,
    block_m: int = 16,
    block_n: int = 32,
) -> torch.Tensor:
    _check_2d_cuda(x)

    y = torch.empty_strided(x.shape, x.stride(), device=x.device, dtype=x.dtype)
    M, N = x.shape
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _copy_2d_kernel[grid](
        x,
        y,
        M,
        N,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )
    return y


def transpose(
    x: torch.Tensor,
    block_m: int = 16,
    block_n: int = 32,
) -> torch.Tensor:
    _check_2d_cuda(x)

    M, N = x.shape
    y = torch.empty((N, M), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _transpose_kernel[grid](
        x,
        y,
        M,
        N,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )
    return y


def rowwise_add(
    x: torch.Tensor,
    bias: torch.Tensor,
    block_m: int = 16,
    block_n: int = 32,
) -> torch.Tensor:
    _check_2d_cuda(x)
    assert bias.is_cuda, "bias must be a CUDA tensor"
    assert bias.ndim == 1, "bias must be a 1D tensor"
    assert bias.numel() == x.shape[1], "bias length must match x.shape[1]"
    assert bias.is_contiguous(), "this lesson expects contiguous bias"

    y = torch.empty_like(x)
    M, N = x.shape
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _rowwise_add_kernel[grid](
        x,
        bias,
        y,
        M,
        N,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )
    return y


def reference_copy_2d(x: torch.Tensor) -> torch.Tensor:
    return x.clone(memory_format=torch.preserve_format)


def reference_transpose(x: torch.Tensor) -> torch.Tensor:
    return x.T.contiguous()


def reference_rowwise_add(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return x + bias


if __name__ == "__main__":
    x = torch.randn((1023, 777), device="cuda")
    bias = torch.randn((777,), device="cuda")
    torch.testing.assert_close(copy_2d(x), reference_copy_2d(x))
    torch.testing.assert_close(transpose(x), reference_transpose(x))
    torch.testing.assert_close(rowwise_add(x, bias), reference_rowwise_add(x, bias))
    print("ok")
