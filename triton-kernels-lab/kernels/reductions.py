import torch
import triton
import triton.language as tl


@triton.jit
def _row_sum_kernel(
    x_ptr,
    out_ptr,
    N: tl.constexpr,
    stride_m: tl.constexpr,
    stride_n: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    x = tl.load(x_ptr + row * stride_m + offs * stride_n, mask=mask, other=0.0)
    x = x.to(tl.float32)
    s = tl.sum(x, axis=0)

    tl.store(out_ptr + row, s)


@triton.jit
def _row_max_kernel(
    x_ptr,
    out_ptr,
    N: tl.constexpr,
    stride_m: tl.constexpr,
    stride_n: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    x = tl.load(x_ptr + row * stride_m + offs * stride_n, mask=mask, other=-float("inf"))
    x = x.to(tl.float32)
    m = tl.max(x, axis=0)

    tl.store(out_ptr + row, m)


@triton.jit
def _row_mean_kernel(
    x_ptr,
    out_ptr,
    N: tl.constexpr,
    stride_m: tl.constexpr,
    stride_n: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    x = tl.load(x_ptr + row * stride_m + offs * stride_n, mask=mask, other=0.0)
    x = x.to(tl.float32)
    mean = tl.sum(x, axis=0) / N

    tl.store(out_ptr + row, mean)


def _check_2d_cuda(x: torch.Tensor) -> None:
    assert x.is_cuda, "input must be a CUDA tensor"
    assert x.ndim == 2, "input must be a 2D tensor"


def _block_n(n: int) -> int:
    return triton.next_power_of_2(n)


def row_sum(x: torch.Tensor, block_n: int | None = None) -> torch.Tensor:
    _check_2d_cuda(x)

    M, N = x.shape
    block_n = _block_n(N) if block_n is None else block_n
    assert block_n >= N, "BLOCK_N must cover the full row in this simple version"

    out = torch.empty((M,), device=x.device, dtype=torch.float32)
    _row_sum_kernel[(M,)](x, out, N, x.stride(0), x.stride(1), BLOCK_N=block_n)
    return out


def row_max(x: torch.Tensor, block_n: int | None = None) -> torch.Tensor:
    _check_2d_cuda(x)

    M, N = x.shape
    block_n = _block_n(N) if block_n is None else block_n
    assert block_n >= N, "BLOCK_N must cover the full row in this simple version"

    out = torch.empty((M,), device=x.device, dtype=torch.float32)
    _row_max_kernel[(M,)](x, out, N, x.stride(0), x.stride(1), BLOCK_N=block_n)
    return out


def row_mean(x: torch.Tensor, block_n: int | None = None) -> torch.Tensor:
    _check_2d_cuda(x)

    M, N = x.shape
    block_n = _block_n(N) if block_n is None else block_n
    assert block_n >= N, "BLOCK_N must cover the full row in this simple version"

    out = torch.empty((M,), device=x.device, dtype=torch.float32)
    _row_mean_kernel[(M,)](x, out, N, x.stride(0), x.stride(1), BLOCK_N=block_n)
    return out


def reference_row_sum(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x, dim=1, dtype=torch.float32)


def reference_row_max(x: torch.Tensor) -> torch.Tensor:
    return torch.max(x, dim=1).values.to(torch.float32)


def reference_row_mean(x: torch.Tensor) -> torch.Tensor:
    return torch.mean(x, dim=1, dtype=torch.float32)
