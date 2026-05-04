import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    x_ptr,
    y_ptr,
    N: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xn: tl.constexpr,
    stride_ym: tl.constexpr,
    stride_yn: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    x = tl.load(x_ptr + row * stride_xm + offs * stride_xn, mask=mask, other=-float("inf"))
    x = x.to(tl.float32)
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    y = num / den

    tl.store(y_ptr + row * stride_ym + offs * stride_yn, y, mask=mask)


def softmax(x: torch.Tensor, block_n: int | None = None) -> torch.Tensor:
    assert x.is_cuda, "input must be a CUDA tensor"
    assert x.ndim == 2, "input must be a 2D tensor"

    M, N = x.shape
    block_n = triton.next_power_of_2(N) if block_n is None else block_n
    assert block_n >= N, "BLOCK_N must cover the full row in this simple version"

    y = torch.empty_like(x)
    _softmax_kernel[(M,)](
        x,
        y,
        N,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_N=block_n,
    )
    return y


def reference_softmax(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)
