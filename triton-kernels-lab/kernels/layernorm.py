import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _layernorm_kernel(
    x_ptr,
    gamma_ptr,
    beta_ptr,
    y_ptr,
    N: tl.constexpr,
    eps: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xn: tl.constexpr,
    stride_ym: tl.constexpr,
    stride_yn: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    x = tl.load(x_ptr + row * stride_xm + offs * stride_xn, mask=mask, other=0.0)
    x = x.to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = tl.rsqrt(var + eps)

    gamma = tl.load(gamma_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * gamma + beta

    tl.store(y_ptr + row * stride_ym + offs * stride_yn, y, mask=mask)


def layernorm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-5,
    block_n: int | None = None,
) -> torch.Tensor:
    assert x.is_cuda, "input must be a CUDA tensor"
    assert gamma.is_cuda and beta.is_cuda, "gamma and beta must be CUDA tensors"
    assert x.ndim == 2, "input must be a 2D tensor"
    assert gamma.ndim == 1 and beta.ndim == 1, "gamma and beta must be 1D tensors"
    assert gamma.numel() == x.shape[1] and beta.numel() == x.shape[1]
    assert gamma.is_contiguous() and beta.is_contiguous()

    M, N = x.shape
    block_n = triton.next_power_of_2(N) if block_n is None else block_n
    assert block_n >= N, "BLOCK_N must cover the full row in this simple version"

    y = torch.empty_like(x)
    _layernorm_kernel[(M,)](
        x,
        gamma,
        beta,
        y,
        N,
        eps,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_N=block_n,
    )
    return y


def reference_layernorm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[1],), gamma, beta, eps)
