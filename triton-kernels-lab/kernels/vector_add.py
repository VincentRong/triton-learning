import torch
import triton
import triton.language as tl


@triton.jit
def _vector_add_kernel(a_ptr, b_ptr, y_ptr, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    y = a + b

    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def _vector_mul_kernel(a_ptr, b_ptr, y_ptr, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    y = a * b

    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def _vector_affine_kernel(
    x_ptr,
    y_ptr,
    scale: tl.constexpr,
    bias: tl.constexpr,
    n: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = x * scale + bias

    tl.store(y_ptr + offsets, y, mask=mask)


def _check_pair_inputs(a: torch.Tensor, b: torch.Tensor) -> None:
    assert a.is_cuda and b.is_cuda, "inputs must be CUDA tensors"
    assert a.shape == b.shape, "input shapes must match"
    assert a.is_contiguous() and b.is_contiguous(), "this lesson expects contiguous tensors"


def _check_single_input(x: torch.Tensor) -> None:
    assert x.is_cuda, "input must be a CUDA tensor"
    assert x.is_contiguous(), "this lesson expects contiguous tensors"


def vector_add(a: torch.Tensor, b: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    _check_pair_inputs(a, b)

    y = torch.empty_like(a)
    n = a.numel()
    grid = (triton.cdiv(n, block_size),)
    _vector_add_kernel[grid](a, b, y, n, BLOCK_SIZE=block_size)
    return y


def vector_mul(a: torch.Tensor, b: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    _check_pair_inputs(a, b)

    y = torch.empty_like(a)
    n = a.numel()
    grid = (triton.cdiv(n, block_size),)
    _vector_mul_kernel[grid](a, b, y, n, BLOCK_SIZE=block_size)
    return y


def vector_affine(
    x: torch.Tensor,
    scale: float,
    bias: float,
    block_size: int = 1024,
) -> torch.Tensor:
    _check_single_input(x)

    y = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, block_size),)
    _vector_affine_kernel[grid](x, y, scale, bias, n, BLOCK_SIZE=block_size)
    return y


def reference_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b


def reference_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a * b


def reference_affine(x: torch.Tensor, scale: float, bias: float) -> torch.Tensor:
    return x * scale + bias


if __name__ == "__main__":
    a = torch.randn((1000,), device="cuda")
    b = torch.randn((1000,), device="cuda")
    torch.testing.assert_close(vector_add(a, b), reference_add(a, b))
    torch.testing.assert_close(vector_mul(a, b), reference_mul(a, b))
    torch.testing.assert_close(vector_affine(a, 2.0, 0.5), reference_affine(a, 2.0, 0.5))
    print("ok")
