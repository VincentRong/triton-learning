import pytest
import torch

from kernels.vector_add import (
    reference_add,
    reference_affine,
    reference_mul,
    vector_add,
    vector_affine,
    vector_mul,
)


SHAPES = [
    (1024,),
    (1000,),
    (1_000_000,),
    (1024, 1024),
]


@pytest.mark.parametrize("shape", SHAPES)
def test_vector_add(shape):
    a = torch.randn(shape, device="cuda")
    b = torch.randn(shape, device="cuda")

    actual = vector_add(a, b)
    expected = reference_add(a, b)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("shape", SHAPES)
def test_vector_mul(shape):
    a = torch.randn(shape, device="cuda")
    b = torch.randn(shape, device="cuda")

    actual = vector_mul(a, b)
    expected = reference_mul(a, b)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("shape", SHAPES)
def test_vector_affine(shape):
    x = torch.randn(shape, device="cuda")
    scale = 1.25
    bias = -0.5

    actual = vector_affine(x, scale, bias)
    expected = reference_affine(x, scale, bias)

    torch.testing.assert_close(actual, expected)
