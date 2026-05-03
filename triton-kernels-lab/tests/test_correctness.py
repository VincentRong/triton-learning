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
from kernels.transpose import (
    copy_2d,
    reference_copy_2d,
    reference_rowwise_add,
    reference_transpose,
    rowwise_add,
    transpose,
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


SHAPES_2D = [
    (32, 32),
    (128, 256),
    (1000, 1024),
    (1023, 777),
]


@pytest.mark.parametrize("shape", SHAPES_2D)
def test_copy_2d(shape):
    x = torch.randn(shape, device="cuda")

    actual = copy_2d(x)
    expected = reference_copy_2d(x)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("shape", SHAPES_2D)
def test_copy_2d_non_contiguous_view(shape):
    base = torch.randn((shape[1], shape[0]), device="cuda")
    x = base.T

    actual = copy_2d(x)
    expected = reference_copy_2d(x)

    assert not x.is_contiguous()
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("shape", SHAPES_2D)
def test_transpose(shape):
    x = torch.randn(shape, device="cuda")

    actual = transpose(x)
    expected = reference_transpose(x)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("shape", SHAPES_2D)
def test_rowwise_add(shape):
    x = torch.randn(shape, device="cuda")
    bias = torch.randn((shape[1],), device="cuda")

    actual = rowwise_add(x, bias)
    expected = reference_rowwise_add(x, bias)

    torch.testing.assert_close(actual, expected)
