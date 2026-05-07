import pytest
import torch

from kernels.fused_matmul import matmul_bias_relu, reference_matmul_bias_relu
from kernels.layernorm import layernorm, reference_layernorm
from kernels.matmul import matmul, reference_matmul
from kernels.reductions import (
    reference_row_max,
    reference_row_mean,
    reference_row_sum,
    row_max,
    row_mean,
    row_sum,
)
from kernels.softmax import reference_softmax, softmax
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


REDUCTION_SHAPES = [
    (4, 16),
    (128, 1024),
    (1023, 777),
]


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
def test_row_sum(shape):
    x = torch.randn(shape, device="cuda")

    actual = row_sum(x)
    expected = reference_row_sum(x)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
def test_row_max(shape):
    x = torch.randn(shape, device="cuda") - 10.0

    actual = row_max(x)
    expected = reference_row_max(x)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
def test_row_mean(shape):
    x = torch.randn(shape, device="cuda")

    actual = row_mean(x)
    expected = reference_row_mean(x)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


SOFTMAX_SHAPES = [
    (16, 128),
    (128, 1024),
    (512, 2048),
]


@pytest.mark.parametrize("shape", SOFTMAX_SHAPES)
def test_softmax(shape):
    x = torch.randn(shape, device="cuda")

    actual = softmax(x)
    expected = reference_softmax(x)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_softmax_large_values_are_stable():
    x = torch.randn((32, 1024), device="cuda") * 20.0 + 1000.0

    actual = softmax(x)
    expected = reference_softmax(x)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


LAYERNORM_SHAPES = [
    (32, 768),
    (32, 1024),
    (16, 4096),
]


@pytest.mark.parametrize("shape", LAYERNORM_SHAPES)
def test_layernorm(shape):
    x = torch.randn(shape, device="cuda")
    gamma = torch.randn((shape[1],), device="cuda")
    beta = torch.randn((shape[1],), device="cuda")

    actual = layernorm(x, gamma, beta)
    expected = reference_layernorm(x, gamma, beta)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


MATMUL_SHAPES = [
    (16, 16, 16),
    (64, 96, 128),
    (127, 129, 65),
]


@pytest.mark.parametrize("M,N,K", MATMUL_SHAPES)
def test_matmul(M, N, K):
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)

    actual = matmul(a, b)
    expected = reference_matmul(a, b)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-1)


@pytest.mark.parametrize("M,N,K", MATMUL_SHAPES)
def test_matmul_bias_relu(M, N, K):
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    bias = torch.randn((N,), device="cuda", dtype=torch.float32)

    actual = matmul_bias_relu(a, b, bias)
    expected = reference_matmul_bias_relu(a, b, bias)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-1)
