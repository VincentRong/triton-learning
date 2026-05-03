import torch
import triton


def test_cuda_available():
    assert torch.cuda.is_available()


def test_basic_tensor_on_gpu():
    x = torch.randn((1024,), device="cuda")
    y = x + 1
    assert y.is_cuda
    assert y.numel() == 1024


def test_triton_import():
    assert triton.__version__ is not None
