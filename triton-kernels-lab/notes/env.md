# Environment

- Date: 2026-05-02
- Workspace: `/home/rwenxiao/dev-learning/Triton/triton-kernels-lab`
- Python: 3.12.3 (`/home/rwenxiao/pytorch-cuda/bin/python`)
- GPU: NVIDIA GeForce RTX 3070
- GPU memory: 8192 MiB
- NVIDIA driver: 576.52
- System CUDA reported by `nvidia-smi`: 12.9
- PyTorch: 2.11.0+cu128
- PyTorch CUDA: 12.8
- Triton: 3.6.0
- CUDA available in PyTorch: True

## Smoke Test

Command:

```bash
/home/rwenxiao/pytorch-cuda/bin/python -m pytest -q
```

Result:

```text
3 passed in 3.83s
```

Note: GPU tests need to run in an environment that can access the NVIDIA driver.
