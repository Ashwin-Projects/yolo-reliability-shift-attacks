"""GPU execution protocol (docs/CONFIRMATORY_DESIGN.md, "Device and execution protocol")."""
import os

import torch

import config

# environment variables that must be set before CUDA or cuBLAS is initialized (see run.py)
CUDA_ENV = {"CUDA_VISIBLE_DEVICES": "0", "CUBLAS_WORKSPACE_CONFIG": config.CUBLAS_WORKSPACE_CONFIG}


def cuda_flags():
    return {"cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_algorithms_warn_only": torch.is_deterministic_algorithms_warn_only_enabled()}


EXPECTED_FLAGS = {"cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False, "cudnn_benchmark": False,
                  "cudnn_deterministic": True, "deterministic_algorithms": True,
                  "deterministic_algorithms_warn_only": False}


def apply_cuda_protocol():
    """float32 without TF32, deterministic cuDNN and algorithms, one GPU; errors are not caught."""
    for k, v in CUDA_ENV.items():
        if os.environ.get(k) != v:
            raise RuntimeError(f"{k} must be {v!r} before CUDA starts (got {os.environ.get(k)!r})")
    if not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"exactly one visible GPU required, found {torch.cuda.device_count()}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)  # an op without a deterministic version raises
    verify_cuda_protocol()


def verify_cuda_protocol():
    """Stop if any library changed a flag after startup."""
    flags = cuda_flags()
    changed = {k: v for k, v in flags.items() if v != EXPECTED_FLAGS[k]}
    if changed:
        raise RuntimeError(f"GPU execution protocol violated: {changed}")
