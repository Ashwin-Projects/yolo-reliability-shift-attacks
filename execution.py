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


def _driver_version():
    import subprocess

    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return out.strip().splitlines()[0] if out.strip() else None


def environment_record():
    """Everything the confirmatory run must reproduce exactly (written by --stage probe).

    Loads the model first: that downloads missing weights (hashed below), and ultralytics sets
    CUBLAS_WORKSPACE_CONFIG when it builds a model, so the record is the same wherever it is taken."""
    import platform

    import numpy
    import scipy
    import torchvision
    import ultralytics

    import attack
    import infer

    infer.load_model()
    cuda = config.DEVICE != "cpu"
    return {
        "device": config.DEVICE,
        "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
        "gpu_count": torch.cuda.device_count() if cuda else 0,
        "driver_version": _driver_version() if cuda else None,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if cuda else None,
        "python": platform.python_version(),
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "ultralytics": ultralytics.__version__, "numpy": numpy.__version__, "scipy": scipy.__version__,
        "dtype": str(torch.get_default_dtype()).replace("torch.", ""),
        "flags": cuda_flags(),
        "env": {k: os.environ.get(k) for k in CUDA_ENV},
        "model": config.MODEL_WEIGHTS, "weights_sha256": attack.weights_sha256(),
    }


def check_manifest(path):
    """Abort unless the manifest exists and the live environment matches it field by field."""
    import json

    if not path.exists():
        raise SystemExit(f"Execution manifest {path} is missing. Run --stage probe on the target machine, "
                         f"commit the manifest as docs/EXECUTION_MANIFEST.json, then rerun.")
    with open(path) as fh:
        saved = json.load(fh)
    live = json.loads(json.dumps(environment_record()))  # same types as the saved JSON
    diffs = {k: (saved.get(k), live.get(k)) for k in sorted(set(saved) | set(live)) if saved.get(k) != live.get(k)}
    if diffs:
        lines = "\n".join(f"  {k}: manifest {a!r} != live {b!r}" for k, (a, b) in diffs.items())
        raise SystemExit(f"Live environment differs from the execution manifest {path}:\n{lines}")
    return live
