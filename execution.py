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


T4_NAMES = {"Tesla T4", "NVIDIA T4"}  # exact, case-sensitive device names (after strip)

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


def pinned_versions():
    """numpy / scipy versions pinned in requirements_gpu.txt (the frozen statistics)."""
    pins = {}
    for line in (config.ROOT / "requirements_gpu.txt").read_text(encoding="utf-8").splitlines():
        name, sep, version = line.strip().partition("==")
        if sep and name in ("numpy", "scipy"):
            pins[name] = version
    if set(pins) != {"numpy", "scipy"}:
        raise SystemExit(f"requirements_gpu.txt must pin numpy and scipy (found {pins})")
    return pins


def validate_production(record):
    """Abort unless an environment record follows the production protocol (docs/CONFIRMATORY_DESIGN.md)."""
    problems = []
    if not str(record.get("device", "")).startswith("cuda"):
        problems.append(f"device is {record.get('device')!r}, not CUDA")
    name = record.get("gpu_name")
    if record.get("gpu_count") != 1 or not isinstance(name, str) or name.strip() not in T4_NAMES:
        problems.append(f"GPU must be one T4 named exactly one of {sorted(T4_NAMES)} "
                        f"(found {record.get('gpu_count')} x {name!r})")
    if record.get("dtype") != "float32":
        problems.append(f"dtype is {record.get('dtype')!r}, not float32")
    flags = record.get("flags") or {}
    problems += [f"flag {k} is {flags.get(k)!r}, must be {v!r}" for k, v in EXPECTED_FLAGS.items()
                 if flags.get(k) != v]
    env = record.get("env") or {}
    problems += [f"environment variable {k} is {env.get(k)!r}, must be {v!r}" for k, v in CUDA_ENV.items()
                 if env.get(k) != v]
    problems += [f"{k} is {record.get(k)!r}, pinned {v!r}" for k, v in pinned_versions().items()
                 if record.get(k) != v]
    if problems:
        raise SystemExit("The execution manifest does not follow the production protocol "
                         "(docs/CONFIRMATORY_DESIGN.md):\n" + "\n".join(f"  - {p}" for p in problems))


def check_manifest(path, production=True):
    """Abort unless the manifest exists and the live environment matches its environment
    record field by field; production=True also requires the production protocol.
    Returns the whole manifest (environment, probe, target_range)."""
    import json

    if not path.exists():
        raise SystemExit(f"Execution manifest {path} is missing. Run --stage probe on the target machine, "
                         f"commit the manifest as docs/EXECUTION_MANIFEST.json, then rerun.")
    with open(path) as fh:
        manifest = json.load(fh)
    saved = manifest.get("environment", {})
    live = json.loads(json.dumps(environment_record()))  # same types as the saved JSON
    diffs = {k: (saved.get(k), live.get(k)) for k in sorted(set(saved) | set(live)) if saved.get(k) != live.get(k)}
    if diffs:
        lines = "\n".join(f"  {k}: manifest {a!r} != live {b!r}" for k, (a, b) in diffs.items())
        raise SystemExit(f"Live environment differs from the execution manifest {path}:\n{lines}")
    if production:
        validate_production(saved)
    return manifest
