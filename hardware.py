"""Hardware diagnostics. Reports what is present; never loads the model.

torch.cuda.is_available() is True for both NVIDIA (CUDA) and AMD (ROCm/HIP) builds of
PyTorch, so the backend is read from torch.version.cuda / torch.version.hip instead.
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path


def _torch_info() -> dict:
    info = {"torch_version": None, "torch_build": None, "cuda_available": False,
            "rocm_available": False, "torch_gpu_name": None, "torch_error": None}
    try:
        import torch
    except Exception as exc:  # torch missing or broken: report, do not fail
        info["torch_error"] = str(exc)
        return info
    info["torch_version"] = torch.__version__
    hip = getattr(torch.version, "hip", None)
    cuda = getattr(torch.version, "cuda", None)
    info["torch_build"] = f"ROCm {hip}" if hip else f"CUDA {cuda}" if cuda else "CPU only"
    try:
        gpu_ready = torch.cuda.is_available()
    except Exception as exc:
        gpu_ready = False
        info["torch_error"] = str(exc)
    info["cuda_available"] = bool(gpu_ready and cuda and not hip)
    info["rocm_available"] = bool(gpu_ready and hip)
    if gpu_ready:
        try:
            info["torch_gpu_name"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
    return info


def _run(cmd: list[str]) -> str:
    if not shutil.which(cmd[0]):
        return ""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return ""


def amd_gpus() -> list[str]:
    """AMD GPU names from rocm-smi or lspci, falling back to sysfs vendor ids."""
    out = _run(["rocm-smi", "--showproductname"])
    names = re.findall(r"Card Series:\s*(.+)", out)
    gfx = re.findall(r"GFX Version:\s*(\S+)", out)
    names = [n.strip() for n in names if n.strip() and n.strip() != "N/A"]
    if names:
        return [f"{n} ({g})" if g else n for n, g in zip(names, gfx + [""] * len(names))]
    out = _run(["lspci"])
    names = [line.split(": ", 1)[1] for line in out.splitlines()
             if re.search(r"VGA|3D|Display", line) and re.search(r"AMD|ATI", line)
             and ": " in line]
    if names:
        return names
    found = []
    for vendor in Path("/sys/class/drm").glob("card*/device/vendor"):
        if vendor.read_text().strip() == "0x1002":
            device = (vendor.parent / "device").read_text().strip()
            found.append(f"AMD GPU (PCI device {device})")
    return found


def nvidia_gpus() -> list[str]:
    """NVIDIA GPU names from nvidia-smi (under WSL2 it lives in /usr/lib/wsl/lib)."""
    exe = shutil.which("nvidia-smi") or (
        "/usr/lib/wsl/lib/nvidia-smi" if Path("/usr/lib/wsl/lib/nvidia-smi").exists() else None)
    if not exe:
        return []
    out = _run([exe, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            gpus.append(f"{parts[0]} ({parts[1]}, driver {parts[2]})")
    return gpus


def is_wsl() -> bool:
    return "microsoft" in platform.release().lower()


def collect(configured_device: str | None = None) -> dict:
    info = {
        "os": f"{platform.system()} {platform.release()}" + (" (WSL)" if is_wsl() else ""),
        "python": sys.version.split()[0],
        **_torch_info(),
        "nvidia_gpus": nvidia_gpus(),
        "amd_gpus": amd_gpus(),
    }
    if configured_device is not None:
        info["configured_device"] = configured_device
    return info


def missing_build_tools() -> list[str]:
    """What Triton needs to JIT-compile its CUDA helper, but is absent."""
    import sysconfig

    missing = []
    if not any(shutil.which(cc) for cc in ("cc", "gcc", "clang")):
        missing.append("a C compiler")
    if not (Path(sysconfig.get_paths()["include"]) / "Python.h").exists():
        missing.append("Python development headers (Python.h)")
    return missing


def _test_triton() -> str | None:
    """Compile and run a tiny Triton kernel. Returns an error message, or None if it works."""
    try:
        import torch
        import triton
        import triton.language as tl

        @triton.jit
        def _add_one(x_ptr, n, BLOCK: tl.constexpr):
            offs = tl.arange(0, BLOCK)
            mask = offs < n
            tl.store(x_ptr + offs, tl.load(x_ptr + offs, mask=mask) + 1, mask=mask)

        x = torch.zeros(16, device="cuda")
        _add_one[(1,)](x, 16, BLOCK=16)
        torch.cuda.synchronize()
        if x.sum().item() != 16:
            return "Triton kernel returned a wrong result"
    except Exception as exc:
        return str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    return None


def test_gpu() -> bool:
    """Run a small bf16 matmul on the GPU. True only if it computes the right answer."""
    try:
        import torch
    except Exception as exc:
        print(f"GPU test: PyTorch not importable: {exc}")
        return False
    if not torch.cuda.is_available():
        build = "ROCm" if getattr(torch.version, "hip", None) else \
            "CUDA" if torch.version.cuda else "CPU-only"
        print(f"GPU test: no usable GPU (PyTorch {torch.__version__}, {build} build).")
        return False
    try:
        name = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info(0)
        a = torch.ones(512, 512, device="cuda", dtype=torch.bfloat16)
        result = (a @ a).float().mean().item()
        torch.cuda.synchronize()
    except Exception as exc:
        print(f"GPU test: GPU present but computation failed: {exc}")
        return False
    if abs(result - 512.0) > 1.0:
        print(f"GPU test: wrong result from the GPU ({result}, expected 512).")
        return False
    print(f"GPU test: OK on {name}, {free / 2**30:.1f} of {total / 2**30:.1f} GiB free.")
    try:
        import fla  # noqa: F401
    except Exception:
        print("GPU test: flash-linear-attention not available; the model uses slower fallbacks.")
        return True
    error = _test_triton()
    if error is None:
        print("GPU test: flash-linear-attention available (fast kernels).")
        return True
    # transformers uses fla whenever it is installed, so a broken Triton crashes the server.
    print(f"GPU test: flash-linear-attention is installed but its Triton kernels fail: {error}")
    missing = missing_build_tools()
    if missing:
        print(f"  Triton needs {' and '.join(missing)} to compile its GPU helper.")
        print("  On Ubuntu/Debian: sudo apt install build-essential python3-dev")
    print("  Or remove the fast kernels and use the GPU without them:")
    print("    .venv/bin/python -m pip uninstall -y flash-linear-attention")
    return False


if __name__ == "__main__":
    from config import get_settings

    if "--test-gpu" in sys.argv:
        sys.exit(0 if test_gpu() else 1)

    data = collect(get_settings().device)
    print("Hardware diagnostics")
    print(f"  Operating system   : {data['os']}")
    print(f"  Python             : {data['python']}")
    print(f"  PyTorch            : {data['torch_version'] or 'not installed'} ({data['torch_build']})")
    print(f"  CUDA available     : {data['cuda_available']}")
    print(f"  ROCm/HIP available : {data['rocm_available']}")
    print(f"  NVIDIA GPU         : {', '.join(data['nvidia_gpus']) or 'none found'}")
    print(f"  AMD GPU            : {', '.join(data['amd_gpus']) or 'none found'}")
    print(f"  Configured device  : {data['configured_device']}")
    if data["amd_gpus"] and not data["rocm_available"]:
        print("  Note: an AMD GPU is present but this PyTorch build cannot use it.")
    if data["nvidia_gpus"] and not data["cuda_available"]:
        print("  Note: an NVIDIA GPU is present but this PyTorch build cannot use it "
              "(run ./setup.sh --cuda).")
