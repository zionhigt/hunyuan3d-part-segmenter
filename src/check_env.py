"""Environment diagnostic: torch / CUDA / VRAM / driver visibility.

Run: `python src/check_env.py`
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys


def _hr(title: str) -> None:
    print(f"\n=== {title} ===")


def _print_kv(key: str, value: object) -> None:
    print(f"  {key:<28} {value}")


def main() -> int:
    _hr("System")
    _print_kv("Python", sys.version.split()[0])
    _print_kv("Platform", platform.platform())
    _print_kv("Executable", sys.executable)

    _hr("PyTorch")
    try:
        import torch  # type: ignore
    except ImportError as exc:
        print(f"  torch not installed: {exc}")
        print("  -> see INSTALL.md (section 'PyTorch')")
        return 1

    _print_kv("torch", torch.__version__)
    _print_kv("compiled CUDA", torch.version.cuda)
    cuda_ok = torch.cuda.is_available()
    _print_kv("cuda.is_available()", cuda_ok)

    if cuda_ok:
        n = torch.cuda.device_count()
        _print_kv("device_count", n)
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            _print_kv(f"  GPU[{i}]", props.name)
            _print_kv(f"  GPU[{i}] VRAM", f"{props.total_memory / 1024**3:.2f} GiB")
            _print_kv(f"  GPU[{i}] compute", f"{props.major}.{props.minor}")
    else:
        print("  CUDA unavailable — check driver and PyTorch CUDA wheel (see INSTALL.md).")

    _hr("nvidia-smi")
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                 "--format=csv,noheader"],
                text=True,
                timeout=10,
            ).strip()
            for line in out.splitlines():
                print(f"  {line}")
        except (subprocess.SubprocessError, OSError) as exc:
            print(f"  nvidia-smi failed: {exc}")
    else:
        print("  nvidia-smi not on PATH")

    _hr("Hunyuan3D-Part importability")
    try:
        import importlib

        for mod in ("P3SAM", "p3sam", "XPart", "xpart"):
            try:
                importlib.import_module(mod)
                _print_kv(mod, "OK")
            except ImportError:
                _print_kv(mod, "not importable")
        print("  (a missing module is OK if its component isn't installed yet)")
    except Exception as exc:  # noqa: BLE001
        print(f"  unexpected error: {exc}")

    print()
    return 0 if cuda_ok else 2


if __name__ == "__main__":
    sys.exit(main())
