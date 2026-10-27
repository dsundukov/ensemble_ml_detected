import os
import platform
import subprocess

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch


def _sysctl(name: str):
    try:
        return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def cpu_name() -> str:
    return _sysctl("machdep.cpu.brand_string") or platform.processor() or platform.machine()


def performance_cores():
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        v = _sysctl("hw.perflevel0.physicalcpu")
        return int(v) if v and v.isdigit() else None
    return None


def setup_cpu(num_threads=None):
    if platform.system() == "Darwin" and platform.machine() != "arm64":
        raise SystemExit("Python запущен через Rosetta (x86_64): обучение на M4 будет в разы медленнее. "
                         "Установите arm64-версию Python.")
    n = num_threads or performance_cores()
    if n:
        torch.set_num_threads(n)
        os.environ["OMP_NUM_THREADS"] = str(n)
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass
    return torch.get_num_threads()


def get_device(preference: str = "cpu") -> str:
    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def describe(device: str) -> str:
    if device == "cuda":
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    if device == "mps":
        return "Apple Silicon GPU (MPS)"
    perf = performance_cores()
    cores = f"{os.cpu_count()} ядер" + (f", из них производительных {perf}" if perf else "")
    return f"CPU {cpu_name()} ({torch.get_num_threads()} потоков PyTorch, {cores})"
