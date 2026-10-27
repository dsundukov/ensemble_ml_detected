import platform
import sys

print(f"Python:      {sys.version.split()[0]}")
print(f"Архитектура: {platform.machine()}")
if platform.system() == "Darwin" and platform.machine() != "arm64":
    print("  [!] Python запущен через Rosetta (x86_64): MPS недоступен. "
          "Установите arm64-версию Python (python.org, Homebrew или Miniforge).")

try:
    from device import describe, get_device
    import torch
    print(f"PyTorch:     {torch.__version__}")
    print(f"MPS собран:  {torch.backends.mps.is_built()}")
    print(f"MPS доступен:{torch.backends.mps.is_available()}")
    from config import Config
    dev = get_device(Config().device)
    x = torch.randn(512, 512, device=dev)
    _ = (x @ x).sum().item()
    print(f"Устройство:  {describe(dev)} — тест вычислений пройден")
except Exception as e:
    print(f"  [!] Ошибка PyTorch: {e}")

for pkg in ("transformers", "datasets", "sklearn", "safetensors", "onnxruntime", "sentencepiece"):
    try:
        mod = __import__(pkg)
        print(f"{pkg:<13}{getattr(mod, '__version__', 'ok')}")
    except ImportError:
        print(f"{pkg:<13}НЕ УСТАНОВЛЕН")
