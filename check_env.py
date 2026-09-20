import importlib
import inspect
import sys

EXPECTED = {
    "gradio": "6.27.0",
    "gradio_client": "2.7.0",
    "fastapi": "0.141.1",
    "starlette": "1.6.0",
    "uvicorn": "0.52.0",
    "jinja2": "3.1.6",
}

ok = True
for name, expected in EXPECTED.items():
    try:
        module = importlib.import_module(name)
        actual = getattr(module, "__version__", "unknown")
        print(f"{name}: {actual} (expected {expected})")
        if not actual.startswith(expected):
            ok = False
    except Exception as exc:
        print(f"{name}: IMPORT FAILED: {exc}")
        ok = False

try:
    import torch
    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA build:", torch.version.cuda)
    cuda_ok = bool(torch.cuda.is_available())
    print("CUDA available:", cuda_ok)
    if cuda_ok:
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("WARNING: CUDA is unavailable in this Python environment; ReverseTagger requires CUDA.")
        ok = False
    if torch.__version__.split("+")[0] != "2.14.0":
        ok = False
except Exception as exc:
    print("PyTorch check failed:", exc)
    ok = False

try:
    import torchvision
    print("torchvision:", torchvision.__version__)
    if not torchvision.__version__.startswith("0.29.0"):
        ok = False
except Exception as exc:
    print("torchvision check failed:", exc)
    ok = False

try:
    import gradio as gr
    sig = inspect.signature(gr.Blocks.launch)
    print("Blocks.launch supports theme:", "theme" in sig.parameters)
    print("Blocks.launch supports css:", "css" in sig.parameters)
    if "theme" not in sig.parameters or "css" not in sig.parameters:
        ok = False
except Exception as exc:
    print("Gradio launch signature check failed:", exc)
    ok = False

print("Python:", sys.version.split()[0])
if sys.version_info[:2] != (3, 10):
    ok = False

raise SystemExit(0 if ok else 1)
