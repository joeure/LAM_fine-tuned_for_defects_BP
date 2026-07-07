import sys, platform
import torch
import deepmd, dpdata

print("Python:", sys.version.splitlines()[0])
print("Executable:", sys.executable)
print("Platform:", platform.platform())

def v(name):
    try:
        m = __import__(name)
        ver = getattr(m, "__version__", getattr(m, "version", "unknown"))
        print(f"{name:10s} {ver}")
    except Exception as e:
        print(f"{name:10s} NOT INSTALLED ({e.__class__.__name__}: {e})")

for pkg in ["torch","deepmd","dpdata","numpy","scipy","pandas","matplotlib"]:
    v(pkg)

# PyTorch CUDA / cuDNN details
try:
    print("\nPyTorch build CUDA:", torch.version.cuda)
    print("PyTorch version:    ", torch.__version__)
    print("cuDNN version:      ", getattr(torch.backends.cudnn, "version", lambda: None)())
    print("CUDA available?:    ", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device count:       ", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            cap = ".".join(map(str, torch.cuda.get_device_capability(i)))
            print(f"  [{i}] {torch.cuda.get_device_name(i)} (cc {cap})")
    # Show compile flags / linked libs
    import torch.__config__ as tc
    print("\n--- torch.__config__.show() ---")
    tc.show()
except Exception as e:
    print("\n(PyTorch extra info unavailable)", e)

# BLAS used by NumPy (MKL/OpenBLAS)
try:
    import numpy, numpy.__config__ as nc
    print("\n--- numpy.__config__.show() ---")
    nc.show()
except Exception:
    pass


from deepmd.infer.deep_eval import DeepEval
m = DeepEval("./DPA-3.1-3M.pt", head=0, model_branch="Alex2D", device="cpu")
print(getattr(m, "type_map", "type_map not exposed in this build"))