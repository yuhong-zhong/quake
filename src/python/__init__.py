try:
    import ctypes
    import sys
    import os
    from pathlib import Path
    
    # Preload conda's libstdc++ to prevent CXXABI version conflicts with system libstdc++
    conda_libstdcpp = Path(sys.prefix) / "lib" / "libstdc++.so.6"
    if conda_libstdcpp.exists():
        ctypes.CDLL(str(conda_libstdcpp), mode=ctypes.RTLD_GLOBAL)

    import torch  # noqa: F401

    from ._bindings import *  # noqa: F401 F403

except ModuleNotFoundError as e:
    print(e)
    if "torch" in str(e):
        print("ImportError: Pytorch not found. Please install pytorch first.")
    else:
        print("Bindings not installed")

