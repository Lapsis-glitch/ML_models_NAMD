"""Native (Python-free) custom-op libraries used by the optimised builds.

NAMD's shim loads libtorch only, so the custom ops of the optimised models are
registered by small C++ libraries, passed to NAMD through
``NAMD_MLFF_EXTRA_LIBS``.  Each library is built once per machine by
``src/fast/native/<name>/build.sh`` against the libtorch NAMD uses
(``TORCH=<libtorch dir>``; default: the torch of the current Python).

    cueq   MACE  (cuEquivariance)    nvrtc + libcue_ops + libcueq_uniform1d_native.so
    oeq    NequIP, SevenNet (OpenEquivariance)               liboeq_native.so
    cuaev  ANI   (cuAEV)                          libcuaev_native_precise.so
"""

import os
import site
import subprocess
import sys
from typing import List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))

_BUILT = {
    "cueq": os.path.join(HERE, "cueq", "libcueq_uniform1d_native.so"),
    "oeq": os.path.join(HERE, "oeq", "liboeq_native.so"),
    "cuaev": os.path.join(HERE, "cuaev", "libcuaev_native_precise.so"),
}


def site_packages() -> str:
    return site.getsitepackages()[0]


def nvrtc_library() -> str:
    """NVRTC of the CUDA version torch was built with, from the nvidia pip
    wheels ("" if they are not installed, e.g. with a system CUDA)."""
    import glob
    import torch
    major = (torch.version.cuda or "").split(".")[0]
    hits = sorted(glob.glob(os.path.join(site_packages(), "nvidia", "*", "lib",
                                         f"libnvrtc.so.{major}*")))
    return hits[0] if major and hits else ""


def cuda_lib_dir() -> str:
    """Directory of the pip-installed NVRTC ("" if none)."""
    lib = nvrtc_library()
    return os.path.dirname(lib) if lib else ""


def lib_paths(name: str) -> List[str]:
    """Every library NAMD must load for ``name``, in load order."""
    if name == "cueq":
        sp = site_packages()
        nvrtc = [nvrtc_library()] if nvrtc_library() else []
        return nvrtc + [os.path.join(sp, "cuequivariance_ops", "lib", "libcue_ops.so"),
                        _BUILT["cueq"]]
    return [_BUILT[name]]


def build(name: str, libtorch: Optional[str] = None) -> None:
    env = dict(os.environ, PY=sys.executable)
    if libtorch:
        env["TORCH"] = libtorch
    print(f"[native] building {name} library ({env.get('TORCH', 'torch of this Python')})")
    subprocess.run(["bash", os.path.join(HERE, name, "build.sh")], env=env, check=True)


def ensure(name: str, libtorch: Optional[str] = None, rebuild: bool = False) -> List[str]:
    """Paths of ``name``'s libraries, building the native one first if needed."""
    if rebuild or not os.path.exists(_BUILT[name]):
        build(name, libtorch)
    missing = [p for p in lib_paths(name) if not os.path.exists(p)]
    if missing:
        raise RuntimeError(f"missing native libraries for {name}: {missing}")
    return lib_paths(name)


def load(paths: List[str]) -> None:
    import torch
    for p in paths:
        torch.ops.load_library(p)


def ensure_cuda_lib_path(module: str) -> None:
    """Re-exec this process with the pip-installed NVRTC on LD_LIBRARY_PATH.

    cuEquivariance and the TorchScript fuser dlopen NVRTC by name; without it
    on the library path cuEquivariance silently falls back to its slow
    kernels, so the optimised build would bake those in.
    """
    d = cuda_lib_dir()
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    if not d or d in cur.split(":"):
        return
    env = dict(os.environ, LD_LIBRARY_PATH=f"{d}:{cur}" if cur else d)
    os.execve(sys.executable, [sys.executable, "-m", module] + sys.argv[1:], env)
