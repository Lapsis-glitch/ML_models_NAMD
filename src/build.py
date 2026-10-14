"""
Build a NAMD-ready model in one call: compile, (build native op libraries,)
wrap.

    python -m src.build MODEL_TYPE SOURCE [--fast] [--out mlff_model.pt] [options]

    python -m src.build mace     MACE-OFF23_medium.model --fast --mace-python <MACE_312 python>
    python -m src.build nequip   nequip.net:mir-group/NequIP-OAM-L:0.1 --r-max 6.0 --fast
    python -m src.build allegro  my_allegro.nequip.zip
    python -m src.build sevennet 7net-0 --fast --d3
    python -m src.build torchani ani2x --fast
    python -m src.build schnet   schnet_scripted.pt --r-max 5.0 --fast
    python -m src.build xmace    fulvene.model --state 1

SOURCE is what the model's compile script takes (a checkpoint, a package, a
pretrained name), or an already compiled TorchScript file, which skips the
compile step.  ``--fast`` builds the optimised model (GPU): same weights, faster
kernels, checked against the stock model while it is built.  When the result
needs custom-op libraries, they are built once against ``--libtorch`` (NAMD's
libtorch) and the line to export for NAMD is printed at the end.
"""

import argparse
import os
import subprocess
import sys
import zipfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_TYPES = ["mace", "nequip", "allegro", "sevennet", "torchani", "schnet", "xmace"]
NO_FAST = {"allegro", "xmace"}


def is_torchscript(path: str) -> bool:
    """Whether ``path`` is a saved TorchScript module (a zip holding
    ``constants.pkl`` and ``code/``, unlike torch.save / torch.package files)."""
    if not os.path.isfile(path) or not zipfile.is_zipfile(path):
        return False
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
    return (any(n.endswith("/constants.pkl") for n in names)
            and any("/code/" in n for n in names))


def inner_path(out: str, model_type: str) -> str:
    stem = out[:-3] if out.endswith(".pt") else out
    return stem + ("_inner.nequip.pth" if model_type in ("nequip", "allegro") else "_inner.pt")


def compile_inner(a, inner: str) -> tuple:
    """Step 1.  Returns (native library set or "", extra src.cli args)."""
    t, src, fast = a.model_type, a.source, a.fast
    if t == "mace":
        if fast:
            from .fast.mace import build_fast_mace
            return build_fast_mace(src, inner, dtype=a.dtype, mace_python=a.mace_python), []
        from .compile_mace import compile_mace
        compile_mace(src, inner, dtype=a.dtype)
        return "", []
    if t in ("nequip", "allegro"):
        if fast:
            from .fast.nequip import build_fast_nequip
            return build_fast_nequip(src, inner), []
        from .compile_nequip import compile_nequip
        compile_nequip(src, inner, device=a.compile_device)
        return "", []
    if t == "sevennet":
        from .compile_sevennet import compile_sevennet
        kind = compile_sevennet(src, inner, modal=a.modal, fast=fast)
        return ("oeq" if kind != "stock" else ""), []
    if t == "torchani":
        from .compile_torchani import compile_torchani
        if fast:
            from .fast.native import ensure
            from .fast.ani import build_fast_ani
            elements = build_fast_ani(src, inner, ensure("cuaev", a.libtorch)[0])
            return "cuaev", ["--elements", ",".join(map(str, elements)), "--fast"]
        elements = compile_torchani(src, inner)
        return "", ["--elements", ",".join(map(str, elements))]
    if t == "xmace":
        from .compile_xmace import compile_xmace
        compile_xmace(src, inner)
        return "", []
    raise SystemExit(f"{t}: SOURCE must be a compiled TorchScript file")


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Compile + wrap a model for NAMD in one call.",
        epilog="Model-specific options are marked [type].")
    p.add_argument("model_type", choices=MODEL_TYPES)
    p.add_argument("source", help="checkpoint / package / pretrained name, or a compiled TorchScript file")
    p.add_argument("--out", default="mlff_model.pt", help="NAMD model file (default: mlff_model.pt)")
    p.add_argument("--fast", action="store_true", help="optimised build (GPU)")
    p.add_argument("--libtorch", default=os.environ.get("TORCH"),
                   help="libtorch NAMD uses, for the native op libraries "
                        "(default: $TORCH, else the torch of this Python)")
    p.add_argument("--compile-device", default="cuda",
                   help="[nequip, allegro] device the model will run on (default: cuda)")
    p.add_argument("--dtype", default="float64", choices=["float64", "float32"],
                   help="[mace] model precision (default: float64)")
    p.add_argument("--mace-python", default=os.environ.get("MACE_PY"),
                   help="[mace --fast] Python that can unpickle the .model (default: $MACE_PY)")
    p.add_argument("--modal", default=None, help="[sevennet] fidelity of a multi-fidelity checkpoint")
    p.add_argument("--r-max", type=float, default=0.0,
                   help="[schnet] training cutoff in Å (required); [nequip] if the model has none")
    p.add_argument("--elements", default=None,
                   help="[torchani] element order of a compiled SOURCE (default: ANI-2x)")
    p.add_argument("--state", type=int, default=0, help="[xmace] electronic state (default: 0)")
    p.add_argument("--d3", action="store_true", help="add D3(BJ) dispersion (needs sevenn installed)")
    p.add_argument("--d3-functional", default="pbe", help="[--d3] functional (default: pbe)")
    a = p.parse_args(argv)

    if a.fast and a.model_type in NO_FAST:
        p.error(f"{a.model_type} has no optimised build; drop --fast")
    if a.fast:
        from .fast.native import ensure_cuda_lib_path
        ensure_cuda_lib_path("src.build")

    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    prebuilt = is_torchscript(a.source)
    if prebuilt and a.fast and a.model_type != "schnet":
        p.error("--fast rebuilds the model from its weights; SOURCE must be the "
                "checkpoint / package / pretrained name, not a compiled file")

    print(f"== 1. compile  ({a.model_type}{', optimised' if a.fast else ''})")
    if prebuilt or a.model_type == "schnet":
        inner, libset, extra = os.path.abspath(a.source), "", []
        if a.model_type == "torchani" and a.elements:
            extra = ["--elements", a.elements]
        print(f"  using the compiled model {inner}")
    else:
        inner = inner_path(out, a.model_type)
        libset, extra = compile_inner(a, inner)

    libs = []
    if libset:
        from .fast.native import ensure
        print(f"== 2. native op libraries ({libset})")
        libs = ensure(libset, a.libtorch)

    print("== 3. wrap")
    wrap = ["--model-type", a.model_type, "--compiled", inner, "--out", out] + extra
    if a.r_max > 0:
        wrap += ["--r-max", str(a.r_max)]
    if a.model_type == "xmace":
        wrap += ["--state", str(a.state)]
    if a.fast and a.model_type == "schnet":
        wrap += ["--fast"]
    if a.d3:
        wrap += ["--d3", "--d3-functional", a.d3_functional]
    if libs:
        # a fresh process that loads only the native libraries, as NAMD does
        wrap += ["--extra-libs", ":".join(libs)]
        subprocess.run([sys.executable, "-m", "src.cli"] + wrap, cwd=REPO, check=True)
    else:
        from .cli import main as wrap_main
        wrap_main(wrap)

    print(f"\nNAMD model: {out}")
    if libs:
        print("Before running NAMD:\n  export NAMD_MLFF_EXTRA_LIBS=" + ":".join(libs))


if __name__ == "__main__":
    main()
