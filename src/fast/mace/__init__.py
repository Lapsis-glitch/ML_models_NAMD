"""Optimised MACE inner: cuEquivariance kernels (fused conv TP) + FastMACE rewrites.

The model is rebuilt under the current mace/e3nn from its weights
(``rebuild.py``), converted to cuEquivariance (``cueq.py`` makes it scriptable),
rewritten by ``FastMACE`` (``fast.py``) and checked against the stock model
before it is saved.

    python -m src.fast.mace MODEL OUT [--kind e3nn|cueq|cueqf] [--dtype float32] ...

exposes the ablation switches; ``python -m src.compile_mace --fast`` is the
recommended build.
"""
import argparse
import os
import subprocess
import tempfile

import torch

from .. import report_parity, water_box

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _is_state(path: str) -> bool:
    try:
        d = torch.load(path, weights_only=False, map_location="cpu")
    except Exception:
        return False
    return isinstance(d, dict) and "config" in d and "state_dict" in d


def state_file(model: str, workdir: str, mace_python: str = None) -> str:
    """A version-neutral weights file for ``model`` (a ``*.model`` or an
    existing dump), extracting it in-process or with ``mace_python``."""
    if _is_state(model):
        return model
    out = os.path.join(workdir, "mace_state.pt")
    if mace_python:
        subprocess.run([mace_python, "-m", "src.fast.mace.extract_state", model, out],
                       cwd=REPO, check=True,
                       env=dict(os.environ, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1"))
        return out
    try:
        from .extract_state import extract_state
        extract_state(model, out)
    except Exception as exc:
        raise RuntimeError(
            f"cannot read {model} in this environment ({type(exc).__name__}: {exc}). "
            "Models pickled by an older e3nn (MACE-OFF23) must be read by the env "
            "that wrote them: pass --mace-python <that env's python>.") from exc
    return out


def _stock(state, dtype, kind, dev):
    from .rebuild import rebuild
    if kind == "e3nn":
        return rebuild(state, dtype, device=dev)
    import cuequivariance_torch  # noqa: F401  registers the cuEq torch ops
    from .cueq import make_scriptable
    m = rebuild(state, dtype, cueq=True, device=dev, conv_fusion=(kind == "cueqf"))
    return make_scriptable(m)


def _inputs(m, n, dtype, dev):
    """Water geometry with Z cycled through every element of the model (an
    unknown Z would get an all-zero one-hot on both sides and pass trivially)."""
    from ...edges import build_edges
    zs = m.atomic_numbers.cpu()
    Z = zs[torch.arange(n) % zs.numel()].to(dev)
    xyz = water_box(n).to(dev)
    ei, _, _ = build_edges(xyz.float(), float(m.r_max))
    E = ei.size(1)
    return {"positions": xyz.to(dtype), "node_attrs": (Z[:, None] == m.atomic_numbers.to(dev)[None]).to(dtype),
            "edge_index": ei,
            "shifts": torch.zeros(E, 3, dtype=dtype, device=dev),
            "unit_shifts": torch.zeros(E, 3, dtype=dtype, device=dev),
            "cell": torch.zeros(3, 3, dtype=dtype, device=dev),
            "batch": torch.zeros(n, dtype=torch.long, device=dev),
            "ptr": torch.tensor([0, n], device=dev)}


def _check(tag, model, ref, sizes, dtype, dev, calls=1):
    tol = 1e-10 if dtype == torch.float64 else 2e-4  # on max|dF| / max(1, |F|max)
    for n in sizes:
        d = _inputs(ref, n, dtype, dev)
        r = ref(dict(d), compute_force=True)
        for _ in range(calls):  # profiling executor: the 3rd call runs the optimised graph
            o = model(dict(d), compute_force=True)
        err = report_parity(tag, n, r["energy"], o["energy"], r["forces"], o["forces"], "eV")
        if err >= tol:
            raise RuntimeError(f"{tag} model differs from the stock model (relative dF {err:.1e})")


def build_fast_mace(model: str, out: str, kind: str = "cueqf", dtype: str = "float64",
                    half_radial: bool = True, species_skip: bool = True, plain_linear: bool = False,
                    cache_species: bool = False, check: int = 300, mace_python: str = None) -> str:
    """Build the optimised MACE inner ``out`` from ``model``.

    Returns the native library set it needs at run time (``"cueq"``) or ``""``.
    Falls back to the kernels without FastMACE if the rewrites don't fit the
    model's structure.
    """
    from e3nn.util import jit as ej
    from .fast import FastMACE

    if not torch.cuda.is_available():
        raise RuntimeError("the optimised MACE build needs a CUDA GPU")
    dt = getattr(torch, dtype)
    dev = "cuda"
    with tempfile.TemporaryDirectory() as tmp:
        state = state_file(model, tmp, mace_python)
        m = _stock(state, dt, kind, dev)
    try:
        fm = FastMACE(m, half_radial=half_radial, species_skip=species_skip,
                      plain_linear=plain_linear, cache_species=cache_species).eval()
        _check("eager FastMACE", fm, m, [check], dt, dev)
    except (AssertionError, NotImplementedError) as exc:
        print(f"[MACE] FastMACE rewrites don't fit this model ({exc}); "
              "building with the kernels only.")
        fm = m
    s = torch.jit.script(ej.compile(fm))
    _check("scripted", s, m, [30, check], dt, dev, calls=3)
    # weights on the CPU so torch.jit.load(map_location=...) works anywhere
    s.to("cpu").save(out)
    print(f"[MACE] optimised inner ({kind}, {dtype})  ->  {out}")
    return "" if kind == "e3nn" else "cueq"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build an optimised MACE inner (ablation switches).")
    ap.add_argument("model", help="MACE *.model, or a dump from src.fast.mace.extract_state")
    ap.add_argument("out")
    ap.add_argument("--kind", default="cueqf", choices=["e3nn", "cueq", "cueqf"],
                    help="e3nn: no custom ops; cueq: cuEquivariance; cueqf: + fused conv (default)")
    ap.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    ap.add_argument("--no-half", action="store_true")
    ap.add_argument("--no-species-skip", action="store_true")
    ap.add_argument("--plain-linear", action="store_true")
    ap.add_argument("--cache-species", action="store_true")
    ap.add_argument("--check", type=int, default=300)
    ap.add_argument("--mace-python", default=os.environ.get("MACE_PY"))
    a = ap.parse_args(argv)
    build_fast_mace(a.model, a.out, kind=a.kind, dtype=a.dtype, half_radial=not a.no_half,
                    species_skip=not a.no_species_skip, plain_linear=a.plain_linear,
                    cache_species=a.cache_species, check=a.check, mace_python=a.mace_python)
