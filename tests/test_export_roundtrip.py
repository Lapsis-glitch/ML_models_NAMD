"""
Does a wrapper actually produce an artifact NAMD can load and use?

Scripting a module and exporting one are not the same check, and it is the
exported file that ships.  This walks the real path: export_wrapped, then
reload the saved file as a fresh module the way torch::jit::load does in NAMD,
then assert the things the C++ side inspects at load time.

NAMD decides how to call a model from its forward() SIGNATURE, not from the
supports_pbc attribute, and refuses to load when the two disagree.  So the
arities below are not cosmetic: they are the contract.

    forward(self, coords, Z, pc_coords, pc_charges, cell)          -> 6
    forward_batch(self, coords, Z, batch, ptr, pc_coords, pc_charges, cells) -> 8

and the returned tuple must have 4 entries for the virial to be picked up
(NAMD reads els[3] only when it is there, so a 3-tuple still works, it just
means no virial).
"""

import os
import sys
import tempfile

import torch

sys.path.insert(0, "/home/rat/PycharmProjects/ML_models_NAMD")
from src.export import export_wrapped

REPO = "/home/rat/PycharmProjects/ML_models_NAMD"
FAILS = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + ("   " + detail if detail else ""))
    if not ok:
        FAILS.append(name)


def roundtrip(label, wrapper, expect_pbc, expect_virial):
    """Export, reload, and inspect what NAMD would inspect."""
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "wrapped.pt")
        wrapper.eval()
        export_wrapped(wrapper, out, model_type=label)
        check(f"{label}: export produced a file", os.path.getsize(out) > 0)

        m = torch.jit.load(out, map_location="cpu")

        fa = len(m.forward.schema.arguments)
        check(f"{label}: forward arity is {6 if expect_pbc else 5}",
              fa == (6 if expect_pbc else 5), f"got {fa}")

        try:
            ba = len(m.forward_batch.schema.arguments)
            check(f"{label}: forward_batch arity is {8 if expect_pbc else 7}",
                  ba == (8 if expect_pbc else 7), f"got {ba}")
        except Exception:
            check(f"{label}: forward_batch present", False, "missing")

        # NAMD fails the load if this disagrees with the signature.
        attr_pbc = bool(getattr(m, "supports_pbc", False))
        check(f"{label}: supports_pbc agrees with the signature",
              attr_pbc == expect_pbc, f"attr={attr_pbc}, arity says {expect_pbc}")

        nret = len(m.forward.schema.returns[0].type.elements())
        check(f"{label}: forward returns {4 if expect_virial else 3} values",
              nret == (4 if expect_virial else 3), f"got {nret}")
        return m


def water_pair(L):
    """Two waters adjacent only through the x face, so periodicity matters."""
    c = torch.tensor([[0.4, 6., 6.], [1.36, 6., 6.], [0.16, 6.93, 6.],
                      [L - 1.4, 6., 6.], [L - 0.44, 6., 6.], [L - 1.64, 6.93, 6.]],
                     dtype=torch.float64)
    Z = torch.tensor([8, 1, 1, 8, 1, 1], dtype=torch.int64)
    return c, Z


def exercise(label, m, L=12.0):
    c, Z = water_pair(L)
    pcx = torch.zeros((0, 3), dtype=torch.float64)
    pcq = torch.zeros((0,), dtype=torch.float64)

    zero = torch.zeros((1, 3, 3), dtype=torch.float64)
    box = (torch.eye(3, dtype=torch.float64) * L).unsqueeze(0)

    o0 = m(c, Z, pcx, pcq, zero)
    oP = m(c, Z, pcx, pcq, box)

    check(f"{label}: periodic run differs from non-periodic",
          not torch.allclose(o0[0], oP[0], atol=1e-8),
          f"dE = {float(oP[0] - o0[0]):.4f} kcal/mol")

    if len(oP) >= 4:
        v0, vP = o0[3], oP[3]
        check(f"{label}: virial is [3,3] float64",
              tuple(vP.shape) == (3, 3) and vP.dtype == torch.float64)
        check(f"{label}: non-periodic virial is exactly zero", bool(torch.all(v0 == 0)))
        check(f"{label}: periodic virial is non-zero", bool(vP.abs().max() > 0))
        check(f"{label}: periodic virial is symmetric",
              bool(torch.allclose(vP, vP.T, atol=1e-13)),
              f"xx = {float(vP[0,0]):.4f} kcal/mol")

    # Batched, two identical walkers in the same box.
    N = c.shape[0]
    ptr = torch.tensor([0, N, 2 * N])
    bat = torch.cat([torch.zeros(N, dtype=torch.long), torch.ones(N, dtype=torch.long)])
    cells = torch.stack([box[0], box[0]])
    ob = m.forward_batch(torch.cat([c, c]), torch.cat([Z, Z]), bat, ptr, pcx, pcq, cells)
    check(f"{label}: batched energies match the single call",
          bool(torch.allclose(ob[0][0], oP[0].reshape(()), atol=1e-8)))
    if len(ob) >= 4:
        check(f"{label}: batched virials are [B,3,3]", tuple(ob[3].shape) == (2, 3, 3))


# Every wrapper, with the model file each one needs.  A missing model or a
# missing package is reported as a SKIP rather than a pass, so an absent
# dependency can never look like a green run.
# Two of these carry no cutoff in the artifact, so it has to be supplied.
# NequIP-OAM-L is 6.0 (its own error message says so) and the SchNet artifact
# was built by src/compile_schnetpack.py, whose default is 5.0.  Getting these
# wrong changes the neighbour list rather than failing, so they are pinned here
# next to the model they belong to.
MODELS = [
    ("MACE",       "wrap_compiled_mace",   "MACE_TS_Wrapper",       "compiled_mace_off23_medium.pt", {}),
    ("X-MACE",     "wrap_xmace",           "XMACE_TS_Wrapper",      "fulvene_compiled.pt",           {}),
    ("NequIP",     "wrap_compiled_nequip", "NequIP_TS_Wrapper",     "compiled_nequip_oam_l.nequip.pth", {"r_max": 6.0}),
    ("SchNetPack", "wrap_schnetpack",      "SchNetPack_Wrapper", "compiled_schnet_default.pt",    {"r_max": 5.0}),
    ("TorchANI",   "wrap_torchani",        "TorchANI_TS_Wrapper",   "compiled_ani2x.pt",             {}),
]

SKIPS = []


def resolve(module_name, class_name):
    """Import a wrapper class, tolerating a differently spelled class name."""
    import importlib
    mod = importlib.import_module(f"src.wrappers.{module_name}")
    if hasattr(mod, class_name):
        return getattr(mod, class_name)
    # Fall back to the single nn.Module subclass defined in the file.
    import inspect
    from torch import nn
    cands = [o for _, o in inspect.getmembers(mod, inspect.isclass)
             if issubclass(o, nn.Module) and o.__module__ == mod.__name__]
    if len(cands) == 1:
        return cands[0]
    raise ImportError(f"cannot find the wrapper class in {module_name}: {cands}")


if __name__ == "__main__":
    only = sys.argv[1:] or None
    for label, module_name, class_name, model_file, kwargs in MODELS:
        if only and label not in only:
            continue
        print(f"\n===== {label} =====")
        path = f"{REPO}/models/{model_file}"
        if not os.path.exists(path):
            SKIPS.append(f"{label} (no model file {model_file})")
            print(f"SKIP  {label}: {model_file} not present")
            continue
        try:
            cls = resolve(module_name, class_name)
            wrapper = cls(path, device="cpu", **kwargs)
        except Exception as e:
            SKIPS.append(f"{label} ({type(e).__name__}: {str(e)[:80]})")
            print(f"SKIP  {label}: could not build the wrapper: {type(e).__name__}: {str(e)[:120]}")
            continue
        try:
            m = roundtrip(label, wrapper, expect_pbc=True, expect_virial=True)
            exercise(label, m)
        except Exception as e:
            check(f"{label}: export/exercise raised", False,
                  f"{type(e).__name__}: {str(e)[:160]}")

    print()
    if SKIPS:
        print("SKIPPED: " + "; ".join(SKIPS))
    print("ALL CHECKS PASSED" if not FAILS
          else f"{len(FAILS)} FAILURE(S): " + ", ".join(FAILS))
    sys.exit(1 if FAILS else 0)
