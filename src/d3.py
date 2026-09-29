"""
Grimme D3(BJ) dispersion in TorchScript, as an add-on to any wrapper.

This reproduces SevenNet's D3 (``sevenn.calculator.D3Calculator``, the CUDA
``pair_d3`` also used from LAMMPS) so that ``SevenNetD3Calculator`` results can
be had inside NAMD.  SevenNet's kernel is a ctypes/CUDA library that NAMD's
shim can't load, so the same maths is written here in plain torch and exported
with the model.  Nothing about it is SevenNet-specific: ``D3_Wrapper`` adds the
dispersion energy, forces and virial to whatever wrapper it is given.

The parameters are read at wrap time from the installed ``sevenn`` package
(``pair_d3_pars.h`` and ``pair_d3_for_ase.cu``) and saved in the exported file,
so the numbers are exactly SevenNet's, down to its unit constants, and NAMD
needs nothing extra at run time.

What is computed (atomic units inside, eV out), for pairs within the cutoff
over all periodic images T, self-images included:

    CN_i  = sum_j,T  1 / (1 + exp(-K1 (rcov_i + rcov_j) / r - 1)),  r^2 <= cn_cutoff
    C6_ij = sum_ab C6ref[a,b] L_ia L_jb / sum_ab L_ia L_jb,  L_ia = exp(K3 (CN_i - CNref_ia)^2)
    E     = -1/2 sum_ij,T  C6_ij (s6 / (r^6 + R0^6) + s8 C8/C6 / (r^8 + R0^8)),  r^2 <= cutoff
    C8/C6 = 3 r2r4_i r2r4_j,   R0 = a1 sqrt(C8/C6) + a2

Two-body only (SevenNet has no three-body term) and BJ damping only.  The
exponent in L factorises, so C6 is evaluated as w_i^T C6ref w_j with per-atom
softmax weights; that is the same number, and it has no underflow fallback to
need (SevenNet's ``c6mem`` branch, which only fires if every reference weight
underflows).  Cutoffs are hard, as in SevenNet, and given in bohr^2 like
SevenNet's ``vdw_cutoff`` / ``cn_cutoff``: the defaults 9000 and 1600 are
50.2 A and 21.2 A.

Cost: every pair-image within 50 A.  That is all pairs for a molecule, and for
a periodic box of side L roughly N^2 (100 / L)^3 pair-images.  Large periodic
boxes should lower ``cutoff``.  Everything runs in float64.
"""

import os
import re
from typing import Dict, List, Tuple

import torch
from torch import nn

from .constants import EV_TO_KCAL
from .edges import cell_is_periodic
from .virial import (make_strain, apply_strain, forces_and_virial, finalize,
                     zero_virial)


# SevenNet's constants, kept verbatim so results match it (they are not CODATA).
AU_TO_ANG = 0.52917726
AU_TO_EV = 27.21138505
K1 = 16.0
K3 = -4.0
MAXC = 5            # reference systems per element
NZ = 95             # table rows: index = Z, 1..94


# -------------------------------------------------------------------
#  Parameters, read from the installed sevenn package
# -------------------------------------------------------------------

def _sevenn_source_dir() -> str:
    try:
        import sevenn
    except ImportError as e:
        raise RuntimeError("D3 parameters are read from the installed sevenn "
                           "package; pip install sevenn") from e
    return os.path.join(os.path.dirname(sevenn.__file__), "pair_e3gnn")


def _floats(text: str) -> List[float]:
    return [float(x) for x in
            re.findall(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", text)]


def _array(src: str, name: str) -> List[float]:
    m = re.search(rf"double {name}\[94\]\s*=\s*\{{(.*?)\}};", src, re.S)
    if m is None:
        raise RuntimeError(f"{name} not found in sevenn's pair_d3 source")
    vals = _floats(m.group(1))
    if len(vals) != 94:
        raise RuntimeError(f"{name}: expected 94 values, got {len(vals)}")
    return vals


def _bj_functionals(src: str) -> Dict[str, Dict[str, float]]:
    """{functional: {s6, s8, a1, a2}} as SevenNet's setfuncpar_bj sets them."""
    body = src[src.index("void PairD3::setfuncpar_bj()"):
               src.index("void PairD3::setfuncpar_zerom()")]
    codes = {int(c): name for name, c in re.findall(r'\{"([^"]+)",\s*(\d+)\}', body)}
    out: Dict[str, Dict[str, float]] = {}
    # Only what runs before `break;` counts: SevenNet's b2-plyp line sets
    # s6 = 0.64 after its break, so SevenNet uses s6 = 1 there.  Same here.
    for code, stmts in re.findall(r"case (\d+):(.*?)break;", body, re.S):
        p = {k: float(v) for k, v in re.findall(r"(\w+)\s*=\s*([-+]?[\d.]+)", stmts)}
        out[codes[int(code)]] = {"s6": p.get("s6", 1.0), "s8": p["s18"],
                                 "a1": p["rs6"], "a2": p["rs18"]}
    return out


def available_functionals() -> List[str]:
    with open(os.path.join(_sevenn_source_dir(), "pair_d3_for_ase.cu")) as f:
        return sorted(_bj_functionals(f.read()))


def load_d3_tables() -> Dict[str, torch.Tensor]:
    """Reference C6 table, reference CNs, rcov and r2r4, indexed by Z."""
    d = _sevenn_source_dir()
    with open(os.path.join(d, "pair_d3_for_ase.cu")) as f:
        src = f.read()
    with open(os.path.join(d, "pair_d3_pars.h")) as f:
        pars = f.read()

    start = pars.index("{", pars.index("#define C6AB_TABLE"))
    rows = _floats(pars[start:])
    if len(rows) % 5 != 0:
        raise RuntimeError("C6AB_TABLE is not a multiple of 5 values")

    c6ref = torch.zeros((NZ, NZ, MAXC, MAXC), dtype=torch.float64)
    filled = torch.zeros((NZ, NZ, MAXC, MAXC), dtype=torch.bool)
    refcn = torch.full((NZ, MAXC), float("nan"), dtype=torch.float64)
    nref = [0] * NZ

    def put_cn(z: int, a: int, cn: float):
        old = refcn[z, a].item()
        if old == old and old != cn:  # not NaN and different
            raise RuntimeError(f"reference CN of Z={z} ref {a} is not unique; "
                               "the factorised C6 would be wrong")
        refcn[z, a] = cn

    for k in range(0, len(rows), 5):
        c6, e1, e2, cn1, cn2 = rows[k:k + 5]
        e1, e2 = int(e1), int(e2)
        a, b = (e1 - 1) // 100, (e2 - 1) // 100       # 0-based reference index
        z1, z2 = (e1 - 1) % 100 + 1, (e2 - 1) % 100 + 1
        c6ref[z1, z2, a, b] = c6
        c6ref[z2, z1, b, a] = c6
        filled[z1, z2, a, b] = True
        filled[z2, z1, b, a] = True
        put_cn(z1, a, cn1)
        put_cn(z2, b, cn2)
        nref[z1] = max(nref[z1], a + 1)
        nref[z2] = max(nref[z2], b + 1)

    # SevenNet skips missing (c6 <= 0) entries inside the reference loops; the
    # factorised form needs every (a, b) present.
    for z1 in range(1, NZ):
        for z2 in range(1, NZ):
            if not bool(filled[z1, z2, :nref[z1], :nref[z2]].all()):
                raise RuntimeError(f"C6 table incomplete for Z={z1},{z2}")

    refmask = torch.zeros((NZ, MAXC), dtype=torch.bool)
    for z in range(1, NZ):
        refmask[z, :nref[z]] = True

    pad = [0.0]
    return {
        "c6ref": c6ref,
        "refcn": torch.nan_to_num(refcn, nan=0.0),
        "refmask": refmask,
        "rcov": torch.tensor(pad + _array(src, "rcov_ref"), dtype=torch.float64),
        "r2r4": torch.tensor(pad + _array(src, "r2r4_ref"), dtype=torch.float64),
    }


def load_d3_functional(functional: str) -> Dict[str, float]:
    with open(os.path.join(_sevenn_source_dir(), "pair_d3_for_ase.cu")) as f:
        table = _bj_functionals(f.read())
    key = functional.lower()
    if key not in table:
        raise ValueError(f"Unknown D3(BJ) functional {functional!r}; "
                         f"choices: {', '.join(sorted(table))}")
    return table[key]


# -------------------------------------------------------------------
#  Energy
# -------------------------------------------------------------------

class D3BJ(nn.Module):
    """
    D3(BJ) energy in eV from positions in Å.

    ``pairs`` builds the half list of pair-images within the cutoff (no grad);
    ``energy`` evaluates it differentiably, so the caller chooses which leaf
    (positions, strain) to differentiate against.
    """

    def __init__(self, functional: str = "pbe", cutoff: float = 9000.0,
                 cn_cutoff: float = 1600.0):
        super().__init__()
        p = load_d3_functional(functional)
        t = load_d3_tables()
        self.register_buffer("c6ref", t["c6ref"])
        self.register_buffer("refcn", t["refcn"])
        self.register_buffer("refmask", t["refmask"])
        self.register_buffer("rcov", t["rcov"])
        self.register_buffer("r2r4", t["r2r4"])

        self.functional: str = functional.lower()
        self.s6: float = p["s6"]
        self.s8: float = p["s8"]
        self.a1: float = p["a1"]
        self.a2: float = p["a2"]
        # Thresholds on r^2, in bohr^2 as SevenNet takes them.
        self.rthr: float = float(cutoff)
        self.cnthr: float = float(cn_cutoff)
        self.cutoff_ang: float = float(cutoff) ** 0.5 * AU_TO_ANG
        self.au_to_ang: float = AU_TO_ANG
        self.au_to_ev: float = AU_TO_EV
        self.k1: float = K1
        self.k3: float = K3

    def _images(self, cell: torch.Tensor) -> torch.Tensor:
        """Integer image offsets [M, 3]: |n_k| <= int(cutoff / height_k) + 1."""
        vol = torch.abs(torch.linalg.det(cell))
        reps: List[int] = []
        for k in range(3):
            c = torch.linalg.cross(cell[(k + 1) % 3], cell[(k + 2) % 3])
            height = vol / torch.sqrt((c * c).sum())
            reps.append(int(self.cutoff_ang / float(height)) + 1)
        rx = torch.arange(-reps[0], reps[0] + 1, device=cell.device)
        ry = torch.arange(-reps[1], reps[1] + 1, device=cell.device)
        rz = torch.arange(-reps[2], reps[2] + 1, device=cell.device)
        g = torch.meshgrid([rx, ry, rz], indexing="ij")
        return torch.stack([g[0].reshape(-1), g[1].reshape(-1), g[2].reshape(-1)], 1)

    def pairs(self, pos: torch.Tensor, cell: torch.Tensor, periodic: bool
              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Half list (i, j, n): i < j with every image n, and i == j with n in the
        positive half-space, each within the cutoff.  n [E, 3] float64 is the
        image such that r_ij = pos[j] - pos[i] + n @ cell for the positions as
        given (wrapped or not).
        """
        dev = pos.device
        N = pos.size(0)
        r2max = self.cutoff_ang * self.cutoff_ang
        ar = torch.arange(N, device=dev)
        upper = ar.unsqueeze(1) < ar.unsqueeze(0)                     # [N, N] i < j

        if not periodic:
            d = pos.unsqueeze(0) - pos.unsqueeze(1)                   # [i, j] = pos[j] - pos[i]
            keep = upper & ((d * d).sum(-1) <= r2max)
            ij = keep.nonzero()
            return (ij[:, 0], ij[:, 1],
                    torch.zeros((ij.size(0), 3), dtype=pos.dtype, device=dev))

        # Wrap into the cell as SevenNet does, so the image range is complete
        # even for unwrapped (NAMD) coordinates, then fold the wrap back into n.
        s = torch.floor(pos @ torch.linalg.inv(cell))
        pw = pos - s @ cell
        imgs = self._images(cell)
        shifts = imgs.to(pos.dtype) @ cell
        positive = ((imgs[:, 0] > 0)
                    | ((imgs[:, 0] == 0) & (imgs[:, 1] > 0))
                    | ((imgs[:, 0] == 0) & (imgs[:, 1] == 0) & (imgs[:, 2] > 0)))
        diag = torch.eye(N, dtype=torch.bool, device=dev)
        d0 = pw.unsqueeze(0) - pw.unsqueeze(1)                        # [N, N, 3]

        chunk = max(1, (1 << 22) // max(1, N * N))
        ii: List[torch.Tensor] = []
        jj: List[torch.Tensor] = []
        kk: List[torch.Tensor] = []
        for c0 in range(0, imgs.size(0), chunk):
            sh = shifts[c0:c0 + chunk]
            d = d0.unsqueeze(0) + sh.view(-1, 1, 1, 3)                # [K, N, N, 3]
            keep = (d * d).sum(-1) <= r2max
            keep = keep & (upper.unsqueeze(0)
                           | (diag.unsqueeze(0) & positive[c0:c0 + chunk].view(-1, 1, 1)))
            kij = keep.nonzero()
            kk.append(kij[:, 0] + c0)
            ii.append(kij[:, 1])
            jj.append(kij[:, 2])
        i = torch.cat(ii)
        j = torch.cat(jj)
        n = imgs.index_select(0, torch.cat(kk)).to(pos.dtype)
        n = n - s.index_select(0, j) + s.index_select(0, i)
        return i, j, n

    def energy(self, pos: torch.Tensor, Z: torch.Tensor, i: torch.Tensor,
               j: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
        """Dispersion energy (eV, 0-dim) over the half list; shifts in Å."""
        N = pos.size(0)
        vec = (pos.index_select(0, j) - pos.index_select(0, i) + shifts) / self.au_to_ang
        r2 = (vec * vec).sum(1)
        r = torch.sqrt(r2)
        Zi = Z.index_select(0, i)
        Zj = Z.index_select(0, j)

        # Coordination numbers.  A self-image edge adds to its atom twice,
        # once for n and once for -n, which is what the full sum does.
        rc = self.rcov.index_select(0, Zi) + self.rcov.index_select(0, Zj)
        damp = 1.0 / (1.0 + torch.exp(-self.k1 * (rc / r - 1.0)))
        damp = torch.where(r2 <= self.cnthr, damp, torch.zeros_like(damp))
        cn = torch.zeros(N, dtype=pos.dtype, device=pos.device)
        cn = cn.index_add(0, i, damp).index_add(0, j, damp)

        # Reference weights per atom, then C6 as w_i^T C6ref[Z_i, Z_j] w_j,
        # contracted over the element types present.
        mask = self.refmask.index_select(0, Z)                        # [N, 5]
        dcn = cn.unsqueeze(1) - self.refcn.index_select(0, Z)
        logits = torch.where(mask, self.k3 * dcn * dcn,
                             torch.full_like(dcn, -1e30))
        w = torch.softmax(logits, dim=1)                              # [N, 5]

        uz, tloc = torch.unique(Z, sorted=True, return_inverse=True)
        U = uz.size(0)
        ref = self.c6ref.index_select(0, Z).index_select(1, uz)       # [N, U, 5, 5]
        left = torch.einsum("na,nuab->nub", w, ref).reshape(N, U * 5)
        onehot = torch.nn.functional.one_hot(tloc, U).to(w.dtype)     # [N, U]
        right = (onehot.unsqueeze(2) * w.unsqueeze(1)).reshape(N, U * 5)
        c6 = (left @ right.t()).reshape(-1).index_select(0, i * N + j)

        rr = 3.0 * self.r2r4.index_select(0, Zi) * self.r2r4.index_select(0, Zj)
        R0 = self.a1 * torch.sqrt(rr) + self.a2
        R0_2 = R0 * R0
        R0_6 = R0_2 * R0_2 * R0_2
        r6 = r2 * r2 * r2
        e = -c6 * (self.s6 / (r6 + R0_6) + self.s8 * rr / (r6 * r2 + R0_6 * R0_2))
        e = torch.where(r2 <= self.rthr, e, torch.zeros_like(e))
        return e.sum() * self.au_to_ev


# -------------------------------------------------------------------
#  Add-on wrapper
# -------------------------------------------------------------------

class D3_Wrapper(nn.Module):
    """
    ``base`` (any wrapper in ``src/wrappers``) plus D3(BJ) dispersion.

    Energies, forces and virials are summed; charges come from ``base``.  D3
    takes its own backward pass, so ``base`` is untouched.
    """

    def __init__(self, base: nn.Module, functional: str = "pbe",
                 cutoff: float = 9000.0, cn_cutoff: float = 1600.0):
        super().__init__()
        self.base = base
        self.d3 = D3BJ(functional, cutoff, cn_cutoff)
        self.ev_to_kcal: float = EV_TO_KCAL

        self.supports_batch: bool = bool(getattr(base, "supports_batch", True))
        self.supports_pbc: bool = bool(getattr(base, "supports_pbc", True))
        # Read by export_wrapped for its diagnostics.
        self.r_max: float = float(getattr(base, "r_max", 0.0))
        self.model_uses_fp32: bool = bool(getattr(base, "model_uses_fp32", False))

    def _dispersion(self, coords: torch.Tensor, Z: torch.Tensor, cell: torch.Tensor
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(energy kcal/mol, forces [N,3], virial [3,3]) of one molecule."""
        pos = coords.detach().to(torch.float64).requires_grad_(True)
        Z = Z.to(torch.int64)
        cell3 = cell.reshape(-1, 3, 3)[0].to(torch.float64)
        periodic = cell_is_periodic(cell3)
        D = make_strain(cell3)

        i, j, n = self.d3.pairs(pos.detach(), cell3, periodic)
        if periodic:
            pos_s, _, shifts = apply_strain(pos, cell3, n, D)
        else:
            pos_s = pos
            shifts = n
        e = self.d3.energy(pos_s, Z, i, j, shifts)
        forces, virial = forces_and_virial(e, pos, D, self.ev_to_kcal)
        virial = finalize(virial) if periodic else zero_virial(coords)
        return e.detach() * self.ev_to_kcal, forces, virial

    def forward(
        self,
        coords: torch.Tensor,
        Z: torch.Tensor,
        pc_coords: torch.Tensor,
        pc_charges: torch.Tensor,
        cell: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        energy, forces, charges, virial = self.base(coords, Z, pc_coords, pc_charges, cell)
        e, f, v = self._dispersion(coords, Z, cell)
        return energy + e, forces + f, charges, virial + v

    @torch.jit.export
    def forward_batch(
        self,
        coords: torch.Tensor,
        Z: torch.Tensor,
        batch: torch.Tensor,
        ptr: torch.Tensor,
        pc_coords: torch.Tensor,
        pc_charges: torch.Tensor,
        cells: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        energies, forces, charges, virials = self.base.forward_batch(
            coords, Z, batch, ptr, pc_coords, pc_charges, cells)
        cells3 = cells.reshape(-1, 3, 3)
        bounds: List[int] = ptr.tolist()
        es: List[torch.Tensor] = []
        fs: List[torch.Tensor] = []
        vs: List[torch.Tensor] = []
        for b in range(len(bounds) - 1):
            lo = bounds[b]
            hi = bounds[b + 1]
            e, f, v = self._dispersion(coords[lo:hi], Z[lo:hi], cells3[b])
            es.append(e)
            fs.append(f)
            vs.append(v)
        return (energies + torch.stack(es), forces + torch.cat(fs), charges,
                virials + torch.stack(vs))
