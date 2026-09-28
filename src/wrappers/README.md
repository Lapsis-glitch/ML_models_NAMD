# Writing a wrapper for a new MLIP

This guide explains how to make a new machine-learning potential loadable by NAMD. A wrapper is a `torch.nn.Module` that puts the model behind the fixed interface NAMD's C++ MLFF backend calls. It is then exported as a single TorchScript file. The wrappers in this directory (`wrap_compiled_mace.py`, `wrap_torchani.py`, …) all follow the pattern described here.

Contents:
1. [What NAMD does with your model](#1-what-namd-does-with-your-model)
2. [Before you write anything](#2-before-you-write-anything)
3. [The template](#3-the-template)
4. [How the pieces work](#4-how-the-pieces-work)
5. [TorchScript rules that will bite you](#5-torchscript-rules-that-will-bite-you)
6. [Register the wrapper](#6-register-the-wrapper)
7. [Test it](#7-test-it)
8. [Load it the way NAMD does](#8-load-it-the-way-namd-does)
9. [Making it fast](#9-making-it-fast)

---

## 1. What NAMD does with your model

NAMD never runs Python. It loads the exported file with libtorch (`torch::jit::load`) inside a small C++ shim (`libnamd_mlff.so`) and then calls it. Everything below comes from what that shim does.

**At load time**
- It counts the arguments of `forward` and `forward_batch` (not counting `self`):
  - `forward` must take 4 inputs, or 5 with a trailing `cell`;
  - `forward_batch` must take 6 inputs, or 7 with a trailing `cells`.
  
  Any other count is a load error. With the cell argument present, NAMD treats the model as PBC-capable.
- Batching is enabled only if a `forward_batch` method exists. It has to be exported with `@torch.jit.export`, or scripting drops it.
- If `NAMD_MLFF_EXTRA_LIBS` is set, the shim `dlopen`s those libraries first. That is the only way to provide custom ops (see [9](#9-making-it-fast)).

**On every step**

| Input | Shape | Dtype | Notes |
|---|---|---|---|
| `coords` | `[N, 3]` | float64 | Å, on the model's device, a leaf tensor with `requires_grad=True` |
| `Z` | `[N]` | int64 | atomic numbers (not type indices) |
| `pc_coords`, `pc_charges` | `[P, 3]`, `[P]` | float64 | MM point charges; currently empty and ignored by every wrapper |
| `cell` | `[1, 3, 3]` | float64 | lattice vectors as **rows**, Å; **all zeros when the system is not periodic** |
| `batch`, `ptr` (batch only) | `[N_total]`, `[B+1]` | int64 | molecule index per atom, molecule boundaries |
| `cells` (batch only) | `[B, 3, 3]` | float64 | one box per molecule |

The call runs **with autograd enabled**, so the wrapper can differentiate.

**What it reads back:** a tuple of **at least 2** tensors, `(energy, forces[, charges[, virial]])`.
- `energy` is reshaped to `[1]` (`[B]` for batches), and `forces` must be `[N, 3]`.
- A third output is taken as per-atom charges, and a fourth as the virial.
- Every wrapper in this repo returns all four, so return all four too.

| Output | Shape (single / batch) | Units | Dtype |
|---|---|---|---|
| energy | scalar or `[1]` / `[B]` | kcal/mol | float64 |
| forces | `[N, 3]` / `[N_total, 3]` | kcal/mol/Å | float64 |
| charges | `[N]` / `[N_total]` | e (zeros if the model has none) | float64 |
| virial | `[3, 3]` / `[B, 3, 3]` | kcal/mol, symmetric, `−dE/dε = Σ rᵢ ⊗ fᵢ`, zeros when not periodic | float64 |

Why the wrapper supplies the virial: NAMD's own `Σ fᵢ ⊗ rᵢ` uses absolute positions, which is wrong for any pair that interacts across a box face. The wrapper computes the virial as the derivative of the energy with respect to a strain of the box, which has no such problem. `src/virial.py` explains the sign and normalisation, and tests pin them.

## 2. Before you write anything

Answer these about the model; they decide which parts of the template you need.

1. **Can it be TorchScript-ed or saved as TorchScript?** If not, it can't run in NAMD. JAX models go the StableHLO route instead (see FeNNiX in the main README).
2. **What does it take as input?**
   - Raw positions plus its own neighbour list (TorchANI does this)?
   - Or an edge list and edge vectors built by the caller (MACE, NequIP, SchNetPack)?
   
   In the second case, use `src/edges.py`.
3. **What does it return?**
   - Energy only: the wrapper does autograd, as in the template.
   - Energy + forces (+ virial) computed internally: the wrapper just converts units, as `wrap_compiled_mace.py` does.
4. **Native units and dtype.** eV or Hartree; float32 or float64 weights. The conversion factors are in `src/constants.py`.
5. **Species encoding.** Raw Z, one-hot over a list of elements (MACE), or 0-based type indices (NequIP, TorchANI)? The wrapper builds this from `Z`.
6. **Cutoff.** Stored on the model (`r_max`), or must it be passed in (SchNetPack `--r-max`)?

## 3. The template

This is a complete wrapper for a hypothetical model that takes an edge list and returns only the energy. Copy it to `src/wrappers/wrap_<name>.py` and adapt the marked parts: the inner call, units, and species encoding.

It has been checked end to end: forces against finite differences (2e-8), the virial against a strain finite difference in a triclinic box (4e-9), the large-box limit against `Σ r ⊗ f` (3e-13), and `forward_batch` against separate `forward` calls (exact, periodic and not). It also loads and runs in NAMD's C++ shim.

```python
"""
NAMD-compatible wrapper for **MyModel**.

The inner model is a TorchScript file with

    inner(Z [N] int64, edge_index [2, E] int64, edge_vec [E, 3], batch [N] int64,
          num_graphs: int) -> energy [B]  (eV)
    inner.r_max: float

Exposes the standard NAMD MLIP interface:
    forward(coords, Z, pc_coords, pc_charges, cell)
        -> (energy_kcal, forces_kcal_A, charges_e, virial_kcal)
    forward_batch(coords, Z, batch, ptr, pc_coords, pc_charges, cells)
        -> (energies, forces, charges, virials)
"""

from typing import Tuple

import torch
from torch import nn

from ..constants import EV_TO_KCAL
from ..edges import (build_edges, build_edges_batched, build_edges_pbc,
                       build_edges_batched_pbc, cell_is_periodic)
from ..virial import (make_strain, apply_strain, forces_and_virial,
                        finalize, zero_virial, zero_virials)


class MyModel_Wrapper(nn.Module):

    def __init__(self, model_path: str, device: str = "cpu"):
        super().__init__()
        self.inner = torch.jit.load(model_path, map_location=device)
        self.inner.eval()

        # Every attribute TorchScript sees needs a fixed type.
        self.r_max: float = float(self.inner.r_max)
        # The model's float dtype, from its first parameter (float64 if it has none).
        self.model_dtype: torch.dtype = torch.float64
        for prm in self.inner.parameters():
            self.model_dtype = prm.dtype
            break
        # Module-level constants can't be read from scripted code: copy them in.
        self.ev_to_kcal: float = EV_TO_KCAL

        # Read by NAMD / checked against the forward() signature.
        self.supports_batch: bool = True
        self.supports_pbc: bool = True

    def _edge_vectors(self, pos: torch.Tensor, edge_index: torch.Tensor,
                      shifts: torch.Tensor) -> torch.Tensor:
        # Recomputed from `pos` so forces can flow back through them.  The
        # vectors build_edges returns are detached and only good for the list.
        return (pos.index_select(0, edge_index[1])
                - pos.index_select(0, edge_index[0]) + shifts)

    # -----------------------------------------------------------------
    #  Single molecule
    # -----------------------------------------------------------------
    def forward(
        self,
        coords: torch.Tensor,       # [N, 3] float64, Å
        Z: torch.Tensor,            # [N]    int64
        pc_coords: torch.Tensor,    # [P, 3] float64 (ignored)
        pc_charges: torch.Tensor,   # [P]    float64 (ignored)
        cell: torch.Tensor,         # [1, 3, 3] float64, rows = lattice vectors; zeros = no box
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dev = coords.device
        N = coords.size(0)
        dt = self.model_dtype

        # Our own leaf in the model's dtype: forces are d(energy)/d(pos).
        pos = coords.detach().to(dt).requires_grad_(True)
        Z = Z.to(torch.int64)
        batch = torch.zeros(N, dtype=torch.int64, device=dev)

        cell3 = cell.reshape(-1, 3, 3)[0].to(dt)
        periodic = cell_is_periodic(cell3)
        D = make_strain(cell3)                   # zero 3x3 handle for the virial

        if periodic:
            edge_index, _, _, unit_shifts = build_edges_pbc(
                pos.detach().to(torch.float32), cell3.to(torch.float32), self.r_max)
            # Deform positions, cell and image shifts by the same (zero) strain,
            # so d(energy)/dD is the virial.
            pos_s, _, shifts = apply_strain(pos, cell3, unit_shifts.to(dt), D)
        else:
            edge_index, _, _ = build_edges(pos.detach().to(torch.float32), self.r_max)
            pos_s = pos
            shifts = torch.zeros((edge_index.size(1), 3), dtype=dt, device=dev)

        edge_vec = self._edge_vectors(pos_s, edge_index, shifts)
        energy_ev = self.inner(Z, edge_index, edge_vec, batch, 1)        # [1]

        # Forces and virial from ONE backward pass, already in kcal/mol.
        forces, virial = forces_and_virial(energy_ev, pos, D, self.ev_to_kcal)

        energy = energy_ev.to(torch.float64).sum() * self.ev_to_kcal
        forces = forces.to(torch.float64)
        charges = torch.zeros(N, dtype=torch.float64, device=dev)
        virial = finalize(virial) if periodic else zero_virial(coords)
        return energy, forces, charges, virial

    # -----------------------------------------------------------------
    #  Batch of molecules (NAMD walkers)
    # -----------------------------------------------------------------
    @torch.jit.export
    def forward_batch(
        self,
        coords: torch.Tensor,       # [N_total, 3] float64
        Z: torch.Tensor,            # [N_total]    int64
        batch: torch.Tensor,        # [N_total]    int64, molecule index per atom
        ptr: torch.Tensor,          # [B + 1]      int64, molecule boundaries
        pc_coords: torch.Tensor,
        pc_charges: torch.Tensor,
        cells: torch.Tensor,        # [B, 3, 3]    one box per molecule
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dev = coords.device
        N_total = coords.size(0)
        B = ptr.size(0) - 1
        dt = self.model_dtype

        pos = coords.detach().to(dt).requires_grad_(True)
        Z = Z.to(torch.int64)
        cells3 = cells.reshape(-1, 3, 3).to(dt)
        # Walkers in one batch always agree about periodicity.
        periodic = cell_is_periodic(cells3[0])

        # One strain per molecule, so one backward pass gives every virial.
        D = torch.zeros((B, 3, 3), dtype=dt, device=dev).requires_grad_(True)

        if periodic:
            edge_index, _, _, unit_shifts = build_edges_batched_pbc(
                pos.detach().to(torch.float32), ptr, cells3.to(torch.float32), self.r_max)
            sym = 0.5 * (D + D.transpose(1, 2))                         # [B, 3, 3]
            pos_s = pos + torch.einsum("ni,nij->nj", pos, sym.index_select(0, batch))
            cells_s = cells3 + torch.bmm(cells3, sym)
            edge_mol = batch.index_select(0, edge_index[0])
            shifts = torch.einsum("ei,eij->ej", unit_shifts.to(dt),
                                  cells_s.index_select(0, edge_mol))
        else:
            edge_index, _, _ = build_edges_batched(
                pos.detach().to(torch.float32), ptr, self.r_max)
            pos_s = pos
            shifts = torch.zeros((edge_index.size(1), 3), dtype=dt, device=dev)

        edge_vec = self._edge_vectors(pos_s, edge_index, shifts)
        energy_ev = self.inner(Z, edge_index, edge_vec, batch, B)        # [B]

        grads = torch.autograd.grad([energy_ev.sum()], [pos, D], allow_unused=True)
        g_pos = grads[0]
        g_D = grads[1]
        if g_pos is None:
            forces = torch.zeros((N_total, 3), dtype=torch.float64, device=dev)
        else:
            forces = (-g_pos * self.ev_to_kcal).to(torch.float64)

        virials = zero_virials(B, coords)
        if periodic and g_D is not None:
            V = (-g_D * self.ev_to_kcal).to(torch.float64)
            virials = 0.5 * (V + V.transpose(1, 2))

        energies = energy_ev.to(torch.float64) * self.ev_to_kcal
        charges = torch.zeros(N_total, dtype=torch.float64, device=dev)
        return energies, forces, charges, virials
```

If your model **computes forces and virial itself**, drop the autograd parts. Call it with the flags that ask for them, convert units, and symmetrise. `wrap_compiled_mace.py` shows this route, including one subtlety: MACE is asked for a virial only when `cell` is a real box.

## 4. How the pieces work

**Neighbour lists (`src/edges.py`).**
- `build_edges(coords, r_max)` and `build_edges_batched(coords, ptr, r_max)` return `(edge_index [2,E], edge_vec [E,3], edge_len [E])`, with the direction convention `vec = pos[edge_index[1]] - pos[edge_index[0]]`.
- The `*_pbc` variants take cells and also return `unit_shifts [E,3]` (integer image offsets). They use the minimum-image convention, so every box width must be at least 2 × `r_max`; `check_min_image(cell, r_max)` asserts this.
- Build the list in float32: it's faster, and the list only decides *which* pairs exist.
- The returned vectors are **not** connected to the autograd graph. If your forces come from autograd, recompute the vectors from `pos` as the template does (`_edge_vectors`).

**Forces.** Make your own leaf, `pos = coords.detach().to(dtype).requires_grad_(True)`, run the model, and differentiate the energy with respect to `pos`.

**Virial (`src/virial.py`).**
- `D = make_strain(cell)` is a zero 3×3 tensor that requires grad.
- `apply_strain(pos, cell, unit_shifts, D)` deforms positions, cell and image shifts together. Straining the positions without the shifts gives a plausible-looking but **wrong** virial.
- `forces_and_virial(energy, pos, D, scale)` returns both from one backward pass.
- For a batch, use one strain per molecule (`D [B,3,3]`, applied through `batch`), as the template does. One backward pass then gives every molecule's virial.
- `finalize` forces float64 and symmetry; `zero_virial(s)` gives the non-periodic answer.

**Periodic or not.** `cell_is_periodic(cell)` is a determinant test on the 3×3 cell. Keep the non-periodic path free of any cell arithmetic: it's the common case, and it should stay the cheap one.

**Batching.**
- `batch[i]` is atom *i*'s molecule and `ptr` holds the molecule boundaries (`ptr[b]:ptr[b+1]`).
- Edges from the batched builders never cross molecules, so a batch is one block-diagonal graph evaluated in one call.
- Each molecule has **its own cell**, because replicas run separate barostats. So image shifts are built per edge from that edge's molecule's cell.
- NAMD guarantees that all walkers in a batch agree about periodicity.

**Units.** Convert once, at the end, and return float64. Copy the factor into a typed attribute (`self.ev_to_kcal: float = EV_TO_KCAL`); see the next section.

## 5. TorchScript rules that will bite you

`src/cli.py` runs `torch.jit.script` on the whole wrapper. These are the failures we hit in this repo:

- **Module-level constants can't be read from scripted code.** For example, using `EV_TO_KCAL` directly inside `forward` fails with *"python value of type 'float' cannot be used as a value"*. Store it as a typed attribute in `__init__`.
- **Every attribute needs a fixed type.** Annotate in `__init__` (`self.r_max: float = …`, `self.cache: torch.Tensor = torch.empty(0)`). An attribute that is sometimes `None` needs `Optional[...]` and an explicit check before use.
- **Annotate non-tensor arguments and return types** (`num_graphs: int`, `-> Tuple[Tensor, Tensor, Tensor, Tensor]`). Unannotated arguments are assumed to be tensors.
- **`forward_batch` needs `@torch.jit.export`.** Otherwise scripting drops it, and NAMD silently runs without batching.
- **Some ops need every argument spelled out.** `x.norm(dim=1)` doesn't script; use `torch.linalg.norm(x, dim=1)`, or `sqrt((x*x).sum(1))`.
- **`torch.autograd.grad` returns `List[Optional[Tensor]]`.** Check each entry for `None` (use `allow_unused=True`), as the template does.
- **No `zip()` over several `ModuleList`s, no `None` inside a `ModuleList`, and no untyped empty lists.** These are what broke upstream X-MACE: they compile, but the loop silently never runs. Index with `enumerate` over one `ModuleList`, and type empty lists (`xs: List[Tensor] = []`).
- **Returning a different tuple length on different paths is an error.** Always return all four outputs, using zeros where there's nothing.

## 6. Register the wrapper

1. **`src/wrappers/__init__.py`:** import the class and add it to `__all__`.
2. **`src/cli.py`:**
   - add the name to `--model-type` `choices`;
   - add a branch that builds your wrapper (`.eval()`), with any model-specific flags prefixed `[<name>]` in their help text;
   - add a label for `export_wrapped`.
3. **`pyproject.toml`:** add an optional-dependency group with the packages needed to *compile* the inner model. Wrapping itself only needs torch.
4. If the inner model needs its own compile step, add a `src/compile_<name>.py` alongside `compile_mace_off.py` / `compile_torchani.py`.

Then:

```bash
python -m src.cli --model-type <name> --compiled inner.pt --out mlff_model.pt
```

## 7. Test it

**Interface compliance, with no real weights.** In `tests/test_interface_compliance.py`:
- add a small mock inner model that returns plausible tensors in your model's native format;
- add a builder that scripts and saves it, then constructs your wrapper;
- add the builder to `_WRAPPER_BUILDERS`.

The parametrised tests then check shapes, dtypes, zero virials for a zero cell, `forward_batch`, `supports_batch` / `supports_pbc`, and that scripting works. The unit suite must stay weight-free, so don't use real models there.

```bash
python -m pytest tests/test_interface_compliance.py -v      # allegro env
```

**Physics checks with the real model.** Write these as a separate test file, guarded with `pytest.skip` when the model file is missing so the unit suite stays weight-free. `tests/test_virial_nequip_ani.py` shows each check below.
- **Forces:** compare against a central finite difference of the energy.
- **Virial sign and scale:**
  - deform positions and cell by `±h` along each strain component, `x → x (I + ε)`, and check `virial ≈ −ΔE / Δε`;
  - put the molecule in a box much larger than the cutoff and check `virial == Σ rᵢ ⊗ fᵢ`. This pins sign, transpose and scale in one comparison.
- **Batching:** `forward_batch` on two molecules with different boxes must equal two `forward` calls.
- **Translation invariance:** shifting every atom must not change the energy, forces or virial.

## 8. Load it the way NAMD does

**In Python**, with only torch, and the native libraries if the model uses custom ops:

```python
import os, torch
for so in filter(None, os.environ.get("NAMD_MLFF_EXTRA_LIBS", "").split(":")):
    torch.ops.load_library(so)
m = torch.jit.load("mlff_model.pt", map_location="cuda")
coords = torch.tensor([[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]],
                      dtype=torch.float64, device="cuda", requires_grad=True)
Z = torch.tensor([8, 1, 1], device="cuda")
pc = torch.zeros(0, 3, dtype=torch.float64, device="cuda"), torch.zeros(0, dtype=torch.float64, device="cuda")
print(m(coords, Z, *pc, torch.zeros(1, 3, 3, dtype=torch.float64, device="cuda")))
```

**Through the real C++ shim**, which catches signature, arity and custom-op problems before a NAMD run:

```bash
source namd_benchmarks/env.sh                     # sets NAMD_MLFF_LIB
namd_benchmarks/lib/mlff_shim_test "$NAMD_MLFF_LIB" mlff_model.pt 0     # GPU 0; -1 = CPU
```

It reports `supports_batch`, runs a single and a batched evaluation, and ends with `ALL CHECKS PASSED`. The NAMD config lines are in the main README (Step 4).

## 9. Making it fast

Start from a correct, tested wrapper, then optimise against it. What paid off for the models in `scripts/opt/` (details in each `scripts/opt/<model>/REPORT.md`):

- **Avoid host syncs in `forward`.**
  - `.item()`, `int(tensor)`, `bool(tensor)`, `torch.equal` and Python `if` on tensor values all stall the GPU once per step.
  - Cache constant tensors keyed on the atom count (a Python int), not on `Z` contents.
  - Use `torch._assert_async` for sanity checks.
- **Keep constants on the device.** Move conversion factors and lookup tables once, not on every call.
- **Build edges in FP32**, and use `build_edges_cell*` (the cell list) for thousands of atoms.
- **Put faster kernels behind an opt-in flag** (`fast=True`, `lean=True`) and keep the stock path as the reference. The optimised build must match the reference to rounding, and the per-model `build_fast.py` scripts assert exactly that.
- **Custom CUDA kernels must be native.** Register the op in C++ with `TORCH_LIBRARY` and build it against the **same libtorch as NAMD's shim**. Load it with `NAMD_MLFF_EXTRA_LIBS` in NAMD, and with `src.cli --extra-libs` when wrapping.
  - An op registered from Python (a `torch.library` decorator, or `import some_package`) can't be used, because NAMD never runs Python.
  - Working examples: `scripts/opt/mace/cueq_native/`, `scripts/opt/nequip/oeq_native/` and `scripts/opt/ani/cuaev_native/`.
- **Measure with `scripts/opt/bench_common.py`.** It feeds artifacts exactly what the shim feeds them and times them interleaved in one process, so ratios are fair. Pass the reference first; parity is reported against it.
  ```bash
  python scripts/opt/bench_common.py --model ref=mlff_model.pt --model fast=mlff_fast.pt \
      --systems 30,300,3000 --walkers 1,4 --jitter 0.02 [--extra-lib lib.so]
  ```
