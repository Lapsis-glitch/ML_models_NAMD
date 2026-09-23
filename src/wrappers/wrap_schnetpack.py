"""
NAMD-compatible wrapper for **SchNetPack** (≥ 2.0) models.

SchNetPack models are exported via ``torch.jit.script`` (or loaded from
a TorchScript archive produced during training).  The wrapper translates
NAMD's ``(coords, Z)`` interface into SchNetPack's input dict format and
converts the output to kcal/mol.

Exposes the standard NAMD MLIP interface:
    forward(coords, Z, pc_coords, pc_charges, cell)
        -> (energy_kcal, forces_kcal_A, charges_e, virial_kcal)
    forward_batch(coords, Z, batch, ptr, pc_coords, pc_charges, cells)
        -> (energies, forces, charges, virials)

WHERE THE PERIODIC VIRIAL COMES FROM
SchNetPack has stress code, but the compiled artifact bakes calc_stress in as
False and its forward takes no runtime flags, so there is nothing to ask for.
The wrapper derives the virial itself with the shared strain helpers in
src/virial.py.

That needs the energy graph intact, and SchNetPack's Forces postprocessor
destroys it on the way to the forces: it calls autograd.grad without
retain_graph, which frees the very intermediates a strain derivative would have
to walk back through.  A second pass would not help, because the same thing
happens again.  So for a periodic box the wrapper stops one module short, runs
the pipeline as far as the energy, and takes forces and virial out of a single
backward of its own.  That is one backward either way, so the virial arrives for
free, and the forces come out bit for bit the same, since it is the same
derivative taken by the same engine.  A cluster calculation still goes through
the model whole.

SchNetPack specifics:
  * Input dict uses ``_positions``, ``_atomic_numbers``, ``_idx_i``,
    ``_idx_j``, ``_offsets``, ``_cell``, ``_n_atoms``, ``_idx_m`` keys.
  * The neighbor list is provided as COO index pairs (``_idx_i``,
    ``_idx_j``) — same topology as our shared ``build_edges`` but with
    different key names.
  * Default native units are eV (configurable at training time).
    The wrapper accepts a ``units`` parameter to handle non-default
    training unit systems.
  * ``_offsets`` is the Cartesian shift added to the neighbour, i.e.
    ``Rij = pos[_idx_j] - pos[_idx_i] + _offsets``, and ``_cell`` carries
    one 3x3 per structure.
"""

import argparse

import torch
from torch import nn
from typing import Dict, List, Optional

from ..constants import EV_TO_KCAL
from ..edges import (build_edges, build_edges_batched, build_edges_pbc,
                     build_edges_batched_pbc, cell_is_periodic,
                     build_edges_cell, build_edges_cell_batched)
from ..export import export_wrapped
from ..virial import apply_strain, finalize, forces_and_virial, make_strain


def _has_pipeline(inner) -> bool:
    """Whether *inner* is laid out like a SchNetPack NeuralNetworkPotential."""
    return (hasattr(inner, "input_modules")
            and hasattr(inner, "representation")
            and hasattr(inner, "output_modules")
            and hasattr(inner, "postprocess"))


class _EnergyBeforeForces(nn.Module):
    """
    Run a SchNetPack model as far as the energy and stop.

    Everything the full forward does is here except the Forces postprocessor,
    which is the one step that cannot be undone: it differentiates the energy
    without asking autograd to keep the graph, and the strain derivative needs
    that graph.  Stopping short costs nothing, because the wrapper takes the
    forces out of its own backward instead, from the same energy.
    """

    def __init__(self, inner, energy_key: str):
        super().__init__()
        self.neighbours = getattr(inner.input_modules, "0")
        self.representation = inner.representation
        self.atomwise = getattr(inner.output_modules, "0")
        # Kept for postprocess(), which is where a model puts unit rescaling and
        # energy offsets.  Skipping it would leave the energy and the virial on
        # different scales.
        self.whole_model = inner
        self.energy_key: str = energy_key

        # Stopping early is only safe if the module we stop at is the one that
        # writes the energy we then read.  A different output order would hand
        # us someone else's tensor and a virial for the wrong quantity.
        written = str(getattr(self.atomwise, "output_key", ""))
        if written != energy_key:
            raise RuntimeError(
                "The first output module writes '" + written + "' but this "
                "wrapper was told the energy lives under '" + energy_key
                + "'. Pass a matching energy_key."
            )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.neighbours(inputs)
        x = self.representation(x)
        x = self.atomwise(x)
        x = self.whole_model.postprocess(x)
        return x[self.energy_key]


class _SchNetFastEnergy(nn.Module):
    """
    Opt-in fast energy route (``fast=True``), used for non-periodic inputs.

    Same submodules, same weights, same math as the stock forward; what it
    drops is overhead around them:

      * the Forces postprocessor / initialize_derivatives / extract_outputs
        dict plumbing (the wrapper takes forces from one autograd.grad of its
        own, exactly like the periodic path);
      * the host sync in SchNetPack's Atomwise, which sizes its per-molecule
        sum with ``int(idx_m[-1]) + 1`` -- a device->host read in the middle
        of the step.  Here the number of molecules is passed in (it is known
        from the input shapes), and the sum is the same ``index_add`` that
        ``schnetpack.nn.scatter.scatter_add`` performs.

    Having no host sync at all is also what makes the step capturable as a
    CUDA graph (see ``SchNetPack_Wrapper.graph_step``).
    """

    def __init__(self, inner, energy_key: str):
        super().__init__()
        atomwise = getattr(inner.output_modules, "0")
        written = str(getattr(atomwise, "output_key", ""))
        if written != energy_key:
            raise RuntimeError(
                "fast=True: the first output module writes '" + written
                + "' but the energy key is '" + energy_key + "'.")
        if not hasattr(atomwise, "outnet"):
            raise RuntimeError("fast=True: the first output module has no 'outnet' "
                               "(not a SchNetPack Atomwise); export without fast.")
        mode = str(getattr(atomwise, "aggregation_mode", "sum"))
        if mode not in ("sum", "avg"):
            raise RuntimeError("fast=True: unsupported Atomwise aggregation_mode '"
                               + mode + "'; export without fast.")
        if getattr(atomwise, "per_atom_output_key", None) is not None:
            raise RuntimeError("fast=True: per-atom outputs are not supported.")
        self.neighbours = getattr(inner.input_modules, "0")
        self.representation = inner.representation
        self.outnet = atomwise.outnet
        self.whole_model = inner
        self.energy_key: str = energy_key
        self.average: bool = mode == "avg"

    def forward(self, pos: torch.Tensor, Z: torch.Tensor, idx_i: torch.Tensor,
                idx_j: torch.Tensor, offsets: torch.Tensor, idx_m: torch.Tensor,
                n_atoms: torch.Tensor, n_mol: int) -> torch.Tensor:
        x: Dict[str, torch.Tensor] = {
            "_positions": pos,
            "_atomic_numbers": Z,
            "_idx_i": idx_i,
            "_idx_j": idx_j,
            "_offsets": offsets,
            "_n_atoms": n_atoms,
            "_idx_m": idx_m,
        }
        x = self.neighbours(x)
        x = self.representation(x)
        return self._head(x, x["scalar_representation"], idx_m, n_atoms, n_mol)

    def energy_from_representation(self, rep: torch.Tensor, idx_m: torch.Tensor,
                                   n_atoms: torch.Tensor, n_mol: int) -> torch.Tensor:
        x: Dict[str, torch.Tensor] = {"_n_atoms": n_atoms, "_idx_m": idx_m,
                                      "scalar_representation": rep}
        return self._head(x, rep, idx_m, n_atoms, n_mol)

    def _head(self, x: Dict[str, torch.Tensor], rep: torch.Tensor, idx_m: torch.Tensor,
              n_atoms: torch.Tensor, n_mol: int) -> torch.Tensor:
        y = self.outnet(rep)                                         # [N, 1]
        e = torch.zeros((n_mol, y.size(1)), dtype=y.dtype, device=y.device)
        e = e.index_add(0, idx_m, y).squeeze(-1)                     # [B]
        if self.average:
            e = e / n_atoms
        x[self.energy_key] = e
        x = self.whole_model.postprocess(x)
        return x[self.energy_key]


class _HalfInteraction(nn.Module):
    """
    One SchNetInteraction evaluated on a HALF edge list (each unordered pair
    i<j once).  Stock SchNet runs the filter network on every directed edge,
    i.e. twice per pair, although W(d_ij) = W(d_ji) (non-periodic: the two
    displacement vectors are exact negatives, so d, the radial basis and the
    cutoff are bit-identical).  Here W is computed once per pair and used for
    both messages:  x_i += h_j * W_ij  and  x_j += h_i * W_ij.  Same weights,
    same sum, half the filter-network work and activation memory.
    """

    def __init__(self, interaction):
        super().__init__()
        self.in2f = interaction.in2f
        self.filter_network = interaction.filter_network
        self.f2out = interaction.f2out

    def forward(self, x: torch.Tensor, f_h: torch.Tensor, rcut_h: torch.Tensor,
                i_h: torch.Tensor, j_h: torch.Tensor) -> torch.Tensor:
        h = self.in2f(x)
        W = self.filter_network(f_h) * rcut_h.unsqueeze(1)                   # [Eh, F]
        agg = torch.zeros((h.size(0), h.size(1)), dtype=h.dtype, device=h.device)
        agg = agg.index_add(0, i_h, h[j_h] * W)
        agg = agg.index_add(0, j_h, h[i_h] * W)
        return self.f2out(agg)


class _SchNetHalfEnergy(nn.Module):
    """
    Opt-in (fast=True, half_filter=True) energy route for NON-periodic input
    on a half edge list, see ``_HalfInteraction``.  Built only for a stock
    SchNet representation (radial_basis, cutoff_fn, embedding, no electronic
    embeddings, SchNetInteraction blocks) and verified numerically against
    the stock representation when the wrapper is constructed.
    """

    def __init__(self, inner, energy_key: str):
        super().__init__()
        rep = inner.representation
        ok = (getattr(rep, "original_name", "") == "SchNet"
              and hasattr(rep, "radial_basis") and hasattr(rep, "cutoff_fn")
              and hasattr(rep, "embedding") and hasattr(rep, "interactions"))
        if ok and hasattr(rep, "electronic_embeddings"):
            ok = len(list(rep.electronic_embeddings.children())) == 0
        inters = list(rep.interactions.children()) if ok else []
        for it in inters:
            ok = ok and getattr(it, "original_name", "") == "SchNetInteraction" \
                and hasattr(it, "in2f") and hasattr(it, "filter_network") and hasattr(it, "f2out")
        if not ok or len(inters) == 0:
            raise RuntimeError("half_filter=True needs a stock SchNet representation; "
                               "export with --no-half-filter.")
        self.neighbours = getattr(inner.input_modules, "0")
        self.radial_basis = rep.radial_basis
        self.cutoff_fn = rep.cutoff_fn
        self.embedding = rep.embedding
        self.blocks = nn.ModuleList([_HalfInteraction(it) for it in inters])
        # Atomwise head + postprocess exactly as in _SchNetFastEnergy.
        self.head = _SchNetFastEnergy(inner, energy_key)

    def forward(self, pos: torch.Tensor, Z: torch.Tensor, i_h: torch.Tensor,
                j_h: torch.Tensor, offsets: torch.Tensor, idx_m: torch.Tensor,
                n_atoms: torch.Tensor, n_mol: int) -> torch.Tensor:
        # Same displacement definition as SchNetPack's PairwiseDistances.
        r = self.neighbours({"_positions": pos, "_idx_i": i_h, "_idx_j": j_h,
                             "_offsets": offsets})["_Rij"]
        d = torch.norm(r, dim=1)
        f = self.radial_basis(d)
        rc = self.cutoff_fn(d)
        x = self.embedding(Z)
        for blk in self.blocks:
            x = x + blk(x, f, rc, i_h, j_h)
        return self.head.energy_from_representation(x, idx_m, n_atoms, n_mol)


class _NoFastRoute(nn.Module):
    """Placeholder so the default (fast=False) wrapper still scripts."""

    def forward(self, pos: torch.Tensor, Z: torch.Tensor, idx_i: torch.Tensor,
                idx_j: torch.Tensor, offsets: torch.Tensor, idx_m: torch.Tensor,
                n_atoms: torch.Tensor, n_mol: int) -> torch.Tensor:
        raise RuntimeError("This SchNet model was exported without fast=True.")


class _NoEnergyRoute(nn.Module):
    """Stand-in for a model whose innards do not look like SchNetPack's."""

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise RuntimeError(
            "This model is not laid out like a SchNetPack "
            "NeuralNetworkPotential, so the wrapper cannot reach its energy "
            "without also triggering the Forces postprocessor, and a periodic "
            "virial cannot be produced. Run it without a cell, or export the "
            "model as a NeuralNetworkPotential."
        )


class SchNetPack_Wrapper(nn.Module):
    """
    Wrap a SchNetPack (≥ 2.0) TorchScript model for NAMD.

    Args:
        model_path:       Path to a scripted SchNetPack ``.pt`` file.
        r_max:            Cutoff radius in Å.  Must match the cutoff
                          used during training.
        device:           ``"cpu"`` or ``"cuda"``.
        energy_key:       Key in the model's output dict for energy.
        forces_key:       Key in the model's output dict for forces.
        energy_units_to_kcal:
            Multiplicative factor to convert the model's native energy
            unit to kcal/mol.  Default ``EV_TO_KCAL`` (23.0621).
        fast:             Opt-in (default off).  Non-periodic calls use
                          ``_SchNetFastEnergy`` (no mid-step host sync, one
                          autograd.grad) and the artifact gains the CUDA-graph
                          API ``graph_capacity`` / ``graph_step`` for a shim
                          that captures and replays the step.  Periodic calls
                          are unchanged.  Same weights and math; outputs agree
                          with the default path to fp32 scatter noise.
        graph_max_atoms:  Largest system the graph API advertises itself for
                          (the shim reads ``graph_max_atoms``).  Measured on
                          the RTX 5080 (half_filter on): graph wins up to
                          ~2000 atoms, ties at 3000; its dense in-graph
                          neighbour list costs O(N^2).
        nl_cell_min_pairs: [fast only] use the O(N) cell-list neighbour list
                          (same edges, same order) once B * n_per_molecule^2
                          reaches this many candidate pairs; the dense build is
                          cheaper below it.  Default 16e6 (~4000 atoms x 1
                          walker, ~2000 atoms x 4 walkers) from RTX 5080
                          timings in scripts/opt/schnet/nl/time_nl.py.
        half_filter:      [fast only] evaluate the filter network once per
                          atom pair instead of once per directed edge (see
                          ``_HalfInteraction``); checked against the stock
                          representation at construction.  Default on.
        half_min_atoms:   [fast, half_filter] plain forward()/forward_batch()
                          use the half-list route from this many total atoms
                          up (RTX 5080: 1200 atoms no gain, 3600 atoms 1.3x);
                          graph_step always uses it.
    """

    def __init__(
        self,
        model_path: str,
        r_max: float,
        device: str = "cpu",
        energy_key: str = "energy",
        forces_key: str = "forces",
        energy_units_to_kcal: float = EV_TO_KCAL,
        fast: bool = False,
        graph_max_atoms: int = 2048,
        nl_cell_min_pairs: int = 16_000_000,
        half_filter: bool = True,
        half_min_atoms: int = 1500,
    ):
        super().__init__()

        self.inner = torch.jit.load(model_path, map_location=device)
        self.inner.eval()

        self.r_max: float = r_max
        self.energy_key: str = energy_key
        self.forces_key: str = forces_key

        self.conv_factor = torch.tensor(
            energy_units_to_kcal, dtype=torch.float64,
        )
        # Same factor as a plain float for the fast path: multiplying by a
        # Python float needs no per-call host->device copy of a CPU tensor
        # (and is capturable in a CUDA graph).  Same double, same result.
        self.conv_float: float = float(energy_units_to_kcal)

        # Cached tensors
        self._cached_N: int = -1
        self._cached_idx_m: torch.Tensor = torch.empty(0)
        self._cached_n_atoms: torch.Tensor = torch.empty(0)
        self._cached_offsets_dummy: torch.Tensor = torch.empty(0)
        self._cached_cell: torch.Tensor = torch.empty(0)

        self.supports_batch: bool = True

        # Says this model takes a cell argument.  NAMD works this out from the
        # forward() signature anyway and only uses the flag to cross-check, so
        # if the two ever disagree the model fails to load rather than running
        # with a mishandled cell.
        self.supports_pbc: bool = True

        # The periodic path needs the energy before the Forces postprocessor has
        # eaten the graph, which means running the pipeline by hand.  A model
        # that is not a SchNetPack NeuralNetworkPotential has no such pipeline
        # to run, and rather than refuse to load it (it may only ever be used on
        # a cluster, where none of this matters) the stand-in below says so if
        # and when a periodic box turns up.
        if _has_pipeline(self.inner):
            self.energy_only = _EnergyBeforeForces(self.inner, self.energy_key)
        else:
            self.energy_only = _NoEnergyRoute()

        # ---- opt-in fast path / CUDA-graph API (default off) ----
        self.fast: bool = bool(fast)
        if self.fast:
            if not _has_pipeline(self.inner):
                raise RuntimeError("fast=True needs a SchNetPack NeuralNetworkPotential.")
            self.fast_energy = _SchNetFastEnergy(self.inner, self.energy_key)
        else:
            self.fast_energy = _NoFastRoute()
        # Half-list (i<j) route: used by graph_step always, and by the plain
        # forward paths from half_min_atoms total atoms up (below that the
        # extra pair-selection kernels cost more than the filter work saved
        # while the step is launch-bound).
        self.half: bool = bool(self.fast and half_filter)
        if self.half:
            self.half_energy = _SchNetHalfEnergy(self.inner, self.energy_key)
        else:
            self.half_energy = _NoFastRoute()
        self.half_min_atoms: int = int(half_min_atoms)
        # Read by a CUDA-graph-capable shim (NAMD_MLFF_CUDA_GRAPH=1).
        self.graph_capable: bool = self.fast
        self.graph_max_atoms: int = int(graph_max_atoms)
        self.nl_cell_min_pairs: int = int(nl_cell_min_pairs)
        if self.half:
            self._check_half_route(device)

    def _check_half_route(self, device: str):
        """Refuse to build if the half-list route disagrees with the stock one."""
        g = torch.Generator().manual_seed(0)
        n = 24
        pos = (torch.rand(n, 3, generator=g, dtype=torch.float64) * 7.0).to(device)
        Z = torch.tensor([8, 1, 1, 6] * (n // 4), dtype=torch.int64, device=device)
        ei, _, _ = build_edges(pos.to(torch.float32), self.r_max)
        idx_m = torch.zeros(n, dtype=torch.long, device=device)
        n_atoms = torch.tensor([n], dtype=torch.long, device=device)
        p32 = pos.to(torch.float32)
        e_full = self.fast_energy(p32, Z, ei[0], ei[1], torch.zeros(ei.size(1), 3, device=device),
                                  idx_m, n_atoms, 1)
        keep = ei[0] < ei[1]
        eh = ei[:, keep]
        e_half = self.half_energy(p32, Z, eh[0], eh[1], torch.zeros(eh.size(1), 3, device=device),
                                  idx_m, n_atoms, 1)
        err = float((e_full - e_half).abs().max())
        scale = max(1.0, float(e_full.abs().max()))
        if not err <= 1e-4 * scale:
            raise RuntimeError("half_filter self-check failed (|dE| = " + str(err)
                               + "); export with --no-half-filter.")

    # -----------------------------------------------------------------
    #  Fast path helpers (fast=True only)
    # -----------------------------------------------------------------

    def _fast_eval(self, coords: torch.Tensor, Z: torch.Tensor,
                   edge_index: torch.Tensor, idx_m: torch.Tensor,
                   n_atoms: torch.Tensor, n_mol: int):
        """Energy [B] and forces [N,3] (float64, kcal) for a non-periodic batch."""
        dev = coords.device
        # Our own fp32 leaf: forces are d/d(pos32), cast up, like the stock path.
        pos = coords.detach().to(torch.float32).requires_grad_(True)
        if self.half and coords.size(0) >= self.half_min_atoms:
            # Each unordered pair once (the builders emit both directions).
            edge_index = edge_index[:, edge_index[0] < edge_index[1]]
            offsets = torch.zeros((edge_index.size(1), 3), dtype=torch.float32, device=dev)
            e = self.half_energy(pos, Z, edge_index[0], edge_index[1], offsets,
                                 idx_m, n_atoms, n_mol)
        else:
            offsets = torch.zeros((edge_index.size(1), 3), dtype=torch.float32, device=dev)
            e = self.fast_energy(pos, Z, edge_index[0], edge_index[1], offsets,
                                 idx_m, n_atoms, n_mol)
        grads = torch.autograd.grad([e.sum()], [pos], allow_unused=True)
        g = grads[0]
        conv = self.conv_float
        # squeeze(-1) like the stock path, so output shapes match exactly
        energy = e.to(torch.float64).squeeze(-1) * conv
        if g is None:
            forces = torch.zeros((coords.size(0), 3), dtype=torch.float64, device=dev)
        else:
            forces = (-g).to(torch.float64) * conv
        return energy, forces

    # -----------------------------------------------------------------
    #  CUDA-graph API (fast=True only; see scripts/opt/schnet/REPORT.md)
    # -----------------------------------------------------------------

    @torch.jit.export
    def graph_capacity(self, coords: torch.Tensor) -> int:
        """
        Edge capacity to capture ``graph_step`` with, for these coordinates:
        the current edge count plus 25% + 64 headroom for MD fluctuations.
        Eager (one host sync); call it only when (re)capturing.
        """
        c = coords.detach().to(torch.float32)
        d = c.unsqueeze(0) - c.unsqueeze(1)
        dist2 = (d * d).sum(dim=2)
        r2 = self.r_max * self.r_max
        mask = (dist2 < r2) & (dist2 > 0.0)
        if self.half:
            ar = torch.arange(c.size(0), device=c.device)
            mask = mask & (ar.unsqueeze(0) > ar.unsqueeze(1))
        n = int(mask.sum())
        return n + n // 4 + 64

    @torch.jit.export
    def graph_step(self, coords: torch.Tensor, Z: torch.Tensor, cap: int):
        """
        One non-periodic single-molecule step with static shapes and no host
        sync, so a shim can capture it once and replay it every MD step.

        The neighbour list is built in-graph with ``nonzero_static`` padded to
        *cap* edges.  Padding slots become self-edges k -> k (spread over all
        atoms so their zero contributions do not serialise one atomic) with a
        2*r_max offset: the cosine cutoff is exactly 0 there, so they add
        exactly 0 to the energy and the forces.

        Returns (energy [1] kcal/mol, forces [N,3] kcal/mol/A, both float64,
        n_edges [] int64; with half_filter these count unordered pairs).  If n_edges > cap the step dropped edges and its
        output is INVALID: the caller must re-capture with a larger
        ``graph_capacity`` and re-run.
        """
        if not self.fast:
            raise RuntimeError("graph_step needs a model exported with fast=True.")
        dev = coords.device
        N = coords.size(0)
        pos = coords.detach().to(torch.float32).requires_grad_(True)
        p = pos.detach()
        d = p.unsqueeze(0) - p.unsqueeze(1)
        dist2 = (d * d).sum(dim=2)
        r2 = self.r_max * self.r_max
        mask = (dist2 < r2) & (dist2 > 0.0)
        if self.half:
            ar = torch.arange(N, device=dev)
            mask = mask & (ar.unsqueeze(0) > ar.unsqueeze(1))           # j > i
        n_edges = mask.sum()
        nz = torch.nonzero_static(mask, size=cap, fill_value=-1)       # [cap, 2]
        valid = nz[:, 0] >= 0
        spread = torch.arange(cap, device=dev) % N
        idx_i = torch.where(valid, nz[:, 0], spread)
        idx_j = torch.where(valid, nz[:, 1], spread)
        pad = (~valid).to(torch.float32) * (2.0 * self.r_max)
        offsets = torch.stack((pad, torch.zeros_like(pad), torch.zeros_like(pad)), dim=1)
        idx_m = torch.zeros((N,), dtype=torch.long, device=dev)
        n_atoms = torch.full((1,), N, dtype=torch.long, device=dev)
        if self.half:
            e = self.half_energy(pos, Z.to(torch.int64), idx_i, idx_j, offsets,
                                 idx_m, n_atoms, 1)
        else:
            e = self.fast_energy(pos, Z.to(torch.int64), idx_i, idx_j, offsets,
                                 idx_m, n_atoms, 1)
        grads = torch.autograd.grad([e.sum()], [pos], allow_unused=True)
        g = grads[0]
        conv = self.conv_float
        energy = e.to(torch.float64) * conv
        if g is None:
            forces = torch.zeros((N, 3), dtype=torch.float64, device=dev)
        else:
            forces = (-g).to(torch.float64) * conv
        return energy, forces, n_edges

    # -----------------------------------------------------------------
    #  Forward (single molecule)
    # -----------------------------------------------------------------

    def forward(
        self,
        coords: torch.Tensor,
        Z: torch.Tensor,
        pc_coords: torch.Tensor,
        pc_charges: torch.Tensor,
        cell: torch.Tensor,
    ):
        dev = coords.device
        N = coords.size(0)

        coords32 = coords.to(torch.float32)
        Z = Z.to(torch.int64)

        # NAMD sends the box as [1, 3, 3], all zeros when the system is not
        # periodic.  Anything else would be a caller bug rather than something
        # to guess about, so just take the first entry.
        cell3 = cell.reshape(-1, 3, 3)[0].to(torch.float32)
        periodic = cell_is_periodic(cell3)

        # Cache invariant tensors
        if N != self._cached_N:
            self._cached_N = N
            self._cached_idx_m   = torch.zeros(N, dtype=torch.long, device=dev)
            self._cached_n_atoms = torch.tensor([N], dtype=torch.long, device=dev)
            self._cached_cell    = torch.zeros((3, 3), dtype=torch.float32, device=dev)

        conv = self.conv_factor.to(dev)

        # Build edges (FP32).  SchNetPack forms its own displacements as
        # pos[_idx_j] - pos[_idx_i] + _offsets, so the offsets are the
        # Cartesian shift of the neighbour, in Angstrom.
        if periodic:
            # Our own leaf to differentiate: the forces and the virial both come
            # off it, and going through the model whole would also flip
            # requires_grad on the caller's tensor, which this avoids.
            coords_leaf = coords.to(torch.float32).detach().requires_grad_(True)
            edge_index, _, _, unit_shifts = build_edges_pbc(
                coords_leaf, cell3, self.r_max
            )
            # Positions, cell and offsets all take the same zero strain, so that
            # differentiating with respect to it gives the virial.  Straining
            # the positions alone would leave every imaged neighbour behind and
            # answer a different question, plausibly.
            D = make_strain(cell3)
            pos_in, cell_s, offsets = apply_strain(
                coords_leaf, cell3, unit_shifts, D
            )

            strained: Dict[str, torch.Tensor] = {
                "_positions":       pos_in,
                "_atomic_numbers":  Z,
                "_idx_i":           edge_index[0],
                "_idx_j":           edge_index[1],
                "_offsets":         offsets,
                # SchNetPack indexes the cell per structure, so one box still
                # has to arrive as [1, 3, 3].
                "_cell":            cell_s.unsqueeze(0),
                "_n_atoms":         self._cached_n_atoms,
                "_idx_m":           self._cached_idx_m,
            }
            energy_raw = self.energy_only(strained)
            # One backward for both, which is why the wrapper bothered to take
            # the energy itself rather than let the model produce the forces.
            forces_raw, virial_raw = forces_and_virial(
                energy_raw, coords_leaf, D, 1.0
            )

            energy = energy_raw.to(torch.float64).squeeze(-1) * conv
            forces = forces_raw.to(torch.float64) * conv
            virial = finalize(virial_raw.to(torch.float64) * conv)
        elif self.fast:
            if N * N >= self.nl_cell_min_pairs:
                edge_index, _, _ = build_edges_cell(coords32, self.r_max)
            else:
                edge_index, _, _ = build_edges(coords32, self.r_max)
            energy, forces = self._fast_eval(
                coords, Z, edge_index, self._cached_idx_m, self._cached_n_atoms, 1)
            virial = torch.zeros((3, 3), dtype=torch.float64, device=dev)
        else:
            edge_index, _, _ = build_edges(coords32, self.r_max)
            offsets = torch.zeros(
                (edge_index.size(1), 3), dtype=torch.float32, device=dev,
            )
            cell_in = self._cached_cell

            idx_i = edge_index[0]
            idx_j = edge_index[1]

            inputs: Dict[str, torch.Tensor] = {
                "_positions":       coords32,
                "_atomic_numbers":  Z,
                "_idx_i":           idx_i,
                "_idx_j":           idx_j,
                "_offsets":         offsets,
                "_cell":            cell_in,
                "_n_atoms":         self._cached_n_atoms,
                "_idx_m":           self._cached_idx_m,
            }

            out = self.inner(inputs)

            energy_raw = out[self.energy_key]
            forces_raw = out[self.forces_key]

            if energy_raw is None:
                raise RuntimeError("SchNetPack model returned energy=None")
            if forces_raw is None:
                raise RuntimeError("SchNetPack model returned forces=None")

            # SchNetPack often returns energy as [B, 1]; squeeze the trailing
            # dim only so we don't accidentally collapse unrelated unit dims.
            energy = energy_raw.to(torch.float64).squeeze(-1) * conv
            forces = forces_raw.to(torch.float64) * conv
            # Nothing to strain, so nothing to report.  NAMD falls back to its
            # own sum here, but the shape has to stay the same either way
            # because TorchScript allows only one return type.
            virial = torch.zeros((3, 3), dtype=torch.float64, device=dev)

        charges = torch.zeros(N, dtype=torch.float64, device=dev)

        return energy, forces, charges, virial

    # -----------------------------------------------------------------
    #  Batched forward
    # -----------------------------------------------------------------

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
    ):
        """
        Evaluate SchNetPack for a batch of molecules.

        Args:
            coords:     [N_total, 3]  float64  concatenated positions.
            Z:          [N_total]     int64    atomic numbers.
            batch:      [N_total]     int64    molecule index per atom.
            ptr:        [B+1]         int64    molecule boundaries.
            pc_coords:  [P, 3]        float64  (ignored).
            pc_charges: [P]           float64  (ignored).
            cells:      [B, 3, 3]     one box per molecule, rows are lattice
                                      vectors; all zeros means non-periodic.

        Returns:
            energies:  [B]           float64  kcal/mol.
            forces:    [N_total, 3]  float64  kcal/mol/Å.
            charges:   [N_total]     float64  e.
            virials:   [B, 3, 3]     float64  kcal/mol, one per molecule,
                                     all zero when the batch is not periodic.
        """
        dev = coords.device
        N_total = coords.size(0)
        B = ptr.size(0) - 1

        coords32 = coords.to(torch.float32)
        Z = Z.to(torch.int64)

        cells32 = cells.reshape(-1, 3, 3).to(torch.float32)
        # NAMD guarantees every walker in a batch agrees about periodicity, so
        # the first cell decides for the whole batch.
        periodic = cell_is_periodic(cells32[0])

        # Per-molecule atom counts
        n_atoms = ptr[1:] - ptr[:-1]  # [B]

        conv = self.conv_factor.to(dev)

        # Block-diagonal edges
        if periodic:
            coords_leaf = coords.to(torch.float32).detach().requires_grad_(True)
            edge_index, _, _, unit_shifts = build_edges_batched_pbc(
                coords_leaf, ptr, cells32, self.r_max,
            )
            # One strain per molecule rather than one for the batch.  A shared
            # 3x3 would only ever give the sum of the walkers' virials, which
            # looks like an answer and is not one, and NAMD needs them apart
            # because each replica runs its own barostat.  Molecule b's energy
            # touches only D[b], so a single backward separates them.
            D = torch.zeros_like(cells32).requires_grad_(True)
            sym = 0.5 * (D + D.transpose(-1, -2))
            pos_in = coords_leaf + torch.einsum(
                "ni,nij->nj", coords_leaf, sym.index_select(0, batch)
            )
            cells_s = cells32 + torch.bmm(cells32, sym)
            # Each edge is shifted by its OWN molecule's cell; the walkers'
            # boxes are not the same tensor.
            edge_cells = cells_s.index_select(0, batch.index_select(0, edge_index[0]))
            offsets = torch.einsum("ei,eij->ej", unit_shifts, edge_cells)

            strained: Dict[str, torch.Tensor] = {
                "_positions":       pos_in,
                "_atomic_numbers":  Z,
                "_idx_i":           edge_index[0],
                "_idx_j":           edge_index[1],
                "_offsets":         offsets,
                "_cell":            cells_s,
                "_n_atoms":         n_atoms,
                "_idx_m":           batch,
            }
            energy_raw = self.energy_only(strained)

            grads: List[Optional[torch.Tensor]] = torch.autograd.grad(
                [energy_raw.sum()], [coords_leaf, D], create_graph=False,
                retain_graph=False, allow_unused=True,
            )
            gp = grads[0]
            gd = grads[1]

            energies = energy_raw.to(torch.float64).squeeze(-1) * conv
            if gp is None:
                forces = torch.zeros(N_total, 3, dtype=torch.float64, device=dev)
            else:
                forces = (-gp).to(torch.float64) * conv
            if gd is None:
                virials = torch.zeros((B, 3, 3), dtype=torch.float64, device=dev)
            else:
                virials = finalize((-gd).to(torch.float64) * conv)
        elif self.fast:
            if B > 0 and (N_total * N_total) // B >= self.nl_cell_min_pairs:
                edge_index, _, _ = build_edges_cell_batched(coords32, ptr, self.r_max)
            else:
                edge_index, _, _ = build_edges_batched(coords32, ptr, self.r_max)
            energies, forces = self._fast_eval(
                coords, Z, edge_index, batch, n_atoms, B)
            virials = torch.zeros((B, 3, 3), dtype=torch.float64, device=dev)
        else:
            edge_index, _, _ = build_edges_batched(coords32, ptr, self.r_max)
            offsets = torch.zeros(
                (edge_index.size(1), 3), dtype=torch.float32, device=dev,
            )
            cell = torch.zeros((3, 3), dtype=torch.float32, device=dev)

            idx_i = edge_index[0]
            idx_j = edge_index[1]

            inputs: Dict[str, torch.Tensor] = {
                "_positions":       coords32,
                "_atomic_numbers":  Z,
                "_idx_i":           idx_i,
                "_idx_j":           idx_j,
                "_offsets":         offsets,
                "_cell":            cell,
                "_n_atoms":         n_atoms,
                "_idx_m":           batch,
            }

            out = self.inner(inputs)

            energy_raw = out[self.energy_key]
            forces_raw = out[self.forces_key]

            if energy_raw is None:
                raise RuntimeError("SchNetPack model returned energy=None")
            if forces_raw is None:
                raise RuntimeError("SchNetPack model returned forces=None")

            # SchNetPack often returns energy as [B, 1]; squeeze to [B].
            energies = energy_raw.to(torch.float64).squeeze(-1) * conv
            forces   = forces_raw.to(torch.float64) * conv
            virials  = torch.zeros((B, 3, 3), dtype=torch.float64, device=dev)

        charges = torch.zeros(N_total, dtype=torch.float64, device=dev)

        return energies, forces, charges, virials


# -------------------------------------------------------------------
#  CLI
# -------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export a wrapped SchNetPack model for NAMD",
    )
    parser.add_argument("--model", required=True,
                        help="Path to scripted SchNetPack .pt file")
    parser.add_argument("--r-max", type=float, required=True,
                        help="Cutoff radius in Å (must match training)")
    parser.add_argument("--out", default="mlff_model.pt",
                        help="Output TorchScript file")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--energy-key", default="energy",
                        help="Output dict key for energy")
    parser.add_argument("--forces-key", default="forces",
                        help="Output dict key for forces")
    parser.add_argument("--fast", action="store_true",
                        help="opt-in fast path + CUDA-graph API (graph_step)")
    parser.add_argument("--graph-max-atoms", type=int, default=2048,
                        help="[--fast] largest system a graph-capable shim should capture")
    parser.add_argument("--no-half-filter", action="store_true",
                        help="[--fast] run the filter network per directed edge (stock layout)")
    parser.add_argument("--half-min-atoms", type=int, default=1500,
                        help="[--fast] total atoms from which forward() uses the half-list filter")
    parser.add_argument("--nl-cell-min-pairs", type=int, default=16_000_000,
                        help="[--fast] B*n^2 above which the cell-list neighbour list is used")

    args = parser.parse_args()

    wrapper = SchNetPack_Wrapper(
        model_path=args.model,
        r_max=args.r_max,
        device=args.device,
        energy_key=args.energy_key,
        forces_key=args.forces_key,
        fast=args.fast,
        graph_max_atoms=args.graph_max_atoms,
        nl_cell_min_pairs=args.nl_cell_min_pairs,
        half_filter=not args.no_half_filter,
        half_min_atoms=args.half_min_atoms,
    ).eval()
    export_wrapped(wrapper, args.out, model_type="SchNetPack")

