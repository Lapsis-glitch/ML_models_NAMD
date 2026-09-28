"""FastANI: an exact inference re-plumbing of a TorchANI ANI ensemble model (e.g. ANI-2x).

Same weights, same architecture, same fp32 network math; only the execution plan changes:

1. AEVs from the cuAEV CUDA kernels (torch.ops.cuaev.run, the 'fused' all-pairs single-molecule path of
   torchani's own extension) instead of the pure-PyTorch all-pairs neighbour list + radial/angular terms.
   ~10 kernels + 2 host syncs per molecule instead of ~150 ops. Autograd (forces) is the extension's own
   C++ backward. Built for NAMD as scripts/opt/ani/cuaev_native/libcuaev_native_<variant>.so.

2. Ensemble fused per element: the 8 members' networks for one element run as ONE addmm for layer 1
   ([n,1008] x [1008, 8*h1], i.e. the concatenated first layers) and batched baddbmm for layers 2..4
   ([8, n, h] x [8, h, h']). Atoms are grouped by element with one bincount (one small D2H copy) + a
   stable argsort, instead of 8 members x 7 elements x (eq + nonzero(sync) + index_select + MLP +
   index_add). No atomic energies are materialised: only per-molecule sums are needed.

3. Energy accumulation (opt-in acc64=True): per-atom network outputs (fp32, as computed) are summed over
   members and atoms in float64, and the (fp32-stored) self energies are added in float64. The stock
   model sums a ~1e4-1e5 Ha total in float32 (quantised to 0.05-4 kcal/mol for 30-6000 atoms); acc64
   removes that rounding, forces are unaffected (d sum / d x_i is 1 either way).

4. Periodic systems (the wrapper calls `.ani((species, coords), cell, pbc, ...)`): cuAEV-fused has no PBC
   and cuAEV's half-neighbour-list op does not propagate gradients to the pair vectors (virial wrong), so
   AEVs come from torchani's own pure-PyTorch terms on torchani's CellList (above 190 atoms and >= 3 cutoff
   buckets per axis, else AllPairs, as torchani's AdaptiveList) instead of the stock all-image AllPairs
   (O(27 N^2) pairs: 3.3 s and 9 GB at 3000 atoms). Coordinates are wrapped with the differentiable cell
   first so forces and virial match the stock path. Networks as in 2./3.

Export layout matches _TorchANIExportWrapper: FastANI.forward(species [B,N], coords [B,N,3]) ->
(species, energies [B]) and FastANI.ani(...) with torchani's ANI.forward signature.
"""
from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn


def _celu(x: Tensor) -> Tensor:
    return torch.nn.functional.celu(x, alpha=0.1)   # torchani TightCELU


class ElementNet(nn.Module):
    """All ensemble members' networks for one element, fused (3 hidden layers + scalar output)."""

    def __init__(self, member_nets):
        super().__init__()
        n_mem = len(member_nets)
        for net in member_nets:
            assert len(net.layers) == 3, "expects the ANI-2x layout (3 hidden layers)"
            assert type(net.activation).__name__ == "TightCELU"
            assert all(l.bias is not None for l in net.layers) and net.final_layer.bias is not None
        # Everything is kept TRANSPOSED (features x atoms) so each layer is a contiguous GEMM with no
        # layout copies: layer 1 is ONE mm [M*h1, D] @ [D, n] for all members, layers 2..4 are bmm over
        # members [M, h', h] @ [M, h, n]. Same dot products as the per-member nn.Linear layers.
        l0 = [net.layers[0] for net in member_nets]
        self.h1 = int(l0[0].out_features)
        self.n_mem = n_mem
        self.W1 = nn.Parameter(torch.cat([l.weight.detach() for l in l0], 0).contiguous(), False)  # [M*h1, D]
        self.b1 = nn.Parameter(torch.cat([l.bias.detach() for l in l0], 0).view(-1, 1).contiguous(), False)

        def stack(j):
            ls = [net.layers[j] if j < 3 else net.final_layer for net in member_nets]
            W = torch.stack([l.weight.detach() for l in ls], 0).contiguous()           # [M, out, in]
            b = torch.stack([l.bias.detach().view(-1, 1) for l in ls], 0).contiguous()  # [M, out, 1]
            return nn.Parameter(W, False), nn.Parameter(b, False)

        self.W2, self.b2 = stack(1)
        self.W3, self.b3 = stack(2)
        self.W4, self.b4 = stack(3)

    def forward(self, xT: Tensor) -> Tensor:
        """xT [D, n] fp32 (AEVs of n atoms, transposed) -> per-member atomic outputs [M, n] (fp32)."""
        n = xT.size(1)
        h = _celu(torch.addmm(self.b1, self.W1, xT))                   # [M*h1, n]
        h = h.view(self.n_mem, self.h1, n)                             # [M, h1, n] (contiguous view)
        h = _celu(torch.baddbmm(self.b2, self.W2, h))                  # [M, h2, n]
        h = _celu(torch.baddbmm(self.b3, self.W3, h))                  # [M, h3, n]
        return torch.baddbmm(self.b4, self.W4, h).squeeze(1)           # [M, n]


class FastANICore(nn.Module):
    _cached_counts: List[int]
    _cuaev_ready: bool
    _g_P: int
    _g_nmax: int
    _cached_N: int

    def __init__(self, ani: nn.Module, acc64: bool = True, cache_species: bool = False,
                 group_max_atoms: int = 0, pbc_cell_list_min_atoms: int = 190):
        super().__init__()
        import torchani
        aev = ani.aev_computer
        # periodic path: torchani's own pure-PyTorch AEV terms (pyaev, differentiable w.r.t. the pair
        # vectors -> correct virial) on a torchani neighbour list; cell list above a size threshold
        aev.set_strategy("pyaev")
        self.aev_computer = aev
        self.all_pairs = torchani.neighbors.AllPairs()
        self.cell_list = torchani.neighbors.CellList()
        self.cutoff = float(ani.cutoff)
        self.pbc_cell_list_min_atoms = int(pbc_cell_list_min_atoms)
        # walkers (B > 1) of <= this many atoms go through cuAEV's batch mode in ONE call (measured faster
        # than one call per walker up to ~1200 atoms at B=4 on the RTX 5080 laptop; hard limit ~2400)
        self.cuaev_batch_max_atoms = 1024
        assert aev._cuaev_cutoff_fn in ("cosine", "smooth")
        nn_ = ani.neural_networks
        assert hasattr(nn_, "members"), "expects an Ensemble"
        symbols = list(nn_.members[0].atomics.keys())
        self.num_species = len(symbols)
        self.n_mem = len(nn_.members)
        self.nets = nn.ModuleList([ElementNet([m.atomics[s] for m in nn_.members]) for s in symbols])
        # cuAEV parameters (buffers so they follow .to(device)); the CuaevComputer custom-class object does
        # not, so it is (re)built lazily on the first forward on the actual device, like torchani does.
        self.register_buffer("EtaR", aev.radial.eta.detach().clone())
        self.register_buffer("ShfR", aev.radial.shifts.detach().clone())
        self.register_buffer("EtaA", aev.angular.eta.detach().clone())
        self.register_buffer("Zeta", aev.angular.zeta.detach().clone())
        self.register_buffer("ShfA", aev.angular.shifts.detach().clone())
        self.register_buffer("ShfZ", aev.angular.sections.detach().clone())
        self.Rcr = float(aev.radial.cutoff)
        self.Rca = float(aev.angular.cutoff)
        self.use_cos_cutoff = aev._cuaev_cutoff_fn == "cosine"
        empty = torch.empty(0)
        self.cuaev_computer = torch.classes.cuaev.CuaevComputer(
            0.0, 0.0, empty, empty, empty, empty, empty, empty, 1, True)
        self._cuaev_ready = False
        self.register_buffer("self_energies", ani.energy_shifter.self_energies.detach().clone())
        self.acc64 = bool(acc64)
        # opt-in: reuse the element grouping while the species tensor is unchanged (checked on the device
        # with an async assert, i.e. a changed topology at the same size aborts instead of being wrong)
        self.cache_species = bool(cache_species)
        self._cached_key = torch.empty(0, dtype=torch.long)
        self._cached_order = torch.empty(0, dtype=torch.long)
        self._cached_counts: List[int] = []
        self._cached_N = -1

        # opt-in "grouped" network pass (needs cache_species): all present elements in ONE chain of batched
        # GEMMs over (element x member), hidden widths zero-padded to the widest element and atoms padded to
        # the most populated element (padded hidden units stay exactly 0 through CELU; padded atoms are masked
        # out of the energy). Fewer launches (small N) at the cost of padded FLOPs, so it is used only while
        # (#present elements) x (max atoms per element) <= group_max_atoms. 0 = never.
        self.group_max_atoms = int(group_max_atoms)
        M = self.n_mem
        D = int(self.nets[0].W1.shape[1])
        H1 = max(int(n.h1) for n in self.nets)
        H2 = max(int(n.W2.shape[1]) for n in self.nets)
        H3 = max(int(n.W3.shape[1]) for n in self.nets)
        S = self.num_species
        W1g = torch.zeros(S, M, H1, D); b1g = torch.zeros(S, M, H1, 1)
        W2g = torch.zeros(S, M, H2, H1); b2g = torch.zeros(S, M, H2, 1)
        W3g = torch.zeros(S, M, H3, H2); b3g = torch.zeros(S, M, H3, 1)
        W4g = torch.zeros(S, M, 1, H3); b4g = torch.zeros(S, M, 1, 1)
        for e, n in enumerate(self.nets):
            h1 = int(n.h1); h2 = int(n.W2.shape[1]); h3 = int(n.W3.shape[1])
            W1g[e, :, :h1] = n.W1.detach().view(M, h1, D); b1g[e, :, :h1] = n.b1.detach().view(M, h1, 1)
            W2g[e, :, :h2, :h1] = n.W2.detach(); b2g[e, :, :h2] = n.b2.detach()
            W3g[e, :, :h3, :h2] = n.W3.detach(); b3g[e, :, :h3] = n.b3.detach()
            W4g[e, :, :, :h3] = n.W4.detach(); b4g[e] = n.b4.detach()
        self.register_buffer("W1g", W1g.view(S, M * H1, D)); self.register_buffer("b1g", b1g.view(S, M * H1, 1))
        for nm, t in (("W2g", W2g), ("b2g", b2g), ("W3g", W3g), ("b3g", b3g), ("W4g", W4g), ("b4g", b4g)):
            self.register_buffer(nm, t)
        self.H1 = H1
        # cached grouped plan (valid while the cached species are): selected weights + gather index + mask
        self._g_P = 0
        self._g_nmax = 0
        self._g_idx = torch.empty(0, dtype=torch.long)
        self._g_mask = torch.empty(0)
        self._g_mol = torch.empty(0, dtype=torch.long)
        self._g_W = [torch.empty(0) for _ in range(8)]

    def _init_cuaev(self) -> None:
        self.cuaev_computer = torch.classes.cuaev.CuaevComputer(
            self.Rcr, self.Rca, self.EtaR, self.ShfR, self.EtaA, self.Zeta, self.ShfA, self.ShfZ,
            self.num_species, self.use_cos_cutoff)
        self._cuaev_ready = True

    def _group(self, key: Tensor, N: int):
        if self.cache_species and key.shape == self._cached_key.shape and N == self._cached_N \
                and key.device == self._cached_key.device:
            torch._assert_async((key == self._cached_key).all(),
                                "FastANI cache_species: species changed at fixed size")
            return self._cached_order, self._cached_counts, True
        counts: List[int] = torch.bincount(key, minlength=self.num_species + 1).tolist()   # 1 D2H sync
        order = torch.argsort(key, stable=True)
        if self.cache_species:
            self._cached_key = key.clone()
            self._cached_order = order
            self._cached_counts = counts
            self._cached_N = N
        return order, counts, False

    def _plan_grouped(self, order: Tensor, counts: List[int], N: int, acc: torch.dtype) -> None:
        S = self.num_species
        present: List[int] = []
        starts: List[int] = []
        cnt: List[int] = []
        start = 0
        for i in range(S):
            if counts[i] > 0:
                present.append(i); starts.append(start); cnt.append(counts[i])
            start += counts[i]
        P = len(present)
        nmax = max(cnt)
        dev = order.device
        Ntot = order.numel()
        j = torch.arange(nmax, device=dev).view(1, nmax)
        st = torch.tensor(starts, device=dev).view(P, 1)
        cn = torch.tensor(cnt, device=dev).view(P, 1)
        valid = j < cn
        pos = torch.clamp(st + j, max=Ntot - 1)
        atom = order[pos]                                                   # [P, nmax] sorted -> atom id
        self._g_idx = torch.where(valid, atom, torch.full_like(atom, Ntot)).flatten()   # Ntot = zero row
        self._g_mask = valid.to(acc)
        self._g_mol = torch.where(valid, torch.div(atom, N, rounding_mode="floor"), torch.zeros_like(atom))
        pi = torch.tensor(present, device=dev)
        M = self.n_mem
        self._g_W = [self.W1g.index_select(0, pi), self.b1g.index_select(0, pi),
                     self.W2g.index_select(0, pi).flatten(0, 1), self.b2g.index_select(0, pi).flatten(0, 1),
                     self.W3g.index_select(0, pi).flatten(0, 1), self.b3g.index_select(0, pi).flatten(0, 1),
                     self.W4g.index_select(0, pi).flatten(0, 1), self.b4g.index_select(0, pi).flatten(0, 1)]
        self._g_P = P
        self._g_nmax = nmax

    def _grouped(self, aevf: Tensor, B: int, acc: torch.dtype) -> Tensor:
        P = self._g_P
        nmax = self._g_nmax
        M = self.n_mem
        W = self._g_W
        aev_ext = torch.cat([aevf, aevf.new_zeros(1, aevf.size(1))], 0)
        X = aev_ext.index_select(0, self._g_idx).view(P, nmax, aevf.size(1))       # [P, nmax, D]
        h = _celu(torch.baddbmm(W[1], W[0], X.transpose(1, 2)))                    # [P, M*H1, nmax]
        h = h.view(P * M, self.H1, nmax)
        h = _celu(torch.baddbmm(W[3], W[2], h))
        h = _celu(torch.baddbmm(W[5], W[4], h))
        out = torch.baddbmm(W[7], W[6], h).view(P, M, nmax)                        # [P, M, nmax]
        e = out.to(acc).sum(1) * self._g_mask                                      # [P, nmax]
        if B == 1:
            return e.sum().view(1)
        return torch.zeros(B, dtype=acc, device=aevf.device).index_add(0, self._g_mol.flatten(), e.flatten())

    def _nn_energies(self, aev: Tensor, species: Tensor) -> Tensor:
        """Ensemble-mean network energy + self energies per molecule (acc dtype)."""
        B = species.size(0)
        N = species.size(1)
        flat = species.flatten()
        aevf = aev.flatten(0, 1)
        S = self.num_species
        key = torch.where(flat < 0, torch.full_like(flat, S), flat)     # padding -> extra bin S (dropped)
        order, counts, cached = self._group(key, N)
        acc = torch.float64 if self.acc64 else aevf.dtype
        use_group = False
        if self.cache_species and self.group_max_atoms > 0:
            if not cached:
                self._plan_grouped(order, counts, N, acc)
            use_group = self._g_P * self._g_nmax <= self.group_max_atoms
        if use_group:
            energies = self._grouped(aevf, B, acc)
        else:
            xT = aevf.index_select(0, order).t().contiguous()             # [D, N_atoms] element-sorted
            energies = torch.zeros(B, dtype=acc, device=aevf.device)
            mol = torch.div(order, N, rounding_mode="floor")
            start = 0
            for i, net in enumerate(self.nets):
                c = counts[i]
                if c > 0:
                    out = net(xT.narrow(1, start, c))                              # [M, c] fp32
                    if B == 1:
                        energies = energies + out.to(acc).sum()
                    else:
                        energies = energies.index_add(0, mol.narrow(0, start, c), out.to(acc).sum(0))
                start += c
        energies = energies / float(self.n_mem)
        sae = self.self_energies.to(acc)[key.clamp(max=S - 1)]
        sae = sae.masked_fill(key == S, 0.0).view(B, N).sum(1)
        return energies + sae

    def _aev_cuaev(self, species: Tensor, coordinates: Tensor) -> Tensor:
        B = species.size(0)
        if not self._cuaev_ready:
            self._init_cuaev()
        sp32 = species.to(torch.int32)
        if B == 1 or species.size(1) <= self.cuaev_batch_max_atoms:
            # B == 1: single-molecule kernels; small molecules: the extension's batch mode (one thread block
            # per molecule, molecule kept in shared memory: 20 B/atom <= 48 KiB, i.e. <= ~2400 atoms)
            return torch.ops.cuaev.run(coordinates.contiguous(), sp32.contiguous(), self.cuaev_computer)
        # larger molecules: one single-molecule call per molecule (one block per molecule gets slow)
        parts: List[Tensor] = []
        for b in range(B):
            parts.append(torch.ops.cuaev.run(coordinates[b:b + 1].contiguous(),
                                             sp32[b:b + 1].contiguous(), self.cuaev_computer))
        return torch.cat(parts, 0)

    def _aev_pbc(self, species: Tensor, coordinates: Tensor, cell: Tensor, pbc: Tensor) -> Tensor:
        # Wrap into the central cell OURSELVES with the (strained, differentiable) cell, so the cell list's
        # own map_to_central (which uses cell.detach()) is a no-op and every pair vector carries the strain
        # consistently (x_j - x_i + shift @ cell): forces AND virial stay those of the stock all-pairs path.
        frac = torch.matmul(coordinates.detach(), torch.linalg.inv(cell.detach()))
        coords = coordinates - torch.matmul(torch.floor(frac), cell)
        lengths = torch.linalg.norm(cell.detach(), dim=1)
        # cell list only where torchani's does not need self-images: >= 3 grid buckets per axis
        use_cl = species.size(1) >= self.pbc_cell_list_min_atoms and \
            bool((lengths >= 3.0 * self.cutoff + 1e-3).all())                    # 1 small D2H sync
        if use_cl:
            neighbors = self.cell_list(self.cutoff, species, coords, cell, pbc)
        else:
            neighbors = self.all_pairs(self.cutoff, species, coords, cell, pbc)
        return self.aev_computer.compute_from_neighbors(species, coords, neighbors)

    def forward(
        self,
        species_coordinates: Tuple[Tensor, Tensor],
        cell: Optional[Tensor] = None,
        pbc: Optional[Tensor] = None,
        charge: int = 0,
        atomic: bool = False,
        ensemble_values: bool = False,
        _molecule_idxs: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Same call signature as torchani's ANI.forward (only the energy path is implemented)."""
        species, coordinates = species_coordinates
        assert charge == 0 and not atomic and not ensemble_values and _molecule_idxs is None
        if pbc is None:
            aev = self._aev_cuaev(species, coordinates)
        else:
            assert cell is not None
            assert species.size(0) == 1, "periodic: one molecule per call"
            aev = self._aev_pbc(species, coordinates, cell, pbc)
        return species, self._nn_energies(aev, species)


class FastANI(nn.Module):
    """Export shape expected by src/wrappers/wrap_torchani.py: forward(species, coords) for clusters,
    `.ani((species, coords), cell, pbc, ...)` for the periodic path."""

    def __init__(self, ani: nn.Module, **kw):
        super().__init__()
        self.ani = FastANICore(ani, **kw)

    def forward(self, species: Tensor, coordinates: Tensor) -> Tuple[Tensor, Tensor]:
        return self.ani((species, coordinates))
