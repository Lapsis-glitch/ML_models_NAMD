"""fp64 reference energies/forces (torchani ANI-2x, pyaev, everything float64) for the geoms.py cases.
Hartree -> kcal/mol with the repo constant. usage: python ref_fp64.py OUT.pt"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
import ani_env; ani_env.setup(cuaev=False)
import torch, torchani
from src.constants import HARTREE_TO_KCAL
import geoms
m = torchani.models.ANI2x(periodic_table_index=True).cuda().double().eval()
out = {}
tot = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(min(1.0, 14 * 2**30 / tot), 0)
for (n, W, pbc) in geoms.CASES:
    if pbc and n >= 3000:
        continue   # fp64 all-image AllPairs needs > 16 GB here; that case is checked against the baseline only
    for g in range(geoms.N_GEOM):
        xs, Z, cell = geoms.case_inputs(n, W, pbc, g)
        Es, Fs, Vs = [], [], []
        for x in xs:
            c = x.cuda()[None].clone().requires_grad_(True)
            if pbc:
                # same strain convention as src/virial.py (apply_strain / forces_and_virial / finalize)
                D = torch.zeros(3, 3, dtype=torch.float64, device="cuda", requires_grad=True)
                sym = 0.5 * (D + D.t())
                cc = cell.cuda()
                e = m((Z.cuda()[None], c + c @ sym), cell=cc + cc @ sym,
                      pbc=torch.ones(3, dtype=torch.bool, device="cuda")).energies
                f, gD = torch.autograd.grad(e.sum(), [c, D])
                V = -gD * HARTREE_TO_KCAL; V = 0.5 * (V + V.t())
            else:
                e = m((Z.cuda()[None], c)).energies
                f, = torch.autograd.grad(e.sum(), c)
                V = torch.zeros(3, 3, dtype=torch.float64)
            Es.append(e.detach().cpu()[0] * HARTREE_TO_KCAL); Fs.append(-f[0].cpu() * HARTREE_TO_KCAL)
            Vs.append(V.detach().cpu())
        out[geoms.key(n, W, pbc, g)] = (torch.stack(Es), torch.cat(Fs, 0), torch.stack(Vs))
torch.save(out, sys.argv[1]); print("saved", sys.argv[1], len(out), "entries")
