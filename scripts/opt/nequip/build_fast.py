"""Build a FastNequIP inner (TorchScript .nequip.pth) from the NequIP-OAM-L package.
usage: build_fast.py PKG OUT [--no-half] [--no-species-sc] [--check N ...]
Loads the eager model, applies enable_OpenEquivariance, then fast_nequip.make_fast surgery, checks eager
parity vs the stock OEQ eager model (python OEQ imported here: BUILD-time only), then scripts + saves
like nequip-compile --mode torchscript (compile_ts.py)."""
import argparse, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
import torch
ap = argparse.ArgumentParser()
ap.add_argument("pkg"); ap.add_argument("out")
ap.add_argument("--no-half", action="store_true"); ap.add_argument("--no-species-sc", action="store_true")
ap.add_argument("--check", type=int, nargs="*", default=[30, 300])
a = ap.parse_args()
import copy
from nequip.utils.compile import conditional_torchscript_mode
from nequip.utils.global_state import set_global_state, get_latest_global_state
from nequip.scripts._workflow_utils import set_workflow_state
from nequip.model.saved_models.load_utils import load_saved_model
from nequip.model.modify_utils import modify
from nequip.model.inference_models.torchscript import save_torchscript_model
from nequip.model.utils import _EAGER_MODEL_KEY
import fast_nequip
import bench_common as bc
from src.edges import build_edges

set_workflow_state("compile")
set_global_state(allow_tf32=False)
dev = torch.device("cuda")
with conditional_torchscript_mode(True):
    model = load_saved_model(a.pkg, _EAGER_MODEL_KEY, "sole_model")
    model = modify(model, [{"modifier": "enable_OpenEquivariance"}]).to(dev)
    ref = copy.deepcopy(model)
    rep = fast_nequip.make_fast(model, half_edges=not a.no_half, species_sc=not a.no_species_sc)
    print("species-sc kron-structure rel err per layer:", {k: f"{v:.1e}" for k, v in rep.items()})

    tn = [str(s) for s in model.type_names] if hasattr(model, "type_names") else None
    def data_for(n):
        xyz, Z = bc.water_system(n)
        pos = (xyz.to(dev) + 0.02 * torch.randn(xyz.shape, dtype=xyz.dtype, device=dev)).float()
        ei = build_edges(pos, 6.0)[0]; E = ei.size(1); N = pos.size(0)
        from src.constants import SYMBOL_TO_Z
        z2t = {SYMBOL_TO_Z[s]: i for i, s in enumerate(model.metadata["type_names"].split())}
        at = torch.tensor([z2t[int(z)] for z in Z], device=dev)
        return {"pos": pos, "edge_index": ei, "atom_types": at, "edge_cell_shift": torch.zeros(E, 3, dtype=torch.float32, device=dev),
                "cell": torch.zeros(1, 3, 3, dtype=torch.float32, device=dev), "batch": torch.zeros(N, dtype=torch.long, device=dev),
                "num_atoms": torch.tensor([N], device=dev)}
    for n in a.check:
        d = data_for(n)
        o0 = ref(dict(d)); o1 = model(dict(d))
        de = (o0["total_energy"] - o1["total_energy"]).abs().max().item() * 23.0605
        df = (o0["forces"] - o1["forces"]).abs().max().item() * 23.0605
        print(f"eager parity N={n}: |dE| {de:.2e} kcal/mol  max|dF| {df:.2e} kcal/mol/A")
    del ref
    md = model.metadata.copy()
    md.update(get_latest_global_state(only_metadata_related=True))
    md = {k: str(int(v)) if isinstance(v, bool) else v for k, v in md.items()}
    save_torchscript_model(model, md, a.out, dev)
set_workflow_state(None)
print("saved", a.out)
