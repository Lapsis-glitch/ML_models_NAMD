"""Count ATen ops (forward + backward) per NequIP submodule type, eager OEQ model (analysis only)."""
import glob, os, sys, collections
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
from torch.utils._python_dispatch import TorchDispatchMode
import bench_common as bc
from src.edges import build_edges
from nequip.model.saved_models.load_utils import load_saved_model
from nequip.model.modify_utils import modify
from nequip.model.utils import _EAGER_MODEL_KEY
z = glob.glob(os.path.expanduser('~/.nequip/model_cache/*.nequip.zip'))[0]
m = modify(load_saved_model(z, _EAGER_MODEL_KEY, "sole_model"), [{"modifier": "enable_OpenEquivariance"}]).cuda().eval()
dev = torch.device("cuda"); xyz, Z = bc.water_system(30); pos = xyz.to(dev).float()
ei = build_edges(pos, 6.0)[0]; E = ei.size(1); N = pos.size(0)
at = torch.tensor([{1: 0, 8: 7}[int(z)] for z in Z], device=dev)
d = {"pos": pos, "edge_index": ei, "atom_types": at, "edge_cell_shift": torch.zeros(E, 3, device=dev),
     "cell": torch.zeros(1, 3, 3, device=dev), "batch": torch.zeros(N, dtype=torch.long, device=dev),
     "num_atoms": torch.tensor([N], device=dev)}
stack = ["<top>"]; counts = collections.Counter(); ops = collections.Counter()
VIEW = {"view", "reshape", "_unsafe_view", "slice", "select", "narrow", "expand", "permute", "transpose",
        "unsqueeze", "squeeze", "t", "alias", "detach", "as_strided", "split", "unbind"}
class M(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        n = func.__name__.split(".")[0]
        counts[(stack[-1], "view" if n in VIEW else "compute")] += 1
        ops[n] += 1
        return func(*args, **(kwargs or {}))
def pre(name):
    return lambda mod, inp: stack.append(name)
def post(name):
    return lambda mod, inp, out: (stack.pop(), None)[1]
for name, sub in m.named_modules():
    t = type(sub).__name__
    if t in ("Linear", "Gate", "FullyConnectedTensorProduct", "OpenEquivarianceTensorProductScatter",
             "ScalarMLPFunction", "SphericalHarmonicEdgeAttrs", "BesselEdgeLengthEncoding", "ZBL",
             "PerTypeScaleShift", "NodeTypeEmbed", "EdgeLengthNormalizer", "AtomwiseReduce", "ScalarMLP"):
        sub.register_forward_pre_hook(pre(t)); sub.register_forward_hook(post(t))
with M():
    out = m(d)
tot = sum(counts.values())
print("total ops (fwd+bwd incl. autograd of forces):", tot)
bytype = collections.defaultdict(lambda: [0, 0])
for (k, kind), v in counts.items():
    bytype[k][0 if kind == "compute" else 1] += v
for k, (c, v) in sorted(bytype.items(), key=lambda x: -sum(x[1])):
    print(f"{k:40s} compute {c:5d}  view {v:5d}")
print(ops.most_common(25))
