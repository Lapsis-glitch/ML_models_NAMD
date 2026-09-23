"""Run in MACE_312 (e3nn 0.4.4): dump MACE-OFF config + state_dict to a
version-neutral file so the model can be rebuilt under e3nn 0.6 (allegro),
where the pickled e3nn-0.4.4 codegen state cannot be unpickled."""
import sys, os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
import torch
from mace.tools.scripts_utils import extract_config_mace_model
src, dst = sys.argv[1], sys.argv[2]
m = torch.load(src, weights_only=False, map_location="cpu")
cfg = extract_config_mace_model(m)
cfg["gate"] = "silu" if cfg["gate"] is torch.nn.functional.silu else cfg["gate"]
cfg["interaction_cls"] = cfg["interaction_cls"].__name__
cfg["interaction_cls_first"] = cfg["interaction_cls_first"].__name__
cfg["hidden_irreps"] = str(cfg["hidden_irreps"])
cfg["MLP_irreps"] = str(cfg["MLP_irreps"])
cfg["atomic_numbers"] = [int(z) for z in cfg["atomic_numbers"]]
torch.save({"class": type(m).__name__, "config": cfg,
            "dtype": str(next(m.parameters()).dtype),
            "state_dict": m.state_dict()}, dst)
print("saved", dst, type(m).__name__, len(m.state_dict()))
