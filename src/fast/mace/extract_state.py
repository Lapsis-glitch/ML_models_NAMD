"""Dump a MACE ``*.model`` to a version-neutral file (class, config, weights).

The optimised build rebuilds the model under the current mace/e3nn.  Models
pickled by an older e3nn (e.g. MACE-OFF23, e3nn 0.4.4) cannot be unpickled
there, so run this in the env that wrote them (``MACE_312``):

    python -m src.fast.mace.extract_state MACE-OFF23_medium.model state.pt
"""
import os
import sys

import torch


def extract_state(model_path: str, out_path: str) -> None:
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    from mace.tools.scripts_utils import extract_config_mace_model
    m = torch.load(model_path, weights_only=False, map_location="cpu")
    cfg = extract_config_mace_model(m)
    cfg["gate"] = "silu" if cfg["gate"] is torch.nn.functional.silu else cfg["gate"]
    cfg["interaction_cls"] = cfg["interaction_cls"].__name__
    cfg["interaction_cls_first"] = cfg["interaction_cls_first"].__name__
    cfg["hidden_irreps"] = str(cfg["hidden_irreps"])
    cfg["MLP_irreps"] = str(cfg["MLP_irreps"])
    cfg["atomic_numbers"] = [int(z) for z in cfg["atomic_numbers"]]
    torch.save({"class": type(m).__name__, "config": cfg,
                "dtype": str(next(m.parameters()).dtype),
                "state_dict": m.state_dict()}, out_path)
    print(f"[MACE] weights of {model_path} ({type(m).__name__})  ->  {out_path}")


if __name__ == "__main__":
    extract_state(sys.argv[1], sys.argv[2])
