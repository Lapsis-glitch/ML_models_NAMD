#!/usr/bin/env python
"""TorchScript-compile a NequIP package in torch>=2.10 (allegro env).

`nequip-compile --mode torchscript` refuses to run on torch>=2.10 (hard check in
nequip/scripts/compile.py), but torch.jit.script itself still works.  This
replicates compile.py's torchscript branch with the check removed and
TorchScript mode forced on for the model build.

usage: compile_ts.py <package.nequip.zip|nequip.net:id> <out.nequip.pth>
           --device cuda [--modifiers enable_CuEquivariance ...] [--tf32]
"""
import argparse
import torch

p = argparse.ArgumentParser()
p.add_argument("input")
p.add_argument("output")
p.add_argument("--device", default="cuda")
p.add_argument("--modifiers", nargs="*", default=[])
p.add_argument("--tf32", action="store_true")
a = p.parse_args()

from nequip.utils.compile import conditional_torchscript_mode
from nequip.utils.global_state import set_global_state, get_latest_global_state
from nequip.scripts._workflow_utils import set_workflow_state
from nequip.model.saved_models.load_utils import load_saved_model
from nequip.model.modify_utils import modify
from nequip.model.inference_models.torchscript import save_torchscript_model
from nequip.model.utils import _EAGER_MODEL_KEY

set_workflow_state("compile")
set_global_state(allow_tf32=a.tf32)
with conditional_torchscript_mode(True):
    model = load_saved_model(a.input, _EAGER_MODEL_KEY, "sole_model")
    model = modify(model, [{"modifier": m} for m in a.modifiers])
    md = model.metadata.copy()
    md.update(get_latest_global_state(only_metadata_related=True))
    md = {k: str(int(v)) if isinstance(v, bool) else v for k, v in md.items()}
    print("metadata:", md)
    save_torchscript_model(model, md, a.output, torch.device(a.device))
set_workflow_state(None)
print("saved", a.output)
