#!/usr/bin/env python
"""Write a FeNNiX artifact variant dir whose compile_options.pb carries XLA option overrides.

The PJRT plugin ignores XLA_FLAGS for these artifacts (compile_options.pb embeds a full DebugOptions
frozen at export time), so per-artifact XLA tuning goes into CompileOptionsProto.env_option_overrides,
which PJRT_Client_Compile applies on top.  The StableHLO (the model) is symlinked, never changed.

usage: make_compile_options.py SRC_DIR OUT_DIR [xla_flag=value ...]   (run in the fennix env)
"""
import json, os, sys, shutil
from jax._src.lib import xla_client as xc

src, out, kv = sys.argv[1], sys.argv[2], sys.argv[3:]
os.makedirs(out, exist_ok=True)
o = xc.CompileOptions.ParseFromString(open(os.path.join(src, "compile_options.pb"), "rb").read())
ov = dict(o.env_option_overrides)
def typed(v):
    if v.lower() in ("true", "false"): return v.lower() == "true"
    for t in (int, float):
        try: return t(v)
        except ValueError: pass
    return v
for s in kv:
    k, v = s.split("=", 1)
    ov[k.lstrip("-")] = typed(v)
o.env_option_overrides = list(ov.items())
open(os.path.join(out, "compile_options.pb"), "wb").write(o.SerializeAsString())
man = json.load(open(os.path.join(src, "manifest.json")))
for key in ("stablehlo_mlir", "hlo_text", "reference_npz", "reference_runtime"):
    f = man["artifacts"][key]
    dst = os.path.join(out, f)
    if os.path.lexists(dst): os.remove(dst)
    os.symlink(os.path.realpath(os.path.join(src, f)), dst)
man.setdefault("opt", {})["xla_env_option_overrides"] = ov
man["opt"]["source_artifact"] = os.path.realpath(src)
json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)
print(out, ov)
