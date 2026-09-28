#!/usr/bin/env python
"""Re-serialize an exported FeNNiX StableHLO artifact as MLIR *bytecode* (same program, no re-export).

Text MLIR at 6000 atoms is 298 MB (dense triu pair-index constants printed as hex) and its parse is a
large part of the NAMD start-up compile.  PJRT_Client_Compile(format="mlir") accepts bytecode as well.
Writes OUT_DIR with the bytecode file + symlinked sidecars + manifest pointing at it.
usage: to_bytecode.py SRC_DIR OUT_DIR   (fennix env)
"""
import json, os, sys, time
from jax._src.interpreters import mlir as jmlir
from jaxlib.mlir import ir

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
man = json.load(open(os.path.join(src, "manifest.json")))
txt = open(os.path.join(src, man["artifacts"]["stablehlo_mlir"])).read()
ctx = jmlir.make_ir_context()
t = time.time()
with ctx:
    mod = ir.Module.parse(txt)
print(f"parsed text MLIR ({len(txt)/1e6:.1f} MB) in {time.time()-t:.2f} s")
bc = "fennix_bio1_eval.stablehlo.mlirbc"
with open(os.path.join(out, bc), "wb") as f:
    mod.operation.write_bytecode(f)
print("bytecode MB", os.path.getsize(os.path.join(out, bc)) / 1e6)
for key in ("compile_options_pb", "hlo_text", "reference_npz", "reference_runtime"):
    fn = man["artifacts"][key]
    dst = os.path.join(out, fn)
    if os.path.lexists(dst): os.remove(dst)
    if key == "compile_options_pb" and os.path.exists(os.path.join(src, fn)) and not os.path.islink(os.path.join(src, fn)):
        pass
    os.symlink(os.path.realpath(os.path.join(src, fn)), dst)
man["artifacts"]["stablehlo_mlir"] = bc
man.setdefault("opt", {})["bytecode_of"] = os.path.realpath(os.path.join(src, "fennix_bio1_eval.stablehlo.mlir"))
json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)
