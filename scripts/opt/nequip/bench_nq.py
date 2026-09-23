"""bench_common with a hard VRAM cap (like mace/bench_capped.py) and optional python imports that
register custom ops (benchmark-only; NOT the NAMD path).
usage: python bench_nq.py [--cap-gib X] [--import MOD ...] <bench_common args...>"""
import importlib, os, runpy, sys
import torch
cap = 15.0
args = sys.argv[1:]
imports = []
while args and args[0] in ("--cap-gib", "--import"):
    if args[0] == "--cap-gib":
        cap = float(args[1])
    else:
        imports.append(args[1])
    args = args[2:]
for m in imports:
    importlib.import_module(m)
tot = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(min(1.0, cap * 2**30 / tot), 0)
bench = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench_common.py")
sys.argv = [bench] + args
runpy.run_path(bench, run_name="__main__")
