"""Run scripts/opt/bench_common.py with a hard VRAM cap (default 15 GiB of the 16 GiB card).
On WSL2 the driver silently spills over-sized allocations into shared system memory instead of
raising OOM, which makes a too-big cell crawl for tens of minutes; the cap turns that into a clean
OOM row. usage: python bench_capped.py [--cap-gib X] <bench_common args...>"""
import os, runpy, sys
import torch
cap = 15.0
if len(sys.argv) > 2 and sys.argv[1] == "--cap-gib":
    cap = float(sys.argv[2]); del sys.argv[1:3]
tot = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(min(1.0, cap * 2**30 / tot), 0)
bench = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench_common.py")
sys.argv[0] = bench
runpy.run_path(bench, run_name="__main__")
