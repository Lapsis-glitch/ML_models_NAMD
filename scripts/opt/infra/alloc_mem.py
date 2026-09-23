#!/usr/bin/env python
"""
Phase-2 (infra): GPU memory of one artifact under MD-like jittered inputs.
Run once per allocator setting (PYTORCH_CUDA_ALLOC_CONF must be set before CUDA
init, so the caller sets it in the environment):

  python scripts/opt/infra/alloc_mem.py <model.pt> <n_atoms> [calls=100]

Prints: allocated peak, reserved peak/final (caching allocator), and the
device-wide used memory delta from cudaMemGetInfo (includes CUDA context,
kernel images, cuBLAS workspace).  Takes the shared GPU lock.
"""
import fcntl
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench_common as bc  # noqa: E402

model, n = sys.argv[1], int(sys.argv[2])
calls = int(sys.argv[3]) if len(sys.argv) > 3 else 100
fh = open(bc.LOCK_PATH, "w")
fcntl.flock(fh, fcntl.LOCK_EX)
dev = torch.device("cuda:0")
free0, total = torch.cuda.mem_get_info(dev)          # context created here
torch.cuda.init()
r = bc.Runner("m", Path(model), dev)
xyz, Z = bc.water_system(n)
bc.JITTER = 0.02
r.prepare(xyz, Z, 1)
for _ in range(calls):
    r.call()
torch.cuda.synchronize()
free1, _ = torch.cuda.mem_get_info(dev)
print(f"RESULT model={Path(model).name} n={n} alloc_conf={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')!r} "
      f"max_alloc_MiB={torch.cuda.max_memory_allocated() / 2**20:.1f} "
      f"max_reserved_MiB={torch.cuda.max_memory_reserved() / 2**20:.1f} "
      f"final_reserved_MiB={torch.cuda.memory_reserved() / 2**20:.1f} "
      f"device_used_delta_MiB={(free0 - free1) / 2**20:.1f}")
