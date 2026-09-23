"""TF32 EXTRA parity: same artifact (nequip_oeq.pt, native op lib), outputs with TF32 off vs on, in one
process (the flag is read at matmul time). 5 jittered geometries per cell. NAMD equivalent: NAMD_MLFF_TF32=1."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
import bench_common as bc
torch.ops.load_library(str(bc.REPO / "scripts/opt/nequip/oeq_native/liboeq_native.so"))
torch.cuda.set_per_process_memory_fraction(15 * 2**30 / torch.cuda.get_device_properties(0).total_memory, 0)
bc.JITTER = 0.02
dev = torch.device("cuda:0")
r = bc.Runner("oeq", bc.REPO / "models/opt/nequip_oeq.pt", dev)
for n, W in ((30, 1), (300, 1), (300, 4), (900, 1), (3000, 1), (6000, 1)):
    xyz, Z = bc.water_system(n); r.prepare(xyz, Z, W)
    de = df = 0.0; fmax = 0.0
    for k in range(5):
        torch.backends.cuda.matmul.allow_tf32 = False
        r._k = k - 1; e0, f0 = r.call(); f0 = f0.clone()
        torch.backends.cuda.matmul.allow_tf32 = True
        r._k = k - 1; e1, f1 = r.call()
        de = max(de, (e1 - e0).abs().max().item()); df = max(df, (f1 - f0).abs().max().item())
        fmax = max(fmax, f0.abs().max().item())
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"N={n:5d} W={W}  TF32 vs fp32: max|dE| {de:.3e} kcal/mol  max|dF| {df:.3e} kcal/mol/A  (max|F| {fmax:.1f})")
