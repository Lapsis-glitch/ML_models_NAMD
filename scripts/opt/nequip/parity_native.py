"""Parity of the NAMD-path NequIP artifacts, in a fresh process that loads ONLY the native OEQ op
library (asserts openequivariance python is never imported).  Reference = baseline; recomp gives the
fp32 noise floor (same e3nn model, recompiled).  N = 30/300 x W = 1/4, 5 jittered geometries each."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
import bench_common as bc
REPO = bc.REPO
torch.ops.load_library(str(REPO / "scripts/opt/nequip/oeq_native/liboeq_native.so"))
dev = torch.device("cuda:0")
arts = {k: REPO / "models/opt" / f"nequip_{k}.pt" for k in ("baseline", "recompiled_cuda", "oeq")}
runners = {k: bc.Runner(k, p, dev) for k, p in arts.items()}
assert not any(m.startswith("openequivariance") for m in sys.modules), "openequivariance imported!"
bc.JITTER = 0.02
ok = True
for n in (30, 300):
    xyz, Z = bc.water_system(n)
    for W in (1, 4):
        for r in runners.values():
            r.prepare(xyz, Z, W)
        worst = {k: [0.0, 0.0] for k in runners if k != "baseline"}
        for _ in range(5):
            eb, fb = runners["baseline"].call(); fb = fb.clone()
            for k, r in runners.items():
                if k == "baseline":
                    continue
                e, f = r.call()
                worst[k][0] = max(worst[k][0], (e - eb).abs().max().item())
                worst[k][1] = max(worst[k][1], (f - fb).abs().max().item())
        for k, (de, df) in worst.items():
            flag = "OK" if df < 5e-3 and de < 5e-2 else "FAIL"
            ok &= flag == "OK"
            print(f"N={n:4d} W={W} {k:16s} max|dE| {de:.2e} kcal/mol  max|dF| {df:.2e} kcal/mol/A  {flag}")
assert not any(m.startswith("openequivariance") for m in sys.modules)
print("ALL OK" if ok else "PARITY FAIL")
