"""Overflow / re-capture test of the CUDA-graph path.

Part 1 (python, model contract): capture graph_step at graph_capacity(A), replay with
denser coords B = 0.85*A -> n_edges > cap must be reported; re-capture at graph_capacity(B)
and compare with forward(B).
Part 2 (C ABI, patched shim via ctypes -- this process never imports torch): two handles on
the same model, graph ON and OFF; evaluate A, B (forces an in-shim re-capture), A; compare.
Run each part in its own process under the GPU flock:
  python test_overflow.py py     |     python test_overflow.py shim
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[4]
MODEL = str(REPO / "models/opt/schnet_fast.pt")
PDB = REPO / "namd_benchmarks/systems/w100_300atoms/qm.pdb"

def read_pdb():
    xyz, Z = [], []
    for l in PDB.read_text().splitlines():
        if l.startswith(("ATOM", "HETATM")):
            xyz.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])
            el = l[76:78].strip() or l[12:16].strip()[0]
            Z.append({"H": 1, "O": 8, "C": 6, "N": 7}[el[0]])
    return xyz, Z

if sys.argv[1] == "py":
    import torch
    dev = torch.device("cuda:0")
    m = torch.jit.load(MODEL, map_location=dev).eval()
    xyz, Z = read_pdb()
    A = torch.tensor(xyz, dtype=torch.float64, device=dev); Zt = torch.tensor(Z, device=dev)
    B = A * 0.85
    pcx = torch.zeros(0, 3, dtype=torch.float64, device=dev); pcq = torch.zeros(0, dtype=torch.float64, device=dev)
    cell = torch.zeros(1, 3, 3, dtype=torch.float64, device=dev)
    static = A.clone()
    cap = int(m.graph_capacity(static))
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(4): m.graph_step(static, Zt, cap)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = m.graph_step(static, Zt, cap)
    static.copy_(B); g.replay(); torch.cuda.synchronize()
    n = int(out[2]); print(f"cap(A)={cap}  replay with B: n_edges={n}  overflow reported: {n > cap}")
    assert n > cap
    del g
    cap2 = int(m.graph_capacity(B)); static.copy_(B)
    with torch.cuda.stream(s):
        for _ in range(4): m.graph_step(static, Zt, cap2)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = m.graph_step(static, Zt, cap2)
    g.replay(); torch.cuda.synchronize()
    e_ref, f_ref = m(B.clone().requires_grad_(True), Zt, pcx, pcq, cell)[:2]
    dE = float((out[0].reshape(()) - e_ref).abs()); dF = float((out[1] - f_ref).abs().max())
    print(f"re-captured cap(B)={cap2} n_edges={int(out[2])}  vs forward(B): dE={dE:.2e} dF={dF:.2e}")
    assert int(out[2]) <= cap2 and dF < 1e-4
    print("PY OVERFLOW TEST PASSED")
else:
    import ctypes, os
    lib = ctypes.CDLL(str(REPO / "scripts/opt/schnet/cudagraph/shim/libnamd_mlff.so"), mode=os.RTLD_NOW | os.RTLD_LOCAL)
    lib.mlff_load.restype = ctypes.c_void_p
    lib.mlff_load.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    D = ctypes.POINTER(ctypes.c_double); L = ctypes.POINTER(ctypes.c_int64)
    lib.mlff_eval.argtypes = [ctypes.c_void_p, D, L, ctypes.c_int, D, D, ctypes.c_int, D, D, D]
    lib.mlff_last_error.restype = ctypes.c_char_p
    sb = ctypes.c_int(0)
    os.environ["NAMD_MLFF_CUDA_GRAPH"] = "1"; hg = lib.mlff_load(MODEL.encode(), 0, ctypes.byref(sb))
    os.environ["NAMD_MLFF_CUDA_GRAPH"] = "0"; hf = lib.mlff_load(MODEL.encode(), 0, ctypes.byref(sb))
    assert hg and hf, lib.mlff_last_error()
    xyz, Z = read_pdb(); n = len(Z)
    Zc = (ctypes.c_int64 * n)(*Z)
    def ev(h, scale):
        c = (ctypes.c_double * (3 * n))(*[v * scale for p in xyz for v in p])
        e = (ctypes.c_double * 1)(); f = (ctypes.c_double * (3 * n))(); q = (ctypes.c_double * n)()
        rc = lib.mlff_eval(h, c, Zc, n, None, None, 0, e, f, q)
        assert rc == 0, lib.mlff_last_error()
        return e[0], list(f)
    ok = True
    for tag, sc in (("A", 1.0), ("B=0.85A (overflow -> recapture)", 0.85), ("B again", 0.85), ("A again", 1.0)):
        eg, fg = ev(hg, sc); ef, ff = ev(hf, sc)
        dF = max(abs(a - b) for a, b in zip(fg, ff))
        print(f"shim {tag:34s} E_graph={eg:.6f} E_fwd={ef:.6f} dE={abs(eg-ef):.2e} dF={dF:.2e}")
        ok &= dF < 1e-4
    print("SHIM OVERFLOW TEST PASSED" if ok else "SHIM OVERFLOW TEST FAILED")
