"""Direct test of the one-step lag: does the force NAMD APPLIED at step n equal the
model's force at the CURRENT positions x_n, or at the PREVIOUS positions x_{n-1}?"""
import sys, struct, numpy as np

def read_dcd(path):
    with open(path, "rb") as fh:
        b = fh.read()
    off = 0
    def rec(o):
        (n,) = struct.unpack_from("<i", b, o); o += 4
        data = b[o:o+n]; o += n
        (n2,) = struct.unpack_from("<i", b, o); o += 4
        assert n == n2, (n, n2)
        return data, o
    hdr, off = rec(off)                      # 'CORD' + 20 int32
    icntrl = struct.unpack_from("<20i", hdr, 4)
    nframes = icntrl[0]; has_box = icntrl[10]
    _title, off = rec(off)
    natom_blk, off = rec(off)
    (natom,) = struct.unpack_from("<i", natom_blk, 0)
    frames = []
    for _ in range(nframes):
        if has_box:
            _box, off = rec(off)
        xb, off = rec(off); yb, off = rec(off); zb, off = rec(off)
        x = np.frombuffer(xb, "<f4"); y = np.frombuffer(yb, "<f4"); z = np.frombuffer(zb, "<f4")
        frames.append(np.stack([x, y, z], axis=1))
    return natom, has_box, np.array(frames)

sysdir = "/home/rat/PycharmProjects/ML_models_NAMD/namd_benchmarks/enzyme/systems/mono_shell5"
na_p, box_p, pos = read_dcd("pos.dcd")     # steps 1..6
na_f, box_f, frc = read_dcd("force.dcd")   # applied forces, steps 1..6
print(f"pos frames {pos.shape} box={box_p} | force frames {frc.shape} box={box_f}")

# x0 = initial coords (step 0), from the qm.pdb order the model expects
import importlib.util as u
sp = u.spec_from_file_location("exp", "/home/rat/PycharmProjects/ML_models_NAMD/scripts/export_fennix_bio1_stablehlo.py")
exp = u.module_from_spec(sp); sp.loader.exec_module(exp)
from pathlib import Path
from fennol import FENNIX
z, x0, *_ = exp.load_pdb_system(Path(f"{sysdir}/qm.pdb"))
z = np.asarray(z); x0 = np.asarray(x0, np.float64); n = len(z)
m = FENNIX.load("/home/rat/PycharmProjects/ML_models_NAMD/models/fennix-bio1S.fnx")
EV2K = 23.0621

def Fmodel(c):  # kcal/mol/A, true force
    _e, f, _ = m.energy_and_forces(species=z, coordinates=np.asarray(c, np.float64),
        natoms=np.array([n], np.int32), batch_index=np.zeros(n, np.int32),
        total_charge=np.array([0], np.int32))
    return np.asarray(f).reshape(n, 3) * EV2K

# Build the position sequence x_0, x_1, ... (x0 + dcd frames which are steps 1..)
X = [x0] + [pos[k] for k in range(pos.shape[0])]      # X[0]=step0, X[1]=step1, ...
def cmp(a, b):
    a = a.ravel(); b = b.ravel()
    return dict(maxabs=float(np.abs(a-b).max()),
                rms=float(np.sqrt(((a-b)**2).mean())),
                corr=float(np.corrcoef(a, b)[0,1]))
print(f"\nfor each applied-force frame (step n), compare to Fmodel(x_n) vs Fmodel(x_n-1):")
print(f"{'step n':>6} {'|Fapp|rms':>10} {'vs Fmodel(x_n)':>30} {'vs Fmodel(x_n-1)':>30}")
for k in range(frc.shape[0]):           # k=0 -> step 1
    step = k + 1
    Fapp = frc[k]
    cur = cmp(Fapp, Fmodel(X[step]))            # current positions
    prv = cmp(Fapp, Fmodel(X[step-1]))          # previous positions
    print(f"{step:>6} {np.sqrt((Fapp**2).mean()):>10.3f} "
          f"  rms={cur['rms']:.3f} corr={cur['corr']:.4f}   "
          f"  rms={prv['rms']:.3f} corr={prv['corr']:.4f}")

print("\n=== ORDERING CHECK: is pos.dcd in the same atom order as qm.pdb (x0)? ===")
x1 = pos[0]                       # step-1 positions (atoms barely moved, should ~= x0)
d = x1 - x0
print(f"  rms|x1-x0| = {np.sqrt((d**2).mean()):.4f} A,  max|x1-x0| = {np.abs(d).max():.4f} A")
print(f"  corr(x1,x0) per-component = {np.corrcoef(x1.ravel(), x0.ravel())[0,1]:.5f}")
# If reordered, find the permutation: nearest x0 atom to each pos atom (by coords)
print(f"  (if rms is ~Angstrom-small => same order; if huge => DCD is reordered)")

print("\n=== KINEMATIC CHECK (independent of force.dcd): F_used ∝ m*(x1-x0) from rest ===")
MASS = {1:1.008,6:12.011,7:14.007,8:15.999,11:22.99,15:30.97,16:32.06,17:35.45}
mass = np.array([MASS.get(int(zz), 2*int(zz)) for zz in z])[:,None]
F_kin = mass * (pos[0] - x0)          # ∝ true force NAMD applied at step 0 (from rest)
Fm0 = Fmodel(x0)
print(f"  corr( m*(x1-x0) , Fmodel(x0) )      = {np.corrcoef(F_kin.ravel(), Fm0.ravel())[0,1]:.4f}")
print(f"  corr( force.dcd[step1] , Fmodel(x0) ) = {np.corrcoef(frc[0].ravel(), Fm0.ravel())[0,1]:.4f}")
print(f"  corr( m*(x1-x0) , force.dcd[step1] )  = {np.corrcoef(F_kin.ravel(), frc[0].ravel())[0,1]:.4f}")
# also: does the MOTION go downhill on the model surface? (F_model . displacement > 0 ?)
disp = (pos[0]-x0)
print(f"  Fmodel(x0) . displacement = {float((Fm0*disp).sum()):+.3f}  (>0 => atoms moved along model force)")

print("\n=== IS NAMD APPLYING THE CLASSICAL MM FORCE under qmReplaceAll? ===")
_, _, frcmm = read_dcd("forcemm.dcd")     # MM-only applied force, step1 ~= F_MM(x0)
Fapp = frc[0]; FMM = frcmm[0]; FQM = Fmodel(x0)
def c(a,b): return np.corrcoef(a.ravel(), b.ravel())[0,1]
print(f"  corr( Fapp_QMrun , F_MM )            = {c(Fapp, FMM):.4f}")
print(f"  corr( Fapp_QMrun , F_QM(model) )      = {c(Fapp, FQM):.4f}")
print(f"  corr( Fapp_QMrun , F_MM + F_QM )      = {c(Fapp, FMM+FQM):.4f}")
print(f"  corr( Fapp_QMrun - F_MM , F_QM )      = {c(Fapp-FMM, FQM):.4f}")
print(f"  rms|Fapp - (F_MM+F_QM)| = {np.sqrt(((Fapp-(FMM+FQM))**2).mean()):.4f}  "
      f"(|Fapp|rms={np.sqrt((Fapp**2).mean()):.3f}, |F_MM|rms={np.sqrt((FMM**2).mean()):.3f}, |F_QM|rms={np.sqrt((FQM**2).mean()):.3f})")

print("\n=== PERMUTATION vs ROTATION of the QM forces? ===")
Fapp = frc[0]; FQM = Fmodel(x0)
magA = np.sort(np.linalg.norm(Fapp,axis=1)); magQ = np.sort(np.linalg.norm(FQM,axis=1))
print(f"  sorted per-atom |F|: max diff = {np.abs(magA-magQ).max():.4f}, "
      f"corr = {np.corrcoef(magA,magQ)[0,1]:.5f}  (==1 => Fapp is a permutation/rotation of F_QM)")
# best-fit global 3x3 rotation R minimizing |Fapp - FQM R|  (Kabsch)
H = FQM.T @ Fapp; U,S,Vt = np.linalg.svd(H); R = U @ Vt
Frot = FQM @ R
print(f"  best-fit global rotation R: corr(Fapp, FQM@R) = {np.corrcoef(Fapp.ravel(),Frot.ravel())[0,1]:.4f}, "
      f"rms|Fapp-FQM@R| = {np.sqrt(((Fapp-Frot)**2).mean()):.4f}")
print(f"  R =\n{np.array2string(R, precision=3)}")
print(f"  det(R) = {np.linalg.det(R):.3f}")

print("\n=== TRANSPOSE/LAYOUT TEST: compare Fapp to the shim's actual output (reference.npz) ===")
ref = np.load("/home/rat/PycharmProjects/ML_models_NAMD/namd_benchmarks/enzyme/models/fennix_enzyme/mono_shell5/reference.npz")
Fjit = np.asarray(ref["forces_jit_ev_a"]).reshape(-1) * EV2K   # shim output at x0, kcal/mol/A, flat
Fapp_flat = frc[0].reshape(-1)
N = n
cands = {
  "as-is [N,3] row-major":            Fjit.reshape(N,3),
  "read [N,3]buf as [3,N] then .T":   Fjit.reshape(3,N).T,
  "F.reshape(N,3).T.flatten->(N,3)":  Fjit.reshape(N,3).T.reshape(N,3),
}
for name,cand in cands.items():
    cc = np.corrcoef(Fapp_flat, cand.reshape(-1))[0,1]
    rms = np.sqrt(((Fapp_flat-cand.reshape(-1))**2).mean())
    print(f"  corr(Fapp, {name:32s}) = {cc:.4f}   rms={rms:.4f}")
# brute: is Fapp a flat-identical multiset to Fjit (same numbers, regrouped)?
print(f"\n  sorted(all Fapp values) vs sorted(all Fjit values): "
      f"max diff = {np.abs(np.sort(Fapp_flat)-np.sort(Fjit)).max():.4f}  "
      f"(==0 => exact same numbers, just regrouped => layout bug)")

print("\n=== SANITY: does my F_model(x0) match the artifact's own recorded force at x0? ===")
fjit = np.asarray(ref["forces_jit_ev_a"]).reshape(n,3) * EV2K
fref = np.asarray(ref["forces_ref_ev_a"]).reshape(n,3) * EV2K
FQ = Fmodel(x0)
print(f"  corr(my F_model(x0), reference forces_jit) = {np.corrcoef(FQ.ravel(), fjit.ravel())[0,1]:.5f}")
print(f"  corr(my F_model(x0), reference forces_ref) = {np.corrcoef(FQ.ravel(), fref.ravel())[0,1]:.5f}")
print(f"  rms|F_model - forces_jit| = {np.sqrt(((FQ-fjit)**2).mean()):.4f}  (should be ~0 if consistent)")
# and the reference coords vs x0 (are they the same geometry?)
rc = np.asarray(ref["coordinates"]).reshape(-1,3)
print(f"  rms|ref.coordinates - x0| = {np.sqrt(((rc-x0)**2).mean()):.5f} A  (0 => same geometry)")
print(f"  corr(reference forces_jit, force.dcd[step1]) = {np.corrcoef(fjit.ravel(), frc[0].ravel())[0,1]:.4f}")
