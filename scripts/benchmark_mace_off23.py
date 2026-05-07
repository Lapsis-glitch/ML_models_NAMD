"""
Benchmark MACE-OFF23 medium using the same underlying OFF23 weights via:
  1. Official MACE ASE interface: mace.calculators.MACECalculator + ase.Atoms
  2. This repo's wrapped TorchScript export

Timing includes both energy and force evaluation on each random structure.
"""

import argparse
import statistics as st
import time

import numpy as np
import torch
from ase import Atoms

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


MODEL_PATH = "/home/rat/.cache/mace/MACE-OFF23_medium.model"
WRAPPED_PT = "/home/rat/PycharmProjects/ML_models_NAMD/models/mace_off23_wrapped_identical.pt"

SEED = 20260508
N_ATOMS = 32
N_WARMUP = 10
N_BENCH = 100
ALLOWED_Z = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
SYMBOLS = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S", 17: "Cl", 35: "Br", 53: "I"}
EV_TO_KCAL = 23.0621


def _fmt(ts):
    s = sorted(ts)
    return (
        f"mean={st.mean(ts) * 1e3:.2f} ms  "
        f"median={st.median(ts) * 1e3:.2f} ms  "
        f"p90={s[int(0.9 * len(s))] * 1e3:.2f} ms  "
        f"throughput={1.0 / st.mean(ts):.1f} struct/s"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--atoms", type=int, default=N_ATOMS)
    parser.add_argument("--warmup", type=int, default=N_WARMUP)
    parser.add_argument("--bench", type=int, default=N_BENCH)
    args = parser.parse_args()
    device = args.device
    n_atoms = args.atoms
    n_warmup = args.warmup
    n_bench = args.bench

    print("Comparing official MACECalculator+ASE vs wrapped TorchScript export")
    print(f"device={device}  atoms={n_atoms}  warmup={n_warmup}  benchmark={n_bench}")
    print(f"model={MODEL_PATH}")
    print(f"wrapper={WRAPPED_PT}")

    rng = np.random.default_rng(SEED)
    structs = []
    for _ in range(n_warmup + n_bench):
        z = rng.choice(ALLOWED_Z, size=n_atoms).tolist()
        pos = (rng.random((n_atoms, 3)) - 0.5) * 10.0
        structs.append((z, pos))

    from mace.calculators import MACECalculator  # type: ignore[import-not-found]

    calc = MACECalculator(model_paths=MODEL_PATH, device=device, default_dtype="float64")

    def ase_eval(z, pos):
        atoms = Atoms(symbols=[SYMBOLS[x] for x in z], positions=pos)
        atoms.calc = calc
        e = atoms.get_potential_energy()
        f = atoms.get_forces()
        return e, f

    print("warming up ASE/MACECalculator ...")
    for z, pos in tqdm(structs[:n_warmup], desc="ASE warmup", leave=False):
        ase_eval(z, pos)

    print("timing ASE/MACECalculator ...")
    ase_times = []
    ase_energies = []
    for z, pos in tqdm(structs[n_warmup:], desc="ASE benchmark"):
        t0 = time.perf_counter()
        e, _ = ase_eval(z, pos)
        ase_times.append(time.perf_counter() - t0)
        ase_energies.append(e)

    wrapped = torch.jit.load(WRAPPED_PT, map_location=device).eval()
    pc_coords = torch.empty((0, 3), dtype=torch.float64, device=device)
    pc_charges = torch.empty((0,), dtype=torch.float64, device=device)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    def wrapped_eval(z, pos):
        z_t = torch.tensor(z, dtype=torch.long, device=device)
        pos_t = torch.tensor(pos, dtype=torch.float64, device=device)
        return wrapped(pos_t, z_t, pc_coords, pc_charges)

    print("warming up wrapper ...")
    for z, pos in tqdm(structs[:n_warmup], desc="Wrapper warmup", leave=False):
        wrapped_eval(z, pos)
        sync()

    print("timing wrapper ...")
    wrap_times = []
    wrap_energies = []
    for z, pos in tqdm(structs[n_warmup:], desc="Wrapper benchmark"):
        t0 = time.perf_counter()
        e, _, _ = wrapped_eval(z, pos)
        sync()
        wrap_times.append(time.perf_counter() - t0)
        wrap_energies.append(float(e.item()))

    print("\n=== MACE-OFF23-medium benchmark ===")
    print(f"native  (MACECalculator+ASE): {_fmt(ase_times)}")
    print(f"wrapper (TorchScript NAMD)  : {_fmt(wrap_times)}")
    ratio = st.mean(wrap_times) / st.mean(ase_times)
    print(f"wrapper/native ratio (mean) : {ratio:.3f}x  ({'faster' if ratio < 1 else 'slower'})")

    ase_kcal = np.array(ase_energies) * EV_TO_KCAL
    wrap_kcal = np.array(wrap_energies)
    d_e = np.abs(ase_kcal - wrap_kcal)
    print(f"\nEnergy parity over {n_bench} structures (kcal/mol):")
    print(f"  mean |dE| = {d_e.mean():.6f}")
    print(f"  max  |dE| = {d_e.max():.6f}")


if __name__ == "__main__":
    main()
