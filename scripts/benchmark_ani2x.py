"""
Benchmark ANI-2x using the same model family via:
  1. Official TorchANI implementation: torchani.models.ANI2x
  2. This repo's wrapped TorchScript export

Timing includes both energy and force evaluation on each random structure.
"""

import argparse
import os
import statistics as st
import time

import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


WRAPPED_PT = "/home/rat/PycharmProjects/ML_models_NAMD/models/ani2x_wrapped_identical.pt"

SEED = 20260508
N_ATOMS = 32
N_WARMUP = 10
N_BENCH = 100
ALLOWED_Z = [1, 6, 7, 8, 16, 17]
HARTREE_TO_KCAL = 627.509474
MIN_DISTANCE = 0.9
BOX_SIZE = 10.0


def _fmt(ts):
    s = sorted(ts)
    return (
        f"mean={st.mean(ts) * 1e3:.2f} ms  "
        f"median={st.median(ts) * 1e3:.2f} ms  "
        f"p90={s[int(0.9 * len(s))] * 1e3:.2f} ms  "
        f"throughput={1.0 / st.mean(ts):.1f} struct/s"
    )


def _sample_positions(rng, n_atoms, min_distance, box_size, max_attempts=10_000):
    pos = np.empty((n_atoms, 3), dtype=np.float64)
    for i in range(n_atoms):
        for _ in range(max_attempts):
            cand = (rng.random(3) - 0.5) * box_size
            if i == 0:
                pos[i] = cand
                break
            d = np.linalg.norm(pos[:i] - cand, axis=1)
            if np.all(d >= min_distance):
                pos[i] = cand
                break
        else:
            raise RuntimeError(
                f"Failed to place atom {i} with min_distance={min_distance} in box_size={box_size}"
            )
    return pos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--atoms", type=int, default=N_ATOMS)
    parser.add_argument("--warmup", type=int, default=N_WARMUP)
    parser.add_argument("--bench", type=int, default=N_BENCH)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--wrapped", default=WRAPPED_PT)
    parser.add_argument("--min-distance", type=float, default=MIN_DISTANCE)
    parser.add_argument("--box-size", type=float, default=BOX_SIZE)
    args = parser.parse_args()

    device = args.device
    n_atoms = args.atoms
    n_warmup = args.warmup
    n_bench = args.bench
    seed = args.seed
    wrapped_path = args.wrapped
    min_distance = args.min_distance
    box_size = args.box_size

    print("Comparing official TorchANI ANI2x vs wrapped TorchScript export")
    print(f"device={device}  atoms={n_atoms}  warmup={n_warmup}  benchmark={n_bench}")
    print(f"min_distance={min_distance} Å  box_size={box_size} Å")
    print("official=torchani.models.ANI2x(periodic_table_index=False)")
    print(f"wrapper={wrapped_path}")

    rng = np.random.default_rng(seed)
    structs = []
    for _ in range(n_warmup + n_bench):
        z = rng.choice(ALLOWED_Z, size=n_atoms).tolist()
        pos = _sample_positions(rng, n_atoms, min_distance=min_distance, box_size=box_size)
        structs.append((z, pos))

    os.environ.setdefault("TORCHANI_NO_WARN_EXTENSIONS", "1")
    import torchani

    native = torchani.models.ANI2x(periodic_table_index=False, device=device).eval()
    wrapped = torch.jit.load(wrapped_path, map_location=device).eval()
    pc_coords = torch.empty((0, 3), dtype=torch.float64, device=device)
    pc_charges = torch.empty((0,), dtype=torch.float64, device=device)
    z_to_species = wrapped.z_to_species.to(device)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    def native_eval(z, pos):
        z_t = torch.tensor(z, dtype=torch.long, device=device)
        species = z_to_species[z_t].unsqueeze(0)
        coords = torch.tensor(pos, dtype=torch.float32, device=device).unsqueeze(0)
        coords = coords.requires_grad_(True)
        out = native((species, coords))
        energies_ha = out.energies if hasattr(out, "energies") else out[1]
        grad = torch.autograd.grad([energies_ha.sum()], [coords], create_graph=False, retain_graph=False)[0]
        forces_ha = -grad.squeeze(0)
        energy_kcal = energies_ha.squeeze().to(torch.float64) * HARTREE_TO_KCAL
        forces_kcal = forces_ha.to(torch.float64) * HARTREE_TO_KCAL
        return float(energy_kcal.item()), forces_kcal

    def wrapped_eval(z, pos):
        z_t = torch.tensor(z, dtype=torch.long, device=device)
        pos_t = torch.tensor(pos, dtype=torch.float64, device=device)
        return wrapped(pos_t, z_t, pc_coords, pc_charges)

    print("warming up official TorchANI ...")
    for z, pos in tqdm(structs[:n_warmup], desc="TorchANI warmup", leave=False):
        native_eval(z, pos)
        sync()

    print("timing official TorchANI ...")
    native_times = []
    native_energies = []
    for z, pos in tqdm(structs[n_warmup:], desc="TorchANI benchmark"):
        t0 = time.perf_counter()
        e, _ = native_eval(z, pos)
        sync()
        native_times.append(time.perf_counter() - t0)
        native_energies.append(e)

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

    print("\n=== ANI2x benchmark ===")
    print(f"native  (official TorchANI): {_fmt(native_times)}")
    print(f"wrapper (TorchScript NAMD): {_fmt(wrap_times)}")
    ratio = st.mean(wrap_times) / st.mean(native_times)
    print(f"wrapper/native ratio (mean): {ratio:.3f}x  ({'faster' if ratio < 1 else 'slower'})")

    native_kcal = np.array(native_energies)
    wrap_kcal = np.array(wrap_energies)
    d_e = np.abs(native_kcal - wrap_kcal)
    print(f"\nEnergy parity over {n_bench} structures (kcal/mol):")
    print(f"  mean |dE| = {d_e.mean():.6f}")
    print(f"  max  |dE| = {d_e.max():.6f}")


if __name__ == "__main__":
    main()
