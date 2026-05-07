"""
Benchmark wrapped MACE-OFF23 fp64 vs fp32 TorchScript artifacts.

Both models are evaluated on the same randomly generated structures so the
timing difference isolates the internal precision change.
"""

import argparse
import statistics as st
import time

import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


FP64_WRAPPED = "/home/rat/PycharmProjects/ML_models_NAMD/models/mace_off23_wrapped_identical.pt"
FP32_WRAPPED = "/home/rat/PycharmProjects/ML_models_NAMD/models/mace_off23_wrapped_fp32_test.pt"

SEED = 20260508
N_ATOMS = 32
N_WARMUP = 10
N_BENCH = 100
ALLOWED_Z = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
MIN_DISTANCE = 0.9
BOX_SIZE = 10.0


def _fmt(ts):
    s = sorted(ts)
    return (
        f"mean={st.mean(ts) * 1e3:.2f} ms  "
        f"median={st.median(ts) * 1e3:.2f} ms  "
        f"p90={s[int(0.9 * len(s))] * 1e3:.2f} ms  "
        f"throughput={1.0 / st.mean(ts):.2f} struct/s"
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
    parser.add_argument("--fp64", default=FP64_WRAPPED)
    parser.add_argument("--fp32", default=FP32_WRAPPED)
    parser.add_argument("--atoms", type=int, default=N_ATOMS)
    parser.add_argument("--warmup", type=int, default=N_WARMUP)
    parser.add_argument("--bench", type=int, default=N_BENCH)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--min-distance", type=float, default=MIN_DISTANCE)
    parser.add_argument("--box-size", type=float, default=BOX_SIZE)
    args = parser.parse_args()

    device = args.device
    fp64_path = args.fp64
    fp32_path = args.fp32
    n_atoms = args.atoms
    n_warmup = args.warmup
    n_bench = args.bench
    seed = args.seed
    min_distance = args.min_distance
    box_size = args.box_size

    print("Comparing wrapped MACE-OFF23 fp64 vs fp32 artifacts")
    print(f"device={device}  atoms={n_atoms}  warmup={n_warmup}  benchmark={n_bench}")
    print(f"min_distance={min_distance} Å  box_size={box_size} Å")
    print(f"fp64={fp64_path}")
    print(f"fp32={fp32_path}")

    rng = np.random.default_rng(seed)
    structs = []
    for _ in range(n_warmup + n_bench):
        z = rng.choice(ALLOWED_Z, size=n_atoms).tolist()
        pos = _sample_positions(rng, n_atoms, min_distance=min_distance, box_size=box_size)
        structs.append((z, pos))

    fp64_model = torch.jit.load(fp64_path, map_location=device).eval()
    fp32_model = torch.jit.load(fp32_path, map_location=device).eval()
    pc_coords = torch.empty((0, 3), dtype=torch.float64, device=device)
    pc_charges = torch.empty((0,), dtype=torch.float64, device=device)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    def run(model, desc):
        times = []
        energies = []
        forces = []
        for i, (z, pos) in enumerate(tqdm(structs, desc=desc)):
            z_t = torch.tensor(z, dtype=torch.long, device=device)
            pos_t = torch.tensor(pos, dtype=torch.float64, device=device)
            t0 = time.perf_counter()
            e, f, _ = model(pos_t, z_t, pc_coords, pc_charges)
            sync()
            dt = time.perf_counter() - t0
            if i >= n_warmup:
                times.append(dt)
                energies.append(float(e.view(-1)[0].item()))
                forces.append(f.detach().cpu())
        return times, np.array(energies), forces

    print("running fp64 wrapper ...")
    fp64_times, fp64_energies, fp64_forces = run(fp64_model, "FP64 benchmark")
    print("running fp32 wrapper ...")
    fp32_times, fp32_energies, fp32_forces = run(fp32_model, "FP32 benchmark")

    ratio = st.mean(fp32_times) / st.mean(fp64_times)
    d_e = np.abs(fp64_energies - fp32_energies)
    mean_force_err = float(np.mean([(a - b).abs().mean().item() for a, b in zip(fp64_forces, fp32_forces)]))
    max_force_err = max((a - b).abs().max().item() for a, b in zip(fp64_forces, fp32_forces))

    print("\n=== Wrapped MACE-OFF23 fp64 vs fp32 benchmark ===")
    print(f"fp64: {_fmt(fp64_times)}")
    print(f"fp32: {_fmt(fp32_times)}")
    print(f"fp32/fp64 ratio (mean): {ratio:.3f}x  ({'faster' if ratio < 1 else 'slower'})")
    print(f"\nParity over {n_bench} structures:")
    print(f"  mean |dE| = {d_e.mean():.6f} kcal/mol")
    print(f"  max  |dE| = {d_e.max():.6f} kcal/mol")
    print(f"  mean |dF| = {mean_force_err:.6f} kcal/mol/A")
    print(f"  max  |dF| = {max_force_err:.6f} kcal/mol/A")


if __name__ == "__main__":
    main()
