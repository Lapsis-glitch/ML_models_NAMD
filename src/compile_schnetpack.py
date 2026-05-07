"""
Build a default SchNetPack (≥ 2.0) model with **random weights** and
TorchScript-compile it for use with the existing
:class:`src.wrappers.wrap_schnetpack.SchNetPack_Wrapper`.

This is intended for **timing / plumbing benchmarks only**.  The model
has not been trained, so its energies and forces are physically
meaningless — but inference cost (and therefore the wrapper overhead +
forward latency) is identical to a trained model with the same
hyperparameters.

Usage::

    python -m src.compile_schnetpack \
        --out models/compiled_schnet_default.pt \
        --r-max 5.0 --n-atom-basis 128 --n-interactions 3 --n-rbf 20

Then wrap for NAMD::

    python -m src.cli --model-type schnet \
        --compiled models/compiled_schnet_default.pt \
        --r-max 5.0 \
        --out models/trpcage_schnet_full.pt
"""

import argparse
from pathlib import Path

import torch


def build_schnet(
    r_max: float = 5.0,
    n_atom_basis: int = 128,
    n_interactions: int = 3,
    n_rbf: int = 20,
):
    import schnetpack as spk

    cutoff_fn = spk.nn.cutoff.CosineCutoff(r_max)
    radial_basis = spk.nn.radial.GaussianRBF(n_rbf=n_rbf, cutoff=r_max)

    schnet = spk.representation.SchNet(
        n_atom_basis=n_atom_basis,
        n_interactions=n_interactions,
        radial_basis=radial_basis,
        cutoff_fn=cutoff_fn,
    )
    energy_head = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key="energy")
    forces_head = spk.atomistic.Forces(energy_key="energy", force_key="forces")

    model = spk.model.NeuralNetworkPotential(
        representation=schnet,
        input_modules=[spk.atomistic.PairwiseDistances()],
        output_modules=[energy_head, forces_head],
        postprocessors=[],
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def compile_schnet(
    out_path: str,
    r_max: float,
    n_atom_basis: int,
    n_interactions: int,
    n_rbf: int,
) -> None:
    model = build_schnet(r_max, n_atom_basis, n_interactions, n_rbf)

    sample = {
        "_positions":      torch.randn(3, 3, dtype=torch.float32),
        "_atomic_numbers": torch.tensor([8, 1, 1], dtype=torch.long),
        "_idx_i":          torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long),
        "_idx_j":          torch.tensor([1, 2, 0, 2, 0, 1], dtype=torch.long),
        "_offsets":        torch.zeros(6, 3, dtype=torch.float32),
        "_cell":           torch.zeros(3, 3, dtype=torch.float32),
        "_n_atoms":        torch.tensor([3], dtype=torch.long),
        "_idx_m":          torch.zeros(3, dtype=torch.long),
    }

    try:
        scripted = torch.jit.script(model)
        mode = "script"
    except Exception as e:
        print(f"[SchNetPack] jit.script failed ({e!r}); falling back to jit.trace")
        scripted = torch.jit.trace(model, (sample,))
        mode = "trace"

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    scripted.save(out_path)
    print(f"[SchNetPack] Compiled (mode={mode})  ->  {out_path}")
    print(f"  r_max:          {r_max} Å")
    print(f"  n_atom_basis:   {n_atom_basis}")
    print(f"  n_interactions: {n_interactions}")
    print(f"  n_rbf:          {n_rbf}")
    print(f"  NOTE: random weights — for timing/plumbing only.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True,
                        help="Output compiled TorchScript .pt file")
    parser.add_argument("--r-max", type=float, default=5.0)
    parser.add_argument("--n-atom-basis", type=int, default=128)
    parser.add_argument("--n-interactions", type=int, default=3)
    parser.add_argument("--n-rbf", type=int, default=20)
    args = parser.parse_args(argv)
    compile_schnet(
        args.out, args.r_max, args.n_atom_basis,
        args.n_interactions, args.n_rbf,
    )


if __name__ == "__main__":
    main()
