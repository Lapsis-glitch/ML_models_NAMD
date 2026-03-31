"""
Train a **TorchANI** model and export it for use with the NAMD wrapper.

Builds a TorchANI model with an AEV computer + per-element networks,
trains with a standard PyTorch loop (Adam + ReduceLROnPlateau, joint
energy + forces loss), then exports via ``torch.jit.script``.

The scripted ``.pt`` is loadable by
:class:`src.wrappers.wrap_torchani.TorchANI_Wrapper`.

Usage::

    python -m src.training.train_torchani \\
        --data-dir ./prepared_data \\
        --output-dir ./results/torchani \\
        --elements H C N O \\
        --max-epochs 200 \\
        --device cuda
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# -------------------------------------------------------------------
#  Element helpers
# -------------------------------------------------------------------

_SYMBOL_TO_Z = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7,
    "O": 8, "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13,
    "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
}

_Z_TO_SYMBOL = {v: k for k, v in _SYMBOL_TO_Z.items()}


# -------------------------------------------------------------------
#  HDF5 Dataset
# -------------------------------------------------------------------

class ANI_H5_Dataset(Dataset):
    """
    Read a split (train/val/test) from the HDF5 file produced by
    :mod:`src.training.prepare_data`.

    Returns:
        species:     [N]    int64   (atomic numbers)
        coordinates: [N, 3] float64 (Å)
        energy:      scalar float64 (Hartree)
        forces:      [N, 3] float64 (Hartree/Å)
    """

    def __init__(self, h5_path: str, split: str = "train"):
        import h5py
        self.h5_path = h5_path
        self.split = split
        with h5py.File(h5_path, "r") as f:
            grp = f[split]
            self.species = torch.tensor(np.array(grp["species"]), dtype=torch.long)
            self.coordinates = torch.tensor(np.array(grp["coordinates"]), dtype=torch.float64)
            self.energies = torch.tensor(np.array(grp["energies"]), dtype=torch.float64)
            self.forces = torch.tensor(np.array(grp["forces"]), dtype=torch.float64)

    def __len__(self):
        return self.species.size(0)

    def __getitem__(self, idx):
        return (
            self.species[idx],
            self.coordinates[idx],
            self.energies[idx],
            self.forces[idx],
        )


# -------------------------------------------------------------------
#  Model builder
# -------------------------------------------------------------------

class _TorchANIExportWrapper(nn.Module):
    """
    Thin export wrapper that converts the ANI model's tuple-based forward
    signature ``((species, coords)) → SpeciesEnergies`` to the two-arg
    form ``(species, coords) → (species, energies)`` expected by
    :class:`src.wrappers.wrap_torchani.TorchANI_Wrapper`.

    Also disables ``periodic_table_index`` so the NAMD wrapper (which
    already maps Z → 0-indexed species) can feed indices directly.
    """

    def __init__(self, ani_model: nn.Module):
        super().__init__()
        self.ani = ani_model
        # Disable periodic-table conversion — the NAMD wrapper already
        # maps atomic numbers to 0-indexed species before calling us.
        if hasattr(self.ani, "periodic_table_index"):
            self.ani.periodic_table_index = False

    def forward(
        self,
        species: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        result = self.ani((species, coordinates))
        return result[0], result[1]


def build_ani_model(
    elements: list,
    Rcr: float = 5.2,
    Rca: float = 3.5,
    EtaR: list = None,
    EtaA: list = None,
    Zeta: list = None,
    ShfR: list = None,
    ShfA: list = None,
    ShfZ: list = None,
    hidden_layers: list = None,
    device: str = "cpu",
):
    """
    Build a TorchANI model from scratch using the Assembler API
    (TorchANI ≥ 2.7).

    Args:
        elements:  List of element symbols (e.g. ["H", "C", "N", "O"]).
        Rcr, Rca:  Radial / angular cutoffs.
        hidden_layers: Per-element network architecture [h1, h2, ...].
    """
    import torchani
    from torchani.arch import Assembler

    if hidden_layers is None:
        hidden_layers = [160, 128, 96]

    # AEV hyperparameters (ANI-2x-like defaults)
    if ShfR is None:
        ShfR = [0.9, 1.17, 1.44, 1.71, 1.98, 2.25, 2.52, 2.79,
                3.06, 3.33, 3.6, 3.87, 4.14, 4.41, 4.68, 4.95]

    if ShfA is None:
        ShfA = [0.9, 1.55, 2.2, 2.85]

    if ShfZ is None:
        ShfZ = [0.19634954, 0.58904862, 0.9817477, 1.3744468,
                1.7671459, 2.159845, 2.552544, 2.945243]

    eta_r = 16.0 if EtaR is None else EtaR[0]
    eta_a = 8.0 if EtaA is None else EtaA[0]
    zeta_val = 32.0 if Zeta is None else Zeta[0]

    species_order = [_SYMBOL_TO_Z[s] for s in elements]

    # Build model via the Assembler (torchani ≥ 2.7).
    # periodic_table_index=True ⟹ model converts raw atomic numbers
    # to 0-indexed species during training.  At export we flip this
    # via _TorchANIExportWrapper.
    assembler = Assembler(
        symbols=elements,
        periodic_table_index=True,
    )
    assembler.set_zeros_as_self_energies()

    radial = torchani.aev.ANIRadial(eta=eta_r, shifts=ShfR, cutoff=Rcr)
    angular = torchani.aev.ANIAngular(
        eta=eta_a, zeta=zeta_val, shifts=ShfA, sections=ShfZ, cutoff=Rca,
    )
    assembler.set_aev_computer(angular=angular, radial=radial)
    assembler.set_atomic_networks(
        ctor="ani2x",
        kwargs={"activation": "celu", "bias": True},
    )

    model = assembler.assemble()
    return model.to(device), species_order


# -------------------------------------------------------------------
#  Training loop
# -------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, forces_weight, device):
    model.train()
    total_loss = 0.0
    n = 0

    for species, coords, energies, forces_ref in loader:
        species = species.to(device)
        coords = coords.to(device, dtype=torch.float32).requires_grad_(True)
        energies = energies.to(device, dtype=torch.float32)
        forces_ref = forces_ref.to(device, dtype=torch.float32)

        _, predicted_e = model((species, coords))
        predicted_e = predicted_e.squeeze(-1)  # [B]

        # Energy loss (per-batch MSE)
        e_loss = torch.nn.functional.mse_loss(predicted_e, energies)

        # Forces via autograd
        grad = torch.autograd.grad(
            predicted_e.sum(), coords,
            create_graph=True,
        )[0]
        predicted_f = -grad  # [B, N, 3]

        f_loss = torch.nn.functional.mse_loss(predicted_f, forces_ref)

        loss = e_loss + forces_weight * f_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * species.size(0)
        n += species.size(0)

    return total_loss / max(n, 1)


@torch.no_grad()
def validate(model, loader, forces_weight, device):
    model.eval()
    total_loss = 0.0
    n = 0

    for species, coords, energies, forces_ref in loader:
        species = species.to(device)
        coords = coords.to(device, dtype=torch.float32).requires_grad_(True)
        energies = energies.to(device, dtype=torch.float32)
        forces_ref = forces_ref.to(device, dtype=torch.float32)

        with torch.enable_grad():
            _, predicted_e = model((species, coords))
            predicted_e = predicted_e.squeeze(-1)
            grad = torch.autograd.grad(
                predicted_e.sum(), coords, create_graph=False,
            )[0]
        predicted_f = -grad

        e_loss = torch.nn.functional.mse_loss(predicted_e, energies)
        f_loss = torch.nn.functional.mse_loss(predicted_f, forces_ref)
        loss = e_loss + forces_weight * f_loss

        total_loss += loss.item() * species.size(0)
        n += species.size(0)

    return total_loss / max(n, 1)


# -------------------------------------------------------------------
#  Main
# -------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train a TorchANI model for NAMD",
    )
    parser.add_argument("--data-dir", required=True,
                        help="Directory with torchani/ subfolder from prepare_data")
    parser.add_argument("--output-dir", default="./results/torchani",
                        help="Where to save training outputs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--elements", nargs="+", default=["H", "C", "N", "O"],
                        help="Element symbols in species order")

    # Model
    parser.add_argument("--Rcr", type=float, default=5.2,
                        help="Radial cutoff (Å)")
    parser.add_argument("--Rca", type=float, default=3.5,
                        help="Angular cutoff (Å)")
    parser.add_argument("--hidden-layers", nargs="+", type=int,
                        default=[160, 128, 96],
                        help="Hidden layer sizes for per-element networks")

    # Training
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--forces-weight", type=float, default=100.0)
    parser.add_argument("--patience", type=int, default=50)

    args = parser.parse_args(argv)

    # Check imports
    try:
        import torchani  # noqa: F401
    except ImportError:
        print("ERROR: torchani not installed. pip install 'torchani>=2.2'")
        sys.exit(1)

    try:
        import h5py  # noqa: F401
    except ImportError:
        print("ERROR: h5py not installed. pip install h5py")
        sys.exit(1)

    # ---- Data ----
    data_dir = Path(args.data_dir)
    h5_path = data_dir / "torchani" / "data.h5"
    if not h5_path.exists():
        print(f"ERROR: {h5_path} not found. Run prepare_data.py first.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = ANI_H5_Dataset(str(h5_path), split="train")
    val_ds = ANI_H5_Dataset(str(h5_path), split="val")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    print(f"Training data: {len(train_ds)} frames")
    print(f"Validation data: {len(val_ds)} frames")

    # ---- Model ----
    device = args.device
    model, species_order = build_ani_model(
        elements=args.elements,
        Rcr=args.Rcr,
        Rca=args.Rca,
        hidden_layers=args.hidden_layers,
        device=device,
    )

    print(f"\nModel: TorchANI")
    print(f"  Elements: {args.elements} (Z={species_order})")
    print(f"  Rcr={args.Rcr}, Rca={args.Rca}")
    print(f"  Hidden layers: {args.hidden_layers}")

    # ---- Training ----
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=args.patience // 2, factor=0.5,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    print(f"\nTraining for up to {args.max_epochs} epochs (patience={args.patience})...\n")

    for epoch in range(1, args.max_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, args.forces_weight, device,
        )
        val_loss = validate(model, val_loader, args.forces_weight, device)
        scheduler.step(val_loss)

        lr = optimizer.param_groups[0]["lr"]
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}  train={train_loss:.6f}  "
                  f"val={val_loss:.6f}  lr={lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            # Save checkpoint
            torch.save(model.state_dict(), str(output_dir / "best_model.pt"))
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(best={best_epoch}, val_loss={best_val_loss:.6f})")
            break

    # ---- Load best and export ----
    print(f"\nLoading best model (epoch {best_epoch})...")
    model.load_state_dict(torch.load(str(output_dir / "best_model.pt"), weights_only=True))
    model.eval()

    # Wrap in _TorchANIExportWrapper so the scripted model has the
    # two-arg forward(species, coordinates) signature expected by
    # TorchANI_Wrapper.  Also disables periodic_table_index (the
    # wrapper already maps Z → 0-indexed species).
    export_model = _TorchANIExportWrapper(model.cpu())
    export_model.eval()

    # Script and save
    scripted = torch.jit.script(export_model)
    scripted_path = output_dir / "torchani_scripted.pt"
    scripted.save(str(scripted_path))

    # Also save metadata for the wrapper
    elements_str = ",".join(str(z) for z in species_order)
    meta_path = output_dir / "metadata.txt"
    with open(meta_path, "w") as f:
        f.write(f"elements={elements_str}\n")
        f.write(f"Rcr={args.Rcr}\n")
        f.write(f"Rca={args.Rca}\n")
        f.write(f"best_epoch={best_epoch}\n")
        f.write(f"best_val_loss={best_val_loss}\n")

    print(f"\n{'='*60}")
    print(f"TorchANI scripted model: {scripted_path}")
    print(f"Elements: {args.elements} → Z={species_order}")
    print(f"To wrap for NAMD:")
    print(f"  python -m src.cli --model-type torchani --compiled {scripted_path} --elements {elements_str} --out mlff_model.pt")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

