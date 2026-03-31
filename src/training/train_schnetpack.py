"""
Train a **SchNetPack** (≥ 2.0) model and export it for use with the
NAMD wrapper.

Uses the SchNetPack Python API to build a SchNet representation +
Atomwise energy head + force derivative, trains with
``schnetpack.train``, then exports via ``torch.jit.script``.

The scripted ``.pt`` is loadable by
:class:`src.wrappers.wrap_schnetpack.SchNetPack_Wrapper`.

Usage::

    python -m src.training.train_schnetpack \\
        --data-dir ./prepared_data \\
        --output-dir ./results/schnetpack \\
        --r-max 5.0 \\
        --max-epochs 200 \\
        --device cuda
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train a SchNetPack model for NAMD",
    )
    parser.add_argument("--data-dir", required=True,
                        help="Directory with schnetpack/ subfolder from prepare_data")
    parser.add_argument("--output-dir", default="./results/schnetpack",
                        help="Where to save training outputs")
    parser.add_argument("--device", default="cpu")

    # Model
    parser.add_argument("--r-max", type=float, default=5.0,
                        help="Cutoff radius in Å")
    parser.add_argument("--n-atom-basis", type=int, default=128,
                        help="Feature vector size")
    parser.add_argument("--n-interactions", type=int, default=3,
                        help="Number of interaction blocks")
    parser.add_argument("--n-rbf", type=int, default=20,
                        help="Number of radial basis functions")

    # Training
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--energy-weight", type=float, default=1.0)
    parser.add_argument("--forces-weight", type=float, default=100.0)

    args = parser.parse_args(argv)

    # ---- Imports (fail fast if not installed) ----
    try:
        import schnetpack as spk
        import schnetpack.transform as trn
        from schnetpack.data import ASEAtomsData, AtomsDataModule
    except ImportError:
        print("ERROR: schnetpack not installed. pip install 'schnetpack>=2.0'")
        sys.exit(1)

    # ---- Monkey-patch ASE 3.28 metadata property bug ----
    # SchNetPack accesses SQLite3Database.metadata outside an active
    # connection context manager, which triggers
    # ``assert self.connection is not None`` in ASE ≥ 3.28.
    # Patch the getter to open a temporary connection when needed.
    from ase.db.sqlite import SQLite3Database
    _orig_metadata_fget = SQLite3Database.metadata.fget

    def _patched_metadata_get(self):
        if self._metadata is None and self.connection is None:
            with self.managed_connection():
                return _orig_metadata_fget(self)
        return _orig_metadata_fget(self)

    SQLite3Database.metadata = SQLite3Database.metadata.getter(_patched_metadata_get)

    try:
        import pytorch_lightning as pl
    except ImportError:
        try:
            import lightning.pytorch as pl
        except ImportError:
            print("ERROR: pytorch_lightning not installed.")
            sys.exit(1)

    # ---- Data ----
    data_dir = Path(args.data_dir)
    train_db = data_dir / "schnetpack" / "train.db"
    val_db = data_dir / "schnetpack" / "val.db"

    for f in [train_db, val_db]:
        if not f.exists():
            print(f"ERROR: {f} not found. Run prepare_data.py first.")
            sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # SchNetPack data module
    # We create a combined DB for the data module, or use train DB directly
    # and point validation to the val DB.
    # SchNetPack's AtomsDataModule expects a single DB with split files.
    # Alternative: use custom data loaders.

    # Simplest approach: create a merged DB with custom splits
    merged_db = output_dir / "merged.db"
    if merged_db.exists():
        merged_db.unlink()

    # Read source DBs and build a merged DB using raw ase.db API
    # (avoids SchNetPack/ASE 3.28 connection-management bug).
    from ase.db import connect as ase_db_connect
    from ase.io import read as ase_read

    # Read frames from the XYZ splits (more robust than reading .db)
    train_xyz = data_dir / "xyz" / "train.xyz"
    val_xyz = data_dir / "xyz" / "val.xyz"
    if train_xyz.exists() and val_xyz.exists():
        train_frames = ase_read(str(train_xyz), index=":", format="extxyz")
        val_frames = ase_read(str(val_xyz), index=":", format="extxyz")
    else:
        # Fallback: read from .db files directly
        train_frames = []
        with ase_db_connect(str(train_db)) as db:
            for row in db.select():
                train_frames.append(row.toatoms())
        val_frames = []
        with ase_db_connect(str(val_db)) as db:
            for row in db.select():
                val_frames.append(row.toatoms())

    n_train = len(train_frames)
    n_val = len(val_frames)

    # Helper to extract energy/forces
    from src.training.prepare_data import get_energy, get_forces

    with ase_db_connect(str(merged_db)) as db:
        db.metadata = {
            "_property_unit_dict": {"energy": "eV", "forces": "eV/Ang"},
            "_distance_unit": "Ang",
            "atomrefs": {},
        }
        for atoms in train_frames + val_frames:
            data = {}
            e = get_energy(atoms)
            if e is not None:
                data["energy"] = np.array([e], dtype=np.float64)
            f = get_forces(atoms)
            if f is not None:
                data["forces"] = np.array(f, dtype=np.float64)
            db.write(atoms, data=data)

    # Write split file
    train_idx = list(range(n_train))
    val_idx = list(range(n_train, n_train + n_val))

    split_path = output_dir / "split.npz"
    np.savez(
        str(split_path),
        train_idx=np.array(train_idx),
        val_idx=np.array(val_idx),
        test_idx=np.array([]),  # no test during training
    )

    # Build data module
    data_module = AtomsDataModule(
        datapath=str(merged_db),
        batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        split_file=str(split_path),
        transforms=[
            trn.ASENeighborList(cutoff=args.r_max),
            trn.CastTo32(),
        ],
        num_workers=0,
        pin_memory=False,
    )
    data_module.prepare_data()
    data_module.setup()

    print(f"Training data: {n_train} frames")
    print(f"Validation data: {n_val} frames")

    # ---- Model ----
    cutoff_fn = spk.nn.cutoff.CosineCutoff(args.r_max)
    radial_basis = spk.nn.radial.GaussianRBF(
        n_rbf=args.n_rbf, cutoff=args.r_max,
    )

    schnet = spk.representation.SchNet(
        n_atom_basis=args.n_atom_basis,
        n_interactions=args.n_interactions,
        radial_basis=radial_basis,
        cutoff_fn=cutoff_fn,
    )

    energy_head = spk.atomistic.Atomwise(
        n_in=args.n_atom_basis,
        output_key="energy",
    )
    forces_head = spk.atomistic.Forces(
        energy_key="energy",
        force_key="forces",
    )

    model = spk.model.NeuralNetworkPotential(
        representation=schnet,
        input_modules=[
            spk.atomistic.PairwiseDistances(),
        ],
        output_modules=[energy_head, forces_head],
        postprocessors=[
            # NOTE: CastTo64() is omitted because it uses a Python-level
            # dict that TorchScript cannot compile.  The NAMD wrapper
            # handles dtype conversion (→ float64) on its side.
        ],
    )

    # ---- Loss and training ----
    energy_loss = spk.task.ModelOutput(
        name="energy",
        loss_fn=torch.nn.MSELoss(),
        loss_weight=args.energy_weight,
        metrics={},
    )
    forces_loss = spk.task.ModelOutput(
        name="forces",
        loss_fn=torch.nn.MSELoss(),
        loss_weight=args.forces_weight,
        metrics={},
    )

    task = spk.task.AtomisticTask(
        model=model,
        outputs=[energy_loss, forces_loss],
        optimizer_cls=torch.optim.Adam,
        optimizer_args={"lr": args.lr},
        scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_args={"patience": 25, "factor": 0.5},
        scheduler_monitor="val_loss",
    )

    # ---- Trainer ----
    # Save the model state dict before training for later checkpoint-free reload
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=str(output_dir / "checkpoints"),
            filename="best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        ),
        pl.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=50,
            mode="min",
        ),
    ]

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        default_root_dir=str(output_dir),
        accelerator="gpu" if args.device.startswith("cuda") else "cpu",
        devices=1,
        enable_progress_bar=True,
    )

    print(f"\nTraining SchNetPack model (r_max={args.r_max}, "
          f"n_atom_basis={args.n_atom_basis}, "
          f"n_interactions={args.n_interactions})...")
    print()

    trainer.fit(task, datamodule=data_module)

    # ---- Export ----
    # Load best checkpoint
    best_path = output_dir / "checkpoints" / "best.ckpt"
    if not best_path.exists():
        # Find any checkpoint
        ckpt_dir = output_dir / "checkpoints"
        ckpts = list(ckpt_dir.glob("*.ckpt"))
        if ckpts:
            best_path = ckpts[0]
        else:
            print("ERROR: no checkpoint found after training")
            sys.exit(1)

    print(f"\nLoading best checkpoint: {best_path}")
    # Use weights_only=False because Lightning checkpoints contain
    # SchNetPack class references that require pickling.
    ckpt = torch.load(str(best_path), map_location="cpu", weights_only=False)
    task.load_state_dict(ckpt["state_dict"])
    best_model = task.model
    best_model.eval()

    # Script and save
    # SchNetPack models may contain modules that TorchScript cannot
    # compile via script mode (e.g. Python-level dicts).  Try script
    # first, fall back to trace.
    scripted_path = output_dir / "schnet_scripted.pt"
    try:
        scripted = torch.jit.script(best_model)
    except Exception:
        # Build a representative sample input for tracing.
        sample_inputs: dict = {
            "_positions": torch.randn(3, 3, dtype=torch.float32),
            "_atomic_numbers": torch.tensor([8, 1, 1], dtype=torch.long),
            "_idx_i": torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long),
            "_idx_j": torch.tensor([1, 2, 0, 2, 0, 1], dtype=torch.long),
            "_offsets": torch.zeros(6, 3, dtype=torch.float32),
            "_cell": torch.zeros(3, 3, dtype=torch.float32),
            "_n_atoms": torch.tensor([3], dtype=torch.long),
            "_idx_m": torch.zeros(3, dtype=torch.long),
        }
        with torch.no_grad():
            scripted = torch.jit.trace(best_model, (sample_inputs,))
    scripted.save(str(scripted_path))

    print(f"\n{'='*60}")
    print(f"SchNetPack scripted model: {scripted_path}")
    print(f"Cutoff used: {args.r_max} Å")
    print(f"To wrap for NAMD:")
    print(f"  python -m src.cli --model-type schnet --compiled {scripted_path} --r-max {args.r_max} --out mlff_model.pt")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

