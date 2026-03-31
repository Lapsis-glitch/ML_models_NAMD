"""
Train a **MACE** model and export it for use with the NAMD wrapper.

Uses ``mace_run_train`` CLI under the hood.  Produces a compiled
TorchScript ``.pt`` file with ``r_max`` and ``atomic_numbers``
attributes — directly loadable by
:class:`src.wrappers.wrap_compiled_mace.MACE_TS_Wrapper`.

Usage::

    python -m src.training.train_mace \\
        --data-dir ./prepared_data \\
        --output-dir ./results/mace \\
        --r-max 5.0 \\
        --max-epochs 200 \\
        --device cuda
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _THIS_DIR / "configs" / "mace_default.yaml"


def build_mace_command(
    config: dict,
    output_dir: Path,
    device: str = "cpu",
) -> list:
    """
    Translate a config dict into ``mace_run_train`` CLI arguments.
    """
    cmd = [sys.executable, "-m", "mace.cli.run_train"]

    # Data
    cmd += ["--train_file", str(config["train_file"])]
    cmd += ["--valid_file", str(config["valid_file"])]
    cmd += ["--test_file", str(config["test_file"])]
    cmd += ["--energy_key", config.get("energy_key", "energy")]
    cmd += ["--forces_key", config.get("forces_key", "forces")]

    # Atomic energies: required in MACE >= 0.3.x.
    # "average" computes per-element E0 from the training set.
    cmd += ["--E0s", config.get("E0s", "average")]

    # Model
    cmd += ["--model", config.get("model", "ScaleShiftMACE")]
    cmd += ["--r_max", str(config["r_max"])]
    cmd += ["--num_radial_basis", str(config.get("num_radial_basis", 8))]
    cmd += ["--num_cutoff_basis", str(config.get("num_cutoff_basis", 5))]
    cmd += ["--max_L", str(config.get("max_L", 1))]
    cmd += ["--correlation", str(config.get("correlation", 3))]
    cmd += ["--hidden_irreps", config.get("hidden_irreps", "128x0e + 128x1o")]
    cmd += ["--num_interactions", str(config.get("num_interactions", 2))]
    cmd += ["--MLP_irreps", config.get("MLP_irreps", "16x0e")]

    # Training
    cmd += ["--batch_size", str(config.get("batch_size", 10))]
    cmd += ["--valid_batch_size", str(config.get("valid_batch_size", 10))]
    cmd += ["--max_num_epochs", str(config.get("max_num_epochs", 200))]
    cmd += ["--lr", str(config.get("lr", 0.01))]
    cmd += ["--patience", str(config.get("patience", 50))]
    cmd += ["--energy_weight", str(config.get("energy_weight", 1.0))]
    cmd += ["--forces_weight", str(config.get("forces_weight", 100.0))]
    cmd += ["--scheduler", config.get("scheduler", "ReduceLROnPlateau")]

    if config.get("ema", True):
        cmd += ["--ema"]
        cmd += ["--ema_decay", str(config.get("ema_decay", 0.99))]

    # Output
    cmd += ["--checkpoints_dir", str(output_dir / "checkpoints")]
    cmd += ["--results_dir", str(output_dir / "results")]
    cmd += ["--model_dir", str(output_dir)]
    cmd += ["--name", "mace_model"]
    # NOTE: --save_model_as was removed in MACE >= 0.3.x.
    # The compiled model is now saved automatically as <name>_compiled.model.

    # Device
    cmd += ["--device", device]
    cmd += ["--default_dtype", "float64"]

    return cmd


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train a MACE model for NAMD",
    )
    parser.add_argument("--data-dir", required=True,
                        help="Directory with xyz/ subfolder from prepare_data")
    parser.add_argument("--output-dir", default="./results/mace",
                        help="Where to save training outputs")
    parser.add_argument("--config", default=None,
                        help="Custom YAML config (overrides defaults)")
    parser.add_argument("--device", default="cpu")

    # Quick overrides for the most common params
    parser.add_argument("--r-max", type=float, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--energy-weight", type=float, default=None)
    parser.add_argument("--forces-weight", type=float, default=None)

    args = parser.parse_args(argv)

    # Load config
    with open(_DEFAULT_CONFIG) as f:
        config = yaml.safe_load(f)

    # Merge user overrides
    if args.config:
        with open(args.config) as f:
            user_config = yaml.safe_load(f)
        config.update(user_config)

    # CLI overrides
    if args.r_max is not None:
        config["r_max"] = args.r_max
    if args.max_epochs is not None:
        config["max_num_epochs"] = args.max_epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.lr is not None:
        config["lr"] = args.lr
    if args.energy_weight is not None:
        config["energy_weight"] = args.energy_weight
    if args.forces_weight is not None:
        config["forces_weight"] = args.forces_weight

    # Fill in data paths
    data_dir = Path(args.data_dir)
    config["train_file"] = str(data_dir / "xyz" / "train.xyz")
    config["valid_file"] = str(data_dir / "xyz" / "val.xyz")
    config["test_file"] = str(data_dir / "xyz" / "test.xyz")

    for f in [config["train_file"], config["valid_file"], config["test_file"]]:
        if not os.path.exists(f):
            print(f"ERROR: data file not found: {f}")
            print("Run prepare_data.py first.")
            sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build and run
    cmd = build_mace_command(config, output_dir, device=args.device)
    print("Running MACE training:")
    print("  " + " ".join(cmd))
    print()

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nMACE training failed (exit code {result.returncode})")
        sys.exit(result.returncode)

    # Find the compiled model
    compiled = output_dir / "mace_model_compiled.model"
    if not compiled.exists():
        # Try alternate naming
        for p in output_dir.glob("*.model"):
            if "compiled" in p.name:
                compiled = p
                break

    if compiled.exists():
        final_path = output_dir / "mace_compiled.pt"
        shutil.copy2(str(compiled), str(final_path))
        print(f"\n{'='*60}")
        print(f"MACE compiled model: {final_path}")
        print(f"To wrap for NAMD:")
        print(f"  python -m src.cli --model-type mace --compiled {final_path} --out mlff_model.pt")
        print(f"{'='*60}")
    else:
        print("\nWARNING: compiled model not found. Check training output.")
        print("You may need to manually compile:")
        print("  from mace.tools.scripts_utils import compile_model")


if __name__ == "__main__":
    main()

