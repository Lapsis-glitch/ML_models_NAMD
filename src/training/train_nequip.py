"""
Train a **NequIP** model and deploy it for use with the NAMD wrapper.

Generates a Hydra-compatible YAML config from the template, runs
``nequip-train``, then calls ``nequip-package build`` +
``nequip-compile`` to produce a TorchScript ``.pth`` file loadable by
:class:`src.wrappers.wrap_compiled_nequip.NequIP_Allegro_Wrapper`.

Compatible with NequIP ≥ 0.17 (Hydra config system).

Usage::

    python -m src.training.train_nequip \\
        --data-dir ./prepared_data \\
        --output-dir ./results/nequip \\
        --chemical-symbols H C N O \\
        --r-max 5.0 \\
        --max-epochs 200
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch
# e3nn 0.4.4 loads constants.pt via torch.load() at import time without
# weights_only=False.  PyTorch ≥ 2.6 defaults to weights_only=True,
# which rejects builtins in that file.  Patch before any e3nn import.
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    if args and isinstance(args[0], str) and "e3nn" in args[0] and "constants.pt" in args[0]:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

import yaml

from ase.io import read as ase_read, write as ase_write


_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _THIS_DIR / "configs" / "nequip_default.yaml"


def _detect_elements(xyz_path: str) -> list:
    """Detect unique chemical symbols from the XYZ file."""
    frames = ase_read(xyz_path, index=":", format="extxyz")
    symbols = set()
    for f in frames:
        symbols.update(f.get_chemical_symbols())
    return sorted(symbols)


def _deep_set(d: dict, keys: list, value):
    """Set a nested dict value given a list of keys."""
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _save_with_metadata(scripted_inner, path, r_max, z_list):
    """Delegate to the shared metadata wrapper module."""
    from src.training._nequip_metadata_wrapper import save_with_metadata
    save_with_metadata(scripted_inner, path, r_max, z_list)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train a NequIP model for NAMD",
    )
    parser.add_argument("--data-dir", required=True,
                        help="Directory with xyz/ subfolder from prepare_data")
    parser.add_argument("--output-dir", default="./results/nequip",
                        help="Where to save training outputs")
    parser.add_argument("--config", default=None,
                        help="Custom YAML config (overrides defaults)")
    parser.add_argument("--chemical-symbols", nargs="+", default=None,
                        help="Chemical symbols (e.g. H C N O). Auto-detected if omitted.")

    # Quick overrides
    parser.add_argument("--r-max", type=float, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--l-max", type=int, default=None)
    parser.add_argument("--num-features", type=int, default=None)

    args = parser.parse_args(argv)

    # Load config template
    with open(_DEFAULT_CONFIG) as f:
        config = yaml.safe_load(f)

    # Merge user config (shallow — for deep override use --config with full YAML)
    if args.config:
        with open(args.config) as f:
            user_config = yaml.safe_load(f)
        config.update(user_config)

    # CLI overrides
    if args.r_max is not None:
        config["cutoff_radius"] = args.r_max
    if args.max_epochs is not None:
        _deep_set(config, ["trainer", "max_epochs"], args.max_epochs)
    if args.batch_size is not None:
        _deep_set(config, ["dataloader", "batch_size"], args.batch_size)
    if args.lr is not None:
        _deep_set(config, ["training_module", "optimizer", "lr"], args.lr)
    if args.num_layers is not None:
        _deep_set(config, ["training_module", "model", "num_layers"], args.num_layers)
    if args.l_max is not None:
        _deep_set(config, ["training_module", "model", "l_max"], args.l_max)
    if args.num_features is not None:
        _deep_set(config, ["training_module", "model", "num_features"], args.num_features)

    # Data paths
    data_dir = Path(args.data_dir)
    train_xyz = data_dir / "xyz" / "train.xyz"
    val_xyz = data_dir / "xyz" / "val.xyz"

    for f in [train_xyz, val_xyz]:
        if not f.exists():
            print(f"ERROR: data file not found: {f}")
            print("Run prepare_data.py first.")
            sys.exit(1)

    # NequIP 0.17 reads a single XYZ and splits via counts.
    # Concatenate train + val into one file.
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_xyz = output_dir / "combined_train_val.xyz"
    train_frames = ase_read(str(train_xyz), index=":", format="extxyz")
    val_frames = ase_read(str(val_xyz), index=":", format="extxyz")
    ase_write(str(combined_xyz), train_frames + val_frames, format="extxyz")

    # Fill in data placeholders
    _deep_set(config, ["data", "split_dataset", "dataset", "file_path"],
              str(combined_xyz))
    _deep_set(config, ["data", "split_dataset", "train"], len(train_frames))
    _deep_set(config, ["data", "split_dataset", "val"], len(val_frames))

    # Chemical symbols
    if args.chemical_symbols:
        config["model_type_names"] = args.chemical_symbols
    elif config.get("model_type_names") == "PLACEHOLDER" or config.get("model_type_names") is None:
        config["model_type_names"] = _detect_elements(str(train_xyz))
        print(f"Auto-detected elements: {config['model_type_names']}")

    # Write final config as config.yaml in output dir (Hydra convention)
    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Config written to: {config_path}")

    # ---- Train ----
    # nequip-train uses @hydra.main(config_path=os.getcwd(), config_name="config")
    # So we run from the output directory where config.yaml lives.
    train_cmd = ["nequip-train"]
    print(f"\nRunning NequIP training (cwd={output_dir}):")
    print(f"  {' '.join(train_cmd)}")
    print()

    # e3nn 0.4.4 calls torch.load("constants.pt") at import time without
    # weights_only=False.  PyTorch >= 2.6 defaults to weights_only=True.
    # Pass TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 so the subprocess can load
    # e3nn without error.
    _sub_env = {**os.environ, "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"}

    result = subprocess.run(train_cmd, cwd=str(output_dir), env=_sub_env)
    if result.returncode != 0:
        print(f"\nNequIP training failed (exit code {result.returncode})")
        sys.exit(result.returncode)

    # ---- Package + Compile ----
    # Find checkpoint
    # Hydra creates outputs/ subdir; the checkpoint is at
    # <hydra_output_dir>/best.ckpt
    ckpt_candidates = list(output_dir.rglob("best.ckpt"))
    if not ckpt_candidates:
        ckpt_candidates = list(output_dir.rglob("last.ckpt"))
    if not ckpt_candidates:
        ckpt_candidates = list(output_dir.rglob("*.ckpt"))

    if not ckpt_candidates:
        print("ERROR: no checkpoint found after training")
        sys.exit(1)

    ckpt_path = ckpt_candidates[0]
    print(f"\nBest checkpoint: {ckpt_path}")

    # Package
    packaged_path = output_dir / "nequip_packaged.nequip.zip"
    pkg_cmd = ["nequip-package", "build", str(ckpt_path), str(packaged_path)]
    print(f"Packaging: {' '.join(pkg_cmd)}")
    result = subprocess.run(pkg_cmd, env=_sub_env)
    if result.returncode != 0:
        print(f"\nnequip-package failed (exit code {result.returncode})")
        sys.exit(result.returncode)

    # Compile to TorchScript (may fail on PyTorch >= 2.10)
    deployed_path = output_dir / "nequip_deployed.pth"
    compile_cmd = [
        "nequip-compile",
        "--mode", "torchscript",
        "--device", "cpu",
        "--target", "pair_nequip",
        str(packaged_path),
        str(deployed_path),
    ]
    print(f"Compiling: {' '.join(compile_cmd)}")
    result = subprocess.run(compile_cmd, capture_output=True, text=True, env=_sub_env)

    ts_ok = result.returncode == 0 and deployed_path.exists()
    if not ts_ok:
        print("nequip-compile --mode torchscript failed "
              "(expected on PyTorch >= 2.10)")
        # Try manual scripting from eager model
        try:
            import torch
            from nequip.model.saved_models import load_saved_model
            model = load_saved_model(
                str(packaged_path),
                compile_mode="eager",
                model_key="sole_model",
            )
            model = model.to("cpu").eval()
            scripted = torch.jit.script(model)

            # Attach metadata that the NAMD wrapper expects
            from ase.data import atomic_numbers as ase_z
            chem_symbols = config.get("model_type_names", [])
            z_list = [ase_z[s] for s in chem_symbols]
            r_max_val = float(config.get("cutoff_radius", 5.0))

            # Save with extra files encoding metadata
            import json
            extra_files = {
                "metadata.json": json.dumps({
                    "r_max": r_max_val,
                    "atomic_numbers": z_list,
                }),
            }
            scripted.save(str(deployed_path), _extra_files=extra_files)

            # Also save a thin wrapper that exposes r_max / atomic_numbers
            # as actual attributes for the NAMD wrapper to read.
            _save_with_metadata(scripted, deployed_path, r_max_val, z_list)

            ts_ok = True
            print(f"Manual TorchScript export succeeded: {deployed_path}")
        except Exception as exc:
            print(f"Manual TorchScript also failed: {type(exc).__name__}: {exc}")

    print(f"\n{'='*60}")
    if ts_ok:
        print(f"NequIP deployed model (TorchScript): {deployed_path}")
        print(f"To wrap for NAMD:")
        print(f"  python -m src.cli --model-type nequip "
              f"--compiled {deployed_path} --out mlff_model.pt")
    else:
        print(f"NequIP packaged model: {packaged_path}")
        print(f"TorchScript export is not supported on this PyTorch version.")
        print(f"Use aotinductor to compile:")
        print(f"  nequip-compile --mode aotinductor --device cpu "
              f"--target pair_nequip {packaged_path} "
              f"{output_dir / 'nequip_deployed.nequip.pt2'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

