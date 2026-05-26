#!/usr/bin/env python3
"""Offline StableHLO export probe for FeNNol FENNIX-BIO1.

This script is intentionally an offline tool.  It uses Python/JAX/FeNNol to
load a `.fnx` model and lower a fixed-shape energy+forces evaluator to
StableHLO MLIR.  The intended runtime experiment is a no-Python C++ PJRT/XLA
shim that consumes the exported artifact.

The export currently specializes to a fixed atom list and fixed atom count.
That matches a first NAMD-source prototype where the ML region composition is
fixed for the run.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np

EV_TO_KCAL = 23.0621


def parse_z_list(text: str) -> np.ndarray:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("--z-list must contain at least one atomic number")
    return np.asarray(values, dtype=np.int32)


def default_coords(n_atoms: int) -> np.ndarray:
    """Return deterministic non-overlapping test coordinates in Angstrom."""
    if n_atoms == 3:
        return np.asarray(
            [
                [0.000000, 0.000000, 0.000000],
                [0.957200, 0.000000, 0.000000],
                [-0.239987, 0.927297, 0.000000],
            ],
            dtype=np.float32,
        )
    coords = np.zeros((n_atoms, 3), dtype=np.float32)
    coords[:, 0] = np.arange(n_atoms, dtype=np.float32) * 1.25
    return coords


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Export a fixed-shape FENNIX-BIO1 energy+forces StableHLO probe",
    )
    parser.add_argument(
        "--model",
        default="models/fennix-bio1S.fnx",
        help="Path to FENNIX-BIO1 .fnx model",
    )
    parser.add_argument(
        "--out-dir",
        default="models/fennix_bio1_stablehlo_n3",
        help="Directory for .stablehlo.mlir, manifest, and reference outputs",
    )
    parser.add_argument(
        "--z-list",
        default="8,1,1",
        help="Comma-separated fixed atom numbers; default is water O,H,H",
    )
    parser.add_argument(
        "--total-charge",
        type=int,
        default=0,
        help="Fixed total charge for the specialized export",
    )
    args = parser.parse_args(argv)

    try:
        jax = importlib.import_module("jax")
        jnp = importlib.import_module("jax.numpy")
        FENNIX = importlib.import_module("fennol").FENNIX
        compiler = importlib.import_module("jax._src.compiler")
    except ImportError as exc:
        raise SystemExit(
            "This offline exporter must be run in an environment with "
            "FeNNol and JAX installed, e.g. `conda run -n fennix python ...`."
        ) from exc

    model_path = Path(args.model).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    Z = parse_z_list(args.z_list)
    n_atoms = int(Z.shape[0])
    coords = default_coords(n_atoms)

    print(f"Loading FeNNol model: {model_path}")
    model = FENNIX.load(str(model_path))
    print(
        "Loaded:",
        f"cutoff={model.cutoff}",
        f"energy_unit={model.energy_unit}",
        f"energy_terms={model.energy_terms}",
    )

    raw_np = {
        "species": Z,
        "coordinates": coords,
        "natoms": np.array([n_atoms], dtype=np.int32),
        "batch_index": np.zeros(n_atoms, dtype=np.int32),
        "total_charge": np.array([args.total_charge], dtype=np.int32),
    }

    # Warm/init preprocessing once.  The JAX process path then uses the fixed
    # neighbor-list capacities in this state, avoiding Python-side checks.
    _ = model.preprocess(**raw_np)
    state = model.preproc_state

    species = jnp.asarray(Z)
    natoms = jnp.array([n_atoms], dtype=jnp.int32)
    batch_index = jnp.zeros(n_atoms, dtype=jnp.int32)
    total_charge = jnp.array([args.total_charge], dtype=jnp.int32)

    def eval_ev(coordinates):
        raw = {
            "species": species,
            "coordinates": coordinates,
            "natoms": natoms,
            "batch_index": batch_index,
            "total_charge": total_charge,
        }
        pre = model.preprocessing.process(state, raw)
        energy_ev, forces_ev_a, _ = model._energy_and_forces(model.variables, pre)
        return energy_ev, forces_ev_a

    jit_eval = jax.jit(eval_ev)

    print("Evaluating Python/JAX reference and lowered executable...")
    energy_ref, forces_ref, _ = model.energy_and_forces(**raw_np)
    energy_jit, forces_jit = jit_eval(jnp.asarray(coords))

    energy_ref_np = np.asarray(energy_ref)
    forces_ref_np = np.asarray(forces_ref)
    energy_jit_np = np.asarray(energy_jit)
    forces_jit_np = np.asarray(forces_jit)

    print("reference energy eV:", energy_ref_np)
    print("jit energy eV:", energy_jit_np)
    print("energy abs diff eV:", np.max(np.abs(energy_jit_np - energy_ref_np)))
    print("force max abs diff eV/A:", np.max(np.abs(forces_jit_np - forces_ref_np)))

    print("Lowering to StableHLO...")
    lowered = jit_eval.lower(jnp.asarray(coords))
    stablehlo_text = str(lowered.compiler_ir(dialect="stablehlo"))
    stablehlo_path = out_dir / "fennix_bio1_eval.stablehlo.mlir"
    stablehlo_path.write_text(stablehlo_text)

    # Save HLO text if available for debugging; in jaxlib 0.7 this is an
    # XlaComputation object and may only expose textual HLO via as_hlo_text().
    hlo_path = out_dir / "fennix_bio1_eval.hlo.txt"
    try:
        hlo = lowered.compiler_ir(dialect="hlo")
        if hasattr(hlo, "as_hlo_text"):
            hlo_path.write_text(hlo.as_hlo_text())
        else:
            hlo_path.write_text(str(hlo))
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        hlo_path.write_text(f"HLO export failed: {exc!r}\n")

    compile_options_path = out_dir / "compile_options.pb"
    compile_options = compiler.get_compile_options(1, 1)
    compile_options_path.write_bytes(compile_options.SerializeAsString())

    np.savez(
        out_dir / "reference.npz",
        Z=Z,
        coordinates=coords,
        energy_ref_ev=energy_ref_np,
        forces_ref_ev_a=forces_ref_np,
        energy_jit_ev=energy_jit_np,
        forces_jit_ev_a=forces_jit_np,
        energy_jit_kcal=energy_jit_np.astype(np.float64) * EV_TO_KCAL,
        forces_jit_kcal_a=forces_jit_np.astype(np.float64) * EV_TO_KCAL,
    )

    runtime_ref_path = out_dir / "reference_runtime.txt"
    runtime_ref_path.write_text(
        "\n".join(
            [
                f"input_dims={','.join(str(int(x)) for x in coords.shape)}",
                f"coordinates={','.join(f'{float(x):.9g}' for x in coords.reshape(-1))}",
                f"energy_ref_ev={','.join(f'{float(x):.9g}' for x in energy_ref_np.reshape(-1))}",
                f"forces_ref_ev_a={','.join(f'{float(x):.9g}' for x in forces_ref_np.reshape(-1))}",
                f"energy_jit_ev={','.join(f'{float(x):.9g}' for x in energy_jit_np.reshape(-1))}",
                f"forces_jit_ev_a={','.join(f'{float(x):.9g}' for x in forces_jit_np.reshape(-1))}",
                f"tol_energy_ev={float(np.max(np.abs(energy_jit_np - energy_ref_np))):.9g}",
                f"tol_forces_ev_a={float(np.max(np.abs(forces_jit_np - forces_ref_np))):.9g}",
            ]
        )
        + "\n"
    )

    manifest = {
        "model_path": str(model_path),
        "model_type": "FENNIX-BIO1",
        "n_atoms": n_atoms,
        "z_list": Z.tolist(),
        "total_charge": args.total_charge,
        "input_signature": [
            {"name": "coordinates", "shape": [n_atoms, 3], "dtype": "float32", "units": "Angstrom"}
        ],
        "output_signature": [
            {"name": "energy", "shape": [1], "dtype": "float32", "units": "eV"},
            {"name": "forces", "shape": [n_atoms, 3], "dtype": "float32", "units": "eV/Angstrom"},
        ],
        "conversion": {"ev_to_kcal": EV_TO_KCAL},
        "fennol": {
            "cutoff": float(model.cutoff),
            "energy_unit": str(model.energy_unit),
            "energy_terms": list(model.energy_terms),
        },
        "jax": {
            "jax_version": jax.__version__,
            "devices": [str(d) for d in jax.devices()],
        },
        "artifacts": {
            "stablehlo_mlir": stablehlo_path.name,
            "hlo_text": hlo_path.name,
            "compile_options_pb": compile_options_path.name,
            "reference_npz": "reference.npz",
            "reference_runtime": runtime_ref_path.name,
        },
        "validation": {
            "energy_max_abs_diff_ev": float(np.max(np.abs(energy_jit_np - energy_ref_np))),
            "forces_max_abs_diff_ev_a": float(np.max(np.abs(forces_jit_np - forces_ref_np))),
        },
        "runtime_note": "This artifact is specialized to fixed atom identity/count. Runtime Python is not required by the intended PJRT C++ consumer.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Wrote StableHLO: {stablehlo_path}")
    print(f"Wrote manifest:  {out_dir / 'manifest.json'}")
    print(f"Wrote compile options: {compile_options_path}")
    print(f"Wrote runtime reference: {runtime_ref_path}")
    print(f"Wrote reference: {out_dir / 'reference.npz'}")


if __name__ == "__main__":
    main()
