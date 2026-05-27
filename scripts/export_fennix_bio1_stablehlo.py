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
import sys

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.constants import SYMBOL_TO_Z

EV_TO_KCAL = 23.0621


def parse_z_list(text: str) -> np.ndarray:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("--z-list must contain at least one atomic number")
    return np.asarray(values, dtype=np.int32)


def _normalize_element_symbol(text: str) -> str | None:
    text = text.strip()
    if not text:
        return None
    symbol = text[0].upper() + text[1:].lower()
    return symbol if symbol in SYMBOL_TO_Z else None


def _infer_element_from_pdb_atom_name(atom_name_field: str) -> str | None:
    atom_name = atom_name_field.strip()
    letters = "".join(ch for ch in atom_name if ch.isalpha())
    if not letters:
        return None

    # PDB atom-name alignment convention:
    # - one-letter elements are usually right-justified (leading space)
    # - two-letter elements start in column 13 (no leading space)
    if atom_name_field[:1].isspace():
        return _normalize_element_symbol(letters[:1])

    if len(letters) >= 2:
        maybe_two_letter = _normalize_element_symbol(letters[:2])
        if maybe_two_letter is not None:
            return maybe_two_letter

    return _normalize_element_symbol(letters[:1])


def _parse_pdb_charge(charge_field: str) -> int:
    charge_text = charge_field.strip()
    if not charge_text:
        return 0
    if len(charge_text) != 2 or charge_text[0] not in "123456789" or charge_text[1] not in "+-":
        raise ValueError(f"Unsupported PDB charge field {charge_text!r}; expected forms like '1+' or '2-'.")
    magnitude = int(charge_text[0])
    return magnitude if charge_text[1] == "+" else -magnitude


def load_pdb_system(pdb_path: Path) -> tuple[np.ndarray, np.ndarray, list[str], int, int]:
    """Load atomic numbers and coordinates from the first model in a PDB file."""
    pdb_path = Path(pdb_path).expanduser().resolve()
    lines = pdb_path.read_text().splitlines()

    z_list: list[int] = []
    coords: list[list[float]] = []
    symbols: list[str] = []
    inferred_total_charge = 0
    explicit_charge_count = 0

    saw_model_record = False
    inside_first_model = False

    for line_number, line in enumerate(lines, start=1):
        record = line[:6].strip()

        if record == "MODEL":
            if saw_model_record:
                break
            saw_model_record = True
            inside_first_model = True
            continue

        if record == "ENDMDL" and inside_first_model:
            break

        if record not in {"ATOM", "HETATM"}:
            continue

        if saw_model_record and not inside_first_model:
            continue

        alt_loc = line[16:17] if len(line) >= 17 else " "
        if alt_loc not in {" ", "", "A"}:
            continue

        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError as exc:
            raise ValueError(
                f"Invalid coordinate fields in {pdb_path} line {line_number}: {line!r}"
            ) from exc

        symbol = _normalize_element_symbol(line[76:78] if len(line) >= 78 else "")
        if symbol is None:
            symbol = _infer_element_from_pdb_atom_name(line[12:16] if len(line) >= 16 else "")
        if symbol is None:
            raise ValueError(
                f"Could not determine element for atom on line {line_number} of {pdb_path}. "
                "Populate PDB element columns 77-78 or use standard PDB atom naming."
            )

        charge_value = _parse_pdb_charge(line[78:80] if len(line) >= 80 else "")
        if (line[78:80] if len(line) >= 80 else "").strip():
            explicit_charge_count += 1
        inferred_total_charge += charge_value

        z_list.append(SYMBOL_TO_Z[symbol])
        coords.append([x, y, z])
        symbols.append(symbol)

    if not z_list:
        raise ValueError(f"No ATOM/HETATM records found in {pdb_path}")

    coords_arr = np.asarray(coords, dtype=np.float32)
    if not np.isfinite(coords_arr).all():
        raise ValueError(f"Non-finite coordinates found in {pdb_path}")

    return np.asarray(z_list, dtype=np.int32), coords_arr, symbols, inferred_total_charge, explicit_charge_count


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


def default_walker_coords(base_coords: np.ndarray, n_walkers: int) -> np.ndarray:
    """Return deterministic per-walker coordinates for a fixed system."""
    if n_walkers < 1:
        raise ValueError("--n-walkers must be at least 1")
    base = np.asarray(base_coords, dtype=np.float32)
    if base.ndim != 2 or base.shape[1] != 3:
        raise ValueError(f"base coordinates must have shape [N,3], got {base.shape}")
    if n_walkers == 1:
        return base.copy()

    walker_coords = np.repeat(base[None, :, :], n_walkers, axis=0)
    for walker_idx in range(n_walkers):
        walker_coords[walker_idx, :, 0] += np.float32(0.05 * walker_idx)
    return walker_coords


def load_walker_coords_npy(path: Path, n_atoms: int) -> np.ndarray:
    coords = np.load(Path(path).expanduser().resolve())
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 3 or coords.shape[2] != 3:
        raise ValueError(
            f"Walker coordinates must have shape [n_walkers, n_atoms, 3], got {coords.shape}"
        )
    if coords.shape[1] != n_atoms:
        raise ValueError(
            f"Walker coordinates atom count {coords.shape[1]} does not match system atom count {n_atoms}"
        )
    if not np.isfinite(coords).all():
        raise ValueError("Walker coordinates contain non-finite values")
    return coords


def default_out_dir(n_atoms: int, pdb_path: Path | None = None, n_walkers: int = 1) -> Path:
    models_dir = _REPO_ROOT / "models"
    if pdb_path is None:
        suffix = f"fennix_bio1_stablehlo_n{n_atoms}"
    else:
        stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in pdb_path.stem)
        suffix = f"fennix_bio1_stablehlo_{stem}_n{n_atoms}"
    if n_walkers > 1:
        suffix += f"_w{n_walkers}"
    return models_dir / suffix


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
        default=None,
        help="Directory for .stablehlo.mlir, manifest, and reference outputs. Defaults to models/fennix_bio1_stablehlo_<system>_n<N>.",
    )
    parser.add_argument(
        "--z-list",
        default=None,
        help="Comma-separated fixed atom numbers; ignored when --pdb is used. Defaults to water O,H,H when neither --pdb nor --z-list is provided.",
    )
    parser.add_argument(
        "--pdb",
        default=None,
        help="PDB file whose full atom list and coordinates should define the fixed-shape export.",
    )
    parser.add_argument(
        "--total-charge",
        type=int,
        default=None,
        help="Fixed total charge for the specialized export. If omitted with --pdb, the exporter sums any formal PDB charge fields and otherwise falls back to 0.",
    )
    parser.add_argument(
        "--n-walkers",
        type=int,
        default=None,
        help="Experimental fixed walker count for same-system batching. Defaults to 1 unless inferred from --walker-coords-npy.",
    )
    parser.add_argument(
        "--walker-coords-npy",
        default=None,
        help="Optional .npy array of reference coordinates with shape [n_walkers, n_atoms, 3]. Overrides deterministic default walker coordinates.",
    )
    args = parser.parse_args(argv)

    if args.pdb is not None and args.z_list is not None:
        raise SystemExit("Use either --pdb or --z-list, not both.")

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

    if args.pdb is not None:
        pdb_path = Path(args.pdb).expanduser().resolve()
        Z, coords, symbols, inferred_pdb_charge, explicit_charge_count = load_pdb_system(pdb_path)
        input_source = {
            "type": "pdb",
            "path": str(pdb_path),
            "symbols": symbols,
            "explicit_formal_charge_records": explicit_charge_count,
        }
    else:
        Z = parse_z_list(args.z_list or "8,1,1")
        coords = default_coords(int(Z.shape[0]))
        inferred_pdb_charge = 0
        explicit_charge_count = 0
        input_source = {
            "type": "z_list",
            "path": None,
            "symbols": None,
            "explicit_formal_charge_records": 0,
        }

    n_atoms = int(Z.shape[0])
    total_charge = int(args.total_charge) if args.total_charge is not None else int(inferred_pdb_charge)
    if args.total_charge is not None:
        total_charge_source = "cli"
    elif args.pdb is not None and explicit_charge_count > 0:
        total_charge_source = "pdb_formal_charge_sum"
    else:
        total_charge_source = "default_zero"

    walker_coords_path = Path(args.walker_coords_npy).expanduser().resolve() if args.walker_coords_npy else None
    if walker_coords_path is not None:
        loaded_walker_coords = load_walker_coords_npy(walker_coords_path, n_atoms=n_atoms)
        inferred_n_walkers = int(loaded_walker_coords.shape[0])
        if args.n_walkers is not None and args.n_walkers != inferred_n_walkers:
            raise SystemExit(
                f"--n-walkers={args.n_walkers} does not match --walker-coords-npy batch size {inferred_n_walkers}"
            )
        n_walkers = inferred_n_walkers
        export_coords = loaded_walker_coords[0] if n_walkers == 1 else loaded_walker_coords
    else:
        n_walkers = int(args.n_walkers or 1)
        export_coords = default_walker_coords(coords, n_walkers=n_walkers)

    if n_walkers < 1:
        raise SystemExit("--n-walkers must be at least 1")

    resolved_out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir is not None
        else default_out_dir(
            n_atoms=n_atoms,
            pdb_path=Path(input_source["path"]) if input_source["path"] else None,
            n_walkers=n_walkers,
        )
    )
    out_dir = resolved_out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Specializing export to {n_atoms} atoms "
        f"from {'PDB ' + str(input_source['path']) if args.pdb is not None else 'explicit --z-list'}"
    )
    print(f"Writing artifacts to {out_dir}")
    print(f"Walker count: {n_walkers}")
    print(f"Using total charge {total_charge} ({total_charge_source})")

    print(f"Loading FeNNol model: {model_path}")
    model = FENNIX.load(str(model_path))
    print(
        "Loaded:",
        f"cutoff={model.cutoff}",
        f"energy_unit={model.energy_unit}",
        f"energy_terms={model.energy_terms}",
    )

    single_raw_np = {
        "species": Z,
        "coordinates": coords,
        "natoms": np.array([n_atoms], dtype=np.int32),
        "batch_index": np.zeros(n_atoms, dtype=np.int32),
        "total_charge": np.array([total_charge], dtype=np.int32),
    }

    # Warm/init preprocessing once.  The JAX process path then uses the fixed
    # neighbor-list capacities in this state, avoiding Python-side checks.
    _ = model.preprocess(**single_raw_np)
    state = model.preproc_state

    species = jnp.asarray(Z)
    natoms = jnp.array([n_atoms], dtype=jnp.int32)
    batch_index = jnp.zeros(n_atoms, dtype=jnp.int32)
    total_charge_jnp = jnp.array([total_charge], dtype=jnp.int32)

    def eval_ev(coordinates):
        raw = {
            "species": species,
            "coordinates": coordinates,
            "natoms": natoms,
            "batch_index": batch_index,
            "total_charge": total_charge_jnp,
        }
        pre = model.preprocessing.process(state, raw)
        energy_ev, forces_ev_a, _ = model._energy_and_forces(model.variables, pre)
        return energy_ev, forces_ev_a

    if n_walkers == 1:
        jit_eval = jax.jit(eval_ev)
        energy_ref, forces_ref, _ = model.energy_and_forces(**single_raw_np)
        energy_jit, forces_jit = jit_eval(jnp.asarray(export_coords))
    else:
        jit_eval = jax.jit(jax.vmap(eval_ev, in_axes=0, out_axes=0))
        ref_energies = []
        ref_forces = []
        for walker_coords in export_coords:
            energy_ref_i, forces_ref_i, _ = model.energy_and_forces(
                species=single_raw_np["species"],
                coordinates=walker_coords,
                natoms=single_raw_np["natoms"],
                batch_index=single_raw_np["batch_index"],
                total_charge=single_raw_np["total_charge"],
            )
            ref_energies.append(np.asarray(energy_ref_i))
            ref_forces.append(np.asarray(forces_ref_i))
        energy_ref = np.stack(ref_energies, axis=0)
        forces_ref = np.stack(ref_forces, axis=0)
        energy_jit, forces_jit = jit_eval(jnp.asarray(export_coords))

    print("Evaluating Python/JAX reference and lowered executable...")

    energy_ref_np = np.asarray(energy_ref)
    forces_ref_np = np.asarray(forces_ref)
    energy_jit_np = np.asarray(energy_jit)
    forces_jit_np = np.asarray(forces_jit)

    print("reference energy eV:", energy_ref_np)
    print("jit energy eV:", energy_jit_np)
    print("energy abs diff eV:", np.max(np.abs(energy_jit_np - energy_ref_np)))
    print("force max abs diff eV/A:", np.max(np.abs(forces_jit_np - forces_ref_np)))

    print("Lowering to StableHLO...")
    lowered = jit_eval.lower(jnp.asarray(export_coords))
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
        coordinates=export_coords,
        symbols=np.asarray(input_source["symbols"] or [], dtype="U4"),
        n_walkers=np.asarray([n_walkers], dtype=np.int32),
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
                f"input_dims={','.join(str(int(x)) for x in export_coords.shape)}",
                f"coordinates={','.join(f'{float(x):.9g}' for x in export_coords.reshape(-1))}",
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
        "n_walkers": n_walkers,
        "z_list": Z.tolist(),
        "total_charge": total_charge,
        "total_charge_source": total_charge_source,
        "source_system": {
            **input_source,
            "walker_coords_npy": str(walker_coords_path) if walker_coords_path is not None else None,
        },
        "input_signature": [
            {"name": "coordinates", "shape": list(export_coords.shape), "dtype": "float32", "units": "Angstrom"}
        ],
        "output_signature": [
            {"name": "energy", "shape": list(energy_jit_np.shape), "dtype": "float32", "units": "eV"},
            {"name": "forces", "shape": list(forces_jit_np.shape), "dtype": "float32", "units": "eV/Angstrom"},
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
        "runtime_note": "This artifact is specialized to fixed atom identity/count and fixed walker count. Runtime Python is not required by the intended PJRT C++ consumer.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Wrote StableHLO: {stablehlo_path}")
    print(f"Wrote manifest:  {out_dir / 'manifest.json'}")
    print(f"Wrote compile options: {compile_options_path}")
    print(f"Wrote runtime reference: {runtime_ref_path}")
    print(f"Wrote reference: {out_dir / 'reference.npz'}")


if __name__ == "__main__":
    main()
