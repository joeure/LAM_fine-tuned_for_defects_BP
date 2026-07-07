#!/usr/bin/env python3
"""
Prepare a single-path DFT-side first-bite scaffold for BP vacancy-hop NEB work.

Current scope
-------------
- single-system / single-event only
- consumes:
  - manifest row from data_center/manifest_test.csv
  - site mapping from data_center/site_matching.csv
  - a prior DFT-relaxed structure under relaxed_structures/BP_1014/*
- emits a first-bite task scaffold under dft_neb_workspace/ that contains:
  - one authoritative relaxed initial endpoint
  - one constructed final-endpoint guess
  - one self-contained final-endpoint relaxation subtask
  - one interpolated NEB image scaffold
  - copied reference VASP inputs and Bohrium job template
  - provenance-rich metadata

Design choice
-------------
This script intentionally stops at:
- endpoints
- image scaffold
- input templates

It does not claim that the task is already the final production DFT-NEB submission.
The immediate goal is to validate the VASP-side workflow and directory layout first.

Dependencies
------------
    pip install ase pandas numpy
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from ase import Atoms
from ase.geometry import find_mic
from ase.io import read, write
from ase.neighborlist import neighbor_list


DEFAULT_NEB_JOB_JSON = {
    "job_type": "container",
    "command": 'bash -lc "source ${VASP_ENV_SCRIPT:?set VASP_ENV_SCRIPT} && mpirun -n ${VASP_MPI_RANKS:?set VASP_MPI_RANKS} ${VASP_STD_BIN:-vasp_std}"',
    "backward_files": [],
    "project_id": 0,
    "platform": "<CLOUD_PLATFORM>",
    "machine_type": "<EXAMPLE_MACHINE_TYPE>",
    "image_address": "<VASP_IMAGE_ADDRESS>",
}


@dataclass
class DftFirstBiteInfo:
    system_id: str
    site_a: int
    site_b: int
    job_name: str
    images_total: int
    images_intermediate: int
    source_dft_dir: str
    source_relaxed_structure: str
    source_incar: str
    source_kpoints: str
    source_potcar: str
    output_dir: str
    initial_poscar: str
    final_guess_poscar: str
    final_endpoint_relax_dir: str
    neb_scaffold_dir: str
    moving_atom_index_0based: int
    moving_atom_element: str
    site_a_target_pristine_A: List[float]
    site_b_source_current_A: List[float]
    note: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=Path("data_center/manifest_test.csv"))
    p.add_argument("--site-map", type=Path, default=Path("data_center/site_matching.csv"))
    p.add_argument("--job-json-ref", type=Path, default=Path("references_codes/DFT/job.json"))
    p.add_argument("--system-id", required=True)
    p.add_argument("--site-a", required=True, type=int)
    p.add_argument("--site-b", required=True, type=int)
    p.add_argument(
        "--job-name-suffix",
        default="",
        help="Optional suffix appended to the generated first-bite job directory.",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path("dft_neb_workspace/first-bite"),
    )
    p.add_argument(
        "--images-total",
        type=int,
        default=7,
        help="Total number of endpoint+image folders including both endpoints.",
    )
    p.add_argument(
        "--interpolation-space",
        choices=["cartesian"],
        default="cartesian",
        help="Current implementation keeps first-bite interpolation intentionally simple.",
    )
    p.add_argument(
        "--force-overwrite",
        action="store_true",
        help="Overwrite an existing first-bite output directory.",
    )
    return p.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_path(base_file: Path, maybe_rel: str) -> Path:
    p = Path(str(maybe_rel))
    if p.is_absolute():
        return p
    return (base_file.parent / p).resolve()


def sanitize_name(text: str) -> str:
    return text.replace("/", "_").replace(" ", "_").replace(":", "_")


def build_job_name(system_id: str, site_a: int, site_b: int, suffix: str = "") -> str:
    name = f"job_{system_id}__event_{site_a}_{site_b}__dft_neb_first_bite"
    if suffix.strip():
        name = f"{name}__{sanitize_name(suffix.strip())}"
    return sanitize_name(name)


def load_manifest_row(manifest_csv: Path, system_id: str) -> pd.Series:
    df = pd.read_csv(manifest_csv)
    hit = df[df["system_id"] == system_id]
    if len(hit) == 0:
        raise ValueError(f"system_id not found in manifest: {system_id}")
    if len(hit) > 1:
        raise ValueError(f"system_id appears multiple times in manifest: {system_id}")
    return hit.iloc[0]


def load_site_rows(site_csv: Path, system_id: str, structure_key: str) -> pd.DataFrame:
    df = pd.read_csv(site_csv)
    sub = df[(df["system_id"] == system_id) & (df["structure"] == structure_key)].copy()
    if sub.empty:
        raise ValueError(
            f"No site-matching rows found for system_id={system_id}, structure={structure_key}"
        )
    return sub


def get_site_row(site_rows: pd.DataFrame, site_index: int) -> pd.Series:
    hit = site_rows[site_rows["site_index"] == site_index]
    if len(hit) == 0:
        raise ValueError(f"site_index={site_index} not found in site mapping")
    if len(hit) > 1:
        raise ValueError(f"site_index={site_index} appears multiple times in site mapping")
    return hit.iloc[0]


def locate_dft_source_dir(manifest_csv: Path, manifest_row: pd.Series) -> Path:
    dft_path = manifest_row.get("dft_path", "")
    if pd.isna(dft_path) or not str(dft_path).strip():
        raise ValueError(
            f"manifest row for {manifest_row['system_id']} has empty dft_path"
        )
    resolved = resolve_path(manifest_csv, str(dft_path))
    return resolved.parent


def locate_relaxed_structure(dft_dir: Path) -> Path:
    for name in ["CONTCAR", "vasprun.xml", "OUTCAR", "POSCAR"]:
        p = dft_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Cannot locate a readable relaxed DFT structure under {dft_dir}"
    )


def sanitize_poscar_species_line(line: str) -> str:
    toks = line.split()
    clean: List[str] = []
    for tok in toks:
        tok = tok.split("/")[0].split("|")[0].split(":")[0]
        match = re.match(r"^[A-Za-z]{1,2}", tok)
        if not match:
            raise ValueError(f"Cannot sanitize POSCAR species token: {tok}")
        clean.append(match.group(0))
    return "  ".join(clean) + "\n"


def parse_vasp_species_order(path: Path) -> List[str]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 7:
        raise ValueError(f"POSCAR/CONTCAR too short: {path}")
    return sanitize_poscar_species_line(lines[5]).split()


def parse_potcar_species_order(path: Path) -> List[str]:
    symbols: List[str] = []
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "TITEL" not in raw_line:
            continue
        # Typical form: TITEL  = PAW_PBE P 06Sep2000
        match = re.search(r"TITEL\s*=\s*\S+\s+([A-Za-z]{1,2})\b", raw_line)
        if match:
            symbols.append(match.group(1))
    if not symbols:
        raise ValueError(f"Could not parse any TITEL species entries from POTCAR: {path}")
    return symbols


def read_poscar_sanitized(path: Path) -> Atoms:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines(True)
    if len(lines) < 7:
        raise ValueError(f"POSCAR/CONTCAR too short: {path}")
    lines[5] = sanitize_poscar_species_line(lines[5])
    with tempfile.NamedTemporaryFile(mode="w", suffix=".POSCAR", delete=False) as tf:
        tf.writelines(lines)
        tmp = Path(tf.name)
    try:
        return read(str(tmp), format="vasp")
    finally:
        tmp.unlink(missing_ok=True)


def read_vasp_atoms(path: Path) -> Atoms:
    if path.name == "vasprun.xml":
        atoms = read(str(path), format="vasp-xml", index=-1)
    elif path.name in {"CONTCAR", "POSCAR"}:
        try:
            atoms = read(str(path), format="vasp")
        except Exception:
            atoms = read_poscar_sanitized(path)
    elif path.name == "OUTCAR":
        atoms = read(str(path), format="vasp-out", index=-1)
    else:
        atoms = read(str(path))
    atoms.set_pbc([True, True, True])
    return atoms


def finite_coord_from_row(row: pd.Series, preferred_prefix: str, fallback_prefix: str) -> np.ndarray:
    preferred = np.array(
        [
            row.get(f"{preferred_prefix}x", np.nan),
            row.get(f"{preferred_prefix}y", np.nan),
            row.get(f"{preferred_prefix}z", np.nan),
        ],
        dtype=float,
    )
    if np.all(np.isfinite(preferred)):
        return preferred

    fallback = np.array(
        [
            row.get(f"{fallback_prefix}x", np.nan),
            row.get(f"{fallback_prefix}y", np.nan),
            row.get(f"{fallback_prefix}z", np.nan),
        ],
        dtype=float,
    )
    if np.all(np.isfinite(fallback)):
        return fallback

    site_index = row.get("site_index", "UNKNOWN")
    raise ValueError(
        f"Cannot resolve finite coordinate for site_index={site_index} "
        f"from prefixes {preferred_prefix} or {fallback_prefix}"
    )


def site_target_coord(row: pd.Series) -> np.ndarray:
    return finite_coord_from_row(
        row=row,
        preferred_prefix="x_pristine_A".replace("x_", ""),  # sentinel, unused
        fallback_prefix="x_final_A".replace("x_", ""),      # sentinel, unused
    )


def coord_from_columns(row: pd.Series, preferred_triplet: Tuple[str, str, str], fallback_triplet: Tuple[str, str, str]) -> np.ndarray:
    preferred = np.array(
        [row.get(preferred_triplet[0], np.nan), row.get(preferred_triplet[1], np.nan), row.get(preferred_triplet[2], np.nan)],
        dtype=float,
    )
    if np.all(np.isfinite(preferred)):
        return preferred
    fallback = np.array(
        [row.get(fallback_triplet[0], np.nan), row.get(fallback_triplet[1], np.nan), row.get(fallback_triplet[2], np.nan)],
        dtype=float,
    )
    if np.all(np.isfinite(fallback)):
        return fallback
    site_index = row.get("site_index", "UNKNOWN")
    raise ValueError(f"Cannot resolve finite coordinate for site_index={site_index}")


def construct_final_guess(
    init_atoms: Atoms,
    site_rows: pd.DataFrame,
    site_a: int,
    site_b: int,
) -> Tuple[Atoms, Dict[str, Any]]:
    row_a = get_site_row(site_rows, site_a)
    row_b = get_site_row(site_rows, site_b)

    if bool(row_b["is_occupied"]) is not True:
        raise ValueError(f"site_b={site_b} is not occupied in the DFT site mapping")
    if pd.isna(row_b["mapped_atom_index_0based"]):
        raise ValueError(
            f"site_b={site_b} has NaN mapped_atom_index_0based although it is occupied"
        )

    moving_atom_idx = int(row_b["mapped_atom_index_0based"])
    moving_atom = init_atoms[moving_atom_idx]
    target = coord_from_columns(
        row_a,
        ("x_pristine_A", "y_pristine_A", "z_pristine_A"),
        ("x_final_A", "y_final_A", "z_final_A"),
    )
    source = coord_from_columns(
        row_b,
        ("x_final_A", "y_final_A", "z_final_A"),
        ("x_pristine_A", "y_pristine_A", "z_pristine_A"),
    )

    final_guess = init_atoms.copy()
    final_guess.positions[moving_atom_idx] = target

    move_meta = {
        "site_a_row": json.loads(pd.Series(row_a).to_json()),
        "site_b_row": json.loads(pd.Series(row_b).to_json()),
        "moving_atom_index_0based": moving_atom_idx,
        "moving_atom_element": moving_atom.symbol,
        "source_coord_current_A": source.tolist(),
        "target_coord_site_geometry_A": target.tolist(),
        "moving_atom_original_coord_from_atoms_A": init_atoms.positions[moving_atom_idx].tolist(),
        "coord_rule": {
            "site_a_target": "prefer pristine site geometry, fallback final",
            "site_b_source": "prefer final occupied-atom coordinate, fallback pristine",
        },
    }
    return final_guess, move_meta


def interpolate_images(init_atoms: Atoms, final_atoms: Atoms, total_images: int) -> List[Atoms]:
    if len(init_atoms) != len(final_atoms):
        raise ValueError("Initial and final structures have different atom counts")
    if total_images < 2:
        raise ValueError("--images-total must be at least 2")

    displacement, _ = find_mic(
        final_atoms.positions - init_atoms.positions,
        cell=init_atoms.cell,
        pbc=init_atoms.pbc,
    )
    images: List[Atoms] = []
    for i in range(total_images):
        t = i / (total_images - 1)
        atoms = init_atoms.copy()
        atoms.positions[:] = init_atoms.positions + t * displacement
        images.append(atoms)
    return images


def shortest_distance_report(atoms: Atoms, cutoff: float = 3.0, max_pairs: int = 6) -> List[Dict[str, Any]]:
    work = atoms.copy()
    work.set_pbc([True, True, True])
    idx_i, idx_j, distances = neighbor_list("ijd", work, cutoff)
    pairs: List[Dict[str, Any]] = []
    for i, j, distance in zip(idx_i, idx_j, distances):
        if int(i) >= int(j):
            continue
        pairs.append(
            {
                "i": int(i),
                "j": int(j),
                "symbols": f"{work[int(i)].symbol}-{work[int(j)].symbol}",
                "distance_A": float(distance),
            }
        )
    return sorted(pairs, key=lambda row: row["distance_A"])[:max_pairs]


def validate_neb_image_geometries(
    images: List[Atoms],
    min_allowed_distance_A: float = 1.2,
    warning_distance_A: float = 1.5,
) -> List[Dict[str, Any]]:
    reports: List[Dict[str, Any]] = []
    bad: List[str] = []
    for idx, atoms in enumerate(images):
        shortest = shortest_distance_report(atoms)
        if not shortest:
            reports.append({"image": idx, "status": "no_pairs_within_cutoff"})
            continue
        min_pair = shortest[0]
        min_distance = float(min_pair["distance_A"])
        if min_distance < min_allowed_distance_A:
            status = "fatal_short_contact"
            bad.append(
                f"{idx:02d}: {min_pair['symbols']} {min_pair['i']}-{min_pair['j']} "
                f"{min_distance:.4f} A"
            )
        elif min_distance < warning_distance_A:
            status = "suspicious_short_contact"
        else:
            status = "ok"
        reports.append(
            {
                "image": idx,
                "status": status,
                "min_distance_A": min_distance,
                "min_pair": min_pair,
                "shortest_pairs": shortest,
            }
        )
    if bad:
        joined = "; ".join(bad[:8])
        raise ValueError(
            "NEB image preflight found unphysical short contacts below "
            f"{min_allowed_distance_A:.2f} A: {joined}"
        )
    return reports


def write_poscar(path: Path, atoms: Atoms) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write(str(path), atoms, format="vasp", direct=False, sort=False)


def read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def parse_incar(text: str) -> Dict[str, str]:
    tags: Dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.split("!", 1)[0].split("#", 1)[0].strip()
        if not line:
            continue
        for segment in line.split(";"):
            chunk = segment.strip()
            if not chunk or "=" not in chunk:
                continue
            key, value = chunk.split("=", 1)
            tags[key.strip().upper()] = value.strip()
    return tags

def build_project_magmom(atoms: Atoms) -> str:
    symbols = list(atoms.get_chemical_symbols())
    unique_in_order: List[str] = []
    for sym in symbols:
        if sym not in unique_in_order:
            unique_in_order.append(sym)

    parts: List[str] = []
    for sym in unique_in_order:
        count = sum(1 for s in symbols if s == sym)
        if sym == "P":
            parts.append(f"{count}*0.1")
        elif sym == "N":
            parts.append(f"{count}*1")
        else:
            parts.append(f"{count}*0.1")
    return " ".join(parts)


def species_order_from_atoms(atoms: Atoms) -> List[str]:
    order: List[str] = []
    for sym in atoms.get_chemical_symbols():
        if sym not in order:
            order.append(sym)
    return order


def validate_species_order(
    source_species_order: List[str],
    potcar_species_order: List[str],
    atoms: Atoms,
    context: str,
) -> Dict[str, Any]:
    generated_order = species_order_from_atoms(atoms)
    if generated_order != source_species_order:
        raise ValueError(
            f"{context}: generated POSCAR species order {generated_order} does not match "
            f"source VASP order {source_species_order}"
        )
    if potcar_species_order != source_species_order:
        raise ValueError(
            f"{context}: POTCAR species order {potcar_species_order} does not match "
            f"source VASP order {source_species_order}"
        )
    return {
        "source_poscar_species_order": list(source_species_order),
        "potcar_species_order": list(potcar_species_order),
        "generated_poscar_species_order": list(generated_order),
    }


def build_endpoint_relax_incar(
    reference_tags: Dict[str, str],
    system_name: str,
    atoms: Atoms,
) -> str:
    keys = [
        "ENCUT",
        "EDIFF",
        "ISMEAR",
        "SIGMA",
        "LREAL",
        "NELM",
        "ALGO",
        "PREC",
        "NBANDS",
        "NCORE",
        "AMIX",
        "BMIX",
        "LWAVE",
        "LCHARG",
        "ISIF",
        "IBRION",
        "NSW",
        "EDIFFG",
        "POTIM",
        "LDIPOL",
        "IDIPOL",
    ]
    lines = [f"SYSTEM = {system_name} final-endpoint-relax"]
    for key in keys:
        if key in reference_tags:
            lines.append(f"{key} = {reference_tags[key]}")
    lines.append(f"ISPIN = {reference_tags.get('ISPIN', '2')}")
    lines.append(f"MAGMOM = {reference_tags.get('MAGMOM', build_project_magmom(atoms))}")
    return "\n".join(lines) + "\n"


def build_neb_incar(
    reference_tags: Dict[str, str],
    system_name: str,
    images_intermediate: int,
    atoms: Atoms,
) -> str:
    keep_keys = [
        "ENCUT",
        "EDIFF",
        "ISMEAR",
        "SIGMA",
        "LREAL",
        "NELM",
        "ALGO",
        "PREC",
        "NBANDS",
        "NCORE",
        "AMIX",
        "BMIX",
        "LDIPOL",
        "IDIPOL",
    ]
    lines = [
        f"SYSTEM = {system_name} dft-neb-first-bite",
        "# First-bite native-VASP / VTST-compatible NEB template.",
        "# Review these tags before production use on the final HPC environment.",
    ]
    for key in keep_keys:
        if key in reference_tags:
            lines.append(f"{key} = {reference_tags[key]}")
    lines.append(f"ISPIN = {reference_tags.get('ISPIN', '2')}")
    lines.append(f"MAGMOM = {reference_tags.get('MAGMOM', build_project_magmom(atoms))}")
    lines.extend(
        [
            "IBRION = 3",
            "POTIM = 0",
            "NSW = 200",
            "EDIFFG = -0.01",
            "ISIF = 2",
            f"IMAGES = {images_intermediate}",
            "SPRING = -5",
            "LCLIMB = .TRUE.",
        ]
    )
    return "\n".join(lines) + "\n"


def load_job_json_template(path: Path) -> Dict[str, Any]:
    if not path.exists():
        job = dict(DEFAULT_NEB_JOB_JSON)
    else:
        job = read_json(path)
    for key, value in DEFAULT_NEB_JOB_JSON.items():
        job.setdefault(key, value)
    return job


def write_status(path: Path, payload: Dict[str, Any]) -> None:
    write_json(path, payload)


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def build_run_script(job_json: Dict[str, Any]) -> str:
    command = str(job_json.get("command", DEFAULT_NEB_JOB_JSON["command"]))
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            "# First-bite helper; the authoritative remote configuration remains job.json.",
            command,
            "",
        ]
    )


def write_readme(path: Path, info: DftFirstBiteInfo) -> None:
    text = f"""# DFT NEB First-Bite Scaffold

System: {info.system_id}
Event: vacancy {info.site_a} -> {info.site_b}

This package is intentionally a first-bite scaffold, not yet the final production
DFT-NEB workflow.

Contents:
- `initial_endpoint/POSCAR`: authoritative relaxed DFT initial endpoint
- `final_endpoint_guess/POSCAR`: constructed final-endpoint guess before DFT relaxation
- `final_endpoint_relax/`: self-contained subtask for relaxing the final endpoint
- `neb_image_scaffold/00..0N/`: interpolated image scaffold from the relaxed initial
  endpoint to the constructed final guess
- `INCAR.neb`: native-VASP / VTST-compatible NEB template for later review
- `meta/meta.json`: provenance and move metadata

Current intent:
- validate endpoint provenance
- validate image scaffold generation
- keep one-path / one-submission directory conventions explicit from the start
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.images_total < 2:
        raise ValueError("--images-total must be at least 2")

    manifest_row = load_manifest_row(args.manifest, args.system_id)
    site_rows = load_site_rows(args.site_map, args.system_id, "dft")
    source_dft_dir = locate_dft_source_dir(args.manifest, manifest_row)
    source_structure = locate_relaxed_structure(source_dft_dir)

    source_incar = source_dft_dir / "INCAR"
    source_kpoints = source_dft_dir / "KPOINTS"
    source_potcar = source_dft_dir / "POTCAR"
    source_species_order = parse_vasp_species_order(source_structure)
    potcar_species_order = parse_potcar_species_order(source_potcar)

    init_atoms = read_vasp_atoms(source_structure)
    final_guess_atoms, move_meta = construct_final_guess(
        init_atoms=init_atoms,
        site_rows=site_rows,
        site_a=args.site_a,
        site_b=args.site_b,
    )
    init_species_validation = validate_species_order(
        source_species_order=source_species_order,
        potcar_species_order=potcar_species_order,
        atoms=init_atoms,
        context="initial endpoint",
    )
    final_guess_species_validation = validate_species_order(
        source_species_order=source_species_order,
        potcar_species_order=potcar_species_order,
        atoms=final_guess_atoms,
        context="final endpoint guess",
    )
    images = interpolate_images(init_atoms, final_guess_atoms, args.images_total)
    image_preflight = validate_neb_image_geometries(images)

    job_name = build_job_name(
        args.system_id,
        args.site_a,
        args.site_b,
        args.job_name_suffix,
    )
    out_dir = args.out_root / job_name
    if out_dir.exists() and args.force_overwrite:
        shutil.rmtree(out_dir)
    if out_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {out_dir}. Use --force-overwrite to replace it."
        )

    meta_dir = ensure_dir(out_dir / "meta")
    initial_dir = ensure_dir(out_dir / "initial_endpoint")
    final_guess_dir = ensure_dir(out_dir / "final_endpoint_guess")
    final_relax_dir = ensure_dir(out_dir / "final_endpoint_relax")
    neb_dir = ensure_dir(out_dir / "neb_image_scaffold")

    write_poscar(initial_dir / "POSCAR", init_atoms)
    write_poscar(final_guess_dir / "POSCAR", final_guess_atoms)

    for i, atoms in enumerate(images):
        write_poscar(neb_dir / f"{i:02d}" / "POSCAR", atoms)

    reference_incar_text = read_text_if_exists(source_incar)
    reference_tags = parse_incar(reference_incar_text)

    (out_dir / "INCAR.neb").write_text(
        build_neb_incar(reference_tags, job_name, args.images_total - 2, init_atoms),
        encoding="utf-8",
    )

    copy_if_exists(source_kpoints, out_dir / "KPOINTS")
    copy_if_exists(source_potcar, out_dir / "POTCAR")

    (final_relax_dir / "INCAR").write_text(
        build_endpoint_relax_incar(reference_tags, job_name, final_guess_atoms),
        encoding="utf-8",
    )
    copy_if_exists(source_kpoints, final_relax_dir / "KPOINTS")
    copy_if_exists(source_potcar, final_relax_dir / "POTCAR")
    write_poscar(final_relax_dir / "POSCAR", final_guess_atoms)

    job_json = load_job_json_template(args.job_json_ref)
    final_relax_job_json = dict(job_json)
    final_relax_job_json["job_name"] = f"{job_name}__final_endpoint_relax"
    write_json(final_relax_dir / "job.json", final_relax_job_json)
    run_text = build_run_script(final_relax_job_json)
    (final_relax_dir / "run.sh").write_text(run_text, encoding="utf-8")

    root_job_json = dict(job_json)
    root_job_json["job_name"] = job_name
    write_json(out_dir / "job.json", root_job_json)
    (out_dir / "run.sh").write_text(build_run_script(root_job_json), encoding="utf-8")

    if reference_incar_text:
        (meta_dir / "INCAR.source.reference").write_text(reference_incar_text, encoding="utf-8")
    copy_if_exists(source_kpoints, meta_dir / "KPOINTS.source.reference")
    (meta_dir / "job.json.reference.json").write_text(
        json.dumps(job_json, indent=2), encoding="utf-8"
    )

    info = DftFirstBiteInfo(
        system_id=args.system_id,
        site_a=args.site_a,
        site_b=args.site_b,
        job_name=job_name,
        images_total=args.images_total,
        images_intermediate=args.images_total - 2,
        source_dft_dir=str(source_dft_dir),
        source_relaxed_structure=str(source_structure),
        source_incar=str(source_incar),
        source_kpoints=str(source_kpoints),
        source_potcar=str(source_potcar),
        output_dir=str(out_dir),
        initial_poscar=str(initial_dir / "POSCAR"),
        final_guess_poscar=str(final_guess_dir / "POSCAR"),
        final_endpoint_relax_dir=str(final_relax_dir),
        neb_scaffold_dir=str(neb_dir),
        moving_atom_index_0based=int(move_meta["moving_atom_index_0based"]),
        moving_atom_element=str(move_meta["moving_atom_element"]),
        site_a_target_pristine_A=list(move_meta["target_coord_site_geometry_A"]),
        site_b_source_current_A=list(move_meta["source_coord_current_A"]),
        note=(
            "Single-path DFT first-bite scaffold. Initial endpoint comes from prior DFT "
            "relaxation. Final endpoint is currently only a constructed guess; the "
            "final_endpoint_relax subtask is written separately. The NEB image scaffold "
            "is intentionally based on init -> final_guess interpolation at this stage."
        ),
    )
    write_json(
        meta_dir / "meta.json",
        {
            "job_info": asdict(info),
            "move_meta": move_meta,
            "species_order_validation": {
                "initial_endpoint": init_species_validation,
                "final_endpoint_guess": final_guess_species_validation,
            },
            "image_interpolation": "mic_unwrapped",
            "image_geometry_preflight": image_preflight,
            "manifest_row": json.loads(pd.Series(manifest_row).to_json()),
            "dft_site_rows_summary": {
                "structure": "dft",
                "nrows": int(len(site_rows)),
                "occupied_sites": int(site_rows["is_occupied"].fillna(False).sum()),
            },
        },
    )
    write_status(
        meta_dir / "prepare_status.json",
        {
            "status": "prepared",
            "task_class": "dft_neb_first_bite_scaffold",
            "ready_for_submission": False,
            "reason": (
                "The package is a first-bite scaffold. "
                "The final endpoint is not yet DFT-relaxed and the final production "
                "DFT-NEB environment may still differ from the current Server first-bite settings."
            ),
        },
    )
    write_readme(out_dir / "README.md", info)

    print(f"Wrote DFT first-bite scaffold: {out_dir}")


if __name__ == "__main__":
    main()
