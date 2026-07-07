#!/usr/bin/env python3
"""Shared helpers for Stage 4.9 DFT endpoint-relax and NEB packaging."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence

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


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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


def sanitize_poscar_species_line(line: str) -> str:
    clean: List[str] = []
    for tok in line.split():
        tok = tok.split("/")[0].split("|")[0].split(":")[0]
        clean.append("".join(ch for ch in tok if ch.isalpha())[:2])
    return "  ".join(clean) + "\n"


def read_poscar_sanitized(path: Path) -> Atoms:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines(True)
    if len(lines) < 7:
        raise ValueError(f"POSCAR/CONTCAR too short: {path}")
    lines[5] = sanitize_poscar_species_line(lines[5])
    with tempfile.NamedTemporaryFile(mode="w", suffix=".POSCAR", delete=False) as handle:
        handle.writelines(lines)
        tmp_path = Path(handle.name)
    try:
        return read(str(tmp_path), format="vasp")
    finally:
        tmp_path.unlink(missing_ok=True)


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


def write_poscar(path: Path, atoms: Atoms) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write(str(path), atoms, format="vasp", direct=False, sort=False)


def copy_required(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing required file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_project_magmom(atoms: Atoms) -> str:
    parts: List[str] = []
    order: List[str] = []
    for sym in atoms.get_chemical_symbols():
        if sym not in order:
            order.append(sym)
    for sym in order:
        count = sum(1 for s in atoms.get_chemical_symbols() if s == sym)
        if sym == "P":
            parts.append(f"{count}*0.1")
        elif sym == "N":
            parts.append(f"{count}*1")
        else:
            parts.append(f"{count}*0.1")
    return " ".join(parts)


def apply_tag_updates(
    incar_text: str,
    updates: Dict[str, str],
    trailing_comments: Sequence[str] | None = None,
    remove_keys: Sequence[str] | None = None,
) -> str:
    lines_out: List[str] = []
    seen: set[str] = set()
    remove_set = {key.upper() for key in (remove_keys or [])}
    for raw in incar_text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            lines_out.append(raw)
            continue
        key = stripped.split("=", 1)[0].strip().upper()
        if key in remove_set:
            continue
        if key in updates:
            lines_out.append(f"{key} = {updates[key]}")
            seen.add(key)
        else:
            lines_out.append(raw)
    for key, value in updates.items():
        if key not in seen:
            lines_out.append(f"{key} = {value}")
    if trailing_comments:
        lines_out.append("")
        lines_out.extend(trailing_comments)
    return "\n".join(lines_out).rstrip() + "\n"


def interpolate_images(init_atoms: Atoms, final_atoms: Atoms, total_images: int) -> List[Atoms]:
    if len(init_atoms) != len(final_atoms):
        raise ValueError("Initial and final structures have different atom counts")
    if total_images < 2:
        raise ValueError("total_images must be at least 2")
    displacement, _ = find_mic(
        final_atoms.positions - init_atoms.positions,
        cell=init_atoms.cell,
        pbc=init_atoms.pbc,
    )
    images: List[Atoms] = []
    for idx in range(total_images):
        t = idx / (total_images - 1)
        image = init_atoms.copy()
        image.positions[:] = init_atoms.positions + t * displacement
        images.append(image)
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
    images: Sequence[Atoms],
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


def write_neb_image_dirs(out_dir: Path, images: Sequence[Atoms]) -> List[Dict[str, Any]]:
    preflight_report = validate_neb_image_geometries(images)
    for idx, atoms in enumerate(images):
        image_dir = out_dir / f"{idx:02d}"
        ensure_dir(image_dir)
        write_poscar(image_dir / "POSCAR", atoms)
    return preflight_report


def load_job_json_template(path: Path) -> Dict[str, Any]:
    job = dict(DEFAULT_NEB_JOB_JSON)
    if path.exists():
        job.update(read_json(path))
    job["project_id"] = DEFAULT_NEB_JOB_JSON["project_id"]
    return job


def build_run_script(job_json: Dict[str, Any]) -> str:
    command = str(job_json.get("command", DEFAULT_NEB_JOB_JSON["command"]))
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            "# Helper script; the scheduler-facing command remains job.json.",
            command,
            "",
        ]
    )


def find_relaxed_endpoint(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = [
        path / "CONTCAR",
        path / "POSCAR",
        path / "OUTCAR",
        path / "vasprun.xml",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    raise FileNotFoundError(f"Could not locate a relaxed endpoint file under {path}")
