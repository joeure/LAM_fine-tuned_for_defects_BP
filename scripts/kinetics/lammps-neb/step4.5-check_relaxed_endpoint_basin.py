#!/usr/bin/env python3
"""
check_relaxed_endpoint_basin.py

Check whether a prepared single-job folder has:
- final_relaxed collapsed back to the initial basin
- or remained in the intended different final basin

This version is compatible with both old and new meta.json formats.

Interpretation:
- site_a = initial vacancy site (target site for the moving atom)
- site_b = original occupied site of the moving atom

Classification:
- same_basin      : moved atom is back near site_b
- different_basin : moved atom is near site_a
- ambiguous       : neither clearly true

Dependencies:
    pip install ase numpy

Example:
    python check_relaxed_endpoint_basin.py \
      --job-dir /path/to/job_xxx \
      --tol 0.8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
from ase import Atoms
from ase.io import read


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--job-dir", required=True, type=Path)
    p.add_argument(
        "--tol",
        type=float,
        default=0.8,
        help="Distance tolerance in Å for deciding whether the moving atom is "
             "near the original site_b or target site_a."
    )
    return p.parse_args()


def read_lammps_atomic_data(path: Path) -> Atoms:
    atoms = read(str(path), format="lammps-data", atom_style="atomic")
    atoms.set_pbc([True, True, True])
    return atoms


def is_finite_coord(x: np.ndarray) -> bool:
    return np.isfinite(x).all()


def require_finite_coord(name: str, x: np.ndarray) -> np.ndarray:
    if x.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got shape={x.shape}, value={x}")
    if not is_finite_coord(x):
        raise ValueError(f"{name} contains non-finite values: {x}")
    return x


def try_get_coord(d: Dict[str, Any], key: str) -> Optional[np.ndarray]:
    if key not in d:
        return None
    val = d[key]
    if val is None:
        return None
    arr = np.array(val, dtype=float)
    if arr.shape != (3,):
        return None
    if not np.isfinite(arr).all():
        return None
    return arr


def resolve_site_coords(meta: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, str]]:
    """
    Resolve:
      coord_site_a = target site geometry for site_a
      coord_site_b = source/current position for moving atom at site_b

    Priority:
    1) new job_info keys
    2) old job_info keys
    3) move_meta keys
    4) move_meta site rows
    """
    job_info = meta.get("job_info", {})
    move_meta = meta.get("move_meta", {})

    coord_a = try_get_coord(job_info, "site_a_coord_target_A")
    coord_b = try_get_coord(job_info, "site_b_coord_source_A")
    src_a = "job_info.site_a_coord_target_A" if coord_a is not None else ""
    src_b = "job_info.site_b_coord_source_A" if coord_b is not None else ""

    if coord_a is None:
        coord_a = try_get_coord(job_info, "site_a_coord_final_A")
        if coord_a is not None:
            src_a = "job_info.site_a_coord_final_A"
    if coord_b is None:
        coord_b = try_get_coord(job_info, "site_b_coord_final_A")
        if coord_b is not None:
            src_b = "job_info.site_b_coord_final_A"

    if coord_a is None:
        coord_a = try_get_coord(move_meta, "target_coord_site_geometry_A")
        if coord_a is not None:
            src_a = "move_meta.target_coord_site_geometry_A"
    if coord_b is None:
        coord_b = try_get_coord(move_meta, "source_coord_current_A")
        if coord_b is not None:
            src_b = "move_meta.source_coord_current_A"

    site_a_row = move_meta.get("site_a_row", {})
    site_b_row = move_meta.get("site_b_row", {})

    if coord_a is None:
        cand = np.array([
            site_a_row.get("x_pristine_A"),
            site_a_row.get("y_pristine_A"),
            site_a_row.get("z_pristine_A"),
        ], dtype=float)
        if cand.shape == (3,) and np.isfinite(cand).all():
            coord_a = cand
            src_a = "move_meta.site_a_row.{x,y,z}_pristine_A"

    if coord_b is None:
        cand = np.array([
            site_b_row.get("x_final_A"),
            site_b_row.get("y_final_A"),
            site_b_row.get("z_final_A"),
        ], dtype=float)
        if cand.shape == (3,) and np.isfinite(cand).all():
            coord_b = cand
            src_b = "move_meta.site_b_row.{x,y,z}_final_A"

    if coord_a is None:
        raise KeyError(
            "Cannot resolve site_a coordinate from meta.json. "
            "Tried new job_info key, old job_info key, move_meta direct key, and site_a_row pristine coords."
        )
    if coord_b is None:
        raise KeyError(
            "Cannot resolve site_b coordinate from meta.json. "
            "Tried new job_info key, old job_info key, move_meta direct key, and site_b_row final coords."
        )

    coord_a = require_finite_coord("coord_site_a", coord_a)
    coord_b = require_finite_coord("coord_site_b", coord_b)

    return coord_a, coord_b, {"site_a_coord_source": src_a, "site_b_coord_source": src_b}


def mic_distance(cell: np.ndarray, pbc: np.ndarray, r1: np.ndarray, r2: np.ndarray) -> float:
    """
    Minimum-image distance between two Cartesian positions.
    Assumes an ASE Atoms cell matrix.
    """
    dr = r1 - r2
    if np.any(pbc):
        frac = np.linalg.solve(cell.T, dr)
        frac -= np.round(frac) * pbc.astype(float)
        dr = cell.T @ frac
    return float(np.linalg.norm(dr))


def classify_basin(
    final_atoms: Atoms,
    moving_atom_idx: int,
    coord_site_a: np.ndarray,
    coord_site_b: np.ndarray,
    tol: float,
) -> Dict[str, Any]:
    pos = final_atoms.get_positions()

    if moving_atom_idx < 0 or moving_atom_idx >= len(pos):
        raise IndexError(
            f"moving_atom_idx={moving_atom_idx} out of range for natoms={len(pos)}"
        )

    r = pos[moving_atom_idx]
    if not np.isfinite(r).all():
        raise ValueError(f"Moving atom final coordinate is non-finite: index={moving_atom_idx}, coord={r}")

    cell = np.array(final_atoms.cell)
    pbc = np.array(final_atoms.pbc, dtype=bool)

    d_to_a = mic_distance(cell, pbc, r, coord_site_a)
    d_to_b = mic_distance(cell, pbc, r, coord_site_b)

    near_a = d_to_a <= tol
    near_b = d_to_b <= tol

    if near_a and not near_b:
        label = "different_basin"
        reason = "moving atom remained near target site_a"
    elif near_b and not near_a:
        label = "same_basin"
        reason = "moving atom relaxed back near original site_b"
    elif near_a and near_b:
        label = "ambiguous"
        reason = "moving atom is within tolerance of both site_a and site_b"
    else:
        label = "ambiguous"
        reason = "moving atom is not close to either site_a or site_b"

    nearest = "a" if d_to_a <= d_to_b else "b"

    return {
        "classification": label,
        "reason": reason,
        "moving_atom_distance_to_site_a_A": d_to_a,
        "moving_atom_distance_to_site_b_A": d_to_b,
        "near_site_a": near_a,
        "near_site_b": near_b,
        "nearest_site": nearest,
        "moving_atom_final_coord_A": r.tolist(),
    }


def main() -> None:
    args = parse_args()
    job_dir = args.job_dir.resolve()

    meta_path = job_dir / "meta" / "meta.json"
    init_path = job_dir / "data" / "init.data"
    final_path = job_dir / "results" / "final_relaxed.data"

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta file: {meta_path}")
    if not init_path.exists():
        raise FileNotFoundError(f"Missing init structure: {init_path}")
    if not final_path.exists():
        raise FileNotFoundError(f"Missing relaxed final structure: {final_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    job_info = meta["job_info"]

    moving_atom_idx = int(job_info["moving_atom_index_0based"])
    coord_site_a, coord_site_b, coord_sources = resolve_site_coords(meta)

    init_atoms = read_lammps_atomic_data(init_path)
    final_atoms = read_lammps_atomic_data(final_path)

    if len(init_atoms) != len(final_atoms):
        raise ValueError(
            f"Atom count changed: init={len(init_atoms)}, final={len(final_atoms)}"
        )

    result = classify_basin(
        final_atoms=final_atoms,
        moving_atom_idx=moving_atom_idx,
        coord_site_a=coord_site_a,
        coord_site_b=coord_site_b,
        tol=args.tol,
    )

    payload = {
        "job_dir": str(job_dir),
        "system_id": job_info["system_id"],
        "model_tag": job_info["model_tag"],
        "site_a": job_info["site_a"],
        "site_b": job_info["site_b"],
        "moving_atom_index_0based": moving_atom_idx,
        "moving_atom_element": job_info["moving_atom_element"],
        "site_a_coord_A": coord_site_a.tolist(),
        "site_b_coord_A": coord_site_b.tolist(),
        "coord_resolution": coord_sources,
        "tolerance_A": args.tol,
        "result": result,
    }

    out_path = job_dir / "meta" / "basin_check.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[BASIN CHECK] {job_dir.name}")
    print(f"  classification       : {result['classification']}")
    print(f"  reason               : {result['reason']}")
    print(f"  site_a_coord source  : {coord_sources['site_a_coord_source']}")
    print(f"  site_b_coord source  : {coord_sources['site_b_coord_source']}")
    print(f"  d(moving->a)         : {result['moving_atom_distance_to_site_a_A']:.6f} Å")
    print(f"  d(moving->b)         : {result['moving_atom_distance_to_site_b_A']:.6f} Å")
    print(f"  nearest_site         : {result['nearest_site']}")
    print(f"  saved                : {out_path}")


if __name__ == "__main__":
    main()