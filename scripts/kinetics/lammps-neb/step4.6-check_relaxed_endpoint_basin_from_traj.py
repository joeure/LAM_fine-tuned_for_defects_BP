#!/usr/bin/env python3
"""
check_relaxed_endpoint_basin_from_traj.py

Trajectory-based basin checker for one prepared job folder.

It reads:
- meta/meta.json
- dumps/relax_final/traj.lammpstrj
- optionally results/final_relaxed.data for consistency checking

It tracks the moved atom through all minimization frames and classifies the
endpoint-relaxation trajectory as:
- same_basin
- different_basin
- other_basin
- ambiguous

Interpretation:
- site_a = initial vacancy site
- site_b = intended final vacancy site
- moving atom should go from site_b -> site_a

If the moving atom stabilizes near:
- site_b  => same_basin
- site_a  => different_basin
- neither => other_basin / ambiguous

Dependencies:
    pip install ase numpy pandas

Example:
    python check_relaxed_endpoint_basin_from_traj.py \
      --job-dir /path/to/job_xxx \
      --tol 0.8 \
      --tail-window 8
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--job-dir", required=True, type=Path)
    p.add_argument(
        "--traj-path",
        type=Path,
        default=None,
        help="Optional explicit trajectory path. "
             "If omitted, uses job-dir/dumps/relax_final/traj.lammpstrj"
    )
    p.add_argument(
        "--tol",
        type=float,
        default=0.8,
        help="Distance tolerance in Å for labeling moving atom as near site_a or site_b."
    )
    p.add_argument(
        "--tail-window",
        type=int,
        default=8,
        help="How many final frames to inspect for stabilized basin assignment."
    )
    p.add_argument(
        "--min-tail-window",
        type=int,
        default=3,
        help="Minimum tail length if the trajectory is very short."
    )
    p.add_argument(
        "--check-final-data",
        action="store_true",
        help="If set, also compare the final trajectory frame against results/final_relaxed.data."
    )
    return p.parse_args()


def read_lammps_atomic_data(path: Path) -> Atoms:
    atoms = read(str(path), format="lammps-data", atom_style="atomic")
    atoms.set_pbc([True, True, True])
    return atoms


def read_lammpstrj_all(path: Path) -> List[Atoms]:
    frames = read(str(path), index=":", format="lammps-dump-text")
    if isinstance(frames, Atoms):
        frames = [frames]
    for at in frames:
        at.set_pbc([True, True, True])
    return frames


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

    # New-format keys
    coord_a = try_get_coord(job_info, "site_a_coord_target_A")
    coord_b = try_get_coord(job_info, "site_b_coord_source_A")
    src_a = "job_info.site_a_coord_target_A" if coord_a is not None else ""
    src_b = "job_info.site_b_coord_source_A" if coord_b is not None else ""

    # Old-format fallback
    if coord_a is None:
        coord_a = try_get_coord(job_info, "site_a_coord_final_A")
        if coord_a is not None:
            src_a = "job_info.site_a_coord_final_A"
    if coord_b is None:
        coord_b = try_get_coord(job_info, "site_b_coord_final_A")
        if coord_b is not None:
            src_b = "job_info.site_b_coord_final_A"

    # move_meta direct keys
    if coord_a is None:
        coord_a = try_get_coord(move_meta, "target_coord_site_geometry_A")
        if coord_a is not None:
            src_a = "move_meta.target_coord_site_geometry_A"
    if coord_b is None:
        coord_b = try_get_coord(move_meta, "source_coord_current_A")
        if coord_b is not None:
            src_b = "move_meta.source_coord_current_A"

    # final fallback from move_meta rows
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
    dr = r1 - r2
    if np.any(pbc):
        frac = np.linalg.solve(cell.T, dr)
        frac -= np.round(frac) * pbc.astype(float)
        dr = cell.T @ frac
    return float(np.linalg.norm(dr))


def label_frame(
    atoms: Atoms,
    moving_atom_idx: int,
    coord_site_a: np.ndarray,
    coord_site_b: np.ndarray,
    tol: float,
) -> Dict[str, Any]:
    pos = atoms.get_positions()
    r = pos[moving_atom_idx]
    cell = np.array(atoms.cell)
    pbc = np.array(atoms.pbc, dtype=bool)

    d_to_a = mic_distance(cell, pbc, r, coord_site_a)
    d_to_b = mic_distance(cell, pbc, r, coord_site_b)

    if d_to_a <= tol and d_to_b > tol:
        basin_label = "near_a"
    elif d_to_b <= tol and d_to_a > tol:
        basin_label = "near_b"
    elif d_to_a <= tol and d_to_b <= tol:
        basin_label = "near_both"
    else:
        basin_label = "other"

    nearest = "a" if d_to_a <= d_to_b else "b"

    return {
        "d_to_a_A": d_to_a,
        "d_to_b_A": d_to_b,
        "nearest_site": nearest,
        "basin_label": basin_label,
    }


def choose_tail_window(nframes: int, tail_window: int, min_tail_window: int) -> int:
    if nframes <= 0:
        return 0
    if nframes < min_tail_window:
        return nframes
    return min(max(min_tail_window, tail_window), nframes)


def classify_from_tail(labels: List[str]) -> Tuple[str, str]:
    cnt = Counter(labels)
    major_label, major_count = cnt.most_common(1)[0]
    frac = major_count / len(labels)

    if major_label == "near_b" and frac >= 0.8:
        return "same_basin", "tail frames stabilized near original site_b"
    if major_label == "near_a" and frac >= 0.8:
        return "different_basin", "tail frames stabilized near target site_a"
    if major_label == "other" and frac >= 0.8:
        return "other_basin", "tail frames stabilized away from both site_a and site_b"

    if major_label == "near_b" and frac >= 0.6:
        return "likely_same_basin", "tail frames mostly near original site_b"
    if major_label == "near_a" and frac >= 0.6:
        return "likely_different_basin", "tail frames mostly near target site_a"
    if major_label == "other" and frac >= 0.6:
        return "likely_other_basin", "tail frames mostly away from both site_a and site_b"

    return "ambiguous", "tail frames do not stabilize to a single basin label"


def first_stable_switch(labels: List[str], target: str, consecutive: int = 3) -> int | None:
    if consecutive <= 1:
        for i, x in enumerate(labels):
            if x == target:
                return i
        return None

    run = 0
    for i, x in enumerate(labels):
        if x == target:
            run += 1
            if run >= consecutive:
                return i - consecutive + 1
        else:
            run = 0
    return None


def compare_final_frame_to_final_data(
    final_frame: Atoms,
    final_data: Atoms,
    moving_atom_idx: int,
) -> Dict[str, Any]:
    if len(final_frame) != len(final_data):
        return {
            "checked": True,
            "consistent": False,
            "reason": f"natoms mismatch: traj_final={len(final_frame)} final_data={len(final_data)}",
        }

    pos1 = final_frame.get_positions()
    pos2 = final_data.get_positions()

    dr_all = np.linalg.norm(pos1 - pos2, axis=1)
    dr_move = float(dr_all[moving_atom_idx])

    return {
        "checked": True,
        "consistent": True,
        "max_abs_cart_diff_A": float(dr_all.max()),
        "mean_abs_cart_diff_A": float(dr_all.mean()),
        "moving_atom_abs_cart_diff_A": dr_move,
    }


def main() -> None:
    args = parse_args()
    job_dir = args.job_dir.resolve()

    meta_path = job_dir / "meta" / "meta.json"
    default_traj = job_dir / "dumps" / "relax_final" / "traj.lammpstrj"
    default_final_data = job_dir / "results" / "final_relaxed.data"
    traj_path = args.traj_path.resolve() if args.traj_path else default_traj

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta file: {meta_path}")
    if not traj_path.exists():
        raise FileNotFoundError(f"Missing trajectory file: {traj_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    job_info = meta["job_info"]

    moving_atom_idx = int(job_info["moving_atom_index_0based"])
    coord_site_a, coord_site_b, coord_sources = resolve_site_coords(meta)

    frames = read_lammpstrj_all(traj_path)
    if len(frames) == 0:
        raise ValueError(f"No frames read from trajectory: {traj_path}")

    rows: List[Dict[str, Any]] = []
    for iframe, atoms in enumerate(frames):
        if moving_atom_idx >= len(atoms):
            raise IndexError(
                f"moving_atom_idx={moving_atom_idx} out of range for frame {iframe} "
                f"with natoms={len(atoms)}"
            )
        lab = label_frame(
            atoms=atoms,
            moving_atom_idx=moving_atom_idx,
            coord_site_a=coord_site_a,
            coord_site_b=coord_site_b,
            tol=args.tol,
        )
        rows.append({
            "frame": iframe,
            **lab,
        })

    df = pd.DataFrame(rows)
    nframes = len(df)
    tail_n = choose_tail_window(nframes, args.tail_window, args.min_tail_window)
    tail_df = df.tail(tail_n).copy()

    classification, reason = classify_from_tail(tail_df["basin_label"].tolist())

    switch_to_a = first_stable_switch(df["basin_label"].tolist(), "near_a", consecutive=3)
    switch_to_b = first_stable_switch(df["basin_label"].tolist(), "near_b", consecutive=3)

    final_data_check = {"checked": False}
    if args.check_final_data and default_final_data.exists():
        final_frame = frames[-1]
        final_data_atoms = read_lammps_atomic_data(default_final_data)
        final_data_check = compare_final_frame_to_final_data(
            final_frame=final_frame,
            final_data=final_data_atoms,
            moving_atom_idx=moving_atom_idx,
        )

    summary = {
        "job_dir": str(job_dir),
        "traj_path": str(traj_path),
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
        "nframes": nframes,
        "tail_window_used": tail_n,
        "classification": classification,
        "reason": reason,
        "tail_label_counts": dict(Counter(tail_df["basin_label"].tolist())),
        "first_stable_switch_to_near_a_frame": switch_to_a,
        "first_stable_switch_to_near_b_frame": switch_to_b,
        "final_frame": {
            "d_to_a_A": float(df.iloc[-1]["d_to_a_A"]),
            "d_to_b_A": float(df.iloc[-1]["d_to_b_A"]),
            "nearest_site": str(df.iloc[-1]["nearest_site"]),
            "basin_label": str(df.iloc[-1]["basin_label"]),
        },
        "final_data_check": final_data_check,
    }

    csv_path = job_dir / "meta" / "basin_trace.csv"
    json_path = job_dir / "meta" / "basin_trace_summary.json"

    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[TRAJ BASIN CHECK] {job_dir.name}")
    print(f"  nframes              : {nframes}")
    print(f"  tail_window          : {tail_n}")
    print(f"  classification       : {classification}")
    print(f"  reason               : {reason}")
    print(f"  site_a_coord source  : {coord_sources['site_a_coord_source']}")
    print(f"  site_b_coord source  : {coord_sources['site_b_coord_source']}")
    print(f"  final d->a           : {df.iloc[-1]['d_to_a_A']:.6f} Å")
    print(f"  final d->b           : {df.iloc[-1]['d_to_b_A']:.6f} Å")
    if final_data_check.get("checked"):
        print(f"  final-data checked   : True")
        if final_data_check.get("consistent", False):
            print(f"  max |traj-data| diff : {final_data_check['max_abs_cart_diff_A']:.6e} Å")
            print(f"  move atom diff       : {final_data_check['moving_atom_abs_cart_diff_A']:.6e} Å")
        else:
            print(f"  final-data status    : {final_data_check.get('reason', 'inconsistent')}")
    print(f"  saved trace          : {csv_path}")
    print(f"  saved summary        : {json_path}")


if __name__ == "__main__":
    main()