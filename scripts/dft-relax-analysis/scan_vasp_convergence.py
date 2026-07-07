#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, sys, re, warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import pandas as pd

# ---- optional deps detection -------------------------------------------------
HAS_PMG = True
try:
    from pymatgen.io.vasp.outputs import Vasprun
except Exception:
    HAS_PMG = False

HAS_ASE = True
try:
    from ase.io import read
except Exception:
    HAS_ASE = False

RE_OUTCAR_IONIC_CONV = re.compile(r"reached required accuracy\s*-\s*stopping structural energy minimisation", re.I)

def find_source(sys_dir: Path) -> Tuple[Optional[Path], Optional[str]]:
    """Return (path, 'vasprun'|'outcar'|None). Prefer vasprun.xml."""
    p = sys_dir / "vasprun.xml"
    if p.is_file():
        return p, "vasprun"
    p = sys_dir / "OUTCAR"
    if p.is_file():
        return p, "outcar"
    return None, None

def count_frames_with_ase(src: Path) -> Optional[int]:
    """Count ionic steps using ASE (works for vasprun.xml and OUTCAR)."""
    if not HAS_ASE:
        return None
    try:
        # Stream generator; don't load all into memory
        n = 0
        for _ in read(str(src), index=":"):
            n += 1
        return n
    except Exception:
        return None

def parse_convergence_vasprun(src: Path) -> Dict[str, Optional[bool]]:
    """Use pymatgen to read convergence flags from vasprun.xml if available."""
    if not HAS_PMG:
        return {"converged_ionic": None, "converged_electronic": None, "converged": None}
    try:
        vr = Vasprun(
            filename=str(src),
            parse_projected_eigen=False,
            parse_dos=False,
            exception_on_bad_xml=False,
        )
        return {
            "converged_ionic": bool(getattr(vr, "converged_ionic", None)),
            "converged_electronic": bool(getattr(vr, "converged_electronic", None)),
            "converged": bool(getattr(vr, "converged", None)),
        }
    except Exception:
        return {"converged_ionic": None, "converged_electronic": None, "converged": None}

def parse_convergence_outcar(src: Path) -> Dict[str, Optional[bool]]:
    """Heuristic ionic convergence from OUTCAR by marker string."""
    try:
        text = src.read_text(errors="ignore")
    except Exception:
        return {"converged_ionic": None, "converged_electronic": None, "converged": None}
    ionic = bool(RE_OUTCAR_IONIC_CONV.search(text))
    # OUTCAR alone cannot reliably tell electronic convergence across all VASP versions
    return {"converged_ionic": ionic, "converged_electronic": None, "converged": ionic}

def scan_system(sys_dir: Path) -> Dict[str, Any]:
    cif_id = sys_dir.name
    src, kind = find_source(sys_dir)
    row: Dict[str, Any] = {
        "cif_id": cif_id,
        "file_used": None,
        "n_frames": None,
        "converged_ionic": None,
        "converged_electronic": None,
        "converged": None,
        "error": None,
    }
    if src is None:
        row["error"] = "missing vasprun.xml and OUTCAR"
        return row

    row["file_used"] = src.name

    # frames
    n_frames = count_frames_with_ase(src)
    row["n_frames"] = n_frames

    # convergence
    if kind == "vasprun":
        conv = parse_convergence_vasprun(src)
    else:
        conv = parse_convergence_outcar(src)
    row.update(conv)

    return row

def main():
    ap = argparse.ArgumentParser(
        description="Scan VASP results under DFT_DIR/{cif_id}/(vasprun.xml|OUTCAR) "
                    "and write per-system convergence stats to CSV."
    )
    ap.add_argument("--dft_dir", required=True, help="Root directory containing per-system subfolders")
    ap.add_argument("--out_csv", required=True, help="Output CSV path")
    ap.add_argument("--warn_missing", action="store_true", help="Print warnings for missing/failed systems")
    args = ap.parse_args()

    root = Path(args.dft_dir)
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory", file=sys.stderr)
        sys.exit(2)

    if not HAS_ASE:
        warnings.warn("ASE not found; n_frames will be None. Install with: pip install ase")
    if not HAS_PMG:
        warnings.warn("pymatgen not found; vasprun convergence flags will be None. Install with: pip install pymatgen")

    rows = []
    for sys_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        r = scan_system(sys_dir)
        rows.append(r)
        if args.warn_missing and r["error"]:
            warnings.warn(f"{sys_dir.name}: {r['error']}")

    df = pd.DataFrame(rows)

    # summary row
    total_systems = len(df)
    total_frames = df["n_frames"].fillna(0).sum()
    n_conv_ionic = int(df["converged_ionic"].fillna(False).sum())
    n_conv_overall = int(df["converged"].fillna(False).sum())
    n_with_files = int(df["file_used"].notna().sum())

    summary = pd.DataFrame([{
        "cif_id": "__TOTAL__",
        "file_used": f"{n_with_files}/{total_systems} have files",
        "n_frames": int(total_frames),
        "converged_ionic": n_conv_ionic,
        "converged_electronic": None,  # not aggregated meaningfully
        "converged": n_conv_overall,
        "error": int(df["error"].notna().sum()),
    }])

    out = pd.concat([df, summary], ignore_index=True)
    out.to_csv(args.out_csv, index=False)
    print(f"[OK] wrote {args.out_csv}")
    print(f"Systems: {total_systems} | with files: {n_with_files} | total frames: {int(total_frames)}")
    print(f"Ionic-converged systems: {n_conv_ionic} | Overall-converged: {n_conv_overall}")

if __name__ == "__main__":
    main()
