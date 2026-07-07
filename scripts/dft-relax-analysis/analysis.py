import os, sys, csv, math, glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from ase.io import read, iread
from ase import Atoms

import argparse, json

# ---- pairing helpers ----

def _has_vasp_frames(sys_dir: Path) -> bool:
    return (sys_dir / "vasprun.xml").is_file() or (sys_dir / "OUTCAR").is_file()

def find_matching_systems(vasp_dir: str, cif_dir: str):
    """
    Returns:
      pairs: dict {system_id: {"vasp_dir": <abs path to sys dir>, "cif": <abs path to cif>}}
      missing_vasp: set of sids present as CIF but no VASP
      missing_cif : set of sids present as VASP but no CIF
    """
    vasp_dir = Path(vasp_dir).resolve()
    cif_dir  = Path(cif_dir).resolve()

    # VASP systems: subdirectories that contain vasprun.xml or OUTCAR
    vasp_sys = {}
    for p in vasp_dir.iterdir():
        if p.is_dir() and _has_vasp_frames(p):
            vasp_sys[p.name] = {"vasp_dir": str(p)}

    # CIF systems: files named {system_id}.cif
    cif_sys = {}
    for cif in cif_dir.glob("*.cif"):
        sid = cif.stem  # basename without .cif
        cif_sys[sid] = {"cif": str(cif)}

    sids_vasp = set(vasp_sys.keys())
    sids_cif  = set(cif_sys.keys())
    common    = sorted(sids_vasp & sids_cif)

    pairs = {}
    for sid in common:
        pairs[sid] = {"vasp_dir": vasp_sys[sid]["vasp_dir"], "cif": cif_sys[sid]["cif"]}

    return pairs, (sids_cif - sids_vasp), (sids_vasp - sids_cif)

# -----------------------------
# Utilities: PBC wrap + metrics
# -----------------------------
def frac_wrap(ds: np.ndarray) -> np.ndarray:
    """Wrap fractional deltas to [-0.5,0.5) per component."""
    return ds - np.round(ds)

def metrics(frame: Atoms, ref: Atoms, perm: Optional[np.ndarray] = None, cell_mode: str = "ref"):
    """
    MAE/RMSD under PBC. cell_mode:
      - 'ref'   : map delta with reference cell (default; good if cell fixed)
      - 'frame' : compare in the instantaneous frame cell (good for ISIF=3)
    """
    if perm is not None:
        frame = frame[perm]

    C_ref = ref.cell.array
    C_frm = frame.cell.array

    if cell_mode == "ref":
        s_ref = ref.get_scaled_positions()
        s_frm = frame.get_scaled_positions()
        ds = frac_wrap(s_frm - s_ref)
        dcart = ds @ C_ref
    elif cell_mode == "frame":
        s_ref_ref = ref.get_scaled_positions()
        r_ref_cart = s_ref_ref @ C_ref
        s_ref_in_frm = r_ref_cart @ np.linalg.inv(C_frm)
        s_frm = frame.get_scaled_positions()
        ds = frac_wrap(s_frm - s_ref_in_frm)
        dcart = ds @ C_frm
    else:
        raise ValueError("cell_mode must be 'ref' or 'frame'")

    mae_xyz = np.mean(np.abs(dcart), axis=0)
    mae_all = float(np.mean(np.abs(dcart)))
    rmsd    = float(np.sqrt(np.mean(np.sum(dcart**2, axis=1))))
    return mae_xyz, mae_all, rmsd

# If you already have a species-based Hungarian mapping, plug it here.
def hungarian_perm_by_species(ref: Atoms, frame0: Atoms) -> Optional[np.ndarray]:
    # Placeholder: identity. Replace with your existing function if needed.
    return None

# --------------
# VASP utilities
# --------------

def find_vasp_source(dir_path: str) -> Optional[str]:
    """Prefer vasprun.xml; fall back to OUTCAR."""
    for name in ("vasprun.xml", "OUTCAR"):
        p = os.path.join(dir_path, name)
        if os.path.isfile(p):
            return p
    return None

def stream_vasp_frames(src: str):
    """Yield all ionic steps from a VASP result."""
    # ase.io.read(..., index=':') streams; iread also works.
    # Prefer using read with ':' to ensure calculators carry over.
    for atoms in read(src, index=":"):
        yield atoms

def frame_fmax_or_inf(at: Atoms) -> float:
    """Return max |F| (eV/Å) for a VASP frame, or +inf if forces unavailable."""
    try:
        F = at.get_forces(apply_constraint=False)  # VASP forces are eV/Å
        return float(np.linalg.norm(F, axis=1).max())
    except Exception:
        return float("inf")

def write_metrics_csv(path: str, rows: List[Tuple]):
    """
    rows of (frame, MAE_x, MAE_y, MAE_z, MAE_all, RMSD, Fmax)
    Writes Fmax blank if not finite.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame","MAE_x(Å)","MAE_y(Å)","MAE_z(Å)",
                    "MAE_all(Å)","RMSD(Å)","Fmax(eV/Å)"])
        for i, mx, my, mz, mall, rmsd, fmax in rows:
            w.writerow([
                i, f"{mx:.6f}", f"{my:.6f}", f"{mz:.6f}",
                f"{mall:.6f}", f"{rmsd:.6f}",
                "" if not np.isfinite(fmax) else f"{fmax:.6f}"
            ])

def truncate_index_by_fmax(fmax_list, mode="global_fmin", tol=0.0, patience=0):
    """
    Return last frame index to KEEP (inclusive).
      - 'global_fmin': keep up to first global minimum of finite Fmax
      - 'no_uprise' : keep while Fmax <= running_min + tol (allow `patience` ups)
    """
    f = np.asarray(fmax_list, float)
    finite = np.isfinite(f)
    if not finite.any():
        return len(f) - 1
    if mode == "global_fmin":
        j = np.where(finite, f, np.inf).argmin()
        return int(j)
    elif mode == "no_uprise":
        i0 = int(np.flatnonzero(finite)[0])
        run_min = f[i0]; strikes = 0
        for i in range(i0 + 1, len(f)):
            if not np.isfinite(f[i]):  # ignore infs
                continue
            if f[i] <= run_min + tol:
                run_min = min(run_min, f[i])
                strikes = 0
            else:
                strikes += 1
                if strikes > patience:
                    return i - 1
        return len(f) - 1
    else:
        raise ValueError("mode must be 'global_fmin' or 'no_uprise'")

# ------------------------------------
# Per-system processing (VASP version)
# ------------------------------------
def process_vasp_system(system_id: str,
                        vasp_dir: str,
                        ref_cif_path: str,
                        outdir_sys: str,
                        outdir_plot: str,
                        map_mode: str = "id",
                        truncate: bool = True,
                        truncate_mode: str = "global_fmin",
                        tol: float = 1e-4,
                        patience: int = 0,
                        cell_mode: str = "ref"):
    """
    Mirrors your LAMMPS analysis, but reading VASP outputs.
    Returns (csv_path, png_path, last_row_tuple).
    """
    os.makedirs(outdir_sys, exist_ok=True)
    os.makedirs(outdir_plot, exist_ok=True)

    src = find_vasp_source(os.path.join(vasp_dir, system_id))
    if src is None:
        print(f"[WARN] {system_id}: no vasprun.xml/OUTCAR found under {vasp_dir}", file=sys.stderr)
        return None

    ref = read(ref_cif_path); ref.pbc = True

    # Stream frames
    frames_iter = stream_vasp_frames(src)
    try:
        first = next(frames_iter)
    except StopIteration:
        print(f"[WARN] {system_id}: empty trajectory in {src}", file=sys.stderr)
        return None

    if len(first) != len(ref):
        print(f"[WARN] {system_id}: atom count mismatch – traj {len(first)} vs ref {len(ref)}", file=sys.stderr)
        return None

    perm = hungarian_perm_by_species(ref, first) if map_mode == "species" else None

    # Collect metrics
    frames_metrics = []  # (i, MAE_x, MAE_y, MAE_z, MAE_all, RMSD, Fmax)
    i = 0
    mae_xyz, mae_all, rmsd = metrics(first, ref, perm, cell_mode=cell_mode)
    fmax0 = frame_fmax_or_inf(first)
    frames_metrics.append((i, *mae_xyz, mae_all, rmsd, fmax0))

    for fr in frames_iter:
        i += 1
        mae_xyz, mae_all, rmsd = metrics(fr, ref, perm, cell_mode=cell_mode)
        fmax = frame_fmax_or_inf(fr)
        frames_metrics.append((i, *mae_xyz, mae_all, rmsd, fmax))

    # Truncation
    fseries = [r[6] for r in frames_metrics]
    fmaxFrame = truncate_index_by_fmax(fseries, mode=truncate_mode, tol=tol, patience=patience)
    cut = fmaxFrame if truncate else (len(frames_metrics) - 1)
    truncated = frames_metrics[:cut+1]

    # CSV (choose full or truncated by flag; or write both if you prefer)
    csv_path = os.path.join(outdir_sys, f"{system_id}_mae_{'truncated' if truncate else 'full'}.csv")
    write_metrics_csv(csv_path, truncated if truncate else frames_metrics)

    # Per-system plot (MAE) with Fmax-min marker
    plt.figure()
    ax = plt.gca()
    xs = [r[0] for r in truncated]
    ys = [r[4] for r in truncated]  # MAE_all
    ax.plot(xs, ys, label="MAE_all" + (" (truncated)" if truncate else ""))
    # Label Fmax minimum
    farr = np.asarray(fseries, float)
    finite = np.isfinite(farr)
    fmin_val = float(np.nanmin(farr[finite])) if finite.any() else float("nan")
    if truncated:
        cut_x = truncated[-1][0]
        ax.axvline(cut_x, linestyle="--", linewidth=1, label=f"cut@{cut_x}")
        ymax = ax.get_ylim()[1]
        ax.annotate(f"Fmax min = {fmin_val:.3f} eV/Å",
                    xy=(cut_x, ymax), xytext=(5, -5),
                    textcoords="offset points", rotation=90,
                    va="top", ha="left", fontsize=8)
    ax.set_xlabel("Relaxation step (ionic iteration)")
    ax.set_ylabel("MAE to reference CIF (Å)")
    ax.set_title(f"MAE vs step — {system_id} (rule: {truncate_mode}, tol={tol}, patience={patience})")
    ax.legend()
    plt.tight_layout()
    png_path = os.path.join(outdir_plot, f"{system_id}_mae_{'truncated' if truncate else 'full'}.png")
    plt.savefig(png_path, dpi=150)
    plt.close()

    return csv_path, png_path, (truncated[-1] if truncated else frames_metrics[-1])

# ---- thin wrapper around your per-system VASP routine ----

def process_vasp_system_by_paths(system_id: str,
                                 vasp_sys_dir: str,  # this is {vasp_dir}/{system_id}
                                 ref_cif_path: str,
                                 outdir_sys: str,
                                 outdir_plot: str,
                                 **kwargs):
    """
    Calls your process_vasp_system(...) but with explicit sys dir & cif path.
    """
    # Your earlier function expects (sid, vasp_root, ref_cif_path, ...)
    vasp_root = str(Path(vasp_sys_dir).parent)
    return process_vasp_system(system_id, vasp_root, ref_cif_path,
                               outdir_sys, outdir_plot, **kwargs)

# ------------------------------------
# Aggregate plot across many systems
# ------------------------------------

def aggregate_plot(series: Dict[str, Tuple[List[int], List[float]]],
                   cuts_by_sid: Optional[Dict[str, int]],
                   out_png: str,
                   show_cuts: bool = True):
    """
    series: {system_id: (frames, mae_all_values)}
    cuts_by_sid: optional {system_id: cut_frame}
    """
    plt.figure()
    ax = plt.gca()
    for sid, (xs, ys) in sorted(series.items()):
        ax.plot(xs, ys, linewidth=1.0)
    if show_cuts and cuts_by_sid:
        cut_steps = [cuts_by_sid[sid] for sid in series.keys() if sid in cuts_by_sid]
        for cx in cut_steps:
            ax.axvline(cx, color="k", alpha=0.08, linewidth=0.8)
        if cut_steps:
            med = int(np.median(cut_steps))
            ax.axvline(med, linestyle="--", linewidth=1.2)
            ymax = ax.get_ylim()[1]
            ax.annotate(f"median cut = {med}", xy=(med, ymax),
                        xytext=(5, -5), textcoords="offset points",
                        rotation=90, va="top", ha="left", fontsize=8)
    ax.set_xlabel("Relaxation step (ionic iteration)")
    ax.set_ylabel("MAE to reference CIF (Å)")
    ax.set_title("MAE vs step — all systems")
    # no legend (keeps clean). If needed, place outside:
    # ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()

# -------------
# Batch driver
# -------------

def process_all_vasp_pairs(vasp_dir: str, cif_dir: str,
                           out_csv_dir: str, out_plot_dir: str,
                           truncate: bool = True,
                           truncate_mode: str = "global_fmin",
                           tol: float = 1e-4,
                           patience: int = 0,
                           cell_mode: str = "ref"):
    """
    Scans both roots, ignores unmatched, runs per-system analysis, then makes the aggregate plot.
    """
    pairs, miss_vasp, miss_cif = find_matching_systems(vasp_dir, cif_dir)
    if miss_vasp:
        print(f"[INFO] {len(miss_vasp)} CIF(s) without VASP results will be skipped.")
    if miss_cif:
        print(f"[INFO] {len(miss_cif)} VASP system(s) without CIF will be skipped.")

    series   = {}
    cuts_map = {}

    for sid, paths in pairs.items():
        res = process_vasp_system_by_paths(
            sid, paths["vasp_dir"], paths["cif"],
            out_csv_dir, out_plot_dir,
            map_mode="id",
            truncate=truncate,
            truncate_mode=truncate_mode,
            tol=tol, patience=patience,
            cell_mode=cell_mode
        )
        if not res:
            continue
        csv_path, png_path, _ = res

        # load the just-written CSV to feed the aggregate plot
        rows = list(csv.reader(open(csv_path)))[1:]  # skip header
        xs = [int(r[0]) for r in rows]
        ys = [float(r[4]) for r in rows]  # MAE_all
        series[sid] = (xs, ys)
        if truncate:
            cuts_map[sid] = xs[-1]

    # aggregate plot (no legend; optional cut markers)
    agg_png = os.path.join(out_plot_dir, "all_systems_mae.png")
    aggregate_plot(series, cuts_map if truncate else None, agg_png, show_cuts=truncate)

    return series, cuts_map, pairs

def main():
    ap = argparse.ArgumentParser(description="Compute MAE/RMSD vs CIF for many LAMMPS dumps and plot MAE vs step.")
    ap.add_argument("--config", type=str, help="Path to the configuration JSON file.")
    ap.add_argument("--map", choices=["id","species"], default="id",
                    help="id: assumes 'dump_modify ... sort id'; species: per-element Hungarian match")
    args = ap.parse_args()
    
    config_path = args.config
    config = []
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"Loaded configuration from {config_path}")
    
    for sysCfg in config:
        vasp_dir = sysCfg["vasp_dir"]
        cif_dir  = sysCfg["cif_dir"]

        series, cuts, pairs = process_all_vasp_pairs(
            vasp_dir, cif_dir,
            out_csv_dir=sysCfg["out_csv_dir"],
            out_plot_dir=sysCfg["out_plot_dir"],
            truncate=sysCfg["truncate"],                 # False = plot full trajectories
            truncate_mode=sysCfg["truncate_mode"],   # or "no_uprise"
            tol=sysCfg["tol"], patience=sysCfg["patience"]
        )
