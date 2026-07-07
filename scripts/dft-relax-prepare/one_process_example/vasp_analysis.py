#!/usr/bin/env python3
"""
Compute MAE/RMSD vs reference CIF for VASP relaxations, from XDATCAR trajectories.

Layout (yours):
  dumps/{systemName}_{systemID}_unrelaxed/traj.lammpstrj      # (old MACE path, not used here)
  inputs/{inputName}_{systemID}_relaxed.cif                   # reference CIFs you already have
  vasp/<something_with_{systemID}>/XDATCAR                    # VASP relaxation outputs

We pair by {systemID}. If your VASP dirs differ, use --xdatcar-glob/--sid-regex to match.
"""
import argparse, os, re, csv, glob, sys
import numpy as np
from collections import defaultdict
from ase.io import iread, read
import matplotlib.pyplot as plt

try:
    from scipy.optimize import linear_sum_assignment
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# Default patterns:
#   - VASP dirs like: vasp/<any>_{SID}_<any>/XDATCAR (SID often looks like UUID or token)
#   - CIF dirs like: 
UUID_TOKEN = r"[0-9A-Fa-f-]{8,36}"
DEFAULT_SID_REGEX = rf".*_(?P<sid>{UUID_TOKEN})(?:_[^/]*)?$"
# CIF_SID_REGEX = re.compile(rf"^(?P<name>.+)_(?P<sid>{UUID_TOKEN})_relaxed\.cif$", re.IGNORECASE)
CIF_SID_REGEX = re.compile(rf"^(?P<name>[^_]+(?:_[^_]+)?)_(?P<sid>{UUID_TOKEN})_relaxed\.cif$", re.IGNORECASE)

def find_pairs(xdatcar_glob, cifs_root, sid_regex):
    sid_re = re.compile(sid_regex)
    # Map SID -> CIF
    cif_map = {}
    for cf in glob.glob(os.path.join(cifs_root, "*_relaxed.cif")):
        m = CIF_SID_REGEX.match(os.path.basename(cf))
        if m:
            cif_map[m["sid"]] = cf
    # Collect XDATCARs and pair by SID found in parent dir name
    pairs = []
    for xd in glob.glob(xdatcar_glob, recursive=True):
        parent = os.path.basename(os.path.dirname(xd))
        m = sid_re.match(parent)
        if not m:
            print(f"[WARN] skip (no SID in dir): {xd}", file=sys.stderr)
            continue
        sid = m["sid"]
        cf = cif_map.get(sid)
        if not cf:
            print(f"[WARN] no CIF for SID={sid} ({xd})", file=sys.stderr)
            continue
        pairs.append((sid, xd, cf))
    return sorted(pairs, key=lambda t: t[0])

def frac_wrap(d):
    return d - np.round(d)

def build_perm_species(ref, frame):
    """One-time per-element Hungarian match from frame -> ref order."""
    if not HAVE_SCIPY:
        raise RuntimeError("SciPy not installed; cannot use --map species.")
    zref, zfrm = ref.numbers, frame.numbers
    if sorted(zref) != sorted(zfrm):
        raise ValueError("Element counts differ between reference and frame.")
    groups_ref, groups_frm = defaultdict(list), defaultdict(list)
    for i,z in enumerate(zref): groups_ref[z].append(i)
    for j,z in enumerate(zfrm): groups_frm[z].append(j)
    perm = np.empty(len(ref), int)
    # Use the *frame* cell to define minimum-image distances for mapping
    s_ref = ref.get_scaled_positions()
    s_frm = frame.get_scaled_positions()
    C_frm = frame.cell.array
    for z in sorted(groups_ref):
        I = np.array(groups_ref[z]); J = np.array(groups_frm[z])
        ds = s_ref[I][:,None,:] - s_frm[J][None,:,:]
        ds = frac_wrap(ds)
        dcart = ds @ C_frm
        r, c = linear_sum_assignment(np.linalg.norm(dcart, axis=2))
        perm[I[r]] = J[c]
    return perm

def metrics(frame, ref, perm=None, cell_mode="frame"):
    """
    MAE_x, MAE_y, MAE_z, MAE_all, RMSD (Å).
    cell_mode:
      'frame'      -> embed reference positions in the *frame* cell (works if cell changed)
      'assume_fix' -> assume same cell; just use fractional deltas in ref cell
    """
    if perm is not None:
        frame = frame[perm]

    s_frm = frame.get_scaled_positions()
    C_frm = frame.cell.array
    s_ref = ref.get_scaled_positions()
    C_ref = ref.cell.array

    if cell_mode == "frame":
        # Convert ref Cartesian positions into fractional coords of the *frame* cell:
        pos_ref_cart = s_ref @ C_ref
        s_ref_in_frame = pos_ref_cart @ np.linalg.inv(C_frm)
        ds = frac_wrap(s_frm - s_ref_in_frame)
        dcart = ds @ C_frm
    else:
        # Assume identical cell; compare in ref cell
        ds = frac_wrap(s_frm - s_ref)
        dcart = ds @ C_ref

    mae_xyz = np.mean(np.abs(dcart), axis=0)
    mae_all = float(np.mean(np.abs(dcart)))
    rmsd = float(np.sqrt(np.mean(np.sum(dcart**2, axis=1))))
    return mae_xyz, mae_all, rmsd

def process_one(sid, xdatcar_path, cif_path, outdir, map_mode, cell_mode):
    os.makedirs(outdir, exist_ok=True)
    # Read reference CIF
    ref = read(cif_path)
    ref.pbc = True

    # Stream XDATCAR frames (ionic steps) — VASP stores trajectory here
    # (Multiple snapshots; iread returns an iterator over Atoms objects.)
    frames = iread(xdatcar_path, format="vasp-xdatcar", index=":")
    # Pull the first frame to set mapping and count
    try:
        first = next(frames)
    except StopIteration:
        print(f"[WARN] empty XDATCAR: {xdatcar_path}", file=sys.stderr)
        return None

    if len(first) != len(ref):
        print(f"[WARN] atom count mismatch for {sid}: XDATCAR {len(first)} vs CIF {len(ref)}", file=sys.stderr)
        return None

    perm = None
    if map_mode == "species":
        perm = build_perm_species(ref, first)

    rows = []
    i = 0
    mae_xyz, mae_all, rmsd = metrics(first, ref, perm, cell_mode=cell_mode)
    rows.append((i, *mae_xyz, mae_all, rmsd))
    for fr in frames:
        i += 1
        mae_xyz, mae_all, rmsd = metrics(fr, ref, perm, cell_mode=cell_mode)
        rows.append((i, *mae_xyz, mae_all, rmsd))

    # CSV per system
    per_csv = os.path.join(outdir, f"{sid}_mae.csv")
    with open(per_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame","MAE_x(Å)","MAE_y(Å)","MAE_z(Å)","MAE_all(Å)","RMSD(Å)"])
        for r in rows:
            w.writerow([r[0]] + [f"{x:.6f}" for x in r[1:]])

    # Plot per system (MAE_all vs ionic step)
    xs = [r[0] for r in rows]
    ys = [r[4] for r in rows]
    plt.figure()
    plt.plot(xs, ys)
    plt.xlabel("Ionic step (from XDATCAR)")
    plt.ylabel("MAE to reference CIF (Å)")
    plt.title(f"MAE vs step — {sid}")
    plt.tight_layout()
    per_png = os.path.join(outdir, f"{sid}_mae.png")
    plt.savefig(per_png, dpi=150)
    plt.close()

    return per_csv, per_png, rows[-1]

def main():
    ap = argparse.ArgumentParser(description="MAE/RMSD vs CIF from VASP XDATCAR trajectories (per-step).")
    ap.add_argument("--xdatcar_glob", default="vasp/*/XDATCAR",
                    help="Glob to find XDATCAR files (default: vasp/*/XDATCAR)")
    ap.add_argument("--cifs_root", default="inputs",
                    help="Folder containing {inputName}_{systemID}_relaxed.cif")
    ap.add_argument("--out_root", default="mae_vasp_results")
    ap.add_argument("--sid_regex", default=DEFAULT_SID_REGEX,
                    help="Regex (with named group 'sid') to extract systemID from XDATCAR parent dir name")
    ap.add_argument("--map", choices=["id","species"], default="id",
                    help="Atom matching: 'id' (assumes consistent order); 'species' uses Hungarian per element")
    ap.add_argument("--cell_mode", choices=["frame","assume_fix"], default="frame",
                    help="'frame': compare in each frame's cell (works for variable cell); "
                         "'assume_fix': assumes identical cell as reference.")
    args = ap.parse_args()

    pairs = find_pairs(args.xdatcar_glob, args.cifs_root, args.sid_regex)
    if not pairs:
        print("No (XDATCAR, CIF) pairs found. Check --xdatcar-glob/--cifs-root/--sid-regex.", file=sys.stderr)
        sys.exit(1)

    per_dir = os.path.join(args.out_root, "per_system")
    agg_dir = os.path.join(args.out_root, "aggregate")
    os.makedirs(per_dir, exist_ok=True)
    os.makedirs(agg_dir, exist_ok=True)

    long_rows, final_rows = [], []
    for sid, xd, cf in pairs:
        print(f"[{sid}] {xd}  <->  {cf}")
        res = process_one(sid, xd, cf, per_dir, args.map, args.cell_mode)
        if res is None:
            continue
        per_csv, per_png, last = res

        # Append to long-form table
        with open(per_csv, "r") as f:
            for row in csv.DictReader(f):
                long_rows.append([sid, int(row["frame"]),
                                  float(row["MAE_x(Å)"]), float(row["MAE_y(Å)"]), float(row["MAE_z(Å)"]),
                                  float(row["MAE_all(Å)"]), float(row["RMSD(Å)"])])

        final_rows.append([sid,
                           os.path.relpath(per_csv, args.out_root),
                           os.path.relpath(per_png, args.out_root),
                           f"{last[1]:.6f}", f"{last[2]:.6f}", f"{last[3]:.6f}",
                           f"{last[4]:.6f}", f"{last[5]:.6f}"])

    # Aggregate CSVs
    long_csv = os.path.join(agg_dir, "all_systems_mae_long.csv")
    with open(long_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["systemID","frame","MAE_x(Å)","MAE_y(Å)","MAE_z(Å)","MAE_all(Å)","RMSD(Å)"])
        for row in sorted(long_rows, key=lambda x: (x[0], x[1])):
            w.writerow(row)

    summary_csv = os.path.join(agg_dir, "summary_final_frame.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["systemID","per_system_csv","per_system_png",
                    "final_MAE_x(Å)","final_MAE_y(Å)","final_MAE_z(Å)","final_MAE_all(Å)","final_RMSD(Å)"])
        for row in sorted(final_rows, key=lambda x: x[0]):
            w.writerow(row)

    # Aggregate plot: MAE_all vs step for every system
    series = {}
    for sid, frame, mx, my, mz, mae_all, rmsd in long_rows:
        series.setdefault(sid, ([], []))
        series[sid][0].append(frame); series[sid][1].append(mae_all)
    plt.figure()
    for sid, (xs, ys) in sorted(series.items()):
        plt.plot(xs, ys, label=sid, linewidth=1.0)
    plt.xlabel("Ionic step (from XDATCAR)")
    plt.ylabel("MAE to reference CIF (Å)")
    plt.title("MAE vs step — all systems")
    plt.legend(fontsize="xx-small", ncol=2)
    plt.tight_layout()
    agg_png = os.path.join(agg_dir, "all_systems_mae.png")
    plt.savefig(agg_png, dpi=150)
    plt.close()

    print("\nOutputs:")
    print(" - Per-system CSV/plots:", per_dir)
    print(" - Aggregate CSVs:", long_csv, "and", summary_csv)
    print(" - Aggregate plot:", agg_png)

if __name__ == "__main__":
    main()
