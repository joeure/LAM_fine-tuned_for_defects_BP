import argparse
import os, re, csv
import pandas as pd

_EDIFFG_RE = re.compile(r'^\s*EDIFFG\s*=\s*([^\s!#;]+)(.*)$', re.IGNORECASE)

def _compute_target_tol(fmax, mode="relative", floor=0.01, lower=0.005, upper=0.03, factor=0.9):
    """Return a per-system force tolerance in eV/Å (positive number)."""
    if fmax is None or not (fmax == fmax) or fmax <= 0:
        return floor
    if mode == "relative":
        t = factor * float(fmax)           # slightly stricter than last fmax
        t = max(lower, min(t, upper))      # clamp
        t = max(t, floor)                  # never looser than floor
        return t
    elif mode == "constant":
        return floor
    else:
        raise ValueError("mode should be 'relative' or 'constant'")

def _read_incar_text(path):
    with open(path, "r") as f:
        return f.read()

def _write_incar_text(path, text):
    with open(path, "w") as f:
        f.write(text)

def _replace_or_insert_ediffg(text, new_value):
    """
    new_value is positive (eV/Å). We write 'EDIFFG = -<new_value>'.
    Preserves trailing comments on the EDIFFG line if present.
    """
    lines = text.splitlines()
    replaced = False
    for i, line in enumerate(lines):
        m = _EDIFFG_RE.match(line)
        if m:
            # keep trailing comment (m.group(2))
            trailing = m.group(2) or ""
            # keep original comment chars if any
            if trailing and not trailing.lstrip().startswith(("#","!",";")):
                trailing = " " + trailing
            lines[i] = f"EDIFFG = {-new_value:.6f}{trailing}"
            replaced = True
            break
    if not replaced:
        # append at end with a helpful comment
        lines.append(f"EDIFFG = {-new_value:.6f}   # set by script (force threshold, eV/Å)")
    return "\n".join(lines), replaced

def set_ediffg_from_fmax_csv(vasp_root, fmax_csv, out_report_csv,
                             mode="relative", floor=0.01, lower=0.005, upper=0.03, factor=0.9,
                             backup=True):
    """
    For each systemID in fmax_csv, edit {vasp}/{systemID}/INCAR and set EDIFFG = -target.
    The CSV must have columns: system_id, fmax_eV_per_A (as produced earlier).
    """
    df = pd.read_csv(fmax_csv)
    if "system_id" not in df.columns or "fmax_eV_per_A" not in df.columns:
        raise ValueError("CSV must contain columns: system_id, fmax_eV_per_A")
    rows = []
    for sid, fmax in zip(df["system_id"], df["fmax_eV_per_A"]):
        incar_path = os.path.join(vasp_root, f"{str(sid)}_unrelaxed", "INCAR")
        if not os.path.isfile(incar_path):
            print(f"[WARN] Missing INCAR: {incar_path}")
            continue
        text = _read_incar_text(incar_path)

        # read old EDIFFG if present
        old_val = None
        for line in text.splitlines():
            m = _EDIFFG_RE.match(line)
            if m:
                try:
                    old_val = float(m.group(1))
                except Exception:
                    old_val = None
                break

        # compute new positive tolerance (eV/Å), write as negative
        tol = _compute_target_tol(fmax, mode=mode, floor=floor, lower=lower, upper=upper, factor=factor)

        if backup:
            bak = incar_path + ".bak"
            if not os.path.exists(bak):
                with open(bak, "w") as fb: fb.write(text)

        new_text, replaced = _replace_or_insert_ediffg(text, tol)
        _write_incar_text(incar_path, new_text)

        rows.append({
            "system_id": sid,
            "fmax_eV_per_A": fmax,
            "old_EDIFFG": old_val,
            "new_EDIFFG": -tol,   # what we actually wrote
            "mode": mode,
            "replaced_existing": bool(replaced)
        })

    pd.DataFrame(rows).to_csv(out_report_csv, index=False)
    print(f"[INFO] Updated {len(rows)} INCAR files. Report → {out_report_csv}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Update EDIFFG in generated VASP INCAR files from a fmax summary CSV."
    )
    parser.add_argument(
        "--vasp-root",
        required=True,
        help="Directory containing <system_id>_unrelaxed/INCAR folders.",
    )
    parser.add_argument(
        "--fmax-csv",
        required=True,
        help="CSV with columns system_id and fmax_eV_per_A.",
    )
    parser.add_argument(
        "--out-report-csv",
        default="ediffg_update_report.csv",
        help="Output CSV report of old/new EDIFFG values.",
    )
    parser.add_argument("--mode", choices=["relative", "constant"], default="relative")
    parser.add_argument("--floor", type=float, default=0.01)
    parser.add_argument("--lower", type=float, default=0.005)
    parser.add_argument("--upper", type=float, default=0.7)
    parser.add_argument("--factor", type=float, default=1.0)
    parser.add_argument("--no-backup", action="store_true", help="Do not create INCAR.bak files.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_ediffg_from_fmax_csv(
        args.vasp_root,
        args.fmax_csv,
        args.out_report_csv,
        mode=args.mode,
        floor=args.floor,
        lower=args.lower,
        upper=args.upper,
        factor=args.factor,
        backup=not args.no_backup,
    )


if __name__ == "__main__":
    main()
