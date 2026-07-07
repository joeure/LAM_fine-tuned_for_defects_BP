#!/usr/bin/env python3
import argparse, re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def read_text_tail(path: Path, max_bytes=8_000_000):
    size = path.stat().st_size
    with path.open("rb") as f:
        if size > max_bytes:
            f.seek(size - max_bytes)
        data = f.read()
    return data.decode("utf-8", errors="ignore")

def parse_outcar(outcar_path: Path):
    if not outcar_path.exists():
        return {"fmax": np.nan, "ediffg": np.nan, "reached": False, "method": "missing"}
    text = read_text_tail(outcar_path)
    reached = ("reached required accuracy" in text.lower())
    m_ed = re.search(r"EDIFFG\s*=\s*([\-+0-9.Ee]+)", text)
    ediffg = float(m_ed.group(1)) if m_ed else np.nan
    # Try "FORCES: max atom, RMS"
    m_forces = list(re.finditer(r"FORCES:\s*max atom,\s*RMS\s+([\-+0-9.Ee]+)\s+([\-+0-9.Ee]+)", text))
    if m_forces:
        last = m_forces[-1]
        try:
            fmax = float(last.group(1))
            return {"fmax": fmax, "ediffg": ediffg, "reached": reached, "method": "FORCES:max"}
        except Exception:
            pass
    # Fallback: last "TOTAL-FORCE (eV/Angst)" table
    headers = list(re.finditer(r"TOTAL-FORCE\s*\(eV/Angst\)", text, flags=re.IGNORECASE))
    if headers:
        start = headers[-1].end()
        tail = text[start:]
        lines = tail.splitlines()
        i = 0
        while i < len(lines) and not re.search(r"^\s*\d+", lines[i]):
            i += 1
        forces = []
        while i < len(lines):
            ln = lines[i].strip()
            if not ln or "total drift" in ln.lower() or "sum of" in ln.lower():
                break
            toks = re.findall(r"([\-+0-9]*\.?[0-9]+(?:[Ee][\-+]?\d+)?)", ln)
            if len(toks) >= 3:
                try:
                    fx, fy, fz = map(float, toks[-3:])
                    forces.append((fx, fy, fz))
                except Exception:
                    pass
            i += 1
        if forces:
            mags = [np.sqrt(fx*fx + fy*fy + fz*fz) for fx,fy,fz in forces]
            fmax = float(np.max(mags))
            return {"fmax": fmax, "ediffg": ediffg, "reached": reached, "method": "TOTAL-FORCE"}
    return {"fmax": np.nan, "ediffg": ediffg, "reached": reached, "method": "not_found"}

def load_ood(path: Path):
    df = pd.read_csv(path)
    if "cif_id" not in df.columns or "ood_knn_cosine" not in df.columns:
        raise ValueError("OOD CSV must include 'cif_id' and 'ood_knn_cosine'.")
    out = df[["cif_id","ood_knn_cosine"]].copy()
    out["ood_knn_cosine"] = pd.to_numeric(out["ood_knn_cosine"], errors="coerce")
    return out

def classify_convergence(parsed, default_thresh=None):
    fmax = parsed["fmax"]; ediffg = parsed["ediffg"]; reached = parsed["reached"]
    if np.isfinite(fmax) and reached:
        return True, "message"
    if np.isfinite(fmax) and np.isfinite(ediffg) and ediffg < 0:
        return (fmax <= abs(ediffg) + 1e-12), f"EDIFFG={ediffg}"
    if default_thresh is not None and np.isfinite(fmax):
        return (fmax <= default_thresh), f"thresh={default_thresh}"
    return False, "unknown"

def main():
    ap = argparse.ArgumentParser(description="Extract final Fmax from VASP OUTCARs and plot vs OOD.")
    ap.add_argument("--dft_dir", required=True, type=Path)
    ap.add_argument("--ood_csv", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("fmax_vs_ood.png"))
    ap.add_argument("--default_force_thresh", type=float)
    ap.add_argument("--save_csv", type=Path, default=Path("fmax_vs_ood.csv"))
    args = ap.parse_args()

    ood = load_ood(args.ood_csv)
    rows = []
    for cif_id in ood["cif_id"]:
        outcar = args.dft_dir / f"{cif_id}_unrelaxed" / "OUTCAR"
        parsed = parse_outcar(outcar)
        conv, crit = classify_convergence(parsed, default_thresh=args.default_force_thresh)
        rows.append({
            "cif_id": cif_id,
            "ood_knn_cosine": ood.loc[ood["cif_id"]==cif_id, "ood_knn_cosine"].values[0],
            "fmax_eV_per_A": parsed["fmax"],
            "ediffg": parsed["ediffg"],
            "reached_msg": parsed["reached"],
            "method": parsed["method"],
            "converged": bool(conv)
        })
    df = pd.DataFrame(rows)
    df.to_csv(args.save_csv, index=False)

    # Plot
    plt.figure(figsize=(8,5))
    uu = df[df["converged"]==False]
    plt.scatter(uu["ood_knn_cosine"], uu["fmax_eV_per_A"], s=3, label="unconverged")
    cc = df[df["converged"]==True]
    plt.scatter(cc["ood_knn_cosine"], cc["fmax_eV_per_A"], s=3, label="converged")
    plt.xlabel("OOD (k-NN cosine)"); plt.ylabel("Final Fmax (eV/Å)")
    plt.title("DFT: Final Fmax vs OOD"); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print("Wrote figure:", args.out); print("Wrote table:", args.save_csv)

if __name__ == "__main__":
    main()
