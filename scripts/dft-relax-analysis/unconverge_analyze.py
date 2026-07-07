#!/usr/bin/env python3
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import re
import os
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

def analyze_vasp_relax(run_dir: str) -> Dict[str, Any]:
    """
    Analyze a VASP relaxation (ionic steps) that did not converge and infer likely reasons.

    Inputs
    ------
    run_dir : str
        Path to a directory containing OUTCAR/OSZICAR/INCAR (any subset works; more files -> better diagnosis).

    Output
    ------
    report : dict
        {
          "status": "unconverged" | "converged" | "unknown",
          "reason_codes": [list of short codes],
          "details": { ... parsed metrics ... },
          "suggestions": [list of concrete actions],
          "notes": [misc textual notes],
          "files_seen": { "INCAR": True/False, "OUTCAR": True/False, "OSZICAR": True/False }
        }

    Notes
    -----
    - We consider ionic convergence primarily force-based (|F|max <= |EDIFFG| if EDIFFG<0, else default 0.02 eV/Å).
    - We try to detect if electronic steps hit NELM, mixing pathologies, NaNs, oscillations, and stagnation.
    """
    p = Path(run_dir)
    files = {
        "INCAR": p / "INCAR",
        "OUTCAR": p / "OUTCAR",
        "OSZICAR": p / "OSZICAR",
    }
    exist = {k: f.is_file() for k, f in files.items()}

    def _read_text(path: Path, tail_only: bool = False, tail_lines: int = 4000) -> str:
        if not path.is_file():
            return ""
        if not tail_only:
            try:
                return path.read_text(errors="ignore")
            except UnicodeDecodeError:
                return path.read_bytes().decode("latin-1", errors="ignore")
        # Tail-read to keep memory down on huge OUTCARs
        with open(path, "rb") as fh:
            try:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                block = 1024 * 256
                chunks = []
                lines = 0
                while size > 0 and lines < tail_lines:
                    read = min(block, size)
                    size -= read
                    fh.seek(size, os.SEEK_SET)
                    chunk = fh.read(read)
                    chunks.append(chunk)
                    lines = b"".join(reversed(chunks)).count(b"\n")
                data = b"".join(reversed(chunks))
                return data.decode("latin-1", errors="ignore")
            except Exception:
                # fallback to full read
                return path.read_text(errors="ignore")

    incar_txt   = _read_text(files["INCAR"]) if exist["INCAR"] else ""
    outcar_tail = _read_text(files["OUTCAR"], tail_only=True, tail_lines=12000) if exist["OUTCAR"] else ""
    outcar_full = _read_text(files["OUTCAR"], tail_only=False) if exist["OUTCAR"] else ""
    oszicar_txt = _read_text(files["OSZICAR"]) if exist["OSZICAR"] else ""

    def _parse_incar(txt: str) -> dict:
        d = {}
        for line in txt.splitlines():
            line = line.split("#",1)[0].split("!",1)[0]
            if "=" in line:
                k,v = line.split("=",1)
                k = k.strip().upper()
                v = v.strip()
                d[k] = v
        return d

    incar = _parse_incar(incar_txt)

    def _get_int(name: str, default: Optional[int]) -> Optional[int]:
        v = incar.get(name)
        if v is None: return default
        try: return int(float(v))
        except: return default

    def _get_float(name: str, default: Optional[float]) -> Optional[float]:
        v = incar.get(name)
        if v is None: return default
        try: return float(v)
        except: return default

    NSW   = _get_int("NSW", None)
    NELM  = _get_int("NELM", None)
    EDIFF = _get_float("EDIFF", None)
    EDIFFG = _get_float("EDIFFG", None)

    # --- Helpers to extract ionic steps & forces from OUTCAR ---
    def _count_ionic_steps_outcar(txt: str) -> int:
        # Count occurrences of "POSITION  TOTAL-FORCE" blocks
        return len(re.findall(r"^\s*POSITION\s+TOTAL-FORCE.*?\n[-\s]+\n", txt, flags=re.IGNORECASE|re.MULTILINE))

    def _last_forces_block(txt: str) -> List[Tuple[float,float,float,float,float,float]]:
        """
        Returns list of rows [x,y,z,fx,fy,fz] from the LAST force block.
        """
        # Find all blocks quickly
        blocks = list(re.finditer(r"^\s*POSITION\s+TOTAL-FORCE.*?\n[-\s]+\n", txt, flags=re.IGNORECASE|re.MULTILINE))
        if not blocks:
            return []
        start = blocks[-1].end()
        # Read lines until a blank line or a non-data section
        lines = []
        for line in txt[start:].splitlines():
            if not line.strip():
                break
            parts = line.split()
            if len(parts) < 6:
                break
            try:
                row = list(map(float, parts[:3] + parts[-3:]))
            except ValueError:
                break
            lines.append(row)  # x,y,z, fx,fy,fz
        return lines

    def _fmax_from_last_step(txt: str) -> Optional[float]:
        rows = _last_forces_block(txt)
        if not rows: return None
        fmax = 0.0
        for _,_,_, fx,fy,fz in rows:
            fn = math.sqrt(fx*fx + fy*fy + fz*fz)
            if fn > fmax:
                fmax = fn
        return fmax

    # --- Electronic (SCF) hints from OUTCAR/OSZICAR ---
    def _scf_hit_nelm(txt: str) -> bool:
        # Heuristic: many lines ending with "DAV:" or "RMM:" with high iteration counts,
        # or explicit messages that NELM exceeded.
        if re.search(r"reached\smx\.?iter|exiting\sscf|EDDDAV|ZBRENT|ZHEGV|BRMIX:\s+very serious", txt, re.IGNORECASE):
            return True
        if NELM is not None:
            # look for patterns like "DAV:  N= 60" matching NELM often
            pat = re.compile(r"(DAV|RMM):\s*N\s*=\s*(\d+)", re.IGNORECASE)
            counts = [int(m.group(2)) for m in pat.finditer(txt)]
            if counts and max(counts) >= NELM:
                return True
        return False

    def _find_pathology_markers(txt: str) -> List[str]:
        markers = []
        patterns = {
            "brmix": r"BRMIX:\s+very serious problems",
            "zhegv": r"ZHEGV|ZHEGV failed",
            "zbrent": r"ZBRENT:\s+fatal error",
            "edwav": r"EDWAV",
            "subspace": r"WARNING:\s+Sub\-?Space",
            "posmap": r"POSMAP",
            "pulay": r"Pulay",
            "nan": r"nan|NaN|inf|\*{3,}",
        }
        for key,pat in patterns.items():
            if re.search(pat, txt, re.IGNORECASE):
                markers.append(key)
        return markers

    def _parse_oszicar_energies(txt: str) -> List[float]:
        # OSZICAR ionic summary lines often look like:
        #  n  F=  -xxx E0=  -yyy  d E =   +zz
        E = []
        for line in txt.splitlines():
            m = re.search(r"\bE0=\s*([\-+0-9\.Ee]+)", line)
            if m:
                try:
                    E.append(float(m.group(1)))
                except:
                    pass
        return E

    def _energy_oscillation(E: List[float]) -> bool:
        if len(E) < 6:
            return False
        # Detect alternating up/down and non-decreasing amplitude
        signs = []
        for i in range(1, len(E)):
            d = E[i] - E[i-1]
            signs.append(1 if d > 0 else (-1 if d < 0 else 0))
        flips = sum(1 for i in range(1,len(signs)) if signs[i]*signs[i-1] < 0)
        # Also check if recent |ΔE| are not decreasing
        recent = [abs(E[i]-E[i-1]) for i in range(max(1,len(E)-8), len(E))]
        if flips >= 3 and (len(recent)>=4 and max(recent[-4:]) >= max(recent[:max(1,len(recent)-4)])):
            return True
        return False

    # --- Gather metrics ---
    ionic_steps_tail = _count_ionic_steps_outcar(outcar_tail) if exist["OUTCAR"] else None
    ionic_steps_full = _count_ionic_steps_outcar(outcar_full) if exist["OUTCAR"] else None
    ionic_steps = ionic_steps_full if ionic_steps_full is not None else ionic_steps_tail
    fmax_last = _fmax_from_last_step(outcar_tail) if exist["OUTCAR"] else None
    markers = _find_pathology_markers(outcar_tail + "\n" + oszicar_txt)

    energies = _parse_oszicar_energies(oszicar_txt) if exist["OSZICAR"] else []
    oscillation = _energy_oscillation(energies) if energies else False

    # Ionic convergence flag (best-effort)
    converged_ionic = False
    if exist["OUTCAR"]:
        if re.search(r"reached required accuracy\s*-\s*stopping structural energy minim(ization|isation)", outcar_full, re.IGNORECASE):
            converged_ionic = True
        # Sometimes OUTCAR states "FORCES smaller than EDIFFG" etc.
        if re.search(r"FORCES?\s+smaller\s+than\s+EDIFFG", outcar_full, re.IGNORECASE):
            converged_ionic = True

    # Reasoning
    reason_codes: List[str] = []
    suggestions: List[str] = []
    notes: List[str] = []

    # Determine effective force threshold to judge "unconverged"
    # If EDIFFG < 0, |EDIFFG| is the force threshold.
    # Otherwise, use a pragmatic default (0.02 eV/Å), overrideable via INCAR.
    default_fthresh = 0.02
    if EDIFFG is not None and EDIFFG < 0:
        f_thresh = abs(EDIFFG)
    else:
        f_thresh = default_fthresh

    # 1) Ionic convergence vs NSW
    if converged_ionic:
        status = "converged"
    else:
        status = "unconverged"
        if NSW is not None and ionic_steps is not None and ionic_steps >= NSW:
            reason_codes.append("hit_nsw_limit")
            suggestions.append("Increase NSW (e.g., x2) or use better initial geometry / pre-relax with cheaper settings.")
        elif ionic_steps is not None and ionic_steps >= 1:
            # didn't explicitly hit NSW, but didn't meet forces either
            reason_codes.append("forces_not_below_threshold")

    # 2) Electronic issues
    if exist["OUTCAR"] or exist["OSZICAR"]:
        if _scf_hit_nelm(outcar_tail + "\n" + oszicar_txt):
            reason_codes.append("scf_not_converged_nelm")
            suggestions += [
                "Reduce mixing amplitude: set e.g. AMIX=0.2; BMIX=0.0001 for metals, enable Kerker: AMIX_MAG if spin.",
                "Try ALGO=Normal or ALGO=All for difficult cases; for metals, set ISMEAR=1 or 2 and adjust SIGMA≈0.2.",
                "Increase NELM (e.g., +40%) and set PREC=Accurate; consider mixing parameter variations (AMIX/BMIX/AMIX_MAG).",
            ]

    # 3) Pathologies (NaN, BRMIX, ZHEGV, ZBRENT, etc.)
    if markers:
        for m in markers:
            reason_codes.append(f"pathology_{m}")
        if "nan" in markers:
            suggestions.append("Check for NaN/infs: try smaller step sizes (IBRION=2, POTIM≈0.2), safer smearing, or re-initialize WAVECAR/CHGCAR.")
        if "brmix" in markers:
            suggestions.append("Severe mixing issues: set ISMEAR=0 or 1, lower AMIX/BMIX, and consider LREAL=Auto→.FALSE. for accuracy.")
        if "zbrent" in markers or "zhegv" in markers:
            suggestions.append("Linear algebra failure: remove WAVECAR/CHGCAR to start clean; try ALGO=Normal; ensure POTCAR set and ENCUT consistent.")

    # 4) Forces & stagnation
    if fmax_last is not None:
        if fmax_last > f_thresh:
            notes.append(f"Last-step Fmax={fmax_last:.4f} eV/Å exceeds threshold ({f_thresh:.4f}).")
            if "forces_not_below_threshold" not in reason_codes:
                reason_codes.append("forces_not_below_threshold")
            suggestions.append("Tighten relaxation strategy: start with IBRION=2 (CG), POTIM≈0.2; consider a pre-relax with lower ENCUT or softer POTCAR then switch to final settings.")
        else:
            notes.append(f"Last-step Fmax={fmax_last:.4f} eV/Å is below default threshold, but VASP did not mark convergence—check EDIFFG sign/value and criteria.")
            suggestions.append("If using energy-based EDIFFG (>0), switch to force-based threshold: set EDIFFG=-0.02 (negative means force criterion).")

    # 5) Energy oscillations / no progress
    if oscillation:
        reason_codes.append("energy_oscillation")
        suggestions += [
            "Damped dynamics first: IBRION=3; POTIM≈0.1–0.2 for a few hundred steps, then switch back to IBRION=2.",
            "Use smaller ionic step: reduce POTIM; increase ISIF suitably; ensure smearing is appropriate for metallic systems.",
        ]
    elif energies:
        # Check lack of progress: recent |ΔE| small but forces still high
        deltas = [abs(energies[i]-energies[i-1]) for i in range(1,len(energies))]
        if len(deltas) >= 5 and fmax_last and fmax_last > f_thresh and max(deltas[-5:]) < 1e-4:
            reason_codes.append("stagnated_no_progress")
            suggestions.append("Change optimizer: try IBRION=1 (quasi-Newton) or IBRION=2 with smaller POTIM; consider ISIF that matches what you want to relax.")

    # 6) Generic advice if little info
    if not reason_codes and status != "converged":
        reason_codes.append("unknown_unconverged")
        suggestions.append("Re-run from a cleaner state (delete WAVECAR & CHGCAR), reduce POTIM, and consider a pre-relax using lower precision, then switch to final PREC=Accurate.")

    details = {
        "ionic_steps_observed": ionic_steps,
        "NSW": NSW,
        "NELM": NELM,
        "EDIFF": EDIFF,
        "EDIFFG": EDIFFG,
        "force_threshold_used": f_thresh,
        "Fmax_last_step": fmax_last,
        "energy_steps": len(energies),
        "pathology_markers": markers,
        "energy_oscillation_detected": oscillation,
    }

    # De-duplicate & tidy suggestions
    seen = set()
    dedup_suggestions = []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            dedup_suggestions.append(s)

    return {
        "status": status,
        "reason_codes": sorted(set(reason_codes)),
        "details": details,
        "suggestions": dedup_suggestions,
        "notes": notes,
        "files_seen": exist,
    }

# ---- Optional: convenience printer ----
def print_vasp_analysis(run_dir: str):
    rep = analyze_vasp_relax(run_dir)
    print(f"[{run_dir}] status: {rep['status']}")
    print("  files:", rep["files_seen"])
    print("  reason_codes:", ", ".join(rep["reason_codes"]) or "(none)")
    for k,v in rep["details"].items():
        print(f"  {k}: {v}")
    if rep["notes"]:
        print("  notes:")
        for n in rep["notes"]:
            print("   -", n)
    if rep["suggestions"]:
        print("  suggestions:")
        for s in rep["suggestions"]:
            print("   -", s)


# --- paste in the two helpers from my previous message if not already imported ---
# analyze_vasp_relax(run_dir: str) -> Dict[str, Any]
# print_vasp_analysis(run_dir: str)

# If you already have analyze_vasp_relax in scope, delete this dummy import guard.
try:
    analyze_vasp_relax  # type: ignore
except NameError:
    raise RuntimeError("Please paste the analyze_vasp_relax() function from my previous message above this script.")

def _flatten_report(cif_id: str, rep: Dict[str, Any]) -> Dict[str, Any]:
    det = rep.get("details", {})
    files_seen = rep.get("files_seen", {})
    row = {
        "cif_id": cif_id,
        "status": rep.get("status"),
        "reason_codes": ";".join(rep.get("reason_codes", [])) or "",
        "notes": " | ".join(rep.get("notes", [])) or "",
        "suggestions": " || ".join(rep.get("suggestions", [])) or "",
        # details
        "ionic_steps_observed": det.get("ionic_steps_observed"),
        "NSW": det.get("NSW"),
        "NELM": det.get("NELM"),
        "EDIFF": det.get("EDIFF"),
        "EDIFFG": det.get("EDIFFG"),
        "force_threshold_used": det.get("force_threshold_used"),
        "Fmax_last_step_eVA": det.get("Fmax_last_step"),
        "energy_steps": det.get("energy_steps"),
        "pathology_markers": ";".join(det.get("pathology_markers", [])) if det.get("pathology_markers") else "",
        "energy_oscillation_detected": det.get("energy_oscillation_detected"),
        # files present
        "has_INCAR": bool(files_seen.get("INCAR")),
        "has_OUTCAR": bool(files_seen.get("OUTCAR")),
        "has_OSZICAR": bool(files_seen.get("OSZICAR")),
        # raw JSON (optional, helpful for debugging)
        "raw_json": json.dumps(rep, ensure_ascii=False),
    }
    return row

def _discover_runs(dft_parent: Path) -> List[Path]:
    """
    Discover run directories. We expect one level: {parent}/{cif_id}/ (with OUTCAR or vasprun.xml etc.).
    We’ll include a dir if it contains at least one of OUTCAR/OSZICAR/vasprun.xml
    """
    runs = []
    for child in sorted(dft_parent.iterdir()):
        if not child.is_dir():
            continue
        has_any = (child/"OUTCAR").exists() or (child/"OSZICAR").exists() or (child/"vasprun.xml").exists()
        if has_any:
            runs.append(child)
    return runs

def build_vasp_unconverged_csv(
    dft_parent: str,
    csv_out: str,
    only_unconverged: bool = False,
) -> None:
    parent = Path(dft_parent)
    if not parent.is_dir():
        raise NotADirectoryError(dft_parent)

    runs = _discover_runs(parent)
    if not runs:
        raise FileNotFoundError(f"No run dirs with OUTCAR/OSZICAR/vasprun.xml found under {dft_parent}")

    rows: List[Dict[str, Any]] = []
    for run_dir in runs:
        cif_id = run_dir.name
        try:
            rep = analyze_vasp_relax(str(run_dir))
        except Exception as e:
            # If the analyzer crashes for some run, capture as an 'error' row
            rep = {
                "status": "error",
                "reason_codes": ["analyzer_exception"],
                "details": {},
                "suggestions": [f"Exception: {e}"],
                "notes": [f"Analyzer failed for {run_dir}"],
                "files_seen": {
                    "INCAR": (run_dir / "INCAR").exists(),
                    "OUTCAR": (run_dir / "OUTCAR").exists(),
                    "OSZICAR": (run_dir / "OSZICAR").exists(),
                },
            }
        if only_unconverged and rep.get("status") == "converged":
            continue
        rows.append(_flatten_report(cif_id, rep))

    # Column order
    fieldnames = [
        "cif_id",
        "status",
        "reason_codes",
        "notes",
        "suggestions",
        "ionic_steps_observed",
        "NSW",
        "NELM",
        "EDIFF",
        "EDIFFG",
        "force_threshold_used",
        "Fmax_last_step_eVA",
        "energy_steps",
        "pathology_markers",
        "energy_oscillation_detected",
        "has_INCAR",
        "has_OUTCAR",
        "has_OSZICAR",
        "raw_json",
    ]

    outp = Path(csv_out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {len(rows)} rows → {outp}")

# ---- optional CLI ----
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: build_vasp_unconverged_csv.py DFT_PARENT CSV_OUT [--only-unconverged]")
        sys.exit(2)
    dft_parent = sys.argv[1]
    csv_out = sys.argv[2]
    only_unconverged = ("--only-unconverged" in sys.argv[3:])
    build_vasp_unconverged_csv(dft_parent, csv_out, only_unconverged)
