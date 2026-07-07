# prep_sp_from_dft.py
import os, csv, json, argparse, warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from ase.data import atomic_numbers, atomic_masses
from ase import Atoms
from ase.io import read as ase_read, write as ase_write
from pymatgen.io.vasp import Vasprun, Outcar
from pymatgen.io.ase import AseAtomsAdaptor

# ---------- tiny utils ----------
_NUM = __import__("re").compile(r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$")

def split_cif_id(cif_id: str) -> Tuple[str, str]:
    # cif_id == "{chem}_{uid}", chem has no '_'
    return cif_id.split("_", 1)[0], cif_id.split("_", 1)[1]

def exists(p: str) -> bool:
    try: return os.path.exists(p)
    except Exception: return False

def read_test_ids(meta_csv: str) -> List[str]:
    out = []
    with open(meta_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            cid = row.get("cif_id") or row.get("atoms_id") or row.get("id")
            if not cid: 
                continue
            is_test = str(row.get("is_test", "True")).strip().lower() in ("1","true","yes","y")
            if is_test:
                out.append(str(cid))
    return sorted(set(out))

def read_dft_final(cif_id: str, dft_dir: str) -> Tuple[Atoms, Optional[float]]:
    """
    Read the final DFT-relaxed structure + total energy (eV).
    Accepts vasprun.xml (preferred) or OUTCAR (+CONTCAR fallback).
    """
    base = os.path.join(dft_dir, f"{cif_id}_unrelaxed")
    vrun = os.path.join(base, "vasprun.xml")
    outc = os.path.join(base, "OUTCAR")

    if exists(vrun):
        v = Vasprun(vrun, parse_potcar_file=False)
        at = AseAtomsAdaptor.get_atoms(v.final_structure)
        e = v.final_energy
        return at, (float(e) if e is not None else None)

    if exists(outc):
        # prefer CONTCAR geometry if present
        pos = os.path.join(base, "CONTCAR")
        if exists(pos):
            at = ase_read(pos)
        else:
            o = Outcar(outc); at = AseAtomsAdaptor.get_atoms(o.structure)
        e = None
        try:
            with open(outc, "r", errors="ignore") as fh:
                for line in fh:
                    if "free  energy   TOTEN" in line:
                        toks = line.strip().split()
                        for t in toks[::-1]:
                            if _NUM.match(t):
                                e = float(t); break
            # keep last seen value
        except Exception:
            e = None
        return at, e

    raise FileNotFoundError(f"DFT files not found for {cif_id}: {base}/vasprun.xml or OUTCAR")

def atomic_numbers_to_symbols(Zs: List[int]) -> List[str]:
    from periodictable import elements as E
    out = []
    for z in Zs:
        try:
            out.append(getattr(E, str(int(z))).symbol)
        except Exception:
            raise ValueError(f"Unknown atomic number {z}")
    return out

def symbol_Z(s: str) -> int:
    from periodictable import elements as E
    return int(getattr(E, s).number)

# ---------- core builder ----------
def build_sp_for_model(
    model_name: str,
    engine: str,                      # "mace" or "deepmd"
    model_file: str,
    system_name: str,
    dft_dir: str,
    meta_csv: str,
    out_root: str,
    log_basename: Optional[str] = None,
    global_specorder: Optional[List[str]] = None
) -> Dict:
    """
    For one model, write:
      - data/*.data  (DFT-final geometries with a consistent specorder)
      - in.singlepoint.lmp  (one script that runs SP for all test systems)
      - manifest.csv  (cif_id, sysname, data_path)
    Returns a small summary (counts, specorder, paths).
    """
    out_dir  = Path(out_root) / f"{model_name}"
    data_dir = out_dir / "data"
    logs_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # 1) collect test ids
    test_ids = read_test_ids(meta_csv)
    if not test_ids:
        raise RuntimeError("No test cif_ids found in meta CSV.")

    # 2) read all DFT-final structures and accumulate union of elements (for global specorder)
    dft_atoms: Dict[str, Atoms] = {}
    union_syms = set()
    skipped = []
    for cid in test_ids:
        try:
            at, _ = read_dft_final(cid, dft_dir)
        except Exception as e:
            warnings.warn(f"[skip] {cid}: {e}")
            skipped.append((cid, str(e)))
            continue
        dft_atoms[cid] = at
        union_syms.update(set(at.get_chemical_symbols()))

    if not dft_atoms:
        raise RuntimeError("Could not load any DFT-relaxed structures.")

    # decide global specorder (stable, by Z) if not provided
    if global_specorder:
        specorder = list(global_specorder)
    else:
        specorder = sorted(list(union_syms), key=symbol_Z)

    # 3) write each DFT-final to LAMMPS data with *consistent* specorder
    manifest_rows = []
    # Build a fast symbol->mass map once
    sym2mass = {sym: float(atomic_masses[atomic_numbers[sym]]) for sym in specorder}
    for cid, at in dft_atoms.items():
        at.set_masses([sym2mass[s] for s in at.get_chemical_symbols()])
        
        chem, uid = split_cif_id(cid)
        sysname = f"{system_name}_{uid}_DFTfinal"
        data_path = data_dir / f"{sysname}.data"
        ase_write(str(data_path), at, format="lammps-data", masses=True,
                  specorder=specorder, atom_style="atomic", units="metal")
        manifest_rows.append({
            "cif_id": cid,
            "sysname": sysname,
            "data_path": str(data_path),
        })

    # 4) write one LAMMPS input that loops over all systems and does a single-point (run 0)
    #    We'll print sentinel lines so parsing is trivial.
    log_file = logs_dir / (log_basename or f"sp_{model_name}.log")
    in_path  = out_dir / "in.singlepoint.lmp"

    # engine-specific pair_style/coeff lines
    def make_potential_block() -> str:
        if engine.lower() == "mace":
            # user example uses mace/kk; keep consistent
            elems = " ".join(specorder)
            return (
                f"pair_style      mace/kk no_domain_decomposition\n"
                f"pair_coeff      * * {model_file} {elems}\n"
            )
        elif engine.lower() == "deepmd":
            elems = " ".join(specorder)
            return (
                f"pair_style      deepmd {model_file}\n"
                f"pair_coeff      * * {elems}\n"
            )
        else:
            raise ValueError(f"Unknown engine '{engine}', expected 'mace' or 'deepmd'.")

    pot_block = make_potential_block()

    with open(in_path, "w") as f:
        f.write(f"# Auto-generated single-point bundle for {model_name}\n")
        f.write(f"variable model string {model_name}\n")
        f.write(f"shell mkdir -p {logs_dir}\n")
        f.write(f"shell mkdir -p dumps/${{model}}\n")
        f.write(f"log {log_file}\n\n")

        for row in manifest_rows:
            sysname  = row["sysname"]              # e.g. BP_spin_500_..._unrelaxed
            data_p   = row["data_path"]

            # --- init ---
            f.write("# ======================= Initialization =======================\n")
            f.write("units           metal\n")
            f.write("atom_style      atomic\n")
            f.write("atom_modify     map yes\n")
            f.write("boundary        p p p\n")
            f.write("neighbor        2.0 bin\n")
            f.write("neigh_modify    every 1 delay 0 check yes\n")
            f.write("newton          on\n\n")

            # --- read structure ---
            f.write("# ====================== Read Structure ======================\n")
            f.write(f"read_data       {data_p}\n\n")

            # --- model potential ---
            f.write("# ================== Model potential ========================\n")
            f.write(pot_block + "\n")   # your pair_style/pair_coeff block

            # --- single point eval + force dump ---
            f.write("# ================= Single-point (no dynamics) ==============\n")
            f.write(f'variable sysname string {sysname}\n')
            f.write(f"shell mkdir -p dumps/${{model}}/${{sysname}}\n")

            # one-frame dump with forces (sorted by atom id for easy joins)
            f.write(f'dump        d1 all custom 1 dumps/${{model}}/${{sysname}}/sp.lammpstrj id type x y z fx fy fz\n')
            f.write("dump_modify d1 first yes sort id\n")

            # concise thermo + energy print markers
            f.write("thermo       1\n")
            f.write("thermo_style custom step pe\n")
            f.write(f'print "__SP_BEGIN__ ${{sysname}}"\n')

            # 'run 0' → compute E/F once (no time steps)
            f.write("run 0 post no\n")

            # emit the energy in a machine-parsable line
            f.write("variable e equal pe\n")
            f.write(f'print "__SP_RESULT__ ${{sysname}} ${{e}}"\n')
            f.write(f'print "__SP_END__ ${{sysname}}"\n')

            # clean up this dump before the next system
            f.write("undump d1\n")
            f.write("clear\n\n")

        f.write('print "End of single-point bundle."\n')

    # 5) write manifest
    man_path = out_dir / "manifest.csv"
    with open(man_path, "w", newline="") as mf:
        w = csv.DictWriter(mf, fieldnames=["cif_id","sysname","data_path","specorder","engine","model_file"])
        w.writeheader()
        for row in manifest_rows:
            row2 = dict(row)
            row2["specorder"] = " ".join(specorder)
            row2["engine"] = engine
            row2["model_file"] = model_file
            w.writerow(row2)

    return {
        "model": model_name,
        "engine": engine,
        "n_systems": len(manifest_rows),
        "specorder": specorder,
        "in_file": str(in_path),
        "log_file": str(log_file),
        "manifest": str(man_path),
        "data_dir": str(data_dir),
        "skipped": skipped,
    }

def main():
    ap = argparse.ArgumentParser(description="Bundle many single-point E_model(R_DFT) jobs into one LAMMPS input per model.")
    ap.add_argument("--dft_dir", required=True, help="Folder with {cif_id}_relaxed/{vasprun.xml|OUTCAR}")
    ap.add_argument("--meta_csv", required=True, help="CSV with column 'cif_id' and 'is_test'")
    ap.add_argument("--system_name", required=True, help="Prefix used in sysname (e.g., 'BP_spin_500')")
    ap.add_argument("--out_root", required=True, help="Output root for in.lmp, data/, logs/, manifest.csv")
    ap.add_argument("--models", required=True,
                    help="Comma list of MODEL specs 'name:engine:path', e.g. 'MACE:mace:foundations/mace-mpa-0-medium.model-lammps.pt,DPA3:deepmd:foundations/frozen_model_MP_traj_v024.pth'")
    ap.add_argument("--specorder", default="", help="Optional comma-ordered global element list (e.g., 'H,C,N,O,P'). If omitted, will be inferred from DFT structures (sorted by Z).")
    ap.add_argument("--log_basename", default=None, help="Optional fixed log basename (default sp_<model>.log)")
    args = ap.parse_args()

    user_specorder = [s.strip() for s in args.specorder.split(",") if s.strip()] or None

    specs = []
    for part in args.models.split(","):
        part = part.strip()
        if not part: continue
        try:
            name, eng, path = part.split(":", 2)
        except ValueError:
            raise SystemExit(f"Bad --models entry '{part}'. Use 'name:engine:path'.")
        if eng.lower() not in ("mace", "deepmd"):
            raise SystemExit(f"Unknown engine '{eng}' in '{part}' (use 'mace' or 'deepmd').")
        if not os.path.exists(path):
            warnings.warn(f"[warn] model file does not exist yet: {path}")
        specs.append((name, eng, path))

    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    summaries = []
    for (name, eng, path) in specs:
        s = build_sp_for_model(
            model_name=name, engine=eng, model_file=path,
            system_name=args.system_name, dft_dir=args.dft_dir, meta_csv=args.meta_csv,
            out_root=args.out_root, log_basename=args.log_basename,
            global_specorder=user_specorder
        )
        summaries.append(s)

    print(json.dumps(summaries, indent=2))

if __name__ == "__main__":
    main()
