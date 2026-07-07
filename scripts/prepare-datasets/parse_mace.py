#!/usr/bin/env python3
import os, warnings, argparse, json
from pathlib import Path
from typing import Iterable, Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
from ase.io import read, write
from sklearn.model_selection import GroupShuffleSplit
from collections import defaultdict

def _prepare_ndef_map(meta_csv: Optional[Path]) -> Optional[Dict[str, int]]:
    """
    Read metadata and return system_id -> ndef (nsubs + nvacs).
    Accepts either:
      (a) columns: ['system','formula','nsubs','nvacs']  -> system_id = system+'_'+formula
      (b) columns: ['cif_id','nsubs','nvacs']            -> system_id = '_'.join(cif_id.split('_')[:-1])
    If multiple rows share the same system_id, we require consistent ndef (warn otherwise and pick the mode).
    """
    if not meta_csv:
        return None
    df = pd.read_csv(meta_csv)

    # Compute ndef
    required_nv = {"nsubs", "nvacs"}
    if not required_nv.issubset(df.columns):
        raise ValueError(f"{meta_csv} must contain columns {required_nv}.")

    df = df.copy()
    df["ndef"] = df["nsubs"].astype(int) + df["nvacs"].astype(int)

    # Derive system_id
    if "cif_id" in df.columns:
        sysid = df["cif_id"].astype(str)
    else:
        raise ValueError(f"{meta_csv} must contain either ('system','formula') or 'cif_id'.")

    df["system_id"] = sysid

    # Resolve to unique mapping; if inconsistent, warn and choose the most frequent ndef
    grp = df.groupby("system_id")["ndef"]
    ndef_map = {}
    for sid, series in grp:
        vals = series.dropna().astype(int).tolist()
        if not vals:
            continue
        if len(set(vals)) > 1:
            warnings.warn(f"[ndef] Inconsistent ndef for {sid}: {sorted(set(vals))}. Using mode.")
        # pick the mode (most frequent); break ties by max (arbitrary but deterministic)
        counts = pd.Series(vals).value_counts()
        mode_val = counts.index[0]
        ndef_map[sid] = int(mode_val)

    return ndef_map

def _attach_labels_for_mace(atoms, add_virials: bool):
    """
    Ensure MACE-visible keys are present:
      atoms.info['REF_energy']   (float, eV)
      atoms.arrays['REF_forces'] (Nx3, eV/Å)
      optional:
        atoms.info['REF_stress'] (Voigt 6, eV/Å^3)
        atoms.info['REF_virial'] (Voigt 6, eV)
    """
    # Energy/forces from calculator:
    atoms.info["REF_energy"] = float(atoms.get_potential_energy())
    atoms.arrays["REF_forces"] = np.array(atoms.get_forces(), dtype=float)
    if add_virials:
        try:
            s6 = atoms.get_stress(voigt=True)  # eV/Å^3
            V = atoms.get_volume()             # Å^3
            atoms.info["REF_stress"] = np.array(s6, dtype=float)
            atoms.info["REF_virial"] = np.array(-s6 * V, dtype=float)
        except Exception:
            warnings.warn("No stress for this frame; writing without virials.")

# ---------- multi-root helpers ----------
def _find_vasp_source_in_dir(sys_dir: Path) -> Optional[Path]:
    """Prefer vasprun.xml, fallback to OUTCAR, else None (for a single system dir)."""
    for name in ("vasprun.xml", "OUTCAR"):
        p = sys_dir / name
        if p.is_file():
            return p
    return None

def _list_all_system_ids(vasp_roots: List[str]) -> List[str]:
    """Union of system ids across all roots (subdir name minus final token after '_')."""
    ids = set()
    for r in vasp_roots:
        root = Path(r)
        if not root.is_dir():
            continue
        for p in sorted([p for p in root.iterdir() if p.is_dir()]):
            if p.name == "P":  # keep your original exclusion
                continue
            sid = "_".join(p.name.split("_")[:-1])
            if sid:
                ids.add(sid)
    return sorted(ids)

def _sources_for_system(vasp_roots: List[str], system_id: str) -> List[Path]:
    """
    Return an ordered list of available sources (vasprun.xml/OUTCAR) for this system
    across the provided roots (root priority order is preserved).
    """
    out = []
    for r in vasp_roots:
        sys_dir = None
        # find the subdir whose prefix matches {system_id}_*
        root = Path(r)
        cand = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(system_id + "_")]
        if not cand:
            continue
        # there should be exactly one; if multiple, take the first by name order
        sys_dir = sorted(cand)[0]
        src = _find_vasp_source_in_dir(sys_dir)
        if src is not None:
            out.append(src)
    return out

def _is_converged_ionic(src: Path) -> Optional[bool]:
    """Best-effort ionic convergence check."""
    try:
        if src.name == "vasprun.xml":
            from pymatgen.io.vasp import Vasprun
            v = Vasprun(str(src), parse_potcar_file=False)
            return bool(getattr(v, "converged_ionic", None))
        else:  # OUTCAR
            with open(src, "r", errors="ignore") as fh:
                for line in fh:
                    s = line.strip().lower()
                    # common VASP stop messages
                    if "reached required accuracy" in s or "stopping structural energy minimisation" in s:
                        return True
            return None  # unknown
    except Exception:
        return None

def _iter_frames_chain(srcs: List[Path], only_chain_if_unconverged=True, pos_tol=1e-4) -> Iterable:
    """
    Yield ASE Atoms across multiple sources for a system.
    - Read all frames from the first src.
    - If it's converged and only_chain_if_unconverged==True, stop.
    - Otherwise, for each following src:
        * read frames, skip its first frame if it's (nearly) the same as the last frame already yielded
          (because it's usually the previous CONTCAR)
        * yield the rest
    """
    from ase.io import read as ase_read
    yielded_last = None
    # first source
    if not srcs:
        return
    conv0 = _is_converged_ionic(srcs[0])
    for a in ase_read(str(srcs[0]), index=":"):
        yielded_last = a
        yield a
    # decide whether to chain
    if only_chain_if_unconverged and conv0 is True:
        return
    # chain remaining
    for src in srcs[1:]:
        frames = list(ase_read(str(src), index=":"))
        if not frames:
            continue
        start = 0
        if yielded_last is not None:
            try:
                # compare first frame of continuation with last yielded
                pa = frames[0].get_positions()
                pb = yielded_last.get_positions()
                if pa.shape == pb.shape and np.max(np.linalg.norm(pa - pb, axis=1)) < pos_tol:
                    start = 1  # skip duplicate first frame
            except Exception:
                pass
        for a in frames[start:]:
            yielded_last = a
            yield a


# ----------------- OOD + splitting helpers -----------------
def _load_test_ids(test_csv: Optional[Path]) -> List[str]:
    if not test_csv:
        return []
    df = pd.read_csv(test_csv)
    col = "atoms_id" if "atoms_id" in df.columns else "cif_id"
    if col not in df.columns:
        raise ValueError(f"{test_csv} must contain 'atoms_id' or 'cif_id'.")
    return df[col].astype(str).tolist()

def _union_with_is_test(test_ids: List[str], ood_df: Optional[pd.DataFrame]) -> List[str]:
    if ood_df is None or "is_test" not in ood_df.columns:
        return test_ids
    extra = ood_df.loc[ood_df["is_test"].astype(bool), "cif_id"].astype(str).tolist()
    return sorted(set(test_ids).union(extra))

def _prepare_ood_map(ood_csv: Optional[Path],
                     prefer_col: str = "ood_centroid_cosine",
                     fallback_col: str = "ood_knn_cosine") -> Optional[pd.DataFrame]:
    if not ood_csv:
        return None
    df = pd.read_csv(ood_csv)
    if "cif_id" not in df.columns:
        raise ValueError(f"{ood_csv} must contain 'cif_id' column.")
    if prefer_col not in df.columns and fallback_col not in df.columns:
        raise ValueError(f"{ood_csv} must contain '{prefer_col}' or '{fallback_col}'.")
    # Decide which column to use
    use_col = prefer_col if prefer_col in df.columns else fallback_col
    df = df[["cif_id", use_col] + ([c for c in ("is_test",) if c in df.columns])]
    df = df.rename(columns={use_col: "OOD"})
    # Higher OOD = more underrepresented → q5 = highest OOD
    return df

def _make_quintiles(train_ids: List[str], ood_df: Optional[pd.DataFrame]) -> Dict[str, int]:
    """
    Return mapping system_id -> quintile (1..5), computed only over TRAIN ids.
    If OOD missing for some ids, assign q3 and warn once.
    """
    if not train_ids:
        return {}
    if ood_df is None:
        warnings.warn("No OOD CSV provided; assigning all training systems to q3.")
        return {sid: 3 for sid in train_ids}

    od = ood_df.set_index("cif_id")["OOD"].to_dict()
    vals, ids_with_vals, ids_missing = [], [], []
    for sid in train_ids:
        v = od.get(sid, None)
        if v is None or pd.isna(v):
            ids_missing.append(sid)
        else:
            ids_with_vals.append(sid)
            vals.append(float(v))
    if ids_missing:
        print(f"id missing: {ids_missing}")
        warnings.warn(f"{len(ids_missing)} train systems missing OOD; assigning q3.")

    # Rank by OOD descending (highest = most OOD)
    order = np.argsort(-np.array(vals)) if vals else np.array([])
    sorted_ids = [ids_with_vals[i] for i in order]

    # Split into 5 nearly equal bins
    quint = {}
    if sorted_ids:
        n = len(sorted_ids)
        # indices for 5 chunks
        cuts = np.linspace(0, n, 6, dtype=int)  # 0..n, 6 points → 5 bins
        for q in range(1, 6):
            lo, hi = cuts[q-1], cuts[q]
            for sid in sorted_ids[lo:hi]:
                quint[sid] = q  # q1..q5 (q5 = highest OOD)
    # Missing → q3
    for sid in ids_missing:
        quint[sid] = 3
    return quint

# ----------------- main writer -----------------

def write_grouped_extxyz_with_quintiles(
    vasp_roots: List[str],            # CHANGED: multiple roots
    out_train: str,
    out_val: str,
    out_test: str,
    out_train_ids_csv: str,
    out_val_ids_csv: str,
    out_test_ids_csv: str,
    *,
    test_csv: Optional[str] = None,
    ood_csv: Optional[str] = None,
    ood_col: str = "ood_centroid_cosine",
    include_virials: bool = False,
    val_size: float = 0.1,
    seed: int = 0,
    shards_dir: Optional[str] = None,
    chain_if_unconverged_only: bool = True,
    ndef_meta_csv: Optional[str] = None,
    ndef_shards_dir: Optional[str] = None,
    train_csv: Optional[str] = None,
    val_csv: Optional[str] = None
):
    # Collect all system ids across roots
    system_ids = _list_all_system_ids(vasp_roots)

    # Test IDs from test.csv and/or OOD is_test
    test_ids_from_file = _load_test_ids(Path(test_csv)) if test_csv else []
    ood_df = _prepare_ood_map(Path(ood_csv), prefer_col=ood_col) if ood_csv else None
    test_ids = _union_with_is_test(test_ids_from_file, ood_df)
    test_ids = sorted(set(test_ids).intersection(system_ids))  # keep only present systems

    # Split remaining into train / val
    if train_csv and val_csv:
        train_ids = pd.read_csv(Path(train_csv))["atoms_id"].astype(str).tolist()
        val_ids = pd.read_csv(Path(val_csv))["atoms_id"].astype(str).tolist()
    else:
        remain_ids = sorted([sid for sid in system_ids if sid not in set(test_ids)])
        if len(remain_ids) == 0:
            raise RuntimeError("No systems left for train/val after removing test set.")
        gss = GroupShuffleSplit(n_splits=1, test_size=float(val_size), random_state=seed + 1)
        ridx = np.arange(len(remain_ids))
        tr_idx, val_idx = next(gss.split(ridx, groups=remain_ids))
        train_ids = [remain_ids[i] for i in tr_idx]
        val_ids   = [remain_ids[i] for i in val_idx]

    # OOD quintiles on TRAIN ONLY
    quint_map = _make_quintiles(train_ids, ood_df)
    
    ndef_map = _prepare_ndef_map(Path(ndef_meta_csv)) if ndef_meta_csv else None
    print(f"ndef_map: {len(ndef_map)}")

    # Save id lists
    pd.DataFrame({"atoms_id": sorted(train_ids)}).to_csv(out_train_ids_csv, index=False)
    pd.DataFrame({"atoms_id": sorted(val_ids)}).to_csv(out_val_ids_csv, index=False)
    pd.DataFrame({"atoms_id": sorted(test_ids)}).to_csv(out_test_ids_csv, index=False)

    # Prepare outputs
    for p in (out_train, out_val, out_test):
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        if Path(p).exists():
            Path(p).unlink()

    shard_paths: Dict[int, Path] = {}
    if shards_dir:
        Path(shards_dir).mkdir(parents=True, exist_ok=True)
        for q in range(1, 6):
            sp = Path(shards_dir) / f"train_q{q}.xyz"
            if sp.exists(): sp.unlink()
            shard_paths[q] = sp

    ndef_shard_paths: Dict[int, Path] = {}
    ndef_counts: Dict[int, int] = defaultdict(int)

    if ndef_shards_dir:
        Path(ndef_shards_dir).mkdir(parents=True, exist_ok=True)
        # Pre-create files only for ndef values present in TRAIN ids (nice-to-have optimization)
        if ndef_map is None:
            warnings.warn("ndef_shards_dir set but no ndef_meta_csv provided; ndef shards will be skipped.")
        else:
            train_ndef_vals = sorted({ndef_map[sid] for sid in ndef_map.keys() if sid in set(train_ids)})
            for n in train_ndef_vals:
                sp = Path(ndef_shards_dir) / f"train_ndef{n}.xyz"
                if sp.exists():
                    sp.unlink()
                ndef_shard_paths[n] = sp

    counts = {"train": 0, "val": 0, "test": 0}
    shard_counts = {q: 0 for q in range(1, 6)}

    train_set, val_set, test_set = set(train_ids), set(val_ids), set(test_ids)

    # Walk systems once, chain frames across roots
    for sid in system_ids:
        if sid in train_set:
            merged_target = out_train
            qbin = quint_map.get(sid, 3)
            shard_target = shard_paths.get(qbin, None)
        elif sid in val_set:
            merged_target = out_val
            shard_target = None
        elif sid in test_set:
            merged_target = out_test
            shard_target = None
        else:
            warnings.warn(f"{sid} not in train/val/test; writing to test.")
            merged_target = out_test
            shard_target = None

        srcs = _sources_for_system(vasp_roots, sid)
        if not srcs:
            warnings.warn(f"Skip {sid}: no vasprun.xml/OUTCAR found in any root")
            continue

        step_idx = 0
        for atoms in _iter_frames_chain(srcs, only_chain_if_unconverged=chain_if_unconverged_only):
            atoms.info["system_id"] = sid
            atoms.info["frame"] = step_idx
            _attach_labels_for_mace(atoms, include_virials)
            write(merged_target, atoms, format="extxyz", append=True)
            if merged_target == out_train:
                counts["train"] += 1
                if shard_target is not None:
                    write(str(shard_target), atoms, format="extxyz", append=True)
                    shard_counts[qbin] += 1
                if ndef_shards_dir and ndef_map is not None:
                    ndef = ndef_map.get(sid, None)
                    # print(f"get sid: {sid}, ndef: {ndef}")
                    if ndef is not None:
                        sp = ndef_shard_paths.get(ndef)
                        if sp is None:
                            # lazy-create path if this ndef wasn't pre-created
                            sp = Path(ndef_shards_dir) / f"train_ndef{ndef}.xyz"
                            if sp.exists():
                                sp.unlink()
                            ndef_shard_paths[ndef] = sp
                        write(str(sp), atoms, format="extxyz", append=True)
                        ndef_counts[ndef] += 1
            elif merged_target == out_val:
                counts["val"] += 1
            else:
                counts["test"] += 1
            step_idx += 1

    print(f"[Done] Frames written: train={counts['train']}, val={counts['val']}, test={counts['test']}")
    if shards_dir:
        for q in range(1, 6):
            sp = Path(shards_dir) / f"train_q{q}.xyz"
            present = sp.exists() and sp.stat().st_size > 0
            print(f"  shard q{q}: frames={shard_counts[q]}  file={'ok' if present else 'empty'}")
    if ndef_shards_dir and ndef_map is not None:
        print("  ndef shards:")
        print(f"ndef_counts: {len(ndef_counts)}")
        for n in sorted(ndef_counts.keys()):
            sp = Path(ndef_shards_dir) / f"train_ndef{n}.xyz"
            present = sp.exists() and sp.stat().st_size > 0
            print(f"    train_ndef{n}: frames={ndef_counts[n]}  file={'ok' if present else 'empty'}")


# ----------------- CLI -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Config Json")
    args = ap.parse_args()
    cfg = json.load(open(args.config, "r"))

    # NEW: list of roots
    vasp_roots = cfg["vasp_roots"] if isinstance(cfg["vasp_roots"], list) else [cfg["vasp_roots"]]

    write_grouped_extxyz_with_quintiles(
        vasp_roots=vasp_roots,
        out_train=cfg["out_train"],
        out_val=cfg["out_val"],
        out_test=cfg["out_test"],
        out_train_ids_csv=cfg["out_train_csv"],
        out_val_ids_csv=cfg["out_val_csv"],
        out_test_ids_csv=cfg["out_test_csv"],
        test_csv=cfg.get("test_csv"),
        ood_csv=cfg.get("ood_csv"),
        ood_col=cfg.get("ood_col", "ood_centroid_cosine"),
        val_size=float(cfg.get("val_size", 0.1)),
        seed=int(cfg.get("seed", 0)),
        include_virials=bool(cfg.get("include_virials", False)),  # FIXED
        shards_dir=cfg.get("shards_dir"),
        chain_if_unconverged_only=bool(cfg.get("chain_if_unconverged_only", True)),
        ndef_meta_csv=cfg.get("ndef_meta_csv"),
        ndef_shards_dir=cfg.get("ndef_shards_dir"),
        train_csv=cfg.get("train_csv", None),
        val_csv=cfg.get("val_csv", None)
    )

if __name__ == "__main__":
    main()
