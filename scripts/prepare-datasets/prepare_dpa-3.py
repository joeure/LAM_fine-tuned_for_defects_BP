#!/usr/bin/env python3
import os, json, argparse, warnings
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
from periodictable import elements as _els
from sklearn.model_selection import GroupShuffleSplit
try:
    import dpdata
except Exception as e:
    raise SystemExit(
        "dpdata is required: pip install dpdata\n"
        f"Import error: {e}"
    )
    
# ---------- multi-root VASP discovery ----------
import shutil
import tempfile

def _find_vasp_source_in_dir(sys_dir: Path) -> tuple[Optional[Path], Optional[str]]:
    """Prefer vasprun.xml (vasp/xml), fallback OUTCAR (vasp/outcar)."""
    p = sys_dir / "vasprun.xml"
    if p.is_file():
        return p, "vasp/xml"
    p = sys_dir / "OUTCAR"
    if p.is_file():
        return p, "vasp/outcar"
    return None, None

def _scan_all_sources_multi(vasp_roots: list[str]) -> dict[str, list[tuple[Path, str]]]:
    """
    Return mapping: cif_id -> [(src_path, dpdata_fmt), ...] in the priority order of vasp_roots.
    Assumes subdirs named '{cif_id}_unrelaxed' under each root.
    """
    out: dict[str, list[tuple[Path, str]]] = {}
    for root in [Path(r) for r in vasp_roots]:
        if not root.is_dir():
            continue
        for child in sorted(p for p in root.iterdir() if p.is_dir()):
            name = child.name
            if not name.endswith("_unrelaxed"):
                continue
            cif_id = name[:-10]
            src, fmt = _find_vasp_source_in_dir(child)
            if src is None:
                continue
            out.setdefault(cif_id, []).append((src, fmt))
    return out

def _decide_first_src_for_typemap(id_to_srcs: dict[str, list[tuple[Path, str]]]) -> dict[str, tuple[Path, str]]:
    """Pick the first available source for each cif_id (for fast element probing)."""
    return {cid: srclist[0] for cid, srclist in id_to_srcs.items() if srclist}

# ---------- deepmd set merging (no in-memory concat needed) ----------
from collections import defaultdict

def _system_id_from_cif(cif_id: str) -> str:
    # Trim the trailing UUID-like token: "P_P126N9_<uuid>" -> "P_P126N9"
    parts = str(cif_id).split("_")
    return "_".join(parts[:-1]) if len(parts) > 1 else str(cif_id)

def _prepare_ndef_maps(meta_csv: Optional[Path]) -> tuple[Dict[str, int], Dict[str, int]]:
    """
    Returns two mappings:
      - ndef_by_cif_id:    {cif_id -> ndef}
      - ndef_by_system_id: {system_id -> ndef}  where system_id = system+'_'+formula  OR  derived from cif_id

    CSV accepted schemas:
      (A) ['cif_id','nsubs','nvacs', ...]
      (B) ['system','formula','nsubs','nvacs', ...]
    If multiple rows map to the same key with inconsistent ndef, pick the mode and warn once.
    """
    if not meta_csv:
        return {}, {}

    df = pd.read_csv(meta_csv)

    if not {"nsubs", "nvacs"}.issubset(df.columns):
        raise ValueError(f"{meta_csv} must contain 'nsubs' and 'nvacs' columns.")

    df = df.copy()
    df["ndef"] = df["nsubs"].astype(int) + df["nvacs"].astype(int)

    ndef_by_cif: Dict[str, int] = {}
    ndef_by_sys: Dict[str, int] = {}

    # Build from cif_id if present
    if "cif_id" in df.columns:
        grp = df.groupby("cif_id")["ndef"]
        for cid, series in grp:
            vals = series.dropna().astype(int).tolist()
            if not vals:
                continue
            if len(set(vals)) > 1:
                warnings.warn(f"[ndef] Inconsistent ndef for cif_id={cid}: {sorted(set(vals))}. Using mode.")
            mode_val = pd.Series(vals).value_counts().index[0]
            ndef_by_cif[str(cid)] = int(mode_val)
            # also fill its system_id projection
            sys_id = _system_id_from_cif(str(cid))
            ndef_by_sys.setdefault(sys_id, int(mode_val))

    # Build from (system, formula) if present
    if {"system", "formula"}.issubset(df.columns):
        tmp = df.copy()
        tmp["system_id"] = tmp["system"].astype(str).str.strip() + "_" + tmp["formula"].astype(str).str.strip()
        grp = tmp.groupby("system_id")["ndef"]
        for sid, series in grp:
            vals = series.dropna().astype(int).tolist()
            if not vals:
                continue
            if len(set(vals)) > 1:
                warnings.warn(f"[ndef] Inconsistent ndef for system_id={sid}: {sorted(set(vals))}. Using mode.")
            mode_val = pd.Series(vals).value_counts().index[0]
            ndef_by_sys[str(sid)] = int(mode_val)

    return ndef_by_cif, ndef_by_sys

def _merge_deepmd_sets(tmp_out: Path, dest: Path, start_k: int) -> int:
    """
    Move set.* folders from tmp_out into dest as set.{start_k + i:03d}.
    Returns the next available index after merging.
    """
    dest.mkdir(parents=True, exist_ok=True)
    k = start_k
    for sub in sorted(tmp_out.iterdir()):
        if not (sub.is_dir() and sub.name.startswith("set.")):
            continue
        new_name = f"set.{k:03d}"
        shutil.move(str(sub), str(dest / new_name))
        k += 1
    # Copy/ensure single type_map.raw at dest root if present in tmp_out
    tm = tmp_out / "type_map.raw"
    if tm.exists() and not (dest / "type_map.raw").exists():
        shutil.copy2(tm, dest / "type_map.raw")
    return k


# ---------- helpers ----------
def _read_test_ids(csv_path: str) -> set[str]:
    """Accepts a CSV that has either 'cif_id' or 'atoms_id' and returns the set of cif_ids."""
    df = pd.read_csv(csv_path)
    if "cif_id" in df.columns:
        ids = df["cif_id"].astype(str).tolist()
    elif "atoms_id" in df.columns:
        ids = df["atoms_id"].astype(str).tolist()
    else:
        raise ValueError("CSV must contain 'cif_id' or 'atoms_id'.")
    return set(ids)

def _find_vasp_source(sys_dir: Path) -> Tuple[Optional[Path], Optional[str]]:
    """Return (path, format_string_for_dpdata). Prefer vasprun.xml, fallback OUTCAR."""
    vasprun = sys_dir / "vasprun.xml"
    if vasprun.is_file():
        return vasprun, "vasp/xml"
    outcar = sys_dir / "OUTCAR"
    if outcar.is_file():
        return outcar, "vasp/outcar"
    return None, None

def _decide_set_size(n_frames: int,
                     strategy: str = "one",
                     fixed_set_size: int | None = None,
                     auto_single_threshold: int = 5000,
                     auto_chunk_size: int = 2000) -> int:
    if n_frames <= 0:
        return 1
    if strategy == "one":
        return n_frames
    if strategy == "fixed":
        if not fixed_set_size or fixed_set_size < 1:
            raise ValueError("fixed_set_size must be a positive integer when strategy='fixed'.")
        return fixed_set_size
    if strategy == "auto":
        return n_frames if n_frames <= auto_single_threshold else max(1, auto_chunk_size)
    raise ValueError(f"Unknown set sharding strategy: {strategy}")

def _elements_from_lsys(lsys: "dpdata.LabeledSystem") -> list[str]:
    try:
        names = list(lsys["atom_names"])
    except Exception:
        names = list(getattr(lsys, "atom_names", []))
    return [str(x) for x in names]

def _atomic_number(sym: str) -> int:
    el = getattr(_els, sym, None)
    return int(el.number) if el is not None else 10_000

def _unified_type_map(all_syms: set[str], user_order: Optional[List[str]] = None) -> list[str]:
    if user_order:
        seen = set(all_syms)
        ordered = [s for s in user_order if s in seen]
        remaining = sorted([s for s in all_syms if s not in set(ordered)], key=_atomic_number)
        return ordered + remaining
    return sorted(all_syms, key=_atomic_number)

def _prepare_ood_map(ood_csv: Optional[Path],
                     prefer_col: str = "ood_knn_cosine",
                     fallback_col: str = "ood_centroid_cosine") -> Optional[pd.DataFrame]:
    if not ood_csv:
        return None
    df = pd.read_csv(ood_csv)
    if "cif_id" not in df.columns:
        raise ValueError(f"{ood_csv} must contain 'cif_id'.")
    use_col = prefer_col if prefer_col in df.columns else fallback_col if fallback_col in df.columns else None
    if use_col is None:
        raise ValueError(f"{ood_csv} must contain '{prefer_col}' or '{fallback_col}'.")
    return df[["cif_id", use_col]].rename(columns={use_col: "OOD"})

def _train_quintiles(train_ids: List[str], ood_df: Optional[pd.DataFrame]) -> Dict[str, int]:
    """Map each TRAIN cif_id -> quintile (1..5), q5 = highest OOD. Missing OOD → q3."""
    if not train_ids:
        return {}
    if ood_df is None:
        warnings.warn("No OOD CSV provided; assigning all training systems to q3.")
        return {sid: 3 for sid in train_ids}
    od = ood_df.set_index("cif_id")["OOD"].to_dict()
    have, vals, miss = [], [], []
    for sid in train_ids:
        v = od.get(sid, None)
        if v is None or pd.isna(v):
            miss.append(sid)
        else:
            have.append(sid); vals.append(float(v))
    if miss:
        warnings.warn(f"{len(miss)} train systems missing OOD; assigning q3.")
    quint = {}
    if have:
        order = np.argsort(-np.array(vals))  # descending OOD
        sorted_ids = [have[i] for i in order]
        n = len(sorted_ids)
        cuts = np.linspace(0, n, 6, dtype=int)
        for q in range(1, 6):
            lo, hi = cuts[q-1], cuts[q]
            for sid in sorted_ids[lo:hi]:
                quint[sid] = q
    for sid in miss:
        quint[sid] = 3
    return quint
# ------------------------------------------------------------------------------

def _write_split(
    split_name: str,
    cids: list[str],
    *,
    set_strategy: str,
    fixed_set_size: int | None,
    auto_single_threshold: int,
    auto_chunk_size: int,
    frame_stride: int,
    max_frames_per_system: int | None,
    type_map: list[str],
    id_to_srcs: dict[str, list[tuple[Path, str]]],
    out_root: Path,
    dry_run: bool = False
) -> tuple[List[Dict], int]:
    """Return (entries, total_frames); entries: list of {cif_id, system_dir, n_frames}"""
    entries: List[Dict] = []
    total_frames = 0
    for j, cid in enumerate(cids):
        srclist = id_to_srcs.get(cid, [])
        if not srclist:
            continue

        dest = out_root / split_name / f"system.{j:05d}_{cid}"
        entries_n = 0
        next_k = 0

        for s_idx, (src, fmt) in enumerate(srclist):
            try:
                lsys = dpdata.LabeledSystem(str(src), fmt=fmt)
            except Exception as e:
                warnings.warn(f"[skip] {cid}: read error at {src} ({e})")
                continue

            if len(lsys) == 0:
                continue

            # For continuations, drop first frame (duplicate CONTCAR restart)
            if s_idx > 0 and len(lsys) > 1:
                lsys = lsys.sub_system(np.arange(1, len(lsys), 1))

            # Apply stride / cap per system-chunk
            n0 = len(lsys)
            if frame_stride > 1 or max_frames_per_system is not None:
                idx = np.arange(0, n0, max(1, int(frame_stride)))
                if max_frames_per_system is not None:
                    idx = idx[:max_frames_per_system]
                if idx.size == 0:
                    idx = np.arange(min(n0, 1))
                lsys = lsys.sub_system(idx)

            if len(lsys) == 0:
                continue

            lsys.map_atom_types(type_map=type_map)
            entries_n += len(lsys)

            # Decide chunk size for this piece (keep your policy)
            set_size_here = _decide_set_size(
                len(lsys),
                strategy=set_strategy,
                fixed_set_size=fixed_set_size,
                auto_single_threshold=auto_single_threshold,
                auto_chunk_size=auto_chunk_size,
            )

            if dry_run:
                continue

            # Write this piece to a temp dir, then merge its set.* into dest
            with tempfile.TemporaryDirectory(prefix="dp_tmp_") as td:
                tmp_out = Path(td)
                lsys.to("deepmd/npy", str(tmp_out), set_size=set_size_here, set_prefix="set")
                # ensure a single type_map.raw (first create wins)
                next_k = _merge_deepmd_sets(tmp_out, dest, start_k=next_k)

        if entries_n > 0:
            entries.append({"cif_id": cid, "system_dir": str(dest), "n_frames": int(entries_n)})
            total_frames += entries_n
    return entries, total_frames

# ---------- main pipeline ----------
def build_dpdata_from_vasp(
    vasp_roots: list[str],
    out_root: str,
    test_csv: str,
    *,
    val_size: float = 0.1,
    seed: int = 0,
    frame_stride: int = 1,
    max_frames_per_system: Optional[int] = None,
    set_strategy: str = "one",
    fixed_set_size: int | None = None,
    auto_single_threshold: int = 5000,
    auto_chunk_size: int = 2000,
    user_type_map: Optional[List[str]] = None,
    ood_csv: Optional[str] = None,
    ood_col_prefer: str = "ood_knn_cosine",
    ood_col_fallback: str = "ood_centroid_cosine",
    shards_dir: Optional[str] = None,
    dry_run: bool = False,
    ndef_meta_csv: Optional[str] = None,
    ndef_shards_dir: Optional[str] = None,
    train_csv: Optional[str] = None,
    val_csv: Optional[str] = None
) -> dict:
    """
    Make DeepMD (DPA-3) systems for train/val/test from VASP relaxations.

    NEW:
      - writes train/val/test IDs CSVs
      - computes TRAIN OOD quintiles (q1..q5) and writes:
          * shards/train_quintiles.csv (cif_id,quintile,system_dir)
          * shards/train_q{1..5}_systems.txt (paths to per-system dirs)
    """
    oroot = Path(out_root).resolve()
    oroot.mkdir(parents=True, exist_ok=True)

    # 1) discover systems across multiple roots
    id_to_srcs = _scan_all_sources_multi(vasp_roots)
    if not id_to_srcs:
        raise RuntimeError(f"No VASP systems found under any of: {vasp_roots}")
    all_ids = sorted(id_to_srcs.keys())
    id_to_src_for_probe = _decide_first_src_for_typemap(id_to_srcs)  # for quick element probing

    # 2) test split (fixed)
    test_ids = _read_test_ids(test_csv)
    test_ids = {cid for cid in all_ids if cid in test_ids}
    remain_ids = [cid for cid in all_ids if cid not in test_ids]

    # 3) group split remain_ids → train/val
    if len(remain_ids) == 0:
        raise RuntimeError("All systems are in the test CSV; nothing left for train/val.")
    if train_csv and val_csv:
        train_ids = pd.read_csv(Path(train_csv))["atoms_id"].astype(str).to_list()
        val_ids = pd.read_csv(Path(val_csv))["atoms_id"].astype(str).to_list()
    else:
        ridx = np.arange(len(remain_ids))
        gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        tr_idx, va_idx = next(gss.split(ridx, groups=remain_ids))
        train_ids = [remain_ids[i] for i in tr_idx]
        val_ids   = [remain_ids[i] for i in va_idx]

    # 4) first pass: collect union of elements for a unified type_map
    union_syms = set()
    probe_ids = train_ids + val_ids + list(test_ids)
    for cid in probe_ids:
        if cid not in id_to_src_for_probe:
            continue
        src, fmt = id_to_src_for_probe[cid]
        try:
            lsys = dpdata.LabeledSystem(str(src), fmt=fmt)
        except Exception as e:
            warnings.warn(f"[skip] {cid}: failed to read {src} ({e})")
            continue
        union_syms.update(_elements_from_lsys(lsys))
    if not union_syms:
        raise RuntimeError("Could not determine element set from any system.")
    type_map = _unified_type_map(union_syms, user_order=user_type_map)
    print(f"id_to_src: {len(id_to_srcs)}")

    # 5) write splits
    train_entries, ntrain = _write_split(
        "train", train_ids,
        set_strategy=set_strategy,
        fixed_set_size=fixed_set_size,
        auto_single_threshold=auto_single_threshold,
        auto_chunk_size=auto_chunk_size,
        frame_stride=frame_stride,
        max_frames_per_system=max_frames_per_system,
        type_map=type_map,
        id_to_srcs=id_to_srcs,
        out_root=oroot,
        dry_run=dry_run,
    )
    val_entries, nval = _write_split(
        "val", val_ids,
        set_strategy=set_strategy,
        fixed_set_size=fixed_set_size,
        auto_single_threshold=auto_single_threshold,
        auto_chunk_size=auto_chunk_size,
        frame_stride=frame_stride,
        max_frames_per_system=max_frames_per_system,
        type_map=type_map,
        id_to_srcs=id_to_srcs,
        out_root=oroot,
        dry_run=dry_run,
    )
    test_entries, ntest = _write_split(
        "test", sorted(list(test_ids)),
        set_strategy=set_strategy,
        fixed_set_size=fixed_set_size,
        auto_single_threshold=auto_single_threshold,
        auto_chunk_size=auto_chunk_size,
        frame_stride=frame_stride,
        max_frames_per_system=max_frames_per_system,
        type_map=type_map,
        id_to_srcs=id_to_srcs,
        out_root=oroot,
        dry_run=dry_run,
    )

    # 6) save systems lists for DeepMD JSON
    train_dirs = [e["system_dir"] for e in train_entries]
    val_dirs   = [e["system_dir"] for e in val_entries]
    test_dirs  = [e["system_dir"] for e in test_entries]
    (oroot / "train_systems.txt").write_text("\n".join(train_dirs) + "\n", encoding="utf-8")
    (oroot / "val_systems.txt").write_text("\n".join(val_dirs) + "\n", encoding="utf-8")
    (oroot / "test_systems.txt").write_text("\n".join(test_dirs) + "\n", encoding="utf-8")

    # 7) NEW: write IDs CSVs (with optional frame counts)
    pd.DataFrame(train_entries).loc[:, ["cif_id", "n_frames"]].to_csv(oroot / "train_ids.csv", index=False)
    pd.DataFrame(val_entries).loc[:,   ["cif_id", "n_frames"]].to_csv(oroot / "val_ids.csv",   index=False)
    pd.DataFrame(test_entries).loc[:,  ["cif_id", "n_frames"]].to_csv(oroot / "test_ids.csv",  index=False)

    # 8) NEW: TRAIN OOD quintiles + shard lists (no data duplication)
    if shards_dir:
        sdir = Path(shards_dir); sdir.mkdir(parents=True, exist_ok=True)
        ood_df = _prepare_ood_map(Path(ood_csv), prefer_col=ood_col_prefer, fallback_col=ood_col_fallback) if ood_csv else None
        quint = _train_quintiles([e["cif_id"] for e in train_entries], ood_df)

        # Save per-system quintile mapping
        qrows = []
        for e in train_entries:
            cid = e["cif_id"]; q = quint.get(cid, 3)
            qrows.append((cid, q, e["system_dir"], e["n_frames"]))
        pd.DataFrame(qrows, columns=["cif_id", "quintile", "system_dir", "n_frames"])\
          .sort_values(["quintile", "cif_id"])\
          .to_csv(sdir / "train_quintiles.csv", index=False)

        # Write shard list files DeepMD can consume (each line = a system dir)
        for q in range(1, 6):
            paths = [e["system_dir"] for e in train_entries if quint.get(e["cif_id"], 3) == q]
            (sdir / f"train_q{q}_systems.txt").write_text("\n".join(paths) + ("\n" if paths else ""), encoding="utf-8")

    # 8b) NEW: TRAIN ndef shards (ndef = nsubs + nvacs), optional and independent of OOD
    if ndef_shards_dir:
        nsdir = Path(ndef_shards_dir); nsdir.mkdir(parents=True, exist_ok=True)
        ndef_by_cif, ndef_by_sys = _prepare_ndef_maps(Path(ndef_meta_csv)) if ndef_meta_csv else ({}, {})
        if not ndef_by_cif and not ndef_by_sys:
            warnings.warn("ndef_shards_dir provided but no valid ndef_meta_csv or no usable columns; skipping ndef shards.")
        else:
            rows = []
            missing = 0
            for e in train_entries:
                cid = e["cif_id"]
                ndef = ndef_by_cif.get(cid)
                if ndef is None:
                    ndef = ndef_by_sys.get(_system_id_from_cif(cid))
                if ndef is None:
                    missing += 1
                    continue
                rows.append((cid, int(ndef), e["system_dir"], int(e["n_frames"])))

            if rows:
                ndef_df = pd.DataFrame(rows, columns=["cif_id", "ndef", "system_dir", "n_frames"])\
                           .sort_values(["ndef", "cif_id"])
                ndef_df.to_csv(nsdir / "train_ndef.csv", index=False)

                # One file per ndef with DeepMD system dirs
                for ndef_val, g in ndef_df.groupby("ndef"):
                    paths = list(g["system_dir"])
                    (nsdir / f"train_ndef{int(ndef_val)}_systems.txt").write_text(
                        "\n".join(paths) + ("\n" if paths else ""), encoding="utf-8"
                    )
                if missing:
                    print(f"[ndef] Train systems without ndef in metadata: {missing}")
            else:
                print("[ndef] No train entries matched metadata; no ndef shards written.")

    # 9) manifest for traceability
    rows = []
    for split, entries in (("train", train_entries), ("val", val_entries), ("test", test_entries)):
        for e in entries:
            rows.append((split, e["cif_id"], e["system_dir"], e["n_frames"]))
    man = pd.DataFrame(rows, columns=["split","cif_id","system_dir","n_frames"])
    man["type_map"] = [",".join(type_map)] * len(man)
    man.to_csv(oroot / "manifest.csv", index=False)

    return {
        "n_train_frames": ntrain,
        "n_val_frames": nval,
        "n_test_frames": ntest,
        "n_train_systems": len(train_entries),
        "n_val_systems": len(val_entries),
        "n_test_systems": len(test_entries),
        "type_map": type_map,
        "root": str(oroot),
        "shards_dir": str(shards_dir) if shards_dir else None,
    }

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Config Json")
    args = ap.parse_args()
    config_file = args.config
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file {config_file} does not exist.")
    with open(config_file, 'r') as file:
        config = json.load(file)

    user_map = [s.strip() for s in config["type_map"].split(",") if s.strip()] or None

    vasp_roots = config["vasp_roots"] if isinstance(config.get("vasp_roots"), list) else [config["vasp_root"]]
    summary = build_dpdata_from_vasp(
        vasp_roots=vasp_roots,
        out_root=config["out_root"],
        test_csv=config["test_csv"],
        val_size=config["val_size"],
        seed=config["seed"],
        frame_stride=1,
        max_frames_per_system=None,
        set_strategy=config["chunk_strategy"], # one: one system on chunk, fixed: same size of chunk, auto: small as one chunk, large splitted
        fixed_set_size=None, # only useful for "fixed"
        auto_single_threshold=config.get("auto_single_threshold", 5000), # if later seen extremely long traj, need to chunk, default 5000
        auto_chunk_size=config.get("auto_chunk_size", 2000),
        user_type_map=user_map, # Firstly None because maybe I need replay later?
        ood_csv=config["ood_csv"],
        ood_col_prefer=config["ood_col_prefer"],
        ood_col_fallback=config["ood_col_fallback"],
        shards_dir=config["shards_dir"],
        dry_run=config["dry_run"],
        ndef_meta_csv=config.get("ndef_meta_csv"),
        ndef_shards_dir=config.get("ndef_shards_dir"),
    )
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
