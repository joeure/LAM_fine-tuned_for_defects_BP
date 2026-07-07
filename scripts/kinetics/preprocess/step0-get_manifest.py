from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def _boolify_is_test(x) -> bool:
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in {"true", "1", "yes", "y", "t"}


def _join_indices(series_val) -> str:
    if pd.isna(series_val):
        return ""
    return " ".join(str(series_val).strip().split())


def _safe_str(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    return s


def _maybe_resolve_dir_to_file(p: str | Path, filename: str) -> str:
    pp = Path(p)
    if pp.is_dir():
        return str(pp / filename)
    return str(pp)


def _id_from_cif_id(cif_id: str, lam_prefix: str) -> str:
    # Keep your existing transformation logic: do NOT change semantics.
    # Examples:
    #   cif_id: P_P126N9_uuid
    #   lam_id: BP_spin_500_P126N9_uuid
    tail = "_".join(cif_id.split("_")[1:])
    return f"{lam_prefix}{tail}"


def generate_manifest(
    csv1_path: str | Path,
    csv2_path: str | Path,
    dft_dir: str | Path,
    base_parent_dir: str | Path,     # NEW: parent folder (contains features/ and results/)
    ft_parent_dir: str | Path,       # NEW: parent folder (contains features/ and results/)
    base_dumps_dir: str | Path | None,
    ft_dumps_dir: str | Path | None,
    unrelaxed_cif_dir: str | Path | None,
    out_csv: str | Path,
    *,
    dft_file: str = "OUTCAR",
    lam_prefix: str = "BP_spin_500_",
    include_only_test_for_lam: bool = True,
    include_non_test_rows: bool = True,
    strict_path_exists: bool = False,

    # ---- pristine inputs (optional)
    pristine_system_id: str | None = "pristine",
    pristine_dft_path: str | None = None,
    pristine_base_lmp: str | None = None,
    pristine_ft_lmp: str | None = None,
    pristine_cif: str | None = None,
    pristine_base_dump: str | None = None,
    pristine_ft_dump: str | None = None,
    pristine_formula: str = "P4",
    pristine_dft_natoms: int = 144,
    pristine_lam_natoms: int = 4,
    pristine_dft_supercell: str = "6x6",
    pristine_lam_supercell: str = "1x1",
) -> Path:
    """
    Build manifest for BP defect dataset + optional pristine row.

    Defect rows:
      - system_id = cif_id (P_PxxxNyyy_uuid)
      - lam_id = BP_spin_500_ + tail(cif_id)
      - cif_unrelaxed = {unrelaxed_cif_dir}/{cif_id}_unrelaxed.cif (optional)
      - dft_path = {dft_dir}/{cif_id}_unrelaxed/{dft_file}

      Base relaxation:
        base_in_lmp  = {base_parent_dir}/features/{lam_id}_unrelaxed.data
        base_out_lmp = {base_parent_dir}/results/{lam_id}_unrelaxed.data
        base_dump    = {base_dumps_dir}/{lam_id}_unrelaxed/traj.lammpstrj (optional)

      FT relaxation:
        ft_in_lmp  = {ft_parent_dir}/features/{lam_id}_unrelaxed.data
        ft_out_lmp = {ft_parent_dir}/results/{lam_id}_unrelaxed.data
        ft_dump    = {ft_dumps_dir}/{lam_id}_unrelaxed/traj.lammpstrj (optional)

    Pristine row:
      - keeps your previous logic; plus optional CIF/dumps.
    """
    csv1_path = Path(csv1_path)
    csv2_path = Path(csv2_path)
    dft_dir = Path(dft_dir)
    base_parent_dir = Path(base_parent_dir)
    ft_parent_dir = Path(ft_parent_dir)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    base_dumps_dir = Path(base_dumps_dir) if base_dumps_dir else None
    ft_dumps_dir = Path(ft_dumps_dir) if ft_dumps_dir else None
    unrelaxed_cif_dir = Path(unrelaxed_cif_dir) if unrelaxed_cif_dir else None

    # ---------- defect systems ----------
    df1 = pd.read_csv(csv1_path)
    required1 = {"cif_id", "formula", "natoms", "nsubs", "nvacs", "is_test"}
    missing1 = required1 - set(df1.columns)
    if missing1:
        raise ValueError(f"CSV1 missing columns: {sorted(missing1)}")

    df1 = df1.copy()
    df1["cif_id"] = df1["cif_id"].astype(str)
    df1["is_test_bool"] = df1["is_test"].map(_boolify_is_test)

    if include_non_test_rows:
        df1_keep = df1
    else:
        df1_keep = df1[df1["is_test_bool"]].copy()

    cif_set = set(df1_keep["cif_id"].tolist())

    df2 = pd.read_csv(csv2_path)
    required2 = {"cif_id", "idx_subs", "idx_vacs"}
    missing2 = required2 - set(df2.columns)
    if missing2:
        raise ValueError(f"CSV2 missing columns: {sorted(missing2)}")

    df2 = df2.copy()
    df2["cif_id"] = df2["cif_id"].astype(str)
    df2 = df2[df2["cif_id"].isin(cif_set)].copy()
    df2 = df2.drop_duplicates(subset=["cif_id"], keep="first")
    df2["idx_subs_str"] = df2["idx_subs"].map(_join_indices)
    df2["idx_vacs_str"] = df2["idx_vacs"].map(_join_indices)

    df = df1_keep.merge(
        df2[["cif_id", "idx_subs_str", "idx_vacs_str"]],
        on="cif_id",
        how="left",
        validate="one_to_one",
    )

    df["split"] = df["is_test_bool"].map(lambda x: "test" if x else "trainval")

    # IDs
    df["lam_id"] = df["cif_id"].map(lambda cid: _id_from_cif_id(cid, lam_prefix))

    # paths
    df["dft_path"] = df["cif_id"].map(lambda cid: str(dft_dir / f"{cid}_unrelaxed" / dft_file))

    if unrelaxed_cif_dir is not None:
        df["cif_unrelaxed"] = df["cif_id"].map(lambda cid: str(unrelaxed_cif_dir / f"{cid}_unrelaxed.cif"))
    else:
        df["cif_unrelaxed"] = ""

    def base_in_path(lam_id: str) -> str:
        return str(base_parent_dir / "features" / f"{lam_id}_unrelaxed.data")

    def base_out_path(lam_id: str) -> str:
        return str(base_parent_dir / "results" / f"{lam_id}_unrelaxed.data")

    def ft_in_path(lam_id: str) -> str:
        return str(ft_parent_dir / "features" / f"{lam_id}_unrelaxed.data")

    def ft_out_path(lam_id: str) -> str:
        return str(ft_parent_dir / "results" / f"{lam_id}_unrelaxed.data")

    def base_dump_path(lam_id: str) -> str:
        if base_dumps_dir is None:
            return ""
        return str(base_dumps_dir / f"{lam_id}_unrelaxed" / "traj.lammpstrj")

    def ft_dump_path(lam_id: str) -> str:
        if ft_dumps_dir is None:
            return ""
        return str(ft_dumps_dir / f"{lam_id}_unrelaxed" / "traj.lammpstrj")

    if include_only_test_for_lam:
        df["base_in_lmp"] = df.apply(lambda r: base_in_path(r["lam_id"]) if r["is_test_bool"] else "", axis=1)
        df["base_out_lmp"] = df.apply(lambda r: base_out_path(r["lam_id"]) if r["is_test_bool"] else "", axis=1)
        df["ft_in_lmp"] = df.apply(lambda r: ft_in_path(r["lam_id"]) if r["is_test_bool"] else "", axis=1)
        df["ft_out_lmp"] = df.apply(lambda r: ft_out_path(r["lam_id"]) if r["is_test_bool"] else "", axis=1)

        df["base_dump"] = df.apply(lambda r: base_dump_path(r["lam_id"]) if r["is_test_bool"] else "", axis=1)
        df["ft_dump"] = df.apply(lambda r: ft_dump_path(r["lam_id"]) if r["is_test_bool"] else "", axis=1)
    else:
        df["base_in_lmp"] = df["lam_id"].map(base_in_path)
        df["base_out_lmp"] = df["lam_id"].map(base_out_path)
        df["ft_in_lmp"] = df["lam_id"].map(ft_in_path)
        df["ft_out_lmp"] = df["lam_id"].map(ft_out_path)
        df["base_dump"] = df["lam_id"].map(base_dump_path)
        df["ft_dump"] = df["lam_id"].map(ft_dump_path)

    # defect row flags
    df["is_pristine"] = False
    df["dft_supercell_tag"] = "6x6"
    df["lam_supercell_tag"] = "6x6"
    df["dft_natoms_expected"] = 144
    df["base_natoms_expected"] = 144
    df["ft_natoms_expected"] = 144
    df["pristine_mode"] = ""

    out_cols = [
        "cif_id",
        "lam_id",
        "split",
        "is_pristine",
        "formula",
        "natoms",
        "nsubs",
        "nvacs",
        "idx_subs_str",
        "idx_vacs_str",
        "cif_unrelaxed",
        "dft_path",
        "base_in_lmp",
        "base_out_lmp",
        "base_dump",
        "ft_in_lmp",
        "ft_out_lmp",
        "ft_dump",
        "dft_supercell_tag",
        "lam_supercell_tag",
        "dft_natoms_expected",
        "base_natoms_expected",
        "ft_natoms_expected",
        "pristine_mode",
    ]

    df_out = df[out_cols].rename(columns={
        "cif_id": "system_id",
        "idx_subs_str": "idx_subs",
        "idx_vacs_str": "idx_vacs",
    })

    # ---------- path existence checks (optional) ----------
    if strict_path_exists:
        def _check_exists(path_str: str, label: str, sid: str):
            if not path_str:
                return
            if not Path(path_str).exists():
                raise FileNotFoundError(f"[missing {label}] {sid}: {path_str}")

        for _, r in df_out.iterrows():
            sid = r["system_id"]
            _check_exists(r["dft_path"], "dft_path", sid)
            if r["split"] == "test":
                _check_exists(r["base_in_lmp"], "base_in_lmp", sid)
                _check_exists(r["base_out_lmp"], "base_out_lmp", sid)
                if base_dumps_dir is not None:
                    _check_exists(r["base_dump"], "base_dump", sid)
                _check_exists(r["ft_in_lmp"], "ft_in_lmp", sid)
                _check_exists(r["ft_out_lmp"], "ft_out_lmp", sid)
                if ft_dumps_dir is not None:
                    _check_exists(r["ft_dump"], "ft_dump", sid)
            if unrelaxed_cif_dir is not None:
                _check_exists(r["cif_unrelaxed"], "cif_unrelaxed", sid)

    # ---------- pristine row ----------
    if pristine_system_id and pristine_dft_path and pristine_base_lmp and pristine_ft_lmp:
        p_dft = _maybe_resolve_dir_to_file(pristine_dft_path, dft_file)

        pristine_row = {
            "system_id": pristine_system_id,
            "lam_id": pristine_system_id,
            "split": "pristine",
            "is_pristine": True,
            "formula": pristine_formula,
            "natoms": pristine_dft_natoms,
            "nsubs": 0,
            "nvacs": 0,
            "idx_subs": "",
            "idx_vacs": "",
            "cif_unrelaxed": _safe_str(pristine_cif),
            "dft_path": str(p_dft),
            "base_in_lmp": "",
            "base_out_lmp": str(Path(pristine_base_lmp)),
            "base_dump": _safe_str(pristine_base_dump),
            "ft_in_lmp": "",
            "ft_out_lmp": str(Path(pristine_ft_lmp)),
            "ft_dump": _safe_str(pristine_ft_dump),
            "dft_supercell_tag": pristine_dft_supercell,
            "lam_supercell_tag": pristine_lam_supercell,
            "dft_natoms_expected": pristine_dft_natoms,
            "base_natoms_expected": pristine_lam_natoms,
            "ft_natoms_expected": pristine_lam_natoms,
            "pristine_mode": "mixed_supercell",
        }
        df_out = pd.concat([df_out, pd.DataFrame([pristine_row])], ignore_index=True)

    df_out.to_csv(out_csv, index=False)
    return out_csv


def main():
    ap = argparse.ArgumentParser(description="Generate manifest.csv with all paths needed for later lattice matching.")
    ap.add_argument("--csv1", required=True)
    ap.add_argument("--csv2", required=True)
    ap.add_argument("--dft-dir", required=True)

    # NEW: parent folders (contain features/ and results/)
    ap.add_argument("--base-parent-dir", required=True)
    ap.add_argument("--ft-parent-dir", required=True)

    # NEW optional: dumps + cif dirs
    ap.add_argument("--base-dumps-dir", default=None)
    ap.add_argument("--ft-dumps-dir", default=None)
    ap.add_argument("--unrelaxed-cif-dir", default=None)

    ap.add_argument("--out", required=True)
    ap.add_argument("--dft-file", default="OUTCAR")
    ap.add_argument("--lam-prefix", default="BP_spin_500_")
    ap.add_argument("--only-test-lam", action="store_true")
    ap.add_argument("--only-test-rows", action="store_true")
    ap.add_argument("--strict-path-exists", action="store_true")

    # pristine args
    ap.add_argument("--pristine-id", default="pristine")
    ap.add_argument("--pristine-dft", default=None)
    ap.add_argument("--pristine-base", default=None)
    ap.add_argument("--pristine-ft", default=None)
    ap.add_argument("--pristine-cif", default=None)
    ap.add_argument("--pristine-base-dump", default=None)
    ap.add_argument("--pristine-ft-dump", default=None)

    ap.add_argument("--pristine-formula", default="P4")
    ap.add_argument("--pristine-dft-natoms", type=int, default=144)
    ap.add_argument("--pristine-lam-natoms", type=int, default=4)
    ap.add_argument("--pristine-dft-supercell", default="6x6")
    ap.add_argument("--pristine-lam-supercell", default="1x1")

    args = ap.parse_args()

    out = generate_manifest(
        csv1_path=args.csv1,
        csv2_path=args.csv2,
        dft_dir=args.dft_dir,
        base_parent_dir=args.base_parent_dir,
        ft_parent_dir=args.ft_parent_dir,
        base_dumps_dir=args.base_dumps_dir,
        ft_dumps_dir=args.ft_dumps_dir,
        unrelaxed_cif_dir=args.unrelaxed_cif_dir,
        out_csv=args.out,
        dft_file=args.dft_file,
        lam_prefix=args.lam_prefix,
        include_only_test_for_lam=args.only_test_lam,
        include_non_test_rows=not args.only_test_rows,
        strict_path_exists=args.strict_path_exists,
        pristine_system_id=args.pristine_id,
        pristine_dft_path=args.pristine_dft,
        pristine_base_lmp=args.pristine_base,
        pristine_ft_lmp=args.pristine_ft,
        pristine_cif=args.pristine_cif,
        pristine_base_dump=args.pristine_base_dump,
        pristine_ft_dump=args.pristine_ft_dump,
        pristine_formula=args.pristine_formula,
        pristine_dft_natoms=args.pristine_dft_natoms,
        pristine_lam_natoms=args.pristine_lam_natoms,
        pristine_dft_supercell=args.pristine_dft_supercell,
        pristine_lam_supercell=args.pristine_lam_supercell,
    )
    print(f"[OK] wrote manifest: {out}")


if __name__ == "__main__":
    main()