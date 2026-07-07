from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Set

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read
from ase.neighborlist import NeighborList
from ase.geometry import find_mic
from scipy.optimize import linear_sum_assignment


# -----------------------------
# Config
# -----------------------------
@dataclass
class CatalogConfig:
    neighbor_cutoff_A: float = 3.2
    ac_zz_ratio_threshold: float = 1.5
    include_cross_pbc_flag: bool = True
    output_format: str = "csv"  # "csv" or "parquet"

    # Directions: either explicit Cartesian, or from cell vectors by index
    ac_direction_cart: Optional[List[float]] = None
    zz_direction_cart: Optional[List[float]] = None
    ac_direction_from_cell_vector_index: Optional[int] = None
    zz_direction_from_cell_vector_index: Optional[int] = None

    # manifest processing controls
    only_splits: Optional[List[str]] = None   # e.g. ["test","pristine"]
    include_pristine: bool = True
    strict_path_exists: bool = True           # if False, skip missing files

    # pristine handling
    pristine_expand_lam_to_supercell: bool = True
    pristine_lam_supercell: Optional[List[int]] = None  # e.g. [6,6,1]
    pristine_lam_expected_prim_natoms: int = 4
    pristine_target_natoms: int = 144

    # NEW: site matching / canonical indexing
    pristine_reference_path: Optional[str] = None  # must be 144-atom pristine (vasprun.xml recommended)
    total_sites: int = 144
    match_max_dist_tol_A: float = 3.0  # only for reporting; no hard fail unless you want


from dataclasses import fields

def load_config(path: str | Path) -> Tuple[CatalogConfig, dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    allowed = {f.name for f in fields(CatalogConfig)}
    filtered = {k: v for k, v in raw.items() if k in allowed}
    return CatalogConfig(**filtered), raw


# -----------------------------
# Structure loading (VASP/LAMMPS)
# -----------------------------
def _sanitize_poscar_species_line(line: str) -> str:
    # 'P/df60...  N/e053...' -> 'P N'
    toks = line.split()
    clean = []
    for t in toks:
        t = t.split("/")[0].split("|")[0].split(":")[0]
        m = re.match(r"^[A-Za-z]{1,2}", t)
        if not m:
            raise ValueError(f"Cannot sanitize POSCAR species token: {t}")
        clean.append(m.group(0))
    return "  ".join(clean) + "\n"

def _vec_to_list(v: np.ndarray) -> List[float]:
    return [float(x) for x in np.asarray(v, dtype=float).reshape(-1)]

def _read_poscar_sanitized(path: Path) -> Atoms:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines(True)
    if len(lines) < 7:
        raise ValueError(f"POSCAR/CONTCAR too short: {path}")
    lines[5] = _sanitize_poscar_species_line(lines[5])
    with tempfile.NamedTemporaryFile(mode="w", suffix=".POSCAR", delete=False) as tf:
        tf.writelines(lines)
        tmp = tf.name
    try:
        return read(tmp, format="vasp")
    finally:
        try:
            Path(tmp).unlink(missing_ok=True)
        except Exception:
            pass


def _pick_vasp_final_structure(workdir: Path) -> Path:
    for name in ["vasprun.xml", "CONTCAR", "POSCAR", "OUTCAR"]:
        p = workdir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No VASP outputs found in {workdir}")


def load_structure_from_path(path: str | Path) -> Atoms:
    p = Path(str(path)).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {p}")

    if p.is_dir():
        p = _pick_vasp_final_structure(p)

    if p.name.upper() == "OUTCAR":
        p = _pick_vasp_final_structure(p.parent)

    try:
        head = p.read_text(encoding="utf-8", errors="ignore")[:2000].lower()
    except Exception:
        head = ""

    if "lammps data file" in head:
        return read(str(p), format="lammps-data", atom_style="atomic")

    if p.name.lower() == "vasprun.xml":
        return read(str(p), format="vasp-xml")

    if p.name.upper() in {"CONTCAR", "POSCAR"}:
        try:
            return read(str(p), format="vasp")
        except Exception:
            return _read_poscar_sanitized(p)

    if p.name.upper() == "OUTCAR":
        return read(str(p), format="vasp-out")

    return read(str(p))

def load_vasp_relaxation_frames(path: str | Path) -> List[Atoms]:
    """
    Return ionic steps from vasprun.xml or OUTCAR/CONTCAR if needed.
    For vasprun.xml, index=':' gives all frames.
    """
    p = Path(path)
    if p.is_dir():
        p = _pick_vasp_final_structure(p)

    # Prefer vasprun.xml if present
    if p.name.upper() == "OUTCAR":
        vrun = p.parent / "vasprun.xml"
        if vrun.exists():
            p = vrun

    if p.name.lower() == "vasprun.xml":
        frames = read(str(p), index=":")
        if not isinstance(frames, list):
            frames = [frames]
        if len(frames) == 0:
            raise ValueError(f"No frames in {p}")
        return frames

    # Fallback: treat as single frame
    return [load_structure_from_path(p)]

def hungarian_map_atomidx_to_sites_with_stats(
    atoms0: Atoms,
    pristine_atoms: Atoms,
    total_sites: int = 144,
) -> Tuple[Dict[int, int], float, float, float]:
    pos_ref = pristine_atoms.get_positions()
    cell_ref = pristine_atoms.get_cell()
    pbc_ref = pristine_atoms.get_pbc()

    pos = atoms0.get_positions()
    N = len(atoms0)
    if N > total_sites:
        raise ValueError(f"natoms={N} > total_sites={total_sites}")

    D = np.empty((N, total_sites), dtype=float)
    for a in range(N):
        disp = pos_ref - pos[a][None, :]
        _, d = find_mic(disp, cell_ref, pbc_ref)
        D[a, :] = d

    row_ind, col_ind = linear_sum_assignment(D)
    assigned = D[row_ind, col_ind]
    dmax = float(np.max(assigned)) if assigned.size else float("nan")
    dp95 = float(np.percentile(assigned, 95)) if assigned.size else float("nan")
    dmean = float(np.mean(assigned)) if assigned.size else float("nan")

    atomidx2site = {int(a): int(s) for a, s in zip(row_ind, col_ind)}
    return atomidx2site, dmax, dp95, dmean

# =============================
# Dump readers: first/last frame
# =============================
def _read_one_frame(f):
    line = f.readline()
    if not line:
        raise EOFError
    if not line.startswith("ITEM: TIMESTEP"):
        raise ValueError(f"Expected 'ITEM: TIMESTEP', got: {line[:50]}")
    if any(k in line.lower() for k in ["xy", "xz", "yz"]):
        raise NotImplementedError("triclinic BOX BOUNDS not supported yet; dump must be orthorhombic with x/y/z")

    timestep = int(f.readline().strip())

    line = f.readline().strip()
    if line != "ITEM: NUMBER OF ATOMS":
        raise ValueError(f"Expected 'ITEM: NUMBER OF ATOMS', got: {line}")
    natoms = int(f.readline().strip())

    line = f.readline().strip()
    if not line.startswith("ITEM: BOX BOUNDS"):
        raise ValueError(f"Expected 'ITEM: BOX BOUNDS', got: {line}")

    bounds = []
    for _ in range(3):
        lo, hi = f.readline().split()[:2]
        bounds.append([float(lo), float(hi)])
    bounds = np.array(bounds, dtype=float)  # (3,2)

    line = f.readline().strip()
    if not line.startswith("ITEM: ATOMS"):
        raise ValueError(f"Expected 'ITEM: ATOMS', got: {line}")

    cols = line.split()[2:]
    for k in ["id", "type", "x", "y", "z"]:
        if k not in cols:
            raise ValueError(f"Dump ATOMS columns missing '{k}'. Got: {cols}")

    id_col = cols.index("id")
    type_col = cols.index("type")
    x_col = cols.index("x")
    y_col = cols.index("y")
    z_col = cols.index("z")

    ids = np.empty(natoms, dtype=int)
    types = np.empty(natoms, dtype=int)
    pos = np.empty((natoms, 3), dtype=float)

    for i in range(natoms):
        parts = f.readline().split()
        ids[i] = int(parts[id_col])
        types[i] = int(parts[type_col])
        pos[i, 0] = float(parts[x_col])
        pos[i, 1] = float(parts[y_col])
        pos[i, 2] = float(parts[z_col])

    return timestep, bounds, ids, types, pos


def read_first_frame_lammpstrj(path: str | Path):
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return _read_one_frame(f)


def read_last_frame_lammpstrj(path: str | Path):
    path = Path(path)
    last = None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        while True:
            try:
                last = _read_one_frame(f)
            except EOFError:
                break
    if last is None:
        raise ValueError(f"No frames read from dump: {path}")
    return last


# =============================
# Hungarian mapping: IDs -> pristine sites
# =============================

def infer_sites_by_hungarian(
    atoms_defect: Atoms,
    pristine_atoms: Atoms,
    *,
    total_sites: int = 144,
) -> Tuple[Dict[int, int], Set[int], Set[int], float, float, float]:
    """
    Returns:
      atom2site: mapping from atom index in defect structure -> site index (0..total_sites-1)
      missing_sites: sites not assigned (vacancies)
      n_sites_found: sites occupied by N atoms
      dist_max, dist_p95, dist_mean
    """
    pos_ref = pristine_atoms.get_positions()
    cell_ref = pristine_atoms.get_cell()
    pbc_ref = pristine_atoms.get_pbc()

    pos = atoms_defect.get_positions()
    N = len(atoms_defect)
    if N > total_sites:
        raise ValueError(f"natoms={N} > total_sites={total_sites}")

    D = np.empty((N, total_sites), dtype=float)
    for a in range(N):
        disp = pos_ref - pos[a][None, :]
        _, d = find_mic(disp, cell_ref, pbc_ref)
        D[a, :] = d

    row_ind, col_ind = linear_sum_assignment(D)
    atom2site = {int(a): int(s) for a, s in zip(row_ind, col_ind)}
    assigned_sites = set(atom2site.values())
    missing_sites = set(range(total_sites)) - assigned_sites

    assigned_dists = D[row_ind, col_ind]
    dist_max = float(np.max(assigned_dists)) if assigned_dists.size else float("nan")
    dist_p95 = float(np.percentile(assigned_dists, 95)) if assigned_dists.size else float("nan")
    dist_mean = float(np.mean(assigned_dists)) if assigned_dists.size else float("nan")

    n_sites_found = set()
    for a, sym in enumerate(atoms_defect.get_chemical_symbols()):
        if sym == "N" and a in atom2site:
            n_sites_found.add(atom2site[a])

    return atom2site, missing_sites, n_sites_found, dist_max, dist_p95, dist_mean

def hungarian_map_ids_to_sites_with_stats(
    ids: np.ndarray,
    pos: np.ndarray,
    pristine_atoms: Atoms,
    total_sites: int = 144,
) -> Tuple[Dict[int, int], float, float, float]:
    """
    Returns:
      id2site, dist_max, dist_p95, dist_mean
    where distances are MIC distances (Å) from frame-0 atoms to assigned pristine sites.
    """
    pos_ref = pristine_atoms.get_positions()
    cell_ref = pristine_atoms.get_cell()
    pbc_ref = pristine_atoms.get_pbc()

    N = pos.shape[0]
    if N > total_sites:
        raise ValueError(f"natoms={N} > total_sites={total_sites}")

    D = np.empty((N, total_sites), dtype=float)
    for a in range(N):
        disp = pos_ref - pos[a][None, :]
        _, d = find_mic(disp, cell_ref, pbc_ref)
        D[a, :] = d

    row_ind, col_ind = linear_sum_assignment(D)

    assigned_dists = D[row_ind, col_ind]
    dist_max = float(np.max(assigned_dists)) if assigned_dists.size else float("nan")
    dist_p95 = float(np.percentile(assigned_dists, 95)) if assigned_dists.size else float("nan")
    dist_mean = float(np.mean(assigned_dists)) if assigned_dists.size else float("nan")

    id2site = {int(ids[r]): int(s) for r, s in zip(row_ind, col_ind)}
    return id2site, dist_max, dist_p95, dist_mean


def _atoms_from_bounds_ids_types_pos(
    bounds: np.ndarray,
    ids: np.ndarray,
    types: np.ndarray,
    pos: np.ndarray,
    type2elem: Dict[int, str],
) -> Atoms:
    """
    Build ASE atoms from a dump frame, including chemical symbols.
    """
    cell = np.diag(bounds[:, 1] - bounds[:, 0])
    pos2 = pos.copy()
    pos2[:, 0] -= bounds[0, 0]
    pos2[:, 1] -= bounds[1, 0]
    pos2[:, 2] -= bounds[2, 0]

    syms = [type2elem[int(t)] for t in types]
    atoms = Atoms(symbols=syms, positions=pos2, cell=cell, pbc=True)
    atoms.wrap()
    return atoms


def build_site_mapping_rows_from_atom2site(
    *,
    atoms_final: Atoms,
    pristine_atoms: Atoms,
    atom2site: Dict[int, int],
    system_id: str,
    structure_tag: str,
    extra_meta: Dict[str, Any],
    assign_dist_max_A: float,
    assign_dist_p95_A: float,
    assign_dist_mean_A: float,
    atom_identifier_kind: str = "atom_index",
    atom_identifiers: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Build one row per canonical site (0..total_sites-1), recording the final relaxed mapping.

    For occupied sites:
      - site_index
      - mapped atom index / dump id
      - final coordinates
      - final element
      - distance from final atom to pristine site (MIC)
    For vacancies:
      - site_index
      - is_vacancy = True
      - pristine site coordinates only
    """
    pos_ref = pristine_atoms.get_positions()
    cell_ref = pristine_atoms.get_cell()
    pbc_ref = pristine_atoms.get_pbc()

    n_sites = len(pristine_atoms)
    pos_fin = atoms_final.get_positions()
    syms_fin = atoms_final.get_chemical_symbols()

    site2atom = {site: atom for atom, site in atom2site.items()}
    rows = []

    for site in range(n_sites):
        ref_xyz = pos_ref[site]

        if site in site2atom:
            a = site2atom[site]
            fin_xyz = pos_fin[a]
            disp = ref_xyz[None, :] - fin_xyz[None, :]
            _, d = find_mic(disp, cell_ref, pbc_ref)
            d_assign = float(d[0])

            atom_id_val = None
            if atom_identifiers is not None:
                atom_id_val = int(atom_identifiers[a])
            else:
                atom_id_val = int(a)

            rows.append({
                **extra_meta,
                "system_id": system_id,
                "structure": structure_tag,
                "site_index": int(site),
                "is_occupied": True,
                "is_vacancy": False,
                "mapped_atom_identifier_kind": atom_identifier_kind,
                "mapped_atom_identifier": atom_id_val,
                "mapped_atom_index_0based": int(a),
                "element": syms_fin[a],
                "x_final_A": float(fin_xyz[0]),
                "y_final_A": float(fin_xyz[1]),
                "z_final_A": float(fin_xyz[2]),
                "x_pristine_A": float(ref_xyz[0]),
                "y_pristine_A": float(ref_xyz[1]),
                "z_pristine_A": float(ref_xyz[2]),
                "assign_dist_site_A": d_assign,
                "assign_dist_max_A": assign_dist_max_A,
                "assign_dist_p95_A": assign_dist_p95_A,
                "assign_dist_mean_A": assign_dist_mean_A,
            })
        else:
            rows.append({
                **extra_meta,
                "system_id": system_id,
                "structure": structure_tag,
                "site_index": int(site),
                "is_occupied": False,
                "is_vacancy": True,
                "mapped_atom_identifier_kind": atom_identifier_kind,
                "mapped_atom_identifier": pd.NA,
                "mapped_atom_index_0based": pd.NA,
                "element": pd.NA,
                "x_final_A": pd.NA,
                "y_final_A": pd.NA,
                "z_final_A": pd.NA,
                "x_pristine_A": float(ref_xyz[0]),
                "y_pristine_A": float(ref_xyz[1]),
                "z_pristine_A": float(ref_xyz[2]),
                "assign_dist_site_A": pd.NA,
                "assign_dist_max_A": assign_dist_max_A,
                "assign_dist_p95_A": assign_dist_p95_A,
                "assign_dist_mean_A": assign_dist_mean_A,
            })

    return pd.DataFrame(rows)

def read_lammps_masses_type_map(lmp_data_path: str | Path) -> Dict[int, float]:
    """
    Parse LAMMPS data file 'Masses' section.
    Returns: {type_id: mass}
    """
    p = Path(lmp_data_path)
    txt = p.read_text(encoding="utf-8", errors="ignore").splitlines()

    # find "Masses" line
    try:
        i0 = next(i for i, line in enumerate(txt) if line.strip().lower() == "masses")
    except StopIteration:
        raise ValueError(f"No 'Masses' section found in {p}")

    masses: Dict[int, float] = {}
    # lines after "Masses" until blank or next section header
    for line in txt[i0 + 1:]:
        s = line.strip()
        if not s:
            if masses:
                break
            continue
        # stop if we hit next section
        low = s.lower()
        if low.startswith(("atoms", "velocities", "bonds", "angles", "dihedrals", "impropers", "pair", "coeffs")):
            break
        parts = s.split()
        if len(parts) < 2:
            continue
        try:
            t = int(parts[0])
            m = float(parts[1])
        except Exception:
            continue
        masses[t] = m

    if not masses:
        raise ValueError(f"Parsed empty Masses section in {p}")
    return masses


def infer_type_to_element_from_masses(masses: Dict[int, float]) -> Dict[int, str]:
    """
    Infer element from atomic mass (works for your P/N case).
    """
    # expected masses (amu)
    ref = {"N": 14.007, "P": 30.974}

    type2elem: Dict[int, str] = {}
    for t, m in masses.items():
        # choose closest ref
        best_elem = min(ref.keys(), key=lambda e: abs(m - ref[e]))
        # guard: ensure it's not wildly off (optional)
        if abs(m - ref[best_elem]) > 3.0:
            # still assign, but this indicates data mismatch
            best_elem = best_elem
        type2elem[int(t)] = best_elem
    return type2elem

# =============================
# Dump -> ASE Atoms (orthorhombic box)
# =============================
def atoms_from_dump(bounds: np.ndarray, pos: np.ndarray) -> Atoms:
    cell = np.diag(bounds[:, 1] - bounds[:, 0])
    pos2 = pos.copy()
    pos2[:, 0] -= bounds[0, 0]
    pos2[:, 1] -= bounds[1, 0]
    pos2[:, 2] -= bounds[2, 0]
    atoms = Atoms(positions=pos2, cell=cell, pbc=True)
    atoms.wrap()
    return atoms


def extract_site_neighbors_from_dump(
    bounds_last: np.ndarray,
    ids_last: np.ndarray,
    types_last: np.ndarray,
    pos_last: np.ndarray,
    id2site: Dict[int, int],
    type2elem: Dict[int, str],
    pristine_atoms: Atoms,
    cfg: CatalogConfig,
    system_id: str,
    structure_tag: str,
    extra_meta: Dict[str, Any],
    *,
    assign_dist_max_A: float,
    assign_dist_p95_A: float,
    assign_dist_mean_A: float,
    dedup_undirected: bool = True,  # NEW: avoid double counting i->j and j->i
) -> pd.DataFrame:
    """
    Efficient neighbor extraction:
      - Candidate neighbors via ASE NeighborList
      - True distance + disp via MIC (find_mic) for robustness
    """
    # Use pristine basis for direction definitions
    e_ac, e_zz = get_ac_zz_unit_vectors(pristine_atoms, cfg)

    atoms = atoms_from_dump(bounds_last, pos_last)  # wrap() inside
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()
    pos = atoms.get_positions()

    # NeighborList candidates
    nl = build_neighbor_list(atoms, float(cfg.neighbor_cutoff_A))
    cutoff = float(cfg.neighbor_cutoff_A)

    # missing/present sites
    present_sites = {id2site[int(i)] for i in ids_last if int(i) in id2site}
    missing_sites = set(range(cfg.total_sites)) - present_sites

    # N substitution sites in last frame
    n_sites_found: Set[int] = set()
    for atom_id, t in zip(ids_last, types_last):
        atom_id = int(atom_id)
        if atom_id in id2site and type2elem.get(int(t), "") == "N":
            n_sites_found.add(id2site[atom_id])

    id_to_type = {int(i): int(t) for i, t in zip(ids_last, types_last)}

    rows = []
    seen_edges = set()  # for dedup undirected edges: store (min_site,max_site,label) or (min_i,max_i)
    N = len(atoms)

    for i in range(N):
        atom_id_i = int(ids_last[i])
        if atom_id_i not in id2site:
            continue

        u = id2site[atom_id_i]
        elem_u = type2elem.get(id_to_type.get(atom_id_i, -1), "")

        js, offs = nl.get_neighbors(i)
        for j, off in zip(js, offs):
            atom_id_j = int(ids_last[j])
            if atom_id_j not in id2site:
                continue

            v = id2site[atom_id_j]
            elem_v = type2elem.get(id_to_type.get(atom_id_j, -1), "")

            # MIC displacement (ignore off for geometry; off only used as a flag if you want)
            disp0 = pos[j] - pos[i]
            disp_mic, d = find_mic(disp0, cell, pbc)
            dist = float(d)

            # robust cutoff check
            if dist > cutoff + 1e-8:
                continue

            # Optional: de-duplicate undirected edges (recommended)
            if dedup_undirected:
                key = (u, v) if u <= v else (v, u)
                # If you want to keep AC/ZZ label in key, compute label first; here we just dedup by site-pair.
                if key in seen_edges:
                    continue
                seen_edges.add(key)

            # Classification: learned or old AC/ZZ
            # PBC metadata should always come from NeighborList offsets, regardless of
            # how bond direction is classified.
            cross_pbc = bool(np.any(off != 0)) and bool(cfg.include_cross_pbc_flag)
            offset_a, offset_b, offset_c = int(off[0]), int(off[1]), int(off[2])

            # Classification: learned or old AC/ZZ
            if getattr(cfg, "_learned_dirs", None) is not None:
                d0, d1, n_hat = cfg._learned_dirs
                label0, label1 = cfg._learned_label_names
                lab, p0, p1, margin = classify_bond_by_learned_dirs(
                    disp_mic, d0, d1, n_hat, mixed_angle_deg=cfg._mixed_angle_deg
                )
                if lab == "D0":
                    label = label0
                elif lab == "D1":
                    label = label1
                else:
                    label = lab  # MIXED / OUT

                proj_ac = p0
                proj_zz = p1
                ratio = margin
            else:
                label, proj_ac, proj_zz, ratio = classify_bond_ac_zz(
                    disp=disp_mic, e_ac=e_ac, e_zz=e_zz, ratio_thr=cfg.ac_zz_ratio_threshold
                )

            rows.append({
                **extra_meta,
                "system_id": system_id,
                "structure": structure_tag,
                "natoms": int(N),
                "u_site": int(u),
                "v_site": int(v),
                "elem_u": elem_u,
                "elem_v": elem_v,
                "distance_A": dist,
                "proj_ac_A": proj_ac,
                "proj_zz_A": proj_zz,
                "abs_ratio_ac_to_zz": ratio,
                "label": label,
                "cross_pbc": cross_pbc,
                "offset_a": offset_a,
                "offset_b": offset_b,
                "offset_c": offset_c,
                "assign_dist_max_A": assign_dist_max_A,
                "assign_dist_p95_A": assign_dist_p95_A,
                "assign_dist_mean_A": assign_dist_mean_A,
                "missing_sites": " ".join(map(str, sorted(missing_sites))),
                "n_sites_found": " ".join(map(str, sorted(n_sites_found))),
            })

    return pd.DataFrame(rows)

def extract_site_indexed_neighbors_from_vasp_path(
    vasp_path: str | Path,
    pristine_atoms: Atoms,
    cfg: CatalogConfig,
    system_id: str,
    structure_tag: str,
    extra_meta: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frames = load_vasp_relaxation_frames(vasp_path)
    atoms0 = frames[0]
    atomsL = frames[-1]

    # frame-0 assignment
    atom2site, dmax0, dp950, dmean0 = hungarian_map_atomidx_to_sites_with_stats(
        atoms0, pristine_atoms, total_sites=cfg.total_sites
    )

    # neighbor list on final frame geometry
    atoms = atomsL.copy()
    # optional safety
    atoms.wrap()

    e_ac, e_zz = get_ac_zz_unit_vectors(pristine_atoms, cfg)
    nl = build_neighbor_list(atoms, cfg.neighbor_cutoff_A)

    pos = atoms.get_positions()
    cell = atoms.get_cell()

    # present/missing sites inferred from mapping range
    assigned_sites = set(atom2site[i] for i in range(len(atoms)) if i in atom2site)
    missing_sites = set(range(cfg.total_sites)) - assigned_sites

    n_sites_found: Set[int] = set()
    syms = atoms.get_chemical_symbols()
    for i, s in enumerate(syms):
        if s == "N" and i in atom2site:
            n_sites_found.add(atom2site[i])

    rows = []
    for i in range(len(atoms)):
        if i not in atom2site:
            continue
        u = atom2site[i]
        indices, offsets = nl.get_neighbors(i)
        for j, off in zip(indices, offsets):
            if j not in atom2site:
                continue
            v = atom2site[j]

            # MIC displacement (recommended; avoids “dist > cutoff” surprises)
            disp0 = pos[j] - pos[i]
            disp_mic, d = find_mic(disp0, cell, atoms.get_pbc())
            dist = float(d)
            if dist > cfg.neighbor_cutoff_A + 1e-6:
                continue

            if getattr(cfg, "_learned_dirs", None) is not None:
                d0, d1, n_hat = cfg._learned_dirs
                label0, label1 = cfg._learned_label_names
                lab, p0, p1, margin = classify_bond_by_learned_dirs(
                    disp_mic, d0, d1, n_hat, mixed_angle_deg=cfg._mixed_angle_deg
                )
                if lab == "D0": lab = label0
                elif lab == "D1": lab = label1

                label = lab
                proj_ac = p0   # you can store as proj_dir0/proj_dir1 instead if you prefer
                proj_zz = p1
                ratio = margin # store margin (deg) into abs_ratio_ac_to_zz, rename later if you like
            else:
                # old classifier
                label, proj_ac, proj_zz, ratio = classify_bond_ac_zz(
                    disp=disp_mic, e_ac=e_ac, e_zz=e_zz, ratio_thr=cfg.ac_zz_ratio_threshold
                )
            
            cross_pbc = bool(np.any(off != 0))

            rows.append({
                **extra_meta,
                "system_id": system_id,
                "structure": structure_tag,
                "natoms": int(len(atoms)),
                "u_site": int(u),
                "v_site": int(v),
                "elem_u": atoms[i].symbol,
                "elem_v": atoms[j].symbol,
                "distance_A": dist,
                "proj_ac_A": float(np.dot(disp_mic, e_ac)),
                "proj_zz_A": float(np.dot(disp_mic, e_zz)),
                "abs_ratio_ac_to_zz": ratio,
                "label": label,
                "cross_pbc": cross_pbc if cfg.include_cross_pbc_flag else False,
                "offset_a": int(off[0]),
                "offset_b": int(off[1]),
                "offset_c": int(off[2]),
                # frame-0 assignment quality
                "assign_dist_max_A": dmax0,
                "assign_dist_p95_A": dp950,
                "assign_dist_mean_A": dmean0,
                "missing_sites": " ".join(map(str, sorted(missing_sites))),
                "n_sites_found": " ".join(map(str, sorted(n_sites_found))),
                "dft_nframes": len(frames),
            })

    neighbor_df = pd.DataFrame(rows)

    mapping_df = build_site_mapping_rows_from_atom2site(
        atoms_final=atomsL,
        pristine_atoms=pristine_atoms,
        atom2site=atom2site,
        system_id=system_id,
        structure_tag=structure_tag,
        extra_meta=extra_meta,
        assign_dist_max_A=dmax0,
        assign_dist_p95_A=dp950,
        assign_dist_mean_A=dmean0,
        atom_identifier_kind="atom_index",
        atom_identifiers=list(range(len(atomsL))),
    )
    mapping_df["dft_nframes"] = len(frames)

    return neighbor_df, mapping_df


# -----------------------------
# AC-ZZ possible rotation handling
# -----------------------------
# from sklearn.cluster import KMeans

def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    # critical: handle nan/inf norms explicitly
    if (not np.isfinite(n)) or (n < eps):
        return np.zeros_like(v)
    return v / n

def _project_to_plane(v: np.ndarray, n_hat: np.ndarray) -> np.ndarray:
    return v - np.dot(v, n_hat) * n_hat

def _plane_normal_from_cell_ab(cell_array: np.ndarray) -> np.ndarray:
    a = cell_array[0]
    b = cell_array[1]
    n = np.cross(a, b)
    return _unit(n)

def _orthonormal_basis_in_plane(cell_array: np.ndarray):
    cell_array = np.asarray(cell_array, dtype=float)
    a = cell_array[0]
    b = cell_array[1]
    e1 = _unit(a)
    n_hat = _unit(np.cross(a, b))
    # if plane normal is degenerate, fail loudly (otherwise KMeans will see NaNs)
    if np.linalg.norm(n_hat) < 1e-8:
        raise ValueError("Degenerate plane normal from cell a×b (too small). Check cell vectors / plane choice.")
    e2 = _unit(np.cross(n_hat, e1))
    if np.linalg.norm(e2) < 1e-8:
        raise ValueError("Degenerate in-plane basis e2. Check cell vectors.")
    return e1, e2, n_hat

from ase.neighborlist import NeighborList

def _kmeans2_unitcircle(vecs: np.ndarray, k: int = 2, n_iter: int = 50, seed: int = 0) -> np.ndarray:
    """
    Simple k-means on 2D vectors (Nx2), returns centers (kx2).
    Assumes vecs are finite float64. No sklearn.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(vecs, dtype=np.float64)
    n = X.shape[0]
    if n < k:
        raise ValueError(f"kmeans: n={n} < k={k}")

    # kmeans++ init
    centers = np.empty((k, 2), dtype=np.float64)
    idx0 = rng.integers(0, n)
    centers[0] = X[idx0]

    d2 = np.sum((X - centers[0])**2, axis=1)
    for ci in range(1, k):
        # Avoid all-zero / NaN
        tot = float(np.sum(d2))
        if not np.isfinite(tot) or tot <= 0:
            centers[ci] = X[rng.integers(0, n)]
        else:
            p = d2 / tot
            idx = rng.choice(n, p=p)
            centers[ci] = X[idx]
        d2 = np.minimum(d2, np.sum((X - centers[ci])**2, axis=1))

    # Lloyd iterations
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(n_iter):
        # assign
        dist2 = ((X[:, None, :] - centers[None, :, :])**2).sum(axis=2)  # (n,k)
        new_labels = dist2.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # update
        for ci in range(k):
            mask = (labels == ci)
            if not np.any(mask):
                centers[ci] = X[rng.integers(0, n)]
            else:
                centers[ci] = X[mask].mean(axis=0)

    return centers


def learn_two_inplane_directions_from_pristine(
    pristine_atoms,
    neighbor_cutoff_A: float,
    *,
    plane: str = "ab",   # accept kwarg; only "ab" implemented
    k: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Learn 2 dominant in-plane bond directions from pristine using a π-periodic embedding:
        feat = [cos(2θ), sin(2θ)]
    Clusters feats into k groups using numpy-only kmeans (no sklearn), then converts
    cluster centers back into 3D directions within the ab-plane.

    Returns:
      dir0_3d, dir1_3d, n_hat
    """
    if plane != "ab":
        raise NotImplementedError(f"plane='{plane}' not supported yet; use plane='ab'")

    pos = pristine_atoms.get_positions()
    cell = pristine_atoms.get_cell()
    pbc = pristine_atoms.get_pbc()

    e1, e2, n_hat = _orthonormal_basis_in_plane(cell.array)

    N = len(pristine_atoms)
    nl = NeighborList([neighbor_cutoff_A]*N, self_interaction=False, bothways=True, skin=0.0)
    nl.update(pristine_atoms)

    feats = []
    for i in range(N):
        js, offs = nl.get_neighbors(i)
        for j, off in zip(js, offs):
            disp = (pos[j] + np.dot(off, cell.array)) - pos[i]
            disp_in = _project_to_plane(disp, n_hat)
            u = _unit(disp_in)
            if np.linalg.norm(u) < 1e-8:
                continue

            x = float(np.dot(u, e1))
            y = float(np.dot(u, e2))
            theta = np.arctan2(y, x)
            feats.append([np.cos(2.0 * theta), np.sin(2.0 * theta)])

    feats = np.asarray(feats, dtype=np.float64)

    # sanitize
    mask = np.isfinite(feats).all(axis=1)
    feats = feats[mask]
    if feats.shape[0] < 10:
        raise ValueError(f"Not enough finite neighbor vectors to learn directions: {feats.shape[0]}")

    # clip tiny drift
    feats = np.clip(feats, -1.0, 1.0)

    # numpy-only kmeans
    centers = _kmeans2_unitcircle(feats, k=k, n_iter=80, seed=0)  # (k,2)

    # convert centers -> theta -> 3D direction in plane
    dirs = []
    for c in centers:
        theta2 = np.arctan2(c[1], c[0])   # center angle in doubled-angle space
        theta = 0.5 * theta2
        u_plane = np.cos(theta)*e1 + np.sin(theta)*e2
        dirs.append(_unit(u_plane))

    if k != 2:
        raise ValueError("This helper currently expects k=2 for downstream naming.")
    return dirs[0], dirs[1], n_hat

def classify_bond_by_learned_dirs(
    disp_mic: np.ndarray,
    dir0: np.ndarray,
    dir1: np.ndarray,
    n_hat: np.ndarray,
    *,
    mixed_angle_deg: float = 10.0,
) -> Tuple[str, float, float, float]:
    """
    Classify by angular proximity to learned directions.
    Returns:
      label in {"D0","D1","MIXED","OUT"},
      proj0, proj1, angle_margin_deg
    """
    disp_in = _project_to_plane(disp_mic, n_hat)
    u = _unit(disp_in)
    if np.linalg.norm(u) < 1e-8:
        return "OUT", 0.0, 0.0, float("nan")

    # use absolute dot => direction sign irrelevant
    c0 = abs(float(np.dot(u, dir0)))
    c1 = abs(float(np.dot(u, dir1)))

    # convert to angles
    a0 = np.degrees(np.arccos(np.clip(c0, -1, 1)))
    a1 = np.degrees(np.arccos(np.clip(c1, -1, 1)))
    margin = abs(a0 - a1)

    if margin <= mixed_angle_deg:
        return "MIXED", c0, c1, margin
    return ("D0" if a0 < a1 else "D1"), c0, c1, margin

def name_dirs_as_ac_zz(dir0: np.ndarray, dir1: np.ndarray, pristine_atoms) -> Dict[str, np.ndarray]:
    """
    Name learned directions by the in-plane lattice periods:
    - shorter in-plane lattice vector  -> ZZ
    - longer in-plane lattice vector   -> AC

    For BP/phosphorene this matches the common convention.
    """
    cell = np.asarray(pristine_atoms.get_cell().array, dtype=float)
    a = _unit(cell[0])
    b = _unit(cell[1])

    len_a = np.linalg.norm(cell[0])
    len_b = np.linalg.norm(cell[1])

    if len_a >= len_b:
        ac_ref = a
        zz_ref = b
    else:
        ac_ref = b
        zz_ref = a

    s0_ac = abs(float(np.dot(dir0, ac_ref)))
    s1_ac = abs(float(np.dot(dir1, ac_ref)))

    if s0_ac >= s1_ac:
        return {"AC": dir0, "ZZ": dir1}
    else:
        return {"AC": dir1, "ZZ": dir0}

from collections import Counter

def bond_label_counts(
    pristine_atoms,
    dir0,
    dir1,
    n_hat,
    cutoff_A: float,
    *,
    mixed_angle_deg: float = 10.0,
) -> Counter:
    pos = pristine_atoms.get_positions()
    cell = pristine_atoms.get_cell()
    N = len(pristine_atoms)

    nl = NeighborList([cutoff_A] * N, self_interaction=False, bothways=True, skin=0.0)
    nl.update(pristine_atoms)

    c = Counter()
    for i in range(N):
        js, offs = nl.get_neighbors(i)
        for j, off in zip(js, offs):
            disp = (pos[j] + np.dot(off, cell.array)) - pos[i]
            lab, *_ = classify_bond_by_learned_dirs(
                disp, dir0, dir1, n_hat, mixed_angle_deg=mixed_angle_deg
            )
            c[lab] += 1
    return c

def bond_label_counts_old(pristine_atoms, cfg: CatalogConfig) -> Counter:
    pos = pristine_atoms.get_positions()
    cell = pristine_atoms.get_cell()
    N = len(pristine_atoms)

    e_ac, e_zz = get_ac_zz_unit_vectors(pristine_atoms, cfg)
    nl = NeighborList([cfg.neighbor_cutoff_A]*N, self_interaction=False, bothways=True, skin=0.0)
    nl.update(pristine_atoms)

    c = Counter()
    for i in range(N):
        js, offs = nl.get_neighbors(i)
        for j, off in zip(js, offs):
            disp = (pos[j] + np.dot(off, cell.array)) - pos[i]
            lab, *_ = classify_bond_ac_zz(disp, e_ac, e_zz, cfg.ac_zz_ratio_threshold)
            c[lab] += 1
    return c

# -----------------------------
# Pristine expansion helper
# -----------------------------
def maybe_expand_pristine_lam(atoms: Atoms, cfg: CatalogConfig, system_id: str, structure_tag: str) -> Atoms:
    n = len(atoms)
    if not cfg.pristine_expand_lam_to_supercell:
        return atoms
    if cfg.pristine_lam_supercell is None:
        raise ValueError("pristine_lam_supercell must be set when pristine_expand_lam_to_supercell=true")
    if n == cfg.pristine_target_natoms:
        return atoms
    if n == cfg.pristine_lam_expected_prim_natoms:
        rep = tuple(int(x) for x in cfg.pristine_lam_supercell)
        atoms2 = atoms.repeat(rep)
        if len(atoms2) != cfg.pristine_target_natoms:
            raise ValueError(
                f"[pristine expand mismatch] {system_id} {structure_tag}: repeat{rep} gives natoms={len(atoms2)}"
            )
        return atoms2
    raise ValueError(
        f"[unexpected pristine LAM natoms] {system_id} {structure_tag}: got natoms={n}"
    )


# -----------------------------
# Direction handling + label
# -----------------------------
def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("Zero-length direction vector.")
    return v / n


def get_ac_zz_unit_vectors(atoms: Atoms, cfg: CatalogConfig) -> Tuple[np.ndarray, np.ndarray]:
    if cfg.ac_direction_cart is not None and cfg.zz_direction_cart is not None:
        e_ac = _normalize(np.array(cfg.ac_direction_cart, dtype=float))
        e_zz = _normalize(np.array(cfg.zz_direction_cart, dtype=float))
        return e_ac, e_zz

    if cfg.ac_direction_from_cell_vector_index is not None and cfg.zz_direction_from_cell_vector_index is not None:
        cell = atoms.get_cell().array
        e_ac = _normalize(cell[int(cfg.ac_direction_from_cell_vector_index)])
        e_zz = _normalize(cell[int(cfg.zz_direction_from_cell_vector_index)])
        return e_ac, e_zz

    raise ValueError("AC/ZZ directions not specified in config.")


def classify_bond_ac_zz(
    disp: np.ndarray,
    e_ac: np.ndarray,
    e_zz: np.ndarray,
    ratio_thr: float,
    eps: float = 1e-12,
) -> Tuple[str, float, float, float]:
    proj_ac = float(np.dot(disp, e_ac))
    proj_zz = float(np.dot(disp, e_zz))
    a = abs(proj_ac)
    z = abs(proj_zz)

    if a < eps and z < eps:
        return "OUT", proj_ac, proj_zz, float("nan")

    # ratio handling: avoid huge finite numbers
    if z < eps:
        ratio = float("inf")   # pure AC direction
    else:
        ratio = a / z

    if ratio >= ratio_thr:
        return "AC", proj_ac, proj_zz, ratio

    # for ZZ, compare inverse safely
    if a < eps:
        inv = float("inf")     # pure ZZ direction
    else:
        inv = z / a

    if inv >= ratio_thr:
        return "ZZ", proj_ac, proj_zz, ratio

    return "MIXED", proj_ac, proj_zz, ratio


# -----------------------------
# Neighbor list
# -----------------------------
def build_neighbor_list(atoms: Atoms, cutoff: float) -> NeighborList:
    cutoffs = [cutoff] * len(atoms)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True, skin=0.0)
    nl.update(atoms)
    return nl

# ----------------------------- # Site-indexed neighbor extraction # -----------------------------

def extract_site_indexed_neighbors(
    atoms: Atoms,
    pristine_atoms: Atoms,
    cfg: CatalogConfig,
    system_id: str,
    structure_tag: str,
    extra_meta: Dict[str, Any],
) -> pd.DataFrame:
    """
    Fallback path (no dump): Hungarian match final structure -> pristine site indices.
    """
    e_ac, e_zz = get_ac_zz_unit_vectors(pristine_atoms, cfg)
    nl = build_neighbor_list(atoms, cfg.neighbor_cutoff_A)

    atom2site, missing_sites, n_sites_found, dmax, dp95, dmean = infer_sites_by_hungarian(
        atoms, pristine_atoms, total_sites=cfg.total_sites
    )

    pos = atoms.get_positions()
    cell = atoms.get_cell()

    rows = []
    for i in range(len(atoms)):
        if i not in atom2site:
            continue
        u = atom2site[i]
        indices, offsets = nl.get_neighbors(i)
        for j, off in zip(indices, offsets):
            if j not in atom2site:
                continue
            v = atom2site[j]

            # Use MIC displacement for both distance and projections
            disp0 = pos[j] - pos[i]
            disp_mic, d = find_mic(disp0, cell, atoms.get_pbc())
            dist = float(d)

            # Hard guard: should never exceed cutoff if NL is consistent
            if dist > cfg.neighbor_cutoff_A + 1e-6:
                continue

            disp = disp_mic
            
            if getattr(cfg, "_learned_dirs", None) is not None:
                d0, d1, n_hat = cfg._learned_dirs
                label0, label1 = cfg._learned_label_names
                lab, p0, p1, margin = classify_bond_by_learned_dirs(
                    disp_mic, d0, d1, n_hat, mixed_angle_deg=cfg._mixed_angle_deg
                )
                if lab == "D0": lab = label0
                elif lab == "D1": lab = label1

                label = lab
                proj_ac = p0   # you can store as proj_dir0/proj_dir1 instead if you prefer
                proj_zz = p1
                ratio = margin # store margin (deg) into abs_ratio_ac_to_zz, rename later if you like
            else:
                # old classifier
                label, proj_ac, proj_zz, ratio = classify_bond_ac_zz(
                    disp=disp, e_ac=e_ac, e_zz=e_zz, ratio_thr=cfg.ac_zz_ratio_threshold
                )
            cross_pbc = bool(np.any(off != 0))

            rows.append({
                **extra_meta,
                "system_id": system_id,
                "structure": structure_tag,
                "natoms": int(len(atoms)),
                "u_site": int(u),
                "v_site": int(v),
                "elem_u": atoms[i].symbol,
                "elem_v": atoms[j].symbol,
                "distance_A": dist,
                "proj_ac_A": proj_ac,
                "proj_zz_A": proj_zz,
                "abs_ratio_ac_to_zz": ratio,
                "label": label,
                "cross_pbc": cross_pbc if cfg.include_cross_pbc_flag else False,
                "offset_a": int(off[0]),
                "offset_b": int(off[1]),
                "offset_c": int(off[2]),
                "assign_dist_max_A": dmax,
                "assign_dist_p95_A": dp95,
                "assign_dist_mean_A": dmean,
                "missing_sites": " ".join(map(str, sorted(missing_sites))),
                "n_sites_found": " ".join(map(str, sorted(n_sites_found))),
            })

    return pd.DataFrame(rows)


# -----------------------------
# Manifest-aware batch processing (site-indexed)
# -----------------------------
def _safe_get_path(row: pd.Series, col: str) -> Optional[str]:
    v = row.get(col, None)
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def generate_tier1_site_neighbor_catalog_from_manifest(
    manifest_path: str | Path,
    config_path: str | Path,
    out_path: str | Path,
    *,
    include_structures: Tuple[str, ...] = ("base", "ft", "dft"),
    mapping_out_path: Optional[str | Path] = None,
) -> Path:
    cfg, raw_json = load_config(config_path)
    if not cfg.pristine_reference_path:
        raise ValueError("config must set pristine_reference_path (144-atom pristine structure)")
    
    print("[cfg] neighbor_cutoff_A =", cfg.neighbor_cutoff_A)

    pristine_atoms = load_structure_from_path(cfg.pristine_reference_path)
    if len(pristine_atoms) != cfg.total_sites:
        raise ValueError(f"pristine_reference_path natoms={len(pristine_atoms)} expected {cfg.total_sites}")
    
    print("cell:", pristine_atoms.get_cell().array)
    print("pbc:", pristine_atoms.get_pbc())
    
    bcfg = raw_json.get("bond_classification", {})
    if bcfg.get("mode","cell") == "learned":
        dir0, dir1, n_hat = learn_two_inplane_directions_from_pristine(
            pristine_atoms,
            neighbor_cutoff_A=cfg.neighbor_cutoff_A,
            plane=bcfg.get("plane","ab"),
            k=int(bcfg.get("k",2)),
        )
        if bcfg.get("name_by_cell_axis", True):
            named = name_dirs_as_ac_zz(dir0, dir1, pristine_atoms)
            cfg._learned_dirs = (named["AC"], named["ZZ"], n_hat)  # stash runtime-only
            cfg._learned_label_names = ("AC","ZZ")
        else:
            cfg._learned_dirs = (dir0, dir1, n_hat)
            cfg._learned_label_names = ("D0","D1")
        cfg._mixed_angle_deg = float(bcfg.get("mixed_angle_deg", 10.0))
        print("[learned dirs] dir0:", cfg._learned_dirs[0], "dir1:", cfg._learned_dirs[1], "n_hat:", cfg._learned_dirs[2])
        # ---- DIAGNOSTICS (add these lines) ----
        counts = bond_label_counts(
            pristine_atoms,
            cfg._learned_dirs[0], cfg._learned_dirs[1], cfg._learned_dirs[2],
            cutoff_A=cfg.neighbor_cutoff_A,
            mixed_angle_deg=cfg._mixed_angle_deg,
        )
        print("[diag pristine learned-label counts]", dict(counts))
        counts_old = bond_label_counts_old(pristine_atoms, cfg)
        print("[diag pristine old-label counts]", dict(counts_old))
        def write_direction_metadata(path, cfg, counts_old, counts_new):
            ac, zz, n_hat = cfg._learned_dirs
            data = {
                "source": "learned",
                "label_names": list(cfg._learned_label_names),
                "ac_vector_3d": [float(x) for x in ac],
                "zz_vector_3d": [float(x) for x in zz],
                "plane_normal_3d": [float(x) for x in n_hat],
                "mixed_angle_deg": float(cfg._mixed_angle_deg),
                "diag_pristine_learned_label_counts": dict(counts_new),
                "diag_pristine_old_label_counts": dict(counts_old),
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        write_direction_metadata(bcfg.get("output_path", "learned_dirs"), cfg, counts_old, counts)

    dfm = pd.read_csv(Path(manifest_path), dtype=str, low_memory=False).fillna("")

    # UPDATED required columns
    required = {
        "system_id", "split", "is_pristine", "dft_path",
        "base_in_lmp", "base_out_lmp", "base_dump",
        "ft_in_lmp", "ft_out_lmp", "ft_dump",
    }
    missing = required - set(dfm.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    dfm["is_pristine"] = (
        dfm["is_pristine"].astype(str).str.strip().str.lower()
        .isin(["true", "1", "t", "yes", "y"])
    )
    dfm["split"] = dfm["split"].astype(str).str.strip()

    if cfg.only_splits:
        dfm = dfm[dfm["split"].isin(cfg.only_splits)].copy()
    if not cfg.include_pristine:
        dfm = dfm[~dfm["is_pristine"]].copy()

    all_parts: List[pd.DataFrame] = []
    all_mapping_parts: List[pd.DataFrame] = []

    for _, row in dfm.iterrows():
        system_id = str(row["system_id"]).strip()
        split = str(row["split"]).strip()
        is_pristine = bool(row["is_pristine"])

        # ---------------- base (prefer dump) ----------------
        if "base" in include_structures:
            p_dump = _safe_get_path(row, "base_dump")
            p_in = _safe_get_path(row, "base_in_lmp")
            p_out = _safe_get_path(row, "base_out_lmp")
            extra_meta = {
                "split": split,
                "is_pristine": is_pristine,
                "formula": row.get("formula", ""),
                "pristine_mode": row.get("pristine_mode", ""),
                "dft_supercell_tag": "",
                "lam_supercell_tag": row.get("lam_supercell_tag", ""),
            }

            if p_dump and Path(p_dump).exists():
                # Use base_out_lmp (or base_in_lmp) to infer type->element via Masses
                mass_src = p_out or p_in
                if not mass_src:
                    raise ValueError(f"{system_id}: base dump exists but no base_in_lmp/base_out_lmp to parse Masses")
                masses = read_lammps_masses_type_map(mass_src)
                type2elem = infer_type_to_element_from_masses(masses)

                # first frame -> id2site
                _, _, ids0, _, pos0 = read_first_frame_lammpstrj(p_dump)
                id2site, dmax0, dp950, dmean0 = hungarian_map_ids_to_sites_with_stats(
                    ids0, pos0, pristine_atoms, total_sites=cfg.total_sites
                )
                # last frame -> neighbors
                _, boundsL, idsL, typesL, posL = read_last_frame_lammpstrj(p_dump)
                
                atomsL = _atoms_from_bounds_ids_types_pos(
                    bounds=boundsL,
                    ids=idsL,
                    types=typesL,
                    pos=posL,
                    type2elem=type2elem,
                )

                # convert id2site -> atom-index-based mapping for atomsL row order
                atom2site_L = {}
                for a_idx, atom_id in enumerate(idsL):
                    atom_id = int(atom_id)
                    if atom_id in id2site:
                        atom2site_L[a_idx] = int(id2site[atom_id])

                all_mapping_parts.append(build_site_mapping_rows_from_atom2site(
                    atoms_final=atomsL,
                    pristine_atoms=pristine_atoms,
                    atom2site=atom2site_L,
                    system_id=system_id,
                    structure_tag="base",
                    extra_meta=extra_meta,
                    assign_dist_max_A=dmax0,
                    assign_dist_p95_A=dp950,
                    assign_dist_mean_A=dmean0,
                    atom_identifier_kind="lammps_id",
                    atom_identifiers=[int(x) for x in idsL],
                ))


                all_parts.append(extract_site_neighbors_from_dump(
                    bounds_last=boundsL,
                    ids_last=idsL,
                    types_last=typesL,
                    pos_last=posL,
                    id2site=id2site,
                    type2elem=type2elem,
                    pristine_atoms=pristine_atoms,
                    cfg=cfg,
                    system_id=system_id,
                    structure_tag="base",
                    extra_meta=extra_meta,
                    assign_dist_max_A=dmax0,
                    assign_dist_p95_A=dp950,
                    assign_dist_mean_A=dmean0,
                ))
            elif p_out:
                # fallback: use final structure file
                if cfg.strict_path_exists and not Path(p_out).exists():
                    raise FileNotFoundError(f"Missing base_out_lmp for {system_id}: {p_out}")
                if Path(p_out).exists():
                    atoms = load_structure_from_path(p_out)
                    if is_pristine:
                        atoms = maybe_expand_pristine_lam(atoms, cfg, system_id, "base")
                    atom2site, missing_sites, n_sites_found, dmax, dp95, dmean = infer_sites_by_hungarian(
                        atoms, pristine_atoms, total_sites=cfg.total_sites
                    )
                    all_mapping_parts.append(build_site_mapping_rows_from_atom2site(
                        atoms_final=atoms,
                        pristine_atoms=pristine_atoms,
                        atom2site=atom2site,
                        system_id=system_id,
                        structure_tag="base",
                        extra_meta=extra_meta,
                        assign_dist_max_A=dmax,
                        assign_dist_p95_A=dp95,
                        assign_dist_mean_A=dmean,
                        atom_identifier_kind="atom_index",
                        atom_identifiers=list(range(len(atoms))),
                    ))
                    all_parts.append(extract_site_indexed_neighbors(
                        atoms, pristine_atoms, cfg, system_id, "base", extra_meta
                    ))

        # ---------------- ft (prefer dump) ----------------
        if "ft" in include_structures:
            p_dump = _safe_get_path(row, "ft_dump")
            p_in = _safe_get_path(row, "ft_in_lmp")
            p_out = _safe_get_path(row, "ft_out_lmp")
            
            extra_meta = {
                "split": split,
                "is_pristine": is_pristine,
                "formula": row.get("formula", ""),
                "pristine_mode": row.get("pristine_mode", ""),
                "dft_supercell_tag": "",
                "lam_supercell_tag": row.get("lam_supercell_tag", ""),
            }

            if p_dump and Path(p_dump).exists():
                mass_src = p_out or p_in
                if not mass_src:
                    raise ValueError(f"{system_id}: ft dump exists but no ft_in_lmp/ft_out_lmp to parse Masses")
                masses = read_lammps_masses_type_map(mass_src)
                type2elem = infer_type_to_element_from_masses(masses)

                # first frame -> id2site
                _, _, ids0, _, pos0 = read_first_frame_lammpstrj(p_dump)
                id2site, dmax0, dp950, dmean0 = hungarian_map_ids_to_sites_with_stats(
                    ids0, pos0, pristine_atoms, total_sites=cfg.total_sites
                )

                # last frame -> neighbors
                _, boundsL, idsL, typesL, posL = read_last_frame_lammpstrj(p_dump)
                
                atomsL = _atoms_from_bounds_ids_types_pos(
                    bounds=boundsL,
                    ids=idsL,
                    types=typesL,
                    pos=posL,
                    type2elem=type2elem,
                )

                atom2site_L = {}
                for a_idx, atom_id in enumerate(idsL):
                    atom_id = int(atom_id)
                    if atom_id in id2site:
                        atom2site_L[a_idx] = int(id2site[atom_id])

                all_mapping_parts.append(build_site_mapping_rows_from_atom2site(
                    atoms_final=atomsL,
                    pristine_atoms=pristine_atoms,
                    atom2site=atom2site_L,
                    system_id=system_id,
                    structure_tag="ft",
                    extra_meta=extra_meta,
                    assign_dist_max_A=dmax0,
                    assign_dist_p95_A=dp950,
                    assign_dist_mean_A=dmean0,
                    atom_identifier_kind="lammps_id",
                    atom_identifiers=[int(x) for x in idsL],
                ))
                
                all_parts.append(extract_site_neighbors_from_dump(
                    bounds_last=boundsL,
                    ids_last=idsL,
                    types_last=typesL,
                    pos_last=posL,
                    id2site=id2site,
                    type2elem=type2elem,
                    pristine_atoms=pristine_atoms,
                    cfg=cfg,
                    system_id=system_id,
                    structure_tag="ft",
                    extra_meta=extra_meta,
                    assign_dist_max_A=dmax0,
                    assign_dist_p95_A=dp950,
                    assign_dist_mean_A=dmean0,
                ))
            elif p_out:
                if cfg.strict_path_exists and not Path(p_out).exists():
                    raise FileNotFoundError(f"Missing ft_out_lmp for {system_id}: {p_out}")
                if Path(p_out).exists():
                    atoms = load_structure_from_path(p_out)
                    if is_pristine:
                        atoms = maybe_expand_pristine_lam(atoms, cfg, system_id, "ft")
                    atom2site, missing_sites, n_sites_found, dmax, dp95, dmean = infer_sites_by_hungarian(
                        atoms, pristine_atoms, total_sites=cfg.total_sites
                    )
                    all_mapping_parts.append(build_site_mapping_rows_from_atom2site(
                        atoms_final=atoms,
                        pristine_atoms=pristine_atoms,
                        atom2site=atom2site,
                        system_id=system_id,
                        structure_tag="base",
                        extra_meta=extra_meta,
                        assign_dist_max_A=dmax,
                        assign_dist_p95_A=dp95,
                        assign_dist_mean_A=dmean,
                        atom_identifier_kind="atom_index",
                        atom_identifiers=list(range(len(atoms))),
                    ))
                    all_parts.append(extract_site_indexed_neighbors(
                        atoms, pristine_atoms, cfg, system_id, "ft", extra_meta
                    ))

        # ---------------- dft (no dump) ----------------
        if "dft" in include_structures:
            p = _safe_get_path(row, "dft_path")
            extra_meta = {
                "split": split,
                "is_pristine": is_pristine,
                "formula": row.get("formula", ""),
                "pristine_mode": row.get("pristine_mode", ""),
                "dft_supercell_tag": row.get("dft_supercell_tag", ""),
                "lam_supercell_tag": "",
            }
            if p:
                if cfg.strict_path_exists and not Path(p).exists():
                    raise FileNotFoundError(f"Missing dft_path for {system_id}: {p}")

                neigh_df, map_df = extract_site_indexed_neighbors_from_vasp_path(
                    vasp_path=p,
                    pristine_atoms=pristine_atoms,
                    cfg=cfg,
                    system_id=system_id,
                    structure_tag="dft",
                    extra_meta=extra_meta,
                )
                all_parts.append(neigh_df)
                all_mapping_parts.append(map_df)


    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not all_parts:
        raise RuntimeError("No structures were processed. Check split filters and path columns.")

    out_df = pd.concat(all_parts, ignore_index=True)
    m = out_df[out_df["structure"].isin(["base","ft"])]["distance_A"].max()
    print("max base/ft distance:", m)

    if cfg.output_format.lower() == "parquet" or out_path.suffix.lower() == ".parquet":
        out_df.to_parquet(out_path, index=False)
    else:
        out_df.to_csv(out_path, index=False)

    if all_mapping_parts:
        mapping_df = pd.concat(all_mapping_parts, ignore_index=True)

        if mapping_out_path is None:
            if out_path.suffix:
                mapping_out = out_path.with_name(out_path.stem + ".site_mapping.csv")
            else:
                mapping_out = Path(str(out_path) + ".site_mapping.csv")
        else:
            mapping_out = Path(mapping_out_path)

        mapping_out.parent.mkdir(parents=True, exist_ok=True)
        mapping_df.to_csv(mapping_out, index=False)
        print(f"[OK] wrote site-mapping CSV: {mapping_out}")

    return out_path


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--structures", default="base,ft,dft")
    parser.add_argument("--mapping_out", default=None)
    args = parser.parse_args()

    structures = tuple([s.strip() for s in args.structures.split(",") if s.strip()])
    out = generate_tier1_site_neighbor_catalog_from_manifest(
        manifest_path=args.manifest,
        config_path=args.config,
        out_path=args.out,
        include_structures=structures,
        mapping_out_path=args.mapping_out,
    )
    print(f"[OK] wrote: {out}")