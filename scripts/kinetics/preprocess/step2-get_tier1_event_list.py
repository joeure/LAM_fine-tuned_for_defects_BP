import json
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# --- optional: if you want the alignment/matching check ---
from ase.io import read
from ase.geometry import find_mic

def mic_dist_between_sites(i: int, j: int, pos_ref: np.ndarray, cell, pbc) -> float:
    disp = pos_ref[j] - pos_ref[i]
    _, d = find_mic(disp, cell, pbc)
    return float(d)

def mic_dist_to_site_set(i: int, targets: Set[int], pos_ref: np.ndarray, cell, pbc) -> float:
    if not targets:
        return float("inf")
    idx = np.fromiter(targets, dtype=int)
    disp = pos_ref[idx] - pos_ref[i][None, :]
    _, d = find_mic(disp, cell, pbc)
    return float(np.min(d))

def d_nearest_N_mic_from_positions(
    r_site: np.ndarray,
    r_N: np.ndarray,
    cell,
    pbc,
) -> float:
    """
    MIC distance from a site position to nearest N atom position.
    """
    if r_N.size == 0:
        return float("inf")
    disp = r_N - r_site[None, :]
    _, d = find_mic(disp, cell, pbc)
    return float(np.min(d))

def parse_idx_list(s) -> List[int]:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return []
    ss = str(s).strip()
    if not ss:
        return []
    return [int(x) for x in ss.split()]


def build_pristine_graph_and_positions(
    df: pd.DataFrame,
    include_labels: Set[str],
) -> Dict[int, Dict[str, Set[int]]]:
    """
    Build pristine neighbor graph from the neighbor CSV.

    Supports:
      - old schema: i/j
      - new schema: u_site/v_site (site-indexed)
    """
    # pick the right edge columns
    if {"u_site", "v_site"}.issubset(df.columns):
        ucol, vcol = "u_site", "v_site"
    elif {"i", "j"}.issubset(df.columns):
        ucol, vcol = "i", "j"
    else:
        raise ValueError(
            "Neighbor CSV missing edge columns. Expected either (u_site,v_site) or (i,j)."
        )

    # filter pristine dft rows; natoms might be str in CSV, so be robust
    p = df[
        (df["is_pristine"] == True)
        & (df["structure"].astype(str).str.strip() == "dft")
        & (df["natoms"].astype(int) == 144)
    ].copy()

    print("pristine-dft label counts:\n", p["label"].astype(str).str.strip().value_counts().head(10))
    
    if p.empty:
        raise ValueError("No pristine dft rows found in neighbor CSV (structure='dft', natoms=144, is_pristine=True).")

    neighbors: Dict[int, Dict[str, Set[int]]] = {}
    for _, r in p.iterrows():
        label = str(r["label"]).strip()
        if label not in include_labels:
            continue
        u = int(r[ucol]); v = int(r[vcol])
        neighbors.setdefault(u, {}).setdefault(label, set()).add(v)
        neighbors.setdefault(v, {}).setdefault(label, set()).add(u)

    return neighbors

def build_unlabeled_adj(pristine_graph: Dict[int, Dict[str, Set[int]]], include_labels: Set[str]) -> Dict[int, Set[int]]:
    adj = {}
    for u, labmap in pristine_graph.items():
        for lab, vs in labmap.items():
            if lab not in include_labels:
                continue
            adj.setdefault(u, set()).update(vs)
    return adj

from collections import deque

def min_hops_to_set(start: int, targets: Set[int], adj: Dict[int, Set[int]], max_depth: int) -> int:
    if start in targets:
        return 0
    q = deque([(start, 0)])
    seen = {start}
    while q:
        u, d = q.popleft()
        if d >= max_depth:
            continue
        for v in adj.get(u, []):
            if v in seen:
                continue
            if v in targets:
                return d + 1
            seen.add(v)
            q.append((v, d + 1))
    return max_depth + 1  # represent "farther than max_depth"

def load_pristine_last_frame(pristine_atoms_path: str | Path):
    p = str(pristine_atoms_path)
    if p.lower().endswith("vasprun.xml"):
        # last ionic step
        return read(p, format="vasp-xml", index=-1)
    # single-frame structures
    return read(p)

def load_pristine_positions_from_atoms(pristine_path: str) -> np.ndarray:
    atoms = read(pristine_path)
    if len(atoms) != 144:
        raise ValueError(f"Pristine Atoms natoms={len(atoms)} but expected 144: {pristine_path}")
    return atoms.get_positions()


def d_nearest_N(site: int, n_sites: List[int], pos: np.ndarray) -> float:
    if not n_sites:
        return float("inf")
    r = pos[site]
    rr = pos[n_sites]
    return float(np.linalg.norm(rr - r[None, :], axis=1).min())

def d_nearest_N_mic(site: int, n_sites: List[int], pos: np.ndarray, cell, pbc) -> float:
    if not n_sites:
        return float("inf")
    r = pos[site]
    rr = pos[n_sites]  # (nN,3)
    disp = rr - r[None, :]
    disp_mic, d = find_mic(disp, cell, pbc)
    return float(np.min(d))

def eval_near_angstrom(site: int, n_set: Set[int], pos_ref, cell_ref, pbc_ref, near_cut: float) -> tuple[bool, float]:
    d = mic_dist_to_site_set(site, n_set, pos_ref, cell_ref, pbc_ref)
    return (d < near_cut), d

def eval_near_graph(site: int, n_set: Set[int], adj_unlabeled, near_hops: int, far_hops: int) -> tuple[bool, int]:
    # we compute hops up to far_hops because escape may need it too
    h = min_hops_to_set(site, n_set, adj_unlabeled, max_depth=far_hops)
    return (h <= near_hops), h

def eval_escape_euclidean(
    d_init: float,
    d_fin: float,
    *,
    far_cut: float,
    req_init_near: bool,
    is_near_init: bool
) -> bool:
    return (d_fin > far_cut) and (is_near_init if req_init_near else True)

def eval_escape_topological(
    hops_init: int,
    hops_fin: int,
    *,
    near_hops: int,
    far_hops: int,
    escape_rule: str,
    req_init_near: bool
) -> bool:
    is_near_init = (hops_init <= near_hops)
    is_near_fin  = (hops_fin  <= near_hops)
    is_far_fin   = (hops_fin  >  far_hops)

    if escape_rule == "near_to_not_near":
        return (is_near_init if req_init_near else True) and (not is_near_fin)
    # default: near_to_far
    return (is_near_init if req_init_near else True) and is_far_fin

def generate_tier1_events_from_neighbors(
    neighbor_csv: str,
    config_json: str,
    pristine_atoms_path: str,
    out_events_csv: str,
    *,
    restrict_splits: Optional[Set[str]] = None,
) -> None:
    df = pd.read_csv(neighbor_csv, low_memory=False)

    df["is_pristine"] = (
        df["is_pristine"].astype(str).str.strip().str.lower()
        .isin(["true", "1", "t", "yes", "y"])
    )

    cfg = json.loads(Path(config_json).read_text())
    evcfg = cfg.get("tier1_event", {})
    include_labels = set(evcfg.get("include_labels", ["AC", "ZZ"]))
    mode_near = evcfg.get("nearN_mode", "angstrom").lower()  # "angstrom" or "graph"
    mode_escape = evcfg.get("escape_mode", "same_as_nearN").lower() # "topological" | "euclidean" | "same_as_nearN"
    escape_rule = evcfg.get("escape_rule", "near_to_far").lower() # "near_to_far" | "near_to_not_near"

    near_cut = float(evcfg.get("nearN_cutoff_A", 3.0))
    far_cut = float(evcfg.get("escape_far_cutoff_A", 6.0))
    req_init_near = bool(evcfg.get("escape_requires_init_nearN", True))

    near_hops = int(evcfg.get("nearN_graph_hops", 1))
    far_hops = int(evcfg.get("escape_graph_far_hops", near_hops + 1))

    allow_n_diffusive = bool(evcfg.get("allow_substitution_hops", False))

    # Filter splits
    if restrict_splits is not None:
        if isinstance(restrict_splits, str):
            restrict_splits = {s.strip() for s in restrict_splits.split(",") if s.strip()}
        df["split"] = df["split"].astype(str).str.strip()
        df = df[df["split"].isin(list(restrict_splits))].copy()

    # Build pristine graph from neighbor CSV
    pristine_graph = build_pristine_graph_and_positions(df, include_labels)
    adj_unlabeled = build_unlabeled_adj(pristine_graph, include_labels)
    ac_nodes = sum(1 for u in pristine_graph if "AC" in pristine_graph[u] and pristine_graph[u]["AC"])
    zz_nodes = sum(1 for u in pristine_graph if "ZZ" in pristine_graph[u] and pristine_graph[u]["ZZ"])
    print("nodes with AC neighbors:", ac_nodes, "nodes with ZZ neighbors:", zz_nodes)

    # Load pristine positions (for site positions)
    pristine_atoms = load_pristine_last_frame(pristine_atoms_path)
    if len(pristine_atoms) != 144:
        raise ValueError(f"pristine_atoms_path must be 144-atom supercell, got {len(pristine_atoms)}")
    pos_ref = pristine_atoms.get_positions()
    cell_ref = pristine_atoms.get_cell()
    pbc_ref = pristine_atoms.get_pbc()


    out_rows = []
    group_cols = ["system_id", "structure"]

    # helper to parse sets from group
    def get_group_sites(g: pd.DataFrame) -> Tuple[List[int], Set[int]]:
        if "missing_sites" not in g.columns or "n_sites_found" not in g.columns:
            raise ValueError("neighbor_csv must contain missing_sites and n_sites_found columns (from step1).")
        vac_sites = parse_idx_list(g["missing_sites"].iloc[0])
        n_set = set(parse_idx_list(g["n_sites_found"].iloc[0]))
        return vac_sites, n_set

    stats_total, stats_init_near, stats_fin_near, stats_fin_far, stats_escape_candidates = 0, 0, 0, 0, 0
    max_fin_given_init_near = -1.0
    max_delta_given_init_near = -1.0
    for (system_id, structure), g in df.groupby(group_cols, sort=False):
        if bool(g["is_pristine"].iloc[0]):
            continue

        vac_sites, n_set = get_group_sites(g)
        if not vac_sites:
            continue  # nothing to do

        missing_set = set(vac_sites)

        def forbid_nb(nb_site: int) -> bool:
            return (not allow_n_diffusive) and (nb_site in n_set)

        for vac_site in vac_sites:
            for label in include_labels:
                nb_set = pristine_graph.get(vac_site, {}).get(label, set())
                for nb_site in nb_set:
                    if nb_site in missing_set:
                        continue
                    if forbid_nb(nb_site):
                        continue
                    # --- compute metrics (always available for switching) ---
                    # Angstrom distances in pristine metric
                    is_near_init_A, d_init = eval_near_angstrom(vac_site, n_set, pos_ref, cell_ref, pbc_ref, near_cut)
                    is_near_fin_A,  d_fin  = eval_near_angstrom(nb_site,  n_set, pos_ref, cell_ref, pbc_ref, near_cut)

                    # Topological hops
                    is_near_init_h, hops_init = eval_near_graph(vac_site, n_set, adj_unlabeled, near_hops, far_hops)
                    is_near_fin_h,  hops_fin  = eval_near_graph(nb_site,  n_set, adj_unlabeled, near_hops, far_hops)

                    if mode_near == "graph":
                        is_near_init = is_near_init_h
                        is_near_fin  = is_near_fin_h
                        metric_init  = hops_init
                        metric_fin   = hops_fin
                    else:
                        # "angstrom" mode in pristine lattice metric (site-to-N-site distance)
                        if not n_set:
                            # No substitutions => cannot define nearN/escape meaningfully
                            d_init = float("inf")
                            d_fin  = float("inf")
                            is_near_init = False
                            is_near_fin  = False
                            is_escape    = False
                        else:
                            is_near_init = is_near_init_A
                            is_near_fin  = is_near_fin_A

                        stats_total += 1
                        if d_init < near_cut: 
                            stats_init_near += 1
                            max_fin_given_init_near = max(max_fin_given_init_near, d_fin)
                            max_delta_given_init_near = max(max_delta_given_init_near, d_fin - d_init)

                        metric_init, metric_fin = d_init, d_fin
                        
                    # --- escape depends on escape_mode ---
                    if mode_escape == "same_as_nearn":
                        mode_escape_eff = mode_near
                    else:
                        mode_escape_eff = mode_escape
                    
                    if mode_escape_eff == "topological" or mode_escape_eff == "graph":
                        is_escape = eval_escape_topological(
                            hops_init, hops_fin,
                            near_hops=near_hops, far_hops=far_hops,
                            escape_rule=escape_rule,
                            req_init_near=req_init_near
                        )
                    elif n_set:
                        is_escape = eval_escape_euclidean(
                            d_init, d_fin,
                            far_cut=far_cut,
                            req_init_near=req_init_near,
                            is_near_init=is_near_init  # IMPORTANT: use the nearN_mode’s notion of init-near
                        )

                    event_type = f"{label}_hop"
                    if is_escape:
                        event_type = "trap_escape"
                    elif is_near_init or is_near_fin:
                        event_type = "nearN_hop"
                    
                    if is_near_init: stats_fin_near += 1
                    if is_near_fin:  stats_fin_far  += 1
                    if is_escape: stats_escape_candidates += 1

                    out_rows.append({
                        "system_id": system_id,
                        "structure": structure,
                        "vac_site_144": vac_site,
                        "neighbor_site_144": nb_site,
                        "hop_label": label,
                        "event_type": event_type,
                        "d_NV_init_A": metric_init,
                        "d_NV_final_A": metric_fin,
                        "d_init_A": d_init, "d_fin_A": d_fin,
                        "hops_init": hops_init, "hops_fin": hops_fin,
                        "is_nearN_init": is_near_init,
                        "is_nearN_final": is_near_fin,
                        "is_escape": is_escape,
                        "split": g["split"].iloc[0],
                    })
    print("[diag] total edges:", stats_total)
    print("[diag] init_near:", stats_init_near, "fin_near:", stats_fin_near, "fin_far:", stats_fin_far)
    print("[diag] escape_candidates (init_near & fin_far):", stats_escape_candidates)
    print("[diag] max d_fin given init_near:", max_fin_given_init_near)
    print("[diag] max (d_fin - d_init) given init_near:", max_delta_given_init_near)

    out_df = pd.DataFrame(out_rows)
    Path(out_events_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_events_csv, index=False)
    print(f"[OK] wrote events: {out_events_csv}  rows={len(out_df)}")


# ---------------- OPTIONAL: mapping check using manifest ----------------
def site_matching_check(
    manifest_csv: str,
    pristine_atoms_path: str,
    config_json: str,
    out_report_csv: str,
    *,
    only_splits: Optional[Set[str]] = None,
    structures_to_check: Set[str] = {"base", "ft", "dft"},
) -> None:
    """
    Geometry-only mapping report (Hungarian + MIC).

    For each (system_id, structure):
      - Build MIC distance matrix D (N_atoms x 144_sites) between defect atoms and pristine sites
      - Hungarian assignment: each defect atom -> unique pristine site (min-sum)
      - missing_sites = all_sites - assigned_sites  (interpretable as vacancy sites)
      - n_sites_found = assigned sites for atoms whose symbol is 'N'
      - Report only quality metrics and inferred sets; no comparison to any meta idx_*.

    Output columns
    --------------
    system_id, structure, split, status, reason,
    natoms, total_sites,
    assigned_dist_max_A, assigned_dist_p95_A, assigned_dist_mean_A, assigned_dist_sum_A,
    within_max_tol, within_p95_tol,
    missing_size, n_found_size,
    missing_sites, n_sites_found
    """
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from ase.io import read
    from ase.geometry import find_mic

    try:
        from scipy.optimize import linear_sum_assignment
    except Exception as e:
        raise ImportError("scipy is required for Hungarian assignment. Install scipy.") from e

    # -----------------------
    # config
    # -----------------------
    cfg = json.loads(Path(config_json).read_text())
    ccfg = cfg.get("site_match_check", {})
    enable = bool(ccfg.get("enable", True))
    if not enable:
        return

    total_sites = int(ccfg.get("total_sites", 144))

    # pass/fail indicators (not hard fail; just report)
    max_dist_tol = float(ccfg.get("max_assign_dist_A", 2.5))
    p95_dist_tol = float(ccfg.get("p95_assign_dist_A", 1.5))

    # normalize only_splits if passed as string
    if isinstance(only_splits, str):
        only_splits = {s.strip() for s in only_splits.split(",") if s.strip()}

    # -----------------------
    # load manifest + pristine
    # -----------------------
    man = pd.read_csv(manifest_csv, dtype=str, low_memory=False).fillna("")
    man["split"] = man.get("split", "").astype(str).str.strip()

    if only_splits is not None:
        man = man[man["split"].isin(list(only_splits))].copy()

    # robust is_pristine
    if "is_pristine" in man.columns:
        man["is_pristine"] = (
            man["is_pristine"].astype(str).str.strip().str.lower()
            .isin(["true", "1", "t", "yes", "y"])
        )
    else:
        man["is_pristine"] = False

    pristine_atoms = load_pristine_last_frame(pristine_atoms_path)
    if len(pristine_atoms) != 144:
        raise ValueError(f"pristine_atoms_path must be 144-atom supercell, got {len(pristine_atoms)}")
    pos_ref = pristine_atoms.get_positions()
    cell_ref = pristine_atoms.get_cell()
    pbc_ref = pristine_atoms.get_pbc()

    # -----------------------
    # helper: load atoms from path
    # -----------------------
    def load_atoms(path_str: str):
        pp = Path(path_str)
        # prefer vasprun.xml if OUTCAR is given
        if pp.name.upper() == "OUTCAR":
            vrun = pp.parent / "vasprun.xml"
            if vrun.exists():
                pp = vrun

        if pp.suffix == ".data":
            return read(str(pp), format="lammps-data", atom_style="atomic")
        if pp.name.lower() == "vasprun.xml":
            return read(str(pp), format="vasp-xml")
        return read(str(pp))

    # -----------------------
    # main
    # -----------------------
    rows = []

    for _, r in man.iterrows():
        if bool(r.get("is_pristine", False)):
            continue

        system_id = str(r.get("system_id", "")).strip()
        split = str(r.get("split", "")).strip()
        if not system_id:
            continue

        paths = {
        "base": str(r.get("base_out_lmp", "")).strip(),
        "ft":  str(r.get("ft_out_lmp", "")).strip(),
        "dft": str(r.get("dft_path", "")).strip(),
        }

        for tag, p in paths.items():
            if tag not in structures_to_check:
                continue
            if not p:
                continue

            try:
                atoms = load_atoms(p)
            except Exception as e:
                rows.append({
                    "system_id": system_id,
                    "structure": tag,
                    "split": split,
                    "status": "fail",
                    "reason": f"load_error: {type(e).__name__}: {e}",
                })
                continue

            N = len(atoms)
            if N > total_sites:
                rows.append({
                    "system_id": system_id,
                    "structure": tag,
                    "split": split,
                    "status": "fail",
                    "reason": f"natoms={N} > total_sites={total_sites}",
                    "natoms": N,
                    "total_sites": total_sites,
                })
                continue

            pos = atoms.get_positions()

            # MIC distance matrix D: (N x total_sites)
            D = np.empty((N, total_sites), dtype=float)
            for a in range(N):
                disp = pos_ref - pos[a][None, :]
                _, d = find_mic(disp, cell_ref, pbc_ref)
                D[a, :] = d

            row_ind, col_ind = linear_sum_assignment(D)
            assigned_sites = set(int(s) for s in col_ind)
            missing_sites = set(range(total_sites)) - assigned_sites

            assigned_dists = D[row_ind, col_ind]
            dist_max = float(np.max(assigned_dists)) if assigned_dists.size else float("nan")
            dist_p95 = float(np.percentile(assigned_dists, 95)) if assigned_dists.size else float("nan")
            dist_mean = float(np.mean(assigned_dists)) if assigned_dists.size else float("nan")
            dist_sum = float(np.sum(assigned_dists)) if assigned_dists.size else float("nan")

            atom2site = {int(a): int(s) for a, s in zip(row_ind, col_ind)}
            syms = atoms.get_chemical_symbols()
            n_sites_found = set()
            for a, sym in enumerate(syms):
                if sym == "N" and a in atom2site:
                    n_sites_found.add(atom2site[a])

            rows.append({
                "system_id": system_id,
                "structure": tag,
                "split": split,
                "status": "ok",
                "reason": "",
                "natoms": N,
                "total_sites": total_sites,
                "assigned_dist_max_A": dist_max,
                "assigned_dist_p95_A": dist_p95,
                "assigned_dist_mean_A": dist_mean,
                "assigned_dist_sum_A": dist_sum,
                "within_max_tol": (dist_max <= max_dist_tol),
                "within_p95_tol": (dist_p95 <= p95_dist_tol),
                "missing_size": len(missing_sites),
                "n_found_size": len(n_sites_found),
                "missing_sites": " ".join(map(str, sorted(missing_sites))),
                "n_sites_found": " ".join(map(str, sorted(n_sites_found))),
            })

    rep = pd.DataFrame(rows)
    Path(out_report_csv).parent.mkdir(parents=True, exist_ok=True)
    rep.to_csv(out_report_csv, index=False)
    print(f"[OK] wrote mapping report: {out_report_csv} rows={len(rep)}")

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--neighbors", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--pristine", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--splits", default="test,pristine")
    parser.add_argument("--match_check", action="store_true")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--match_out")
    
    args = parser.parse_args()
    
    # parse splits argument: "test" or "test,pristine"
    split_set = {s.strip() for s in args.splits.split(",") if s.strip()}

    generate_tier1_events_from_neighbors(
        neighbor_csv=args.neighbors,
        config_json=args.config,
        pristine_atoms_path=args.pristine,
        out_events_csv=args.out,
        restrict_splits=split_set
    )
    print(f"[OK] wrote: {args.out}")
    if args.match_check:
        site_matching_check(
            manifest_csv=args.manifest,
            pristine_atoms_path=args.pristine,
            config_json=args.config,
            out_report_csv=args.match_out,
            only_splits=split_set
        )
        print(f"[OK] wrote matching report: {args.match_out}")
    