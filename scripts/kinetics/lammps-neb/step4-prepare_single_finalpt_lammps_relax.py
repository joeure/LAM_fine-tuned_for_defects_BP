#!/usr/bin/env python3
"""
prepare_one_neb_endpoints.py

Prepare one event folder for a LAM-LAMMPS vacancy-hop endpoint test.

Packaging modes
---------------
1) --package-mode single_job
   One self-contained submission root for exactly one event.

2) --package-mode batch_shared_model
   One batch submission root containing:
     - models/
     - jobs/<job_name>/
     - inputs/in.relax_bundle.lammps
     - run.sh
   All file paths written into the LAMMPS input are relative to the batch root.

Current scope
-------------
- prepare init.data and final_guess.data
- prepare final endpoint relaxation input
- no NEB image generation yet
- no automatic basin-distinctness check yet

Dependencies
------------
    pip install ase pandas jinja2 numpy
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from ase import Atoms
from ase.geometry import find_mic
from ase.io import read, write
from jinja2 import Template

ELEMENT_MASSES = {
    "H": 1.00784,
    "He": 4.002602,
    "Li": 6.94,
    "Be": 9.0121831,
    "B": 10.81,
    "C": 12.011,
    "N": 14.0067,
    "O": 15.999,
    "F": 18.998403163,
    "Ne": 20.1797,
    "Na": 22.98976928,
    "Mg": 24.305,
    "Al": 26.9815385,
    "Si": 28.085,
    "P": 30.973761998,
    "S": 32.06,
    "Cl": 35.45,
    "Ar": 39.948,
}

## ATTENTION: 
# KOKKOS 在 GPU 上默认是 neigh = full、newton = off；并且当 neigh/thread on 时，newton pair 必须是 off
# BUT Pair style mace requires newton pair on neigh half
# So We need to reset the default settings for the KOKKOS
# 作为备选方案，也可以去掉 KOKKOS，考虑是否必要？ 这里的问题是 MACE 提供 mace/kk 选项，因此KOKKOS应该还是提供了加速和统一映射的作用，但是MACE 本身对 Torch层的直接依赖导致了二者的双重约束

SINGLE_TEMPLATE = r"""
# ======================= Initialization =======================
units           metal
atom_style      atomic
atom_modify     map yes
newton          on
boundary        p p p

neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes

# ====================== Read Structure ======================
read_data {{ structure_file }}

# ===== MACE potential =====================
pair_style      {{ pair_style }}
pair_coeff      * * {{ model_file }} {% for elem in elements %}{{ elem }} {% endfor %}

{{ mass_lines }}

variable sysname string {{ dump_name }}

shell mkdir -p dumps/${sysname}
shell mkdir -p results

dump        d1 all custom 1 dumps/${sysname}/traj.lammpstrj id type x y z fx fy fz
dump_modify d1 first yes sort id

thermo      1
thermo_style custom step pe fmax fnorm

min_style   {{ min_style }}
min_modify  dmax 0.03 line quadratic
minimize    1.0e-8 {{ ftol }} 10000 100000

write_data  {{ output_file }}
print       "Relaxation complete -> {{ output_file }}"
"""

BATCH_BLOCK_TEMPLATE = r"""
# ============================================================
# Job: {{ job_name }}
# System: {{ system_id }}
# Event: vacancy {{ site_a }} -> {{ site_b }}
# Model: {{ model_tag }}
# ============================================================
units           metal
atom_style      atomic
atom_modify     map yes
newton          on
boundary        p p p

neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes

read_data {{ structure_file }}

pair_style      {{ pair_style }}
pair_coeff      * * {{ model_file }} {% for elem in elements %}{{ elem }} {% endfor %}

{{ mass_lines }}

variable sysname string {{ job_name }}

shell mkdir -p dumps/${sysname}
shell mkdir -p results

dump        d1 all custom 1 dumps/${sysname}/traj.lammpstrj id type x y z fx fy fz
dump_modify d1 first yes sort id

thermo      1
thermo_style custom step pe fmax fnorm

min_style   {{ min_style }}
min_modify  dmax 0.03 line quadratic
minimize    1.0e-8 {{ ftol }} 10000 100000

write_data  {{ output_file }}
print       "Relaxation complete -> {{ output_file }}"

clear
"""


@dataclass
class JobInfo:
    system_id: str
    model_tag: str
    package_mode: str
    batch_name: str
    source_relaxed_init: str
    site_a: int
    site_b: int
    moving_atom_index_0based: int
    moving_atom_element: str
    site_a_coord_target_A: List[float]
    site_b_coord_source_A: List[float]
    output_dir: str
    init_data: str
    final_guess_data: str
    final_relaxed_data: str
    model_file_input: str
    model_file_written_path: str
    note: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--package-mode", required=True,
                   choices=["single_job", "batch_shared_model"])
    p.add_argument("--batch-name", default="",
                   help="Required in batch_shared_model mode, e.g. batch_submit_foundation")
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--site-map", required=True, type=Path)
    p.add_argument("--system-id", required=True)
    p.add_argument("--site-a", required=True, type=int,
                   help="Initial vacancy site index on pristine graph")
    p.add_argument("--site-b", required=True, type=int,
                   help="Final vacancy site index on pristine graph; atom at site-b moves to site-a")
    p.add_argument("--model-tag", required=True, choices=["foundation", "finetuned"])
    p.add_argument("--model-file", required=True, type=Path)
    p.add_argument("--out-root", default=Path("neb_runs"), type=Path)
    p.add_argument("--elements", nargs="+", default=["N", "P"])
    p.add_argument("--ftol", type=float, default=0.01)
    
    p.add_argument("--lmp-bin", default="lmp")
    p.add_argument("--env-script", default="")
    p.add_argument("--venv-activate", default="")
    
    p.add_argument("--copy-source-relaxed", action="store_true")
    p.add_argument("--force-overwrite-bundle", action="store_true",
                   help="In batch mode, overwrite existing inputs/in.relax_bundle.lammps instead of appending.")
    p.add_argument("--kokkos", choices=["on", "off"], default="on")
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--suffix", default="kk")
    p.add_argument("--retries", type=int, default=1)
    p.add_argument("--retry-delay", type=int, default=15)
    return p.parse_args()


def resolve_path(base_file: Path, maybe_rel: str) -> Path:
    p = Path(str(maybe_rel))
    if p.is_absolute():
        return p
    return (base_file.parent / p).resolve()


def model_to_manifest_columns(model_tag: str) -> Tuple[str, str, str]:
    if model_tag == "foundation":
        return "base_out_lmp", "base_dump", "base"
    if model_tag == "finetuned":
        return "ft_out_lmp", "ft_dump", "ft"
    raise ValueError(f"Unknown model_tag={model_tag}")


def load_manifest_row(manifest_csv: Path, system_id: str) -> pd.Series:
    df = pd.read_csv(manifest_csv)
    hit = df[df["system_id"] == system_id]
    if len(hit) == 0:
        raise ValueError(f"system_id not found in manifest: {system_id}")
    if len(hit) > 1:
        raise ValueError(f"system_id appears multiple times in manifest: {system_id}")
    return hit.iloc[0]


def load_site_rows(site_csv: Path, system_id: str, structure_key: str) -> pd.DataFrame:
    df = pd.read_csv(site_csv)
    sub = df[(df["system_id"] == system_id) & (df["structure"] == structure_key)].copy()
    if sub.empty:
        raise ValueError(
            f"No site-mapping rows found for system_id={system_id}, structure={structure_key}"
        )
    return sub


def build_symbols_from_site_map(site_rows: pd.DataFrame, natoms: int) -> List[str]:
    occ = site_rows[site_rows["is_occupied"] == True].copy()
    occ = occ.sort_values("mapped_atom_index_0based")
    idx = occ["mapped_atom_index_0based"].astype(int).to_numpy()
    elems = occ["element"].astype(str).to_numpy()

    if len(idx) != natoms:
        raise ValueError(
            f"Occupied site rows = {len(idx)} but Atoms natoms = {natoms}. "
            "Site map and structure do not match."
        )

    expected = np.arange(natoms)
    if not np.array_equal(idx, expected):
        raise ValueError(
            "mapped_atom_index_0based is not exactly 0..natoms-1 after sorting."
        )

    return elems.tolist()


def read_relaxed_init_atoms(relaxed_data_path: Path) -> Atoms:
    atoms = read(str(relaxed_data_path), format="lammps-data", atom_style="atomic")
    atoms.set_pbc([True, True, True])
    return atoms


def render_mass_lines(elements: List[str]) -> str:
    lines = []
    for i, elem in enumerate(elements, start=1):
        if elem not in ELEMENT_MASSES:
            raise ValueError(
                f"No mass registered for element '{elem}'. "
                "Please add it to ELEMENT_MASSES."
            )
        lines.append(f"mass            {i} {ELEMENT_MASSES[elem]}")
    return "\n".join(lines)

def get_site_row(site_rows: pd.DataFrame, site_index: int) -> pd.Series:
    hit = site_rows[site_rows["site_index"] == site_index]
    if len(hit) == 0:
        raise ValueError(f"site_index={site_index} not found in site map")
    if len(hit) > 1:
        raise ValueError(f"site_index={site_index} appears multiple times in site map")
    return hit.iloc[0]


def coord_from_row(row: pd.Series, prefix: str = "final") -> np.ndarray:
    if prefix == "final":
        return np.array([row["x_final_A"], row["y_final_A"], row["z_final_A"]], dtype=float)
    if prefix == "pristine":
        return np.array([row["x_pristine_A"], row["y_pristine_A"], row["z_pristine_A"]], dtype=float)
    raise ValueError(prefix)


def is_finite_coord(x: np.ndarray) -> bool:
    return np.isfinite(x).all()


def get_site_geometry_coord(row: pd.Series) -> np.ndarray:
    """
    Return the geometric coordinate of a lattice site.

    For vacancy-hop target construction, this should represent the destination
    lattice site itself, not the 'current atom final position'. Therefore we
    intentionally use pristine coordinates as the primary source of truth.

    Fallback to final only if pristine is unexpectedly unavailable.
    """
    pristine = coord_from_row(row, prefix="pristine")
    if is_finite_coord(pristine):
        return pristine

    final = coord_from_row(row, prefix="final")
    if is_finite_coord(final):
        return final

    site_index = row.get("site_index", "UNKNOWN")
    raise ValueError(
        f"Cannot resolve finite site geometry coordinate for site_index={site_index}. "
        f"Both pristine and final coordinates are invalid. "
        f"pristine={pristine}, final={final}"
    )


def get_atom_current_coord(row: pd.Series) -> np.ndarray:
    """
    Return the current coordinate of an occupied atom in the chosen structure.

    For the moving atom at site_b, current structure coordinate should normally
    come from final coordinates in the site-map row. Fallback to pristine only
    if needed.
    """
    final = coord_from_row(row, prefix="final")
    if is_finite_coord(final):
        return final

    pristine = coord_from_row(row, prefix="pristine")
    if is_finite_coord(pristine):
        return pristine

    site_index = row.get("site_index", "UNKNOWN")
    raise ValueError(
        f"Cannot resolve finite current atom coordinate for site_index={site_index}. "
        f"Both final and pristine coordinates are invalid. "
        f"final={final}, pristine={pristine}"
    )


def validate_all_positions_finite(atoms: Atoms, label: str) -> None:
    pos = atoms.get_positions()
    bad_mask = ~np.isfinite(pos).all(axis=1)
    if bad_mask.any():
        bad_indices = np.where(bad_mask)[0].tolist()
        bad_positions = pos[bad_mask].tolist()
        raise ValueError(
            f"Non-finite coordinates found in {label}. "
            f"Bad atom indices (0-based): {bad_indices}. "
            f"Bad positions: {bad_positions}"
        )


def wrap_target_via_minimum_image(
    source_coord: np.ndarray,
    target_coord: np.ndarray,
    anchor_coord: np.ndarray,
    atoms: Atoms,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the periodic image of target_coord that is closest to the source/anchor.

    We compute the minimum-image displacement from the source site to the target
    site, then apply that displacement to the actual moving atom coordinate in
    the current structure. This keeps cross-PBC hops geometrically local.
    """
    displacement = target_coord - source_coord
    mic_disp, _ = find_mic(displacement, cell=atoms.cell, pbc=atoms.pbc)
    wrapped_target = anchor_coord + np.asarray(mic_disp, dtype=float)
    return wrapped_target, np.asarray(mic_disp, dtype=float)


def prepare_final_guess(init_atoms: Atoms, site_rows: pd.DataFrame, site_a: int, site_b: int) -> Tuple[Atoms, Dict]:
    """
    Construct a final-state guess for vacancy hop:
      atom at occupied site_b moves to vacancy site_a

    Robust rule:
    - site_a target coordinate: use site geometry coordinate (prefer pristine)
    - site_b source coordinate: use current atom coordinate (prefer final)
    - reject any non-finite coordinates before writing output
    """
    row_a = get_site_row(site_rows, site_a)
    row_b = get_site_row(site_rows, site_b)

    if bool(row_a["is_vacancy"]) is not True:
        raise ValueError(f"site_a={site_a} is not marked as a vacancy in the chosen structure.")
    if bool(row_b["is_occupied"]) is not True:
        raise ValueError(f"site_b={site_b} is not occupied in the chosen structure.")

    if pd.isna(row_b["mapped_atom_index_0based"]):
        raise ValueError(f"site_b={site_b} has NaN mapped_atom_index_0based although it is occupied.")

    moving_atom_idx = int(row_b["mapped_atom_index_0based"])
    moving_elem = str(row_b["element"])

    source_coord = get_atom_current_coord(row_b)
    target_coord = get_site_geometry_coord(row_a)

    if not is_finite_coord(source_coord):
        raise ValueError(f"Resolved source coordinate is not finite for site_b={site_b}: {source_coord}")
    if not is_finite_coord(target_coord):
        raise ValueError(f"Resolved target coordinate is not finite for site_a={site_a}: {target_coord}")

    final_guess = init_atoms.copy()
    validate_all_positions_finite(final_guess, label="init_atoms_before_move")

    pos = final_guess.get_positions().copy()

    if moving_atom_idx < 0 or moving_atom_idx >= len(pos):
        raise IndexError(
            f"moving_atom_index_0based={moving_atom_idx} out of range for natoms={len(pos)}"
        )

    original_coord = pos[moving_atom_idx].copy()
    wrapped_target_coord, mic_disp = wrap_target_via_minimum_image(
        source_coord=source_coord,
        target_coord=target_coord,
        anchor_coord=original_coord,
        atoms=final_guess,
    )
    pos[moving_atom_idx] = wrapped_target_coord
    final_guess.set_positions(pos)

    validate_all_positions_finite(final_guess, label="final_guess_after_move")

    meta = {
        "site_a_row": row_a.to_dict(),
        "site_b_row": row_b.to_dict(),
        "moving_atom_index_0based": moving_atom_idx,
        "moving_atom_element": moving_elem,
        "source_coord_current_A": source_coord.tolist(),
        "target_coord_site_geometry_A": target_coord.tolist(),
        "target_coord_wrapped_for_guess_A": wrapped_target_coord.tolist(),
        "source_to_target_mic_displacement_A": mic_disp.tolist(),
        "moving_atom_original_coord_from_atoms_A": original_coord.tolist(),
        "coord_rule": {
            "site_a_target": "prefer pristine site geometry, fallback final",
            "site_b_source": "prefer final occupied-atom coordinate, fallback pristine",
            "pbc_wrapping": "apply minimum-image source->target displacement, anchored at the actual moving-atom coordinate",
        },
    }
    return final_guess, meta


def write_lammps_data_with_specorder(atoms: Atoms, outpath: Path, specorder: List[str]) -> None:
    validate_all_positions_finite(atoms, label=f"atoms_before_write:{outpath}")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    write(
        str(outpath),
        atoms,
        format="lammps-data",
        atom_style="atomic",
        specorder=specorder,
    )


def sanitize_name(text: str) -> str:
    return text.replace("/", "_").replace(" ", "_").replace(":", "_")


def build_job_name(system_id: str, site_a: int, site_b: int, model_tag: str) -> str:
    return sanitize_name(f"job_{system_id}_event_{site_a}_{site_b}_{model_tag}")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def pair_style_for_kokkos(kokkos: str) -> str:
    if kokkos == "on":
        return "mace/kk no_domain_decomposition"
    return "mace no_domain_decomposition"


def min_style_for_kokkos(kokkos: str) -> str:
    if kokkos == "on":
        return "cg/kk"
    return "cg"


def render_single_input(model_relpath: str, elements: List[str], ftol: float, kokkos: str) -> str:
    tmpl = Template(SINGLE_TEMPLATE)
    return tmpl.render(
        structure_file="data/final_guess.data",
        model_file=model_relpath,
        pair_style=pair_style_for_kokkos(kokkos),
        min_style=min_style_for_kokkos(kokkos),
        elements=elements,
        mass_lines=render_mass_lines(elements),
        ftol=ftol,
        output_file="results/final_relaxed.data",
        dump_name="relax_final",
    )


def render_batch_block(job_name: str, system_id: str, site_a: int, site_b: int,
                       model_tag: str, model_relpath: str, elements: List[str], ftol: float,
                       kokkos: str) -> str:
    tmpl = Template(BATCH_BLOCK_TEMPLATE)
    return tmpl.render(
        job_name=job_name,
        system_id=system_id,
        site_a=site_a,
        site_b=site_b,
        model_tag=model_tag,
        structure_file=f"jobs/{job_name}/final_guess.data",
        model_file=model_relpath,
        pair_style=pair_style_for_kokkos(kokkos),
        min_style=min_style_for_kokkos(kokkos),
        elements=elements,
        mass_lines=render_mass_lines(elements),
        ftol=ftol,
        output_file=f"results/{job_name}__final_relaxed.data",
    )


def append_unique_block(bundle_file: Path, marker: str, block: str, overwrite: bool = False) -> None:
    if overwrite or not bundle_file.exists():
        bundle_file.write_text("", encoding="utf-8")

    existing = bundle_file.read_text(encoding="utf-8") if bundle_file.exists() else ""
    if marker in existing:
        return

    prefix = "" if existing.endswith("\n") or existing == "" else "\n"
    bundle_file.write_text(existing + prefix + block + "\n", encoding="utf-8")


def write_root_run_script(
    run_path: Path,
    lmp_bin: str,
    input_relpath: str,
    log_name: str,
    kokkos: str = "on",
    gpus: int = 1,
    suffix: str = "kk",
    retries: int = 1,
    retry_delay: int = 15,
    env_script: str = "",
    venv_activate: str = "",
) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Optionally load a user-provided runtime environment.",
        f'if [ -n "{env_script}" ] && [ -f "{env_script}" ]; then',
        f'  source "{env_script}"',
        "fi",
        "",
        f'if [ -n "{venv_activate}" ] && [ -f "{venv_activate}" ]; then',
        f'  source "{venv_activate}"',
        "fi",
        "",
        f'LMP_BIN="{lmp_bin}"',
        f'INPUT="{input_relpath}"',
        f'LOGFILE="{log_name}"',
        f'RETRIES={retries}',
        f'RETRY_DELAY={retry_delay}',
        'RESULT_FILE="results/final_relaxed.data"',
        "",
        'echo "Using LAMMPS binary: $LMP_BIN"',
        'if [ ! -x "$LMP_BIN" ]; then',
        '  echo "ERROR: LAMMPS binary not found or not executable: $LMP_BIN"',
        "  exit 127",
        "fi",
        "",
        'if [ -s "$RESULT_FILE" ]; then',
        '  echo "Existing result detected at $RESULT_FILE; skipping rerun."',
        "  exit 0",
        "fi",
        "",
        '"$LMP_BIN" -h | head -n 5 || true',
        "",
        "attempt=1",
        'while [ "$attempt" -le "$RETRIES" ]; do',
        '  echo "▶ attempt ${attempt}/${RETRIES}"',
        "",
        '  if "$LMP_BIN" \\',
        '      -log "$LOGFILE" \\',
    ]

    if kokkos == "on":
        lines.extend([
            f'      -k on g {gpus} \\',
            f'      -sf {suffix} \\',
            '      -pk kokkos newton on neigh half neigh/thread off \\',
        ])

    lines.extend([
        '      -in "$INPUT" ; then',
        '    echo "LAMMPS run completed successfully."',
        "    exit 0",
        "  fi",
        "",
        '  echo "LAMMPS run failed. Sleeping ${RETRY_DELAY}s before retry..."',
        '  sleep "$RETRY_DELAY"',
        '  attempt=$((attempt + 1))',
        "done",
        "",
        'echo "LAMMPS run failed after ${RETRIES} attempts."',
        "exit 1",
        "",
    ])

    run_path.write_text("\n".join(lines), encoding="utf-8")
    run_path.chmod(0o755)


def main() -> None:
    args = parse_args()

    manifest_row = load_manifest_row(args.manifest, args.system_id)
    relaxed_col, dump_col, structure_key = model_to_manifest_columns(args.model_tag)

    relaxed_init_raw = manifest_row.get(relaxed_col, None)
    if pd.isna(relaxed_init_raw) or str(relaxed_init_raw).strip() == "":
        raise ValueError(
            f"Manifest column {relaxed_col} is empty for system_id={args.system_id}."
        )

    relaxed_init_path = resolve_path(args.manifest, str(relaxed_init_raw))
    if not relaxed_init_path.exists():
        raise FileNotFoundError(f"Resolved relaxed initial data path does not exist:\n{relaxed_init_path}")

    site_rows = load_site_rows(args.site_map, args.system_id, structure_key)
    init_atoms = read_relaxed_init_atoms(relaxed_init_path)

    symbols = build_symbols_from_site_map(site_rows, len(init_atoms))
    init_atoms.set_chemical_symbols(symbols)
    validate_all_positions_finite(init_atoms, label="init_atoms_after_read")

    final_guess_atoms, move_meta = prepare_final_guess(
        init_atoms=init_atoms,
        site_rows=site_rows,
        site_a=args.site_a,
        site_b=args.site_b,
    )

    job_name = build_job_name(args.system_id, args.site_a, args.site_b, args.model_tag)
    model_src = args.model_file.resolve()
    if not model_src.exists():
        raise FileNotFoundError(f"Model file not found: {model_src}")

    if args.package_mode == "single_job":
        submit_root = ensure_dir((args.out_root / job_name).resolve())
        model_dir = ensure_dir(submit_root / "model")
        data_dir = ensure_dir(submit_root / "data")
        ensure_dir(submit_root / "results")
        ensure_dir(submit_root / "dumps")
        meta_dir = ensure_dir(submit_root / "meta")

        model_dst = model_dir / model_src.name
        shutil.copy2(model_src, model_dst)
        model_written_path = f"model/{model_dst.name}"

        init_data = data_dir / "init.data"
        final_guess_data = data_dir / "final_guess.data"
        write_lammps_data_with_specorder(init_atoms, init_data, specorder=args.elements)
        write_lammps_data_with_specorder(final_guess_atoms, final_guess_data, specorder=args.elements)

        in_file = submit_root / "in.relax_final.lammps"
        in_file.write_text(
            render_single_input(model_written_path, args.elements, args.ftol, args.kokkos),
            encoding="utf-8",
        )

        run_file = submit_root / "run.sh"
        write_root_run_script(
            run_file,
            args.lmp_bin,
            "in.relax_final.lammps",
            "log.relax_final",
            kokkos=args.kokkos,
            gpus=args.gpus,
            suffix=args.suffix,
            retries=args.retries,
            retry_delay=args.retry_delay,
        )

        if args.copy_source_relaxed:
            shutil.copy2(relaxed_init_path, meta_dir / f"source_{relaxed_init_path.name}")

        info = JobInfo(
            system_id=args.system_id,
            model_tag=args.model_tag,
            package_mode=args.package_mode,
            batch_name="",
            source_relaxed_init=str(relaxed_init_path),
            site_a=args.site_a,
            site_b=args.site_b,
            moving_atom_index_0based=move_meta["moving_atom_index_0based"],
            moving_atom_element=move_meta["moving_atom_element"],
            site_a_coord_target_A=move_meta["target_coord_site_geometry_A"],
            site_b_coord_source_A=move_meta["source_coord_current_A"],
            output_dir=str(submit_root),
            init_data="data/init.data",
            final_guess_data="data/final_guess.data",
            final_relaxed_data="results/final_relaxed.data",
            model_file_input=str(model_src),
            model_file_written_path=model_written_path,
            note="Single self-contained submission root.",
        )

        (meta_dir / "meta.json").write_text(json.dumps({
            "job_info": asdict(info),
            "move_meta": move_meta,
            "manifest_row": {k: (None if pd.isna(v) else v) for k, v in manifest_row.to_dict().items()},
            "elements_specorder": args.elements,
        }, indent=2), encoding="utf-8")

        print(f"[OK] Prepared single-job submission root:\n{submit_root}")
        print("[Next]")
        print(f"  cd {submit_root}")
        print("  bash run.sh")
        return

    if args.package_mode == "batch_shared_model":
        if not args.batch_name.strip():
            raise ValueError("--batch-name is required for batch_shared_model")

        submit_root = ensure_dir((args.out_root / args.batch_name).resolve())
        models_dir = ensure_dir(submit_root / "models")
        jobs_dir = ensure_dir(submit_root / "jobs")
        inputs_dir = ensure_dir(submit_root / "inputs")
        ensure_dir(submit_root / "results")
        ensure_dir(submit_root / "dumps")
        ensure_dir(submit_root / "logs")

        model_dst = models_dir / model_src.name
        if not model_dst.exists():
            shutil.copy2(model_src, model_dst)
        model_written_path = f"models/{model_dst.name}"

        job_dir = ensure_dir(jobs_dir / job_name)
        init_data = job_dir / "init.data"
        final_guess_data = job_dir / "final_guess.data"
        meta_json = job_dir / "meta.json"

        write_lammps_data_with_specorder(init_atoms, init_data, specorder=args.elements)
        write_lammps_data_with_specorder(final_guess_atoms, final_guess_data, specorder=args.elements)

        if args.copy_source_relaxed:
            shutil.copy2(relaxed_init_path, job_dir / f"source_{relaxed_init_path.name}")

        bundle_file = inputs_dir / "in.relax_bundle.lammps"
        block_marker = f"# Job: {job_name}"
        block = render_batch_block(
            job_name=job_name,
            system_id=args.system_id,
            site_a=args.site_a,
            site_b=args.site_b,
            model_tag=args.model_tag,
            model_relpath=model_written_path,
            elements=args.elements,
            ftol=args.ftol,
            kokkos=args.kokkos,
        )
        append_unique_block(bundle_file, block_marker, block, overwrite=args.force_overwrite_bundle)

        run_file = submit_root / "run.sh"
        write_root_run_script(
            run_file,
            args.lmp_bin,
            "inputs/in.relax_bundle.lammps",
            "logs/log.relax_bundle",
            kokkos=args.kokkos,
            gpus=args.gpus,
            suffix=args.suffix,
            retries=args.retries,
            retry_delay=args.retry_delay,
        )

        info = JobInfo(
            system_id=args.system_id,
            model_tag=args.model_tag,
            package_mode=args.package_mode,
            batch_name=args.batch_name,
            source_relaxed_init=str(relaxed_init_path),
            site_a=args.site_a,
            site_b=args.site_b,
            moving_atom_index_0based=move_meta["moving_atom_index_0based"],
            moving_atom_element=move_meta["moving_atom_element"],
            site_a_coord_target_A=move_meta["target_coord_site_geometry_A"],
            site_b_coord_source_A=move_meta["source_coord_current_A"],
            output_dir=str(job_dir),
            init_data=f"jobs/{job_name}/init.data",
            final_guess_data=f"jobs/{job_name}/final_guess.data",
            final_relaxed_data=f"results/{job_name}__final_relaxed.data",
            model_file_input=str(model_src),
            model_file_written_path=model_written_path,
            note="Batch submission root; all paths in bundle file are relative to the batch root.",
        )

        meta_json.write_text(json.dumps({
            "job_info": asdict(info),
            "move_meta": move_meta,
            "manifest_row": {k: (None if pd.isna(v) else v) for k, v in manifest_row.to_dict().items()},
            "elements_specorder": args.elements,
        }, indent=2), encoding="utf-8")

        print(f"[OK] Prepared/updated batch submission root:\n{submit_root}")
        print(f"[Added job] jobs/{job_name}")
        print(f"[Bundle input] inputs/in.relax_bundle.lammps")
        print("[Next]")
        print(f"  cd {submit_root}")
        print("  bash run.sh")
        return

    raise ValueError(f"Unsupported package mode: {args.package_mode}")


if __name__ == "__main__":
    main()
