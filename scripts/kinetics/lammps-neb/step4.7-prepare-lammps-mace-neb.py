#!/usr/bin/env python3
"""
Prepare a LAMMPS+MACE NEB calculation package from one validated Stage 4 job.

Current scope
-------------
- consume one Stage 4 single-job directory as upstream input
- reuse:
  - data/init.data
  - results/final_relaxed.data
  - meta/meta.json
- emit one new server-submission folder for a NEB / CI-NEB calculation
- support:
  - --image-mode lammps_interpolate
  - --image-mode explicit_images
- write machine-readable preparation status for later Stage 4.8 batch orchestration

KOKKOS toggle policy
--------------------
In this project, `--kokkos` is intended to switch:
- `pair_style mace/kk` vs `pair_style mace`
- and the KOKKOS-related launch flags in `run.sh`

It does not remove `no_domain_decomposition`.
For the expected near-term usage here:
- one LAMMPS job per GPU,
- relatively small systems,
- and throughput from many independent tasks,
we keep `no_domain_decomposition` in both modes.

Dependencies
------------
    pip install ase numpy
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from ase import Atoms
from ase.io import read, write

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

DEFAULT_JOB_JSON = {
    "job_type": "container",
    "command": "chmod +x ./run.sh && ./run.sh",
    "backward_files": [
        "data",
        "dumps",
        "meta",
        "log.neb",
        "run.sh",
        "job.json",
        "submit.log",
    ],
    "project_id": 0,
    "platform": "<CLOUD_PLATFORM>",
    "machine_type": "<EXAMPLE_MACHINE_TYPE>",
    "image_address": "<LAMMPS_MACE_IMAGE_ADDRESS>",
}


@dataclass
class NebJobInfo:
    source_job_dir: str
    source_stage: str
    system_id: str
    model_tag: str
    package_mode: str
    batch_name: str
    image_mode: str
    neb_job_name: str
    site_a: int
    site_b: int
    moving_atom_index_0based: int
    moving_atom_element: str
    init_data: str
    final_relaxed_data: str
    model_file_input: str
    model_file_written_path: str
    output_dir: str
    nreplicas_total: int
    nimages_intermediate: int
    neb_type: str
    spring_constant: float
    parallel_mode: str
    perp_spring: float
    etol: float
    ftol: float
    n1_steps: int
    n2_steps: int
    print_every: int
    min_style: str
    timestep: float
    kokkos: str
    gpus: int
    allow_mpirun_as_root: bool
    total_ranks: int
    ranks_per_replica: int
    note: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--package-mode",
        required=True,
        choices=["single_job", "batch_shared_model"],
    )
    p.add_argument(
        "--source-job-dir",
        required=True,
        type=Path,
        help="Existing Stage 4 single-job directory with validated endpoint relaxation.",
    )
    p.add_argument(
        "--batch-name",
        default="",
        help="Required in batch_shared_model mode.",
    )
    p.add_argument(
        "--out-root",
        default=Path("lammps_neb_workspace/neb_runs"),
        type=Path,
    )
    p.add_argument(
        "--image-mode",
        required=True,
        choices=["lammps_interpolate", "explicit_images"],
    )
    p.add_argument(
        "--images",
        type=int,
        default=7,
        help="Total number of replicas including both endpoints.",
    )
    p.add_argument("--spring-constant", type=float, default=1.0)
    p.add_argument("--parallel-mode", choices=["neigh"], default="neigh")
    p.add_argument("--perp-spring", type=float, default=0.0)
    p.add_argument("--neb-type", choices=["ci-neb", "plain-neb"], default="ci-neb")
    p.add_argument("--etol", type=float, default=0.0)
    p.add_argument("--ftol", type=float, default=0.02)
    p.add_argument("--n1-steps", type=int, default=3000)
    p.add_argument("--n2-steps", type=int, default=3000)
    p.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="Shared NEB cadence parameter used for thermo, dump, and the 5th integer argument of the LAMMPS neb command. Current finetuned-safe default is 1.",
    )
    p.add_argument("--min-style", default="fire")
    p.add_argument(
        "--timestep",
        type=float,
        default=0.01,
        help="Explicit NEB timestep used by damped-dynamics minimizers such as quickmin/fire.",
    )
    p.add_argument("--mpirun-bin", default="mpirun")
    p.add_argument("--lmp-bin", default="lmp")
    p.add_argument(
        "--allow-mpirun-as-root",
        action="store_true",
        help="Write OpenMPI root-override environment variables into run.sh. "
             "Use only when the remote container actually launches mpirun as root.",
    )
    p.add_argument("--ranks-per-replica", type=int, default=1)
    p.add_argument(
        "--total-ranks",
        type=int,
        default=None,
        help="Defaults to images * ranks-per-replica.",
    )
    p.add_argument("--kokkos", choices=["on", "off"], default="on")
    p.add_argument(
        "--gpus",
        type=int,
        default=0,
        help="Number of GPUs requested by KOKKOS launch flags. If --kokkos on and this is left as 0, the script promotes it to 1 for the expected single-GPU first-bite workflow.",
    )
    p.add_argument("--suffix", default="kk")
    p.add_argument(
        "--job-name-suffix",
        default="",
        help="Optional suffix appended to the generated NEB job directory and job_name, e.g. 'kokkos-on'.",
    )
    p.add_argument("--env-script", default="")
    p.add_argument("--venv-activate", default="")
    p.add_argument(
        "--allow-unvalidated-final",
        action="store_true",
        help="Prepare a NEB job even if basin-validation files are missing or not distinct.",
    )
    p.add_argument(
        "--force-overwrite-bundle",
        action="store_true",
        help="In batch mode, overwrite inputs/in.neb_bundle.lammps instead of appending.",
    )
    return p.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_name(text: str) -> str:
    return text.replace("/", "_").replace(" ", "_").replace(":", "_")


def build_neb_job_name(
    system_id: str,
    site_a: int,
    site_b: int,
    model_tag: str,
    neb_type: str,
    job_name_suffix: str = "",
) -> str:
    suffix = "cineb" if neb_type == "ci-neb" else "neb"
    job_name = f"job_{system_id}__event_{site_a}_{site_b}__{model_tag}__{suffix}"
    if job_name_suffix.strip():
        job_name = f"{job_name}__{sanitize_name(job_name_suffix.strip())}"
    return sanitize_name(job_name)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_lammps_atomic_data(path: Path) -> Atoms:
    atoms = read(str(path), format="lammps-data", atom_style="atomic")
    atoms.set_pbc([True, True, True])
    return atoms


def apply_elements_specorder(atoms: Atoms, elements: List[str]) -> None:
    atom_types = atoms.arrays.get("type", None)
    if atom_types is None:
        raise ValueError("LAMMPS data read did not expose atom 'type' array")

    mapped_symbols: List[str] = []
    for atom_type in atom_types.astype(int):
        if atom_type < 1 or atom_type > len(elements):
            raise ValueError(
                f"Atom type {atom_type} is out of bounds for elements_specorder={elements}"
            )
        mapped_symbols.append(elements[atom_type - 1])
    atoms.set_chemical_symbols(mapped_symbols)


def validate_finite_positions(atoms: Atoms, label: str) -> None:
    pos = atoms.get_positions()
    if not np.isfinite(pos).all():
        raise ValueError(f"Non-finite coordinates found in {label}")


def build_mass_lines(elements: List[str]) -> str:
    lines = []
    for i, elem in enumerate(elements, start=1):
        if elem not in ELEMENT_MASSES:
            raise ValueError(f"No mass registered for element '{elem}'")
        lines.append(f"mass            {i} {ELEMENT_MASSES[elem]}")
    return "\n".join(lines)


def write_lammps_data_with_specorder(atoms: Atoms, outpath: Path, specorder: List[str]) -> None:
    validate_finite_positions(atoms, str(outpath))
    outpath.parent.mkdir(parents=True, exist_ok=True)
    write(
        str(outpath),
        atoms,
        format="lammps-data",
        atom_style="atomic",
        specorder=specorder,
    )


def allclose_cells(a: Atoms, b: Atoms, tol: float = 1.0e-8) -> bool:
    return (
        np.allclose(np.array(a.cell), np.array(b.cell), atol=tol, rtol=0.0)
        and np.array_equal(np.array(a.pbc, dtype=bool), np.array(b.pbc, dtype=bool))
    )


def image_fractional_dr(cell: np.ndarray, pbc: np.ndarray, dr_cart: np.ndarray) -> np.ndarray:
    frac = np.linalg.solve(cell.T, dr_cart)
    for axis, periodic in enumerate(pbc):
        if periodic:
            frac[axis] -= np.round(frac[axis])
    return frac


def align_final_endpoint_to_init(init_atoms: Atoms, final_atoms: Atoms) -> Tuple[Atoms, np.ndarray]:
    if len(init_atoms) != len(final_atoms):
        raise ValueError(
            f"Atom count mismatch between endpoints: init={len(init_atoms)} final={len(final_atoms)}"
        )
    if not allclose_cells(init_atoms, final_atoms):
        raise ValueError("Cell or PBC mismatch between init and final endpoints")

    init_pos = init_atoms.get_positions()
    final_pos = final_atoms.get_positions()
    cell = np.array(init_atoms.cell)
    pbc = np.array(init_atoms.pbc, dtype=bool)

    aligned = final_atoms.copy()
    aligned_pos = final_pos.copy()
    mic_displacements = np.zeros_like(init_pos)
    for i in range(len(init_atoms)):
        frac = image_fractional_dr(cell, pbc, final_pos[i] - init_pos[i])
        dr_cart = frac @ cell
        aligned_pos[i] = init_pos[i] + dr_cart
        mic_displacements[i] = dr_cart
    aligned.set_positions(aligned_pos)
    validate_finite_positions(aligned, "aligned final endpoint")
    return aligned, mic_displacements


def interpolate_replicas(init_atoms: Atoms, final_atoms: Atoms, nreplicas: int) -> List[Atoms]:
    if nreplicas < 2:
        raise ValueError(f"nreplicas must be >= 2, got {nreplicas}")
    if len(init_atoms) != len(final_atoms):
        raise ValueError(
            f"Atom count mismatch between endpoints: init={len(init_atoms)} final={len(final_atoms)}"
        )
    if not allclose_cells(init_atoms, final_atoms):
        raise ValueError("Cell or PBC mismatch between init and final endpoints")

    init_pos = init_atoms.get_positions()
    final_pos = final_atoms.get_positions()
    cell = np.array(init_atoms.cell)
    pbc = np.array(init_atoms.pbc, dtype=bool)

    dr_frac = np.zeros_like(init_pos)
    for i in range(len(init_atoms)):
        dr_frac[i] = image_fractional_dr(cell, pbc, final_pos[i] - init_pos[i])

    replicas: List[Atoms] = []
    init_frac = init_atoms.get_scaled_positions(wrap=False)
    for replica_index in range(nreplicas):
        t = replica_index / (nreplicas - 1)
        frac = init_frac + t * dr_frac
        replica = init_atoms.copy()
        replica.set_scaled_positions(frac)
        replica.wrap()
        validate_finite_positions(replica, f"replica_{replica_index}")
        replicas.append(replica)
    return replicas


def load_source_context(source_job_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    meta_path = source_job_dir / "meta" / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing source meta file: {meta_path}")

    meta = read_json(meta_path)
    job_info = meta.get("job_info", {})
    elements = meta.get("elements_specorder", ["N", "P"])
    if not isinstance(elements, list) or len(elements) == 0:
        raise ValueError("elements_specorder missing or invalid in source meta.json")
    return meta, job_info, elements


def resolve_basin_status(source_job_dir: Path) -> Dict[str, Any]:
    check_path = source_job_dir / "meta" / "basin_check.json"
    trace_path = source_job_dir / "meta" / "basin_trace_summary.json"

    basin_check = read_json(check_path) if check_path.exists() else None
    basin_trace = read_json(trace_path) if trace_path.exists() else None

    check_label = None
    if basin_check is not None:
        check_label = basin_check.get("result", {}).get("classification")

    trace_label = None
    if basin_trace is not None:
        trace_label = basin_trace.get("classification")

    preferred = trace_label or check_label
    return {
        "basin_check_path": str(check_path) if check_path.exists() else None,
        "basin_trace_summary_path": str(trace_path) if trace_path.exists() else None,
        "basin_check_classification": check_label,
        "basin_trace_classification": trace_label,
        "preferred_classification": preferred,
        "ready_for_neb": preferred == "different_basin",
    }


def build_prepare_status(
    source_job_dir: Path,
    job_info: Dict[str, Any],
    args: argparse.Namespace,
    effective_gpus: int,
    basin_status: Dict[str, Any],
    prepared: bool,
    output_dir: Optional[Path],
    reason: str,
) -> Dict[str, Any]:
    return {
        "source_job_dir": str(source_job_dir),
        "source_stage": "stage4_endpoint_relaxation",
        "system_id": job_info.get("system_id"),
        "model_tag": job_info.get("model_tag"),
        "site_a": job_info.get("site_a"),
        "site_b": job_info.get("site_b"),
        "image_mode": args.image_mode,
        "package_mode": args.package_mode,
        "neb_type": args.neb_type,
        "nreplicas_total": args.images,
        "nimages_intermediate": args.images - 2,
        "kokkos": args.kokkos,
        "gpus": effective_gpus,
        "allow_mpirun_as_root": bool(args.allow_mpirun_as_root),
        "prepared": prepared,
        "output_dir": str(output_dir) if output_dir is not None else None,
        "reason": reason,
        "allow_unvalidated_final": bool(args.allow_unvalidated_final),
        "readiness": basin_status,
    }


def write_status_to_source(source_job_dir: Path, status: Dict[str, Any]) -> None:
    write_json(source_job_dir / "meta" / "neb_prepare_status.json", status)


def resolve_pair_style_line(kokkos: str) -> str:
    if kokkos == "on":
        return "pair_style      mace/kk no_domain_decomposition"
    return "pair_style      mace no_domain_decomposition"


def render_replica_load_lines(image_mode: str, nreplicas: int, replica_prefix_relpath: str, fallback_read_target: str) -> List[str]:
    if image_mode != "explicit_images":
        return [f"read_data {fallback_read_target}"]

    lines = [
        "variable replica index 0",
    ]
    for replica_index in range(nreplicas):
        lines.append(f'if "${{replica}} == {replica_index}" then "read_data {replica_prefix_relpath}{replica_index:02d}.data"')
    return lines


def render_single_input(
    image_mode: str,
    nreplicas: int,
    final_data_relpath: str,
    replica_prefix_relpath: str,
    model_relpath: str,
    kokkos: str,
    elements: List[str],
    spring_constant: float,
    parallel_mode: str,
    perp_spring: float,
    etol: float,
    ftol: float,
    n1_steps: int,
    n2_steps: int,
    print_every: int,
    min_style: str,
    timestep: float,
) -> str:
    world_values = " ".join(str(i) for i in range(nreplicas))
    if image_mode == "explicit_images":
        neb_file_style = "none"
    else:
        neb_file_style = f"final {final_data_relpath}"
    read_lines = render_replica_load_lines(
        image_mode=image_mode,
        nreplicas=nreplicas,
        replica_prefix_relpath=replica_prefix_relpath,
        fallback_read_target="data/init.data",
    )

    return "\n".join(
        [
            "# ======================= CI-NEB Setup =======================",
            "units           metal",
            "atom_style      atomic",
            "atom_modify     map yes",
            "newton          on",
            "boundary        p p p",
            "",
            "neighbor        2.0 bin",
            "neigh_modify    every 1 delay 0 check yes",
            "",
            f"variable replica world {world_values}",
            *read_lines,
            "",
            resolve_pair_style_line(kokkos),
            f"pair_coeff      * * {model_relpath} " + " ".join(elements),
            "",
            build_mass_lines(elements),
            "",
            f"fix             nebfix all neb {spring_constant} parallel {parallel_mode} perp {perp_spring}",
            "",
            f"thermo          {print_every}",
            "thermo_style    custom step pe fmax fnorm",
            "",
            f"dump            d1 all custom {print_every} dumps/replica_${{replica}}.lammpstrj id type x y z fx fy fz",
            "dump_modify     d1 first yes sort id",
            "",
            f"timestep        {timestep}",
            f"min_style       {min_style}",
            f"neb             {etol} {ftol} {n1_steps} {n2_steps} {print_every} {neb_file_style}",
            "",
            f'print           "NEB finished using image-mode={image_mode}"',
            "",
        ]
    )


def render_batch_block(
    job_name: str,
    image_mode: str,
    nreplicas: int,
    final_data_relpath: str,
    replica_prefix_relpath: str,
    model_relpath: str,
    kokkos: str,
    elements: List[str],
    spring_constant: float,
    parallel_mode: str,
    perp_spring: float,
    etol: float,
    ftol: float,
    n1_steps: int,
    n2_steps: int,
    print_every: int,
    min_style: str,
    timestep: float,
) -> str:
    world_values = " ".join(str(i) for i in range(nreplicas))
    if image_mode == "explicit_images":
        neb_file_style = "none"
    else:
        neb_file_style = f"final {final_data_relpath}"
    read_lines = render_replica_load_lines(
        image_mode=image_mode,
        nreplicas=nreplicas,
        replica_prefix_relpath=replica_prefix_relpath,
        fallback_read_target=f"jobs/{job_name}/data/init.data",
    )

    return "\n".join(
        [
            "# ============================================================",
            f"# Job: {job_name}",
            "# ============================================================",
            "clear",
            "units           metal",
            "atom_style      atomic",
            "atom_modify     map yes",
            "newton          on",
            "boundary        p p p",
            "",
            "neighbor        2.0 bin",
            "neigh_modify    every 1 delay 0 check yes",
            "",
            f"variable replica world {world_values}",
            *read_lines,
            "",
            resolve_pair_style_line(kokkos),
            f"pair_coeff      * * {model_relpath} " + " ".join(elements),
            "",
            build_mass_lines(elements),
            "",
            f"fix             nebfix all neb {spring_constant} parallel {parallel_mode} perp {perp_spring}",
            "",
            f"thermo          {print_every}",
            "thermo_style    custom step pe fmax fnorm",
            "",
            f"dump            d1 all custom {print_every} dumps/{job_name}__replica_${{replica}}.lammpstrj id type x y z fx fy fz",
            "dump_modify     d1 first yes sort id",
            "",
            f"timestep        {timestep}",
            f"min_style       {min_style}",
            f"neb             {etol} {ftol} {n1_steps} {n2_steps} {print_every} {neb_file_style}",
            "",
            f'print           "NEB finished for {job_name} using image-mode={image_mode}"',
            "",
        ]
    )


def append_unique_block(bundle_file: Path, marker: str, block: str, overwrite: bool = False) -> None:
    if overwrite or not bundle_file.exists():
        bundle_file.write_text("", encoding="utf-8")

    existing = bundle_file.read_text(encoding="utf-8") if bundle_file.exists() else ""
    if marker in existing:
        return

    prefix = "" if existing.endswith("\n") or existing == "" else "\n"
    bundle_file.write_text(existing + prefix + block + "\n", encoding="utf-8")


def write_single_run_script(
    run_path: Path,
    mpirun_bin: str,
    lmp_bin: str,
    input_relpath: str,
    total_ranks: int,
    nreplicas: int,
    ranks_per_replica: int,
    kokkos: str,
    gpus: int,
    allow_mpirun_as_root: bool,
    suffix: str,
    env_script: str,
    venv_activate: str,
) -> None:
    command_parts = [
        '"$MPIRUN_BIN"',
        "--oversubscribe",
        '-np "$TOTAL_RANKS"',
        '"$LMP_BIN"',
        '-partition "${REPLICAS}x${RANKS_PER_REPLICA}"',
        "-echo both",
        "-nonbuf",
    ]
    if kokkos == "on":
        command_parts.append("-k on")
        if gpus > 0:
            command_parts.append(f"g {gpus}")
        command_parts.extend(
            [
                f"-sf {suffix}",
                "-pk kokkos newton on neigh half neigh/thread off",
            ]
        )
    command_parts.append('-in "$INPUT"')

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f'if [ -f "{env_script}" ]; then',
        f'  source "{env_script}"',
        "fi",
        "",
        f'if [ -f "{venv_activate}" ]; then',
        f'  source "{venv_activate}"',
        "fi",
        "",
        f'MPIRUN_BIN="{mpirun_bin}"',
        f'LMP_BIN="{lmp_bin}"',
        f'INPUT="{input_relpath}"',
        f"TOTAL_RANKS={total_ranks}",
        f"REPLICAS={nreplicas}",
        f"RANKS_PER_REPLICA={ranks_per_replica}",
        "",
        'echo "Using mpirun: $MPIRUN_BIN"',
        'echo "Using LAMMPS binary: $LMP_BIN"',
        "",
    ]
    if allow_mpirun_as_root:
        lines.extend(
            [
                "# OpenMPI root override for containerized server environments",
                "export OMPI_ALLOW_RUN_AS_ROOT=1",
                "export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1",
                "",
            ]
        )
    lines.extend(
        [
            "# Disable OpenMPI help aggregation so the failing rank prints its own message",
            "export OMPI_MCA_orte_base_help_aggregate=0",
            "",
        ]
    )
    lines.append("CMD=(")
    for part in command_parts:
        lines.append(f"  {part}")
    lines.extend(
        [
            ")",
            "",
            'LMP_HELP_OUTPUT="$("$LMP_BIN" -h 2>&1 || true)"',
            'if [[ "$LMP_HELP_OUTPUT" != *"REPLICA"* ]]; then',
            '  echo "ERROR: The LAMMPS binary does not report the REPLICA package in -h output."',
            '  echo "NEB/CI-NEB requires fix neb from the REPLICA package."',
            '  echo "$LMP_HELP_OUTPUT"',
            '  exit 2',
            "fi",
            "",
            'printf "Command:"',
            'printf " %q" "${CMD[@]}"',
            'printf "\\n"',
            '"${CMD[@]}" 2>&1 | tee log.neb',
            "",
        ]
    )
    run_path.write_text("\n".join(lines), encoding="utf-8")
    run_path.chmod(0o755)


def write_job_json(path: Path, job_name: str) -> None:
    payload = dict(DEFAULT_JOB_JSON)
    payload["job_name"] = job_name
    write_json(path, payload)


def copy_model(source_job_dir: Path, destination: Path, package_mode: str) -> Tuple[Path, str]:
    src_model_dir = source_job_dir / "model"
    model_files = [p for p in src_model_dir.iterdir() if p.is_file()]
    if len(model_files) == 0:
        raise FileNotFoundError(f"No model file found in {src_model_dir}")
    if len(model_files) > 1:
        raise ValueError(f"Expected exactly one model file in {src_model_dir}, found {len(model_files)}")

    model_src = model_files[0]
    if package_mode == "single_job":
        model_dst = destination / "model" / model_src.name
        ensure_dir(model_dst.parent)
        shutil.copy2(model_src, model_dst)
        return model_src, f"model/{model_dst.name}"

    model_dst = destination / "models" / model_src.name
    ensure_dir(model_dst.parent)
    if not model_dst.exists():
        shutil.copy2(model_src, model_dst)
    return model_src, f"models/{model_dst.name}"


def validate_total_ranks(total_ranks: int, nreplicas: int, ranks_per_replica: int) -> None:
    expected = nreplicas * ranks_per_replica
    if total_ranks != expected:
        raise ValueError(
            f"total_ranks must equal images * ranks_per_replica = {expected}, got {total_ranks}"
        )


def resolve_requested_gpus(kokkos: str, gpus: int) -> int:
    if kokkos == "on" and gpus <= 0:
        return 1
    return gpus


def resolve_effective_min_style(kokkos: str, min_style: str) -> str:
    if kokkos == "on" and min_style in {"fire", "quickmin"}:
        return "fire/kk"
    return min_style


def main() -> None:
    args = parse_args()

    if args.images < 2:
        raise ValueError("--images must be at least 2")
    if args.package_mode == "batch_shared_model" and not args.batch_name.strip():
        raise ValueError("--batch-name is required for batch_shared_model")

    source_job_dir = args.source_job_dir.resolve()
    if not source_job_dir.exists():
        raise FileNotFoundError(f"Source job directory does not exist: {source_job_dir}")

    meta, source_job_info, elements = load_source_context(source_job_dir)
    basin_status = resolve_basin_status(source_job_dir)

    total_ranks = args.total_ranks
    if total_ranks is None:
        total_ranks = args.images * args.ranks_per_replica
    validate_total_ranks(total_ranks, args.images, args.ranks_per_replica)
    requested_gpus = resolve_requested_gpus(args.kokkos, args.gpus)
    effective_min_style = resolve_effective_min_style(args.kokkos, args.min_style)

    init_path = source_job_dir / "data" / "init.data"
    final_relaxed_path = source_job_dir / "results" / "final_relaxed.data"
    if not init_path.exists():
        raise FileNotFoundError(f"Missing init endpoint file: {init_path}")
    if not final_relaxed_path.exists():
        raise FileNotFoundError(f"Missing final relaxed endpoint file: {final_relaxed_path}")

    ready_for_neb = basin_status["ready_for_neb"]
    if not ready_for_neb and not args.allow_unvalidated_final:
        reason = (
            "Source job is not NEB-ready because basin validation did not classify the "
            "final endpoint as different_basin. Re-run with --allow-unvalidated-final "
            "to override."
        )
        status = build_prepare_status(
            source_job_dir=source_job_dir,
            job_info=source_job_info,
            args=args,
            effective_gpus=requested_gpus,
            basin_status=basin_status,
            prepared=False,
            output_dir=None,
            reason=reason,
        )
        write_status_to_source(source_job_dir, status)
        print(f"[SKIP] {reason}")
        print(f"[Status] {source_job_dir / 'meta' / 'neb_prepare_status.json'}")
        return

    init_atoms = read_lammps_atomic_data(init_path)
    final_atoms = read_lammps_atomic_data(final_relaxed_path)
    apply_elements_specorder(init_atoms, elements)
    apply_elements_specorder(final_atoms, elements)
    validate_finite_positions(init_atoms, "init endpoint")
    validate_finite_positions(final_atoms, "final relaxed endpoint")

    if len(init_atoms) != len(final_atoms):
        raise ValueError(
            f"Endpoint atom count mismatch: init={len(init_atoms)} final={len(final_atoms)}"
        )

    aligned_final_atoms, mic_displacements = align_final_endpoint_to_init(init_atoms, final_atoms)

    neb_job_name = build_neb_job_name(
        source_job_info["system_id"],
        int(source_job_info["site_a"]),
        int(source_job_info["site_b"]),
        source_job_info["model_tag"],
        args.neb_type,
        args.job_name_suffix,
    )

    if args.package_mode == "single_job":
        submit_root = ensure_dir((args.out_root / neb_job_name).resolve())
        ensure_dir(submit_root / "data")
        ensure_dir(submit_root / "results")
        ensure_dir(submit_root / "dumps")
        meta_dir = ensure_dir(submit_root / "meta")

        model_src, model_relpath = copy_model(source_job_dir, submit_root, args.package_mode)

        shutil.copy2(init_path, submit_root / "data" / "init.data")
        write_lammps_data_with_specorder(
            aligned_final_atoms,
            submit_root / "data" / "final_relaxed.data",
            specorder=elements,
        )

        replica_prefix_relpath = "data/replicas/replica_"
        if args.image_mode == "explicit_images":
            replicas = interpolate_replicas(init_atoms, aligned_final_atoms, args.images)
            for idx, replica in enumerate(replicas):
                outpath = submit_root / "data" / "replicas" / f"replica_{idx:02d}.data"
                write_lammps_data_with_specorder(replica, outpath, specorder=elements)

        in_file = submit_root / "in.neb.lammps"
        in_file.write_text(
            render_single_input(
                image_mode=args.image_mode,
                nreplicas=args.images,
                final_data_relpath="data/final_relaxed.data",
                replica_prefix_relpath=replica_prefix_relpath,
                model_relpath=model_relpath,
                kokkos=args.kokkos,
                elements=elements,
                spring_constant=args.spring_constant,
                parallel_mode=args.parallel_mode,
                perp_spring=args.perp_spring,
                etol=args.etol,
                ftol=args.ftol,
                n1_steps=args.n1_steps,
                n2_steps=args.n2_steps if args.neb_type == "ci-neb" else 0,
                print_every=args.print_every,
                min_style=effective_min_style,
                timestep=args.timestep,
            ),
            encoding="utf-8",
        )

        write_single_run_script(
            run_path=submit_root / "run.sh",
            mpirun_bin=args.mpirun_bin,
            lmp_bin=args.lmp_bin,
            input_relpath="in.neb.lammps",
            total_ranks=total_ranks,
            nreplicas=args.images,
            ranks_per_replica=args.ranks_per_replica,
            kokkos=args.kokkos,
            gpus=requested_gpus,
            allow_mpirun_as_root=args.allow_mpirun_as_root,
            suffix=args.suffix,
            env_script=args.env_script,
            venv_activate=args.venv_activate,
        )
        write_job_json(submit_root / "job.json", neb_job_name)

        neb_info = NebJobInfo(
            source_job_dir=str(source_job_dir),
            source_stage="stage4_endpoint_relaxation",
            system_id=source_job_info["system_id"],
            model_tag=source_job_info["model_tag"],
            package_mode=args.package_mode,
            batch_name="",
            image_mode=args.image_mode,
            neb_job_name=neb_job_name,
            site_a=int(source_job_info["site_a"]),
            site_b=int(source_job_info["site_b"]),
            moving_atom_index_0based=int(source_job_info["moving_atom_index_0based"]),
            moving_atom_element=str(source_job_info["moving_atom_element"]),
            init_data="data/init.data",
            final_relaxed_data="data/final_relaxed.data",
            model_file_input=str(model_src),
            model_file_written_path=model_relpath,
            output_dir=str(submit_root),
            nreplicas_total=args.images,
            nimages_intermediate=args.images - 2,
            neb_type=args.neb_type,
            spring_constant=args.spring_constant,
            parallel_mode=args.parallel_mode,
            perp_spring=args.perp_spring,
            etol=args.etol,
            ftol=args.ftol,
            n1_steps=args.n1_steps,
            n2_steps=args.n2_steps if args.neb_type == "ci-neb" else 0,
            print_every=args.print_every,
            min_style=effective_min_style,
            timestep=args.timestep,
            kokkos=args.kokkos,
            gpus=requested_gpus,
            allow_mpirun_as_root=args.allow_mpirun_as_root,
            total_ranks=total_ranks,
            ranks_per_replica=args.ranks_per_replica,
            note="Single self-contained NEB submission root.",
        )

        status = build_prepare_status(
            source_job_dir=source_job_dir,
            job_info=source_job_info,
            args=args,
            effective_gpus=requested_gpus,
            basin_status=basin_status,
            prepared=True,
            output_dir=submit_root,
            reason="Prepared NEB submission root.",
        )
        write_status_to_source(source_job_dir, status)

        write_json(
            meta_dir / "meta.json",
            {
                "neb_job_info": asdict(neb_info),
                "source_job_info": source_job_info,
                "source_move_meta": meta.get("move_meta", {}),
                "source_manifest_row": meta.get("manifest_row", {}),
                "elements_specorder": elements,
                "pbc_alignment": {
                    "aligned_final_written": "data/final_relaxed.data",
                    "alignment_rule": "per-atom minimum-image displacement relative to init endpoint",
                    "max_abs_mic_displacement_A": float(np.max(np.abs(mic_displacements))),
                },
                "readiness": basin_status,
            },
        )
        write_json(meta_dir / "neb_prepare_status.json", status)

        print(f"[OK] Prepared single-job NEB submission root:\n{submit_root}")
        print("[Next]")
        print(f"  cd {submit_root}")
        print("  bash run.sh")
        return

    submit_root = ensure_dir((args.out_root / args.batch_name).resolve())
    jobs_dir = ensure_dir(submit_root / "jobs")
    inputs_dir = ensure_dir(submit_root / "inputs")
    ensure_dir(submit_root / "results")
    ensure_dir(submit_root / "dumps")
    ensure_dir(submit_root / "logs")

    model_src, model_relpath = copy_model(source_job_dir, submit_root, args.package_mode)

    job_dir = ensure_dir(jobs_dir / neb_job_name)
    ensure_dir(job_dir / "data")
    meta_dir = ensure_dir(job_dir / "meta")
    shutil.copy2(init_path, job_dir / "data" / "init.data")
    write_lammps_data_with_specorder(
        aligned_final_atoms,
        job_dir / "data" / "final_relaxed.data",
        specorder=elements,
    )

    replica_prefix_relpath = f"jobs/{neb_job_name}/data/replicas/replica_"
    if args.image_mode == "explicit_images":
        replicas = interpolate_replicas(init_atoms, aligned_final_atoms, args.images)
        for idx, replica in enumerate(replicas):
            outpath = job_dir / "data" / "replicas" / f"replica_{idx:02d}.data"
            write_lammps_data_with_specorder(replica, outpath, specorder=elements)

    bundle_file = inputs_dir / "in.neb_bundle.lammps"
    block_marker = f"# Job: {neb_job_name}"
    block = render_batch_block(
        job_name=neb_job_name,
        image_mode=args.image_mode,
        nreplicas=args.images,
        final_data_relpath=f"jobs/{neb_job_name}/data/final_relaxed.data",
        replica_prefix_relpath=replica_prefix_relpath,
        model_relpath=model_relpath,
        kokkos=args.kokkos,
        elements=elements,
        spring_constant=args.spring_constant,
        parallel_mode=args.parallel_mode,
        perp_spring=args.perp_spring,
        etol=args.etol,
        ftol=args.ftol,
        n1_steps=args.n1_steps,
        n2_steps=args.n2_steps if args.neb_type == "ci-neb" else 0,
        print_every=args.print_every,
        min_style=effective_min_style,
        timestep=args.timestep,
    )
    append_unique_block(bundle_file, block_marker, block, overwrite=args.force_overwrite_bundle)

    write_single_run_script(
        run_path=submit_root / "run.sh",
        mpirun_bin=args.mpirun_bin,
        lmp_bin=args.lmp_bin,
        input_relpath="inputs/in.neb_bundle.lammps",
        total_ranks=total_ranks,
        nreplicas=args.images,
        ranks_per_replica=args.ranks_per_replica,
        kokkos=args.kokkos,
        gpus=requested_gpus,
        allow_mpirun_as_root=args.allow_mpirun_as_root,
        suffix=args.suffix,
        env_script=args.env_script,
        venv_activate=args.venv_activate,
    )
    write_job_json(submit_root / "job.json", args.batch_name)

    neb_info = NebJobInfo(
        source_job_dir=str(source_job_dir),
        source_stage="stage4_endpoint_relaxation",
        system_id=source_job_info["system_id"],
        model_tag=source_job_info["model_tag"],
        package_mode=args.package_mode,
        batch_name=args.batch_name,
        image_mode=args.image_mode,
        neb_job_name=neb_job_name,
        site_a=int(source_job_info["site_a"]),
        site_b=int(source_job_info["site_b"]),
        moving_atom_index_0based=int(source_job_info["moving_atom_index_0based"]),
        moving_atom_element=str(source_job_info["moving_atom_element"]),
        init_data=f"jobs/{neb_job_name}/data/init.data",
        final_relaxed_data=f"jobs/{neb_job_name}/data/final_relaxed.data",
        model_file_input=str(model_src),
        model_file_written_path=model_relpath,
        output_dir=str(job_dir),
        nreplicas_total=args.images,
        nimages_intermediate=args.images - 2,
        neb_type=args.neb_type,
        spring_constant=args.spring_constant,
        parallel_mode=args.parallel_mode,
        perp_spring=args.perp_spring,
        etol=args.etol,
        ftol=args.ftol,
        n1_steps=args.n1_steps,
        n2_steps=args.n2_steps if args.neb_type == "ci-neb" else 0,
        print_every=args.print_every,
        min_style=effective_min_style,
        timestep=args.timestep,
        kokkos=args.kokkos,
        gpus=requested_gpus,
        allow_mpirun_as_root=args.allow_mpirun_as_root,
        total_ranks=total_ranks,
        ranks_per_replica=args.ranks_per_replica,
        note="Batch submission root; all paths in bundle file are relative to the batch root.",
    )

    status = build_prepare_status(
        source_job_dir=source_job_dir,
        job_info=source_job_info,
        args=args,
        effective_gpus=requested_gpus,
        basin_status=basin_status,
        prepared=True,
        output_dir=job_dir,
        reason="Prepared NEB job under shared batch root.",
    )
    write_status_to_source(source_job_dir, status)

    write_json(
        meta_dir / "meta.json",
        {
            "neb_job_info": asdict(neb_info),
            "source_job_info": source_job_info,
            "source_move_meta": meta.get("move_meta", {}),
            "source_manifest_row": meta.get("manifest_row", {}),
            "elements_specorder": elements,
            "pbc_alignment": {
                "aligned_final_written": f"jobs/{neb_job_name}/data/final_relaxed.data",
                "alignment_rule": "per-atom minimum-image displacement relative to init endpoint",
                "max_abs_mic_displacement_A": float(np.max(np.abs(mic_displacements))),
            },
            "readiness": basin_status,
        },
    )
    write_json(meta_dir / "neb_prepare_status.json", status)

    print(f"[OK] Prepared/updated batch NEB submission root:\n{submit_root}")
    print(f"[Added job] jobs/{neb_job_name}")
    print("[Next]")
    print(f"  cd {submit_root}")
    print("  bash run.sh")


if __name__ == "__main__":
    main()
