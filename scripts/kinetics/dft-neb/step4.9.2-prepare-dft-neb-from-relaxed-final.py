#!/usr/bin/env python3
"""Prepare a formal DFT-NEB package from a first-bite scaffold and a relaxed final endpoint."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from step4_9_dft_neb_common import (
    apply_tag_updates,
    build_project_magmom,
    build_run_script,
    copy_required,
    ensure_dir,
    find_relaxed_endpoint,
    interpolate_images,
    load_job_json_template,
    read_json,
    read_vasp_atoms,
    write_json,
    write_neb_image_dirs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaffold-dir", required=True, type=Path)
    parser.add_argument(
        "--final-relax-dir",
        required=True,
        type=Path,
        help="Directory or file containing the DFT-relaxed final endpoint, typically CONTCAR.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Defaults to <scaffold-dir>__neb_hpc",
    )
    parser.add_argument("--job-name-suffix", default="formal-neb-hpc")
    parser.add_argument("--images-total", type=int, default=None)
    parser.add_argument("--nsw", type=int, default=200)
    parser.add_argument("--ediffg", type=float, default=-0.02)
    parser.add_argument("--climb", choices=["on", "off"], default="on")
    parser.add_argument("--spring", type=int, default=-5)
    parser.add_argument(
        "--neb-mpi-ranks",
        type=int,
        default=None,
        help=(
            "MPI ranks for the VASP NEB run. Defaults to IMAGES so one rank group "
            "is assigned per intermediate image."
        ),
    )
    parser.add_argument("--nbands", type=int, default=None, help="Explicit NBANDS override for the generated INCAR.")
    parser.add_argument("--ncore", type=int, default=None, help="Explicit NCORE override for the generated INCAR.")
    parser.add_argument(
        "--drop-performance-tags",
        action="store_true",
        help="Remove inherited performance-tuning tags such as NBANDS/NCORE unless explicitly re-added.",
    )
    parser.add_argument("--force-overwrite", action="store_true")
    return parser.parse_args()


def resolve_meta_path(scaffold_dir: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    for parent in [scaffold_dir, *scaffold_dir.parents]:
        resolved = (parent / candidate).resolve()
        if resolved.exists():
            return resolved
    repo_root = scaffold_dir.parents[2]
    return (repo_root / candidate).resolve()


def build_neb_vasp_command(mpi_ranks: int) -> str:
    return (
        'bash -lc "source ${VASP_ENV_SCRIPT:?set VASP_ENV_SCRIPT} && '
        'mpirun -n ${VASP_MPI_RANKS:?set VASP_MPI_RANKS} ${VASP_STD_BIN:-vasp_std}"'
    )


def main() -> None:
    args = parse_args()
    scaffold_dir = args.scaffold_dir.resolve()
    meta_path = scaffold_dir / "meta" / "meta.json"
    incar_path = scaffold_dir / "INCAR.neb"
    if not meta_path.exists() or not incar_path.exists():
        raise FileNotFoundError("Scaffold must contain meta/meta.json and INCAR.neb")

    out_dir = (args.out_dir.resolve() if args.out_dir else Path(f"{scaffold_dir}__neb_hpc").resolve())
    if out_dir.exists():
        if not args.force_overwrite:
            raise FileExistsError(f"Output directory exists: {out_dir}")
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)

    meta = read_json(meta_path)
    job_info = meta["job_info"]
    images_total = args.images_total or int(job_info["images_total"])
    if images_total < 3:
        raise ValueError("images_total must include two endpoints and at least one intermediate image")
    intermediate_images = images_total - 2
    neb_mpi_ranks = args.neb_mpi_ranks or intermediate_images
    if neb_mpi_ranks < 1:
        raise ValueError("neb_mpi_ranks must be positive")
    if neb_mpi_ranks % intermediate_images != 0:
        raise ValueError(
            f"neb_mpi_ranks={neb_mpi_ranks} must be divisible by IMAGES={intermediate_images}"
        )

    init_path = resolve_meta_path(scaffold_dir, str(job_info["initial_poscar"]))
    final_path = find_relaxed_endpoint(args.final_relax_dir.resolve())
    init_atoms = read_vasp_atoms(init_path)
    final_atoms = read_vasp_atoms(final_path)
    if len(init_atoms) != len(final_atoms):
        raise ValueError("Initial and relaxed final endpoints have different atom counts")

    images = interpolate_images(init_atoms, final_atoms, images_total)
    image_preflight = write_neb_image_dirs(out_dir, images)

    copy_required(scaffold_dir / "KPOINTS", out_dir / "KPOINTS")
    copy_required(scaffold_dir / "POTCAR", out_dir / "POTCAR")

    job_name = f"{job_info['job_name']}__{args.job_name_suffix}"
    incar_text = incar_path.read_text(encoding="utf-8")
    updates = {
        "SYSTEM": f"{job_name} dft-neb-formal",
        "MAGMOM": build_project_magmom(init_atoms),
        "NSW": str(args.nsw),
        "EDIFFG": str(args.ediffg),
        "IMAGES": str(intermediate_images),
        "SPRING": str(args.spring),
        "LCLIMB": ".TRUE." if args.climb == "on" else ".FALSE.",
    }
    if args.nbands is not None:
        updates["NBANDS"] = str(args.nbands)
    if args.ncore is not None:
        updates["NCORE"] = str(args.ncore)

    remove_keys = ["NBANDS", "NCORE"] if args.drop_performance_tags else []

    incar_text = apply_tag_updates(
        incar_text,
        updates,
        trailing_comments=[
            "# Formal DFT NEB package generated from a DFT-relaxed final endpoint.",
            "# This package supersedes the first-bite init -> final_guess scaffold for production runs.",
        ],
        remove_keys=remove_keys,
    )
    (out_dir / "INCAR").write_text(incar_text, encoding="utf-8")

    base_job = load_job_json_template(scaffold_dir / "job.json")
    base_job["job_name"] = job_name
    base_job["command"] = build_neb_vasp_command(neb_mpi_ranks)
    base_job["backward_files"] = [f"{idx:02d}" for idx in range(images_total)] + [
        "INCAR",
        "KPOINTS",
        "OUTCAR",
        "OSZICAR",
        "STDOUTERR",
        "vasprun.xml",
    ]
    write_json(out_dir / "job.json", base_job)
    (out_dir / "run.sh").write_text(build_run_script(base_job), encoding="utf-8")

    neb_meta = {
        "task_class": "dft_neb_formal_hpc",
        "source_scaffold_dir": str(scaffold_dir),
        "source_relaxed_final_endpoint": str(final_path),
        "job_name": job_name,
        "system_id": job_info["system_id"],
        "site_a": job_info["site_a"],
        "site_b": job_info["site_b"],
        "images_total": images_total,
        "images_intermediate": intermediate_images,
        "neb_mpi_ranks": neb_mpi_ranks,
        "uses_final_endpoint_relaxed": True,
        "final_endpoint_source": str(final_path),
        "drop_performance_tags": args.drop_performance_tags,
        "nbands": args.nbands,
        "ncore": args.ncore,
        "image_interpolation": "mic_unwrapped",
        "image_geometry_preflight": image_preflight,
        "note": "Formal DFT NEB package built from the DFT-relaxed final endpoint rather than the first-bite guessed final endpoint.",
    }
    write_json(out_dir / "prepare_info.json", neb_meta)
    (out_dir / "README.md").write_text(
        "\n".join(
            [
                "# Formal DFT NEB Package",
                "",
                f"Source scaffold: `{scaffold_dir}`",
                f"Relaxed final endpoint: `{final_path}`",
                "",
                "This package is the production-oriented DFT NEB input set.",
                "",
                "Assumptions:",
                "- The final endpoint relaxation has already completed and was sanity-checked.",
                "- The current image directories were interpolated from the relaxed initial endpoint to the relaxed final endpoint.",
                "",
                "Before submission on the target HPC system, review:",
                "- MPI layout and per-image core counts",
                "- NCORE / KPAR / NPAR choices for that machine",
                "- production convergence tags in INCAR",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[DFT NEB FORMAL] created {out_dir}")


if __name__ == "__main__":
    main()
