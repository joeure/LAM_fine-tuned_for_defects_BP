#!/usr/bin/env python3
"""Prepare a reusable final-endpoint relaxation package from a Stage 4.9 scaffold."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from step4_9_dft_neb_common import (
    apply_tag_updates,
    build_run_script,
    copy_required,
    ensure_dir,
    load_job_json_template,
    read_json,
    read_vasp_atoms,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaffold-dir", required=True, type=Path)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Defaults to <scaffold-dir>__final_relax_hpc",
    )
    parser.add_argument("--job-name-suffix", default="hpc-final-relax")
    parser.add_argument("--nsw", type=int, default=200)
    parser.add_argument("--ediffg", type=float, default=-0.02)
    parser.add_argument("--ibrion", type=int, default=2)
    parser.add_argument("--potim", type=float, default=0.03)
    parser.add_argument("--nbands", type=int, default=None, help="Explicit NBANDS override for the generated INCAR.")
    parser.add_argument("--ncore", type=int, default=None, help="Explicit NCORE override for the generated INCAR.")
    parser.add_argument(
        "--drop-performance-tags",
        action="store_true",
        help="Remove inherited performance-tuning tags such as NBANDS/NCORE unless explicitly re-added.",
    )
    parser.add_argument("--force-overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scaffold_dir = args.scaffold_dir.resolve()
    meta_path = scaffold_dir / "meta" / "meta.json"
    relax_src = scaffold_dir / "final_endpoint_relax"
    if not meta_path.exists() or not relax_src.exists():
        raise FileNotFoundError("Scaffold must contain meta/meta.json and final_endpoint_relax/")

    out_dir = (args.out_dir.resolve() if args.out_dir else Path(f"{scaffold_dir}__final_relax_hpc").resolve())
    if out_dir.exists():
        if not args.force_overwrite:
            raise FileExistsError(f"Output directory exists: {out_dir}")
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)

    meta = read_json(meta_path)
    job_info = meta["job_info"]
    relax_src_job = read_json(relax_src / "job.json")
    base_job = load_job_json_template(scaffold_dir / "job.json")
    base_job.update(relax_src_job)
    job_name = f"{job_info['job_name']}__{args.job_name_suffix}"
    base_job["job_name"] = job_name
    base_job["backward_files"] = ["INCAR", "KPOINTS", "OUTCAR", "OSZICAR", "POSCAR", "CONTCAR", "STDOUTERR", "vasprun.xml"]

    atoms = read_vasp_atoms(relax_src / "POSCAR")
    _ = atoms  # forces an early format/read validation

    incar_text = (relax_src / "INCAR").read_text(encoding="utf-8")
    updates = {
        "SYSTEM": f"{job_name} final-endpoint-relax",
        "NSW": str(args.nsw),
        "EDIFFG": str(args.ediffg),
        "IBRION": str(args.ibrion),
        "POTIM": str(args.potim),
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
            "# Prepared for a standalone final-endpoint relaxation prior to formal DFT NEB.",
            "# Use the resulting CONTCAR/POSCAR as the final endpoint input for the NEB packager.",
        ],
        remove_keys=remove_keys,
    )

    copy_required(relax_src / "POSCAR", out_dir / "POSCAR")
    copy_required(relax_src / "KPOINTS", out_dir / "KPOINTS")
    copy_required(relax_src / "POTCAR", out_dir / "POTCAR")
    (out_dir / "INCAR").write_text(incar_text, encoding="utf-8")
    write_json(out_dir / "job.json", base_job)
    (out_dir / "run.sh").write_text(build_run_script(base_job), encoding="utf-8")

    hpc_meta = {
        "task_class": "dft_final_endpoint_relax_hpc",
        "source_scaffold_dir": str(scaffold_dir),
        "source_first_bite_relax_dir": str(relax_src),
        "job_name": job_name,
        "system_id": job_info["system_id"],
        "site_a": job_info["site_a"],
        "site_b": job_info["site_b"],
        "expected_output_endpoint": "CONTCAR",
        "drop_performance_tags": args.drop_performance_tags,
        "nbands": args.nbands,
        "ncore": args.ncore,
        "note": "Run this final-endpoint relaxation first. Do not generate formal NEB images from the guessed final endpoint.",
    }
    write_json(out_dir / "prepare_info.json", hpc_meta)
    (out_dir / "README.md").write_text(
        "\n".join(
            [
                "# DFT Final Endpoint Relax",
                "",
                f"Source scaffold: `{scaffold_dir}`",
                "",
                "This package is the standalone endpoint-relaxation stage for formal DFT NEB.",
                "",
                "Workflow:",
                "1. Run this directory to relax the constructed final endpoint guess.",
                "2. Use the relaxed `CONTCAR` or final `POSCAR` as the NEB final endpoint.",
                "3. Generate the formal NEB package with `step4.9.2-prepare-dft-neb-from-relaxed-final.py`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[DFT FINAL RELAX] created {out_dir}")


if __name__ == "__main__":
    main()
