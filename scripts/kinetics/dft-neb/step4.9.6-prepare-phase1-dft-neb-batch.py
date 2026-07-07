#!/usr/bin/env python3
"""Batch-prepare formal DFT-NEB packages from Phase-1 final-relax results."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase1-summary-csv",
        type=Path,
        default=Path("data_center/step4.9.5_phase1_final_relax_postcheck_summary_formal_v1.csv"),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("dft_neb_workspace/phase1_shortlist_formal_v1/formal_neb_jobs"),
    )
    parser.add_argument(
        "--batch-manifest-csv",
        type=Path,
        default=Path("data_center/step4.9.6_phase1_formal_neb_batch_manifest_formal_v1.csv"),
    )
    parser.add_argument(
        "--job-name-suffix",
        default="phase1-formal-neb-hpc",
    )
    parser.add_argument("--images-total", type=int, default=None)
    parser.add_argument("--nsw", type=int, default=200)
    parser.add_argument("--ediffg", type=float, default=-0.02)
    parser.add_argument("--climb", choices=["on", "off"], default="on")
    parser.add_argument("--spring", type=int, default=-5)
    parser.add_argument(
        "--neb-mpi-ranks",
        type=int,
        default=None,
        help="MPI ranks for VASP NEB. Defaults to the number of intermediate images.",
    )
    parser.add_argument(
        "--keep-performance-tags",
        action="store_true",
        help="By default, inherited NBANDS/NCORE tags are dropped.",
    )
    parser.add_argument("--force-overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_checked(cmd: list[str], dry_run: bool) -> None:
    print("[RUN]", " ".join(str(part) for part in cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    scripts_dir = repo_root / "scripts"
    out_root = args.out_root.resolve()

    if out_root.exists():
        if not args.force_overwrite:
            raise FileExistsError(f"Output root already exists: {out_root}")
        if not args.dry_run:
            shutil.rmtree(out_root)
    if args.batch_manifest_csv.exists():
        if not args.force_overwrite:
            raise FileExistsError(f"Batch manifest already exists: {args.batch_manifest_csv}")
        if not args.dry_run:
            args.batch_manifest_csv.unlink()

    summary_rows = read_csv_rows(args.phase1_summary_csv)
    selected_rows = [row for row in summary_rows if row["phase2_decision"] == "prepare_formal_neb"]
    if not selected_rows:
        raise ValueError("No Phase-1 rows are currently marked prepare_formal_neb")

    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(selected_rows, start=1):
        scaffold_dir = Path(row["scaffold_dir"]).resolve()
        final_relax_dir = Path(row["final_relax_dir"]).resolve()
        out_dir = out_root / f"{scaffold_dir.name}__formal_neb_hpc"

        cmd = [
            sys.executable,
            str(scripts_dir / "step4.9.2-prepare-dft-neb-from-relaxed-final.py"),
            "--scaffold-dir",
            str(scaffold_dir),
            "--final-relax-dir",
            str(final_relax_dir),
            "--out-dir",
            str(out_dir),
            "--job-name-suffix",
            args.job_name_suffix,
            "--nsw",
            str(args.nsw),
            "--ediffg",
            str(args.ediffg),
            "--climb",
            args.climb,
            "--spring",
            str(args.spring),
        ]
        if args.images_total is not None:
            cmd.extend(["--images-total", str(args.images_total)])
        if args.neb_mpi_ranks is not None:
            cmd.extend(["--neb-mpi-ranks", str(args.neb_mpi_ranks)])
        if not args.keep_performance_tags:
            cmd.append("--drop-performance-tags")
        if args.force_overwrite:
            cmd.append("--force-overwrite")

        print(
            f"[{idx}/{len(selected_rows)}] prepare formal DFT-NEB for "
            f"{row['system_id']} event {row['site_a']}->{row['site_b']}"
        )
        run_checked(cmd, args.dry_run)
        neb_mpi_ranks = "" if args.neb_mpi_ranks is None else args.neb_mpi_ranks
        prepare_info_path = out_dir / "prepare_info.json"
        if not args.dry_run and prepare_info_path.exists():
            neb_mpi_ranks = json.loads(prepare_info_path.read_text(encoding="utf-8")).get(
                "neb_mpi_ranks", neb_mpi_ranks
            )

        manifest_rows.append(
            {
                "system_id": row["system_id"],
                "formula": row["formula"],
                "event_type": row["event_type"],
                "event_family": row["event_family"],
                "story_role": row["story_role"],
                "event_pair_key": row["event_pair_key"],
                "site_a": row["site_a"],
                "site_b": row["site_b"],
                "source_final_relax_job_id": row["job_id"],
                "source_final_relax_dir": str(final_relax_dir),
                "source_scaffold_dir": str(scaffold_dir),
                "formal_neb_dir": str(out_dir),
                "formal_neb_job_json": str(out_dir / "job.json"),
                "formal_neb_run_sh": str(out_dir / "run.sh"),
                "job_name_suffix": args.job_name_suffix,
                "neb_nsw": args.nsw,
                "neb_ediffg": args.ediffg,
                "neb_climb": args.climb,
                "neb_spring": args.spring,
                "neb_mpi_ranks": neb_mpi_ranks,
                "drop_performance_tags": not args.keep_performance_tags,
                "status": "prepared(dry-run)" if args.dry_run else "prepared",
            }
        )

    if not args.dry_run:
        write_csv(args.batch_manifest_csv, manifest_rows)
        write_json(
            out_root / "batch_prepare_info.json",
            {
                "task_class": "phase1_dft_formal_neb_batch",
                "source_phase1_summary_csv": str(args.phase1_summary_csv.resolve()),
                "selected_rows": len(selected_rows),
                "out_root": str(out_root),
                "batch_manifest_csv": str(args.batch_manifest_csv.resolve()),
                "note": (
                    "This batch only includes Phase-1 rows marked prepare_formal_neb. "
                    "Endpoint-adjudication-only rows and active reruns are intentionally excluded."
                ),
            },
        )

    print(f"[OK] Phase-1 formal DFT-NEB tasks listed: {len(manifest_rows)}")
    if args.dry_run:
        print("[DRY-RUN] no directories or manifest were written")
    else:
        print(f"[OK] wrote batch manifest: {args.batch_manifest_csv.resolve()}")


if __name__ == "__main__":
    main()
