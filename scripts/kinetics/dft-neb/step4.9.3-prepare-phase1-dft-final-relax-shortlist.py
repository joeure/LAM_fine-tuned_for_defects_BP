#!/usr/bin/env python3
"""Prepare Phase-1 DFT final-relax task directories from the Step 4.8 shortlist.

This script bridges the shortlist-oriented Stage 4.8 analysis to the existing
single-event Stage 4.9 VASP packaging workflow.

For each shortlisted event it:
1. builds a reusable DFT scaffold with ``step4.9-prepare-single-dft-neb-first-bite.py``
2. exports a standalone HPC-oriented final-endpoint-relax package with
   ``step4.9.1-prepare-dft-final-endpoint-relax.py``
3. writes a batch manifest CSV for downstream submission and tracking

Current scope:
- Phase 1 only
- final-endpoint relaxation only
- no formal DFT-NEB generation yet
"""

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
        "--phase1-shortlist-csv",
        type=Path,
        default=Path("data_center/step4.8.11_phase1_first_wave_dft_shortlist_formal_v1.csv"),
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=Path("data_center/manifest_test.csv"),
    )
    parser.add_argument(
        "--site-map-csv",
        type=Path,
        default=Path("data_center/site_matching.csv"),
    )
    parser.add_argument(
        "--job-json-ref",
        type=Path,
        default=Path("references_codes/DFT/job.json"),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("dft_neb_workspace/phase1_shortlist_formal_v1"),
        help="Root directory for generated scaffold and final-relax packages.",
    )
    parser.add_argument(
        "--batch-manifest-csv",
        type=Path,
        default=Path("data_center/step4.9.3_phase1_final_relax_batch_manifest_formal_v1.csv"),
    )
    parser.add_argument(
        "--job-name-suffix",
        default="phase1-shortlist-formal-v1",
        help="Suffix appended to the reusable per-event DFT scaffold name.",
    )
    parser.add_argument(
        "--final-relax-job-name-suffix",
        default="hpc-final-relax",
        help="Suffix appended by step4.9.1 to the standalone final-relax job name.",
    )
    parser.add_argument("--images-total", type=int, default=7)
    parser.add_argument("--nsw", type=int, default=200)
    parser.add_argument("--ediffg", type=float, default=-0.02)
    parser.add_argument("--ibrion", type=int, default=2)
    parser.add_argument("--potim", type=float, default=0.03)
    parser.add_argument("--nbands", type=int, default=None)
    parser.add_argument("--ncore", type=int, default=None)
    parser.add_argument(
        "--drop-performance-tags",
        action="store_true",
        help="Drop inherited NBANDS/NCORE in the generated final-relax package.",
    )
    parser.add_argument(
        "--submission-project-tag",
        default="phase1_dft_final_relax_first_wave",
        help="Human-readable label written into the batch metadata only.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for incremental validation or partial preparation.",
    )
    parser.add_argument(
        "--force-overwrite",
        action="store_true",
        help="Replace the target out-root and batch manifest if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the actions without generating directories.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def sanitize_name(text: str) -> str:
    return text.replace("/", "_").replace(" ", "_").replace(":", "_")


def build_scaffold_job_name(system_id: str, site_a: int, site_b: int, suffix: str) -> str:
    base = f"job_{system_id}__event_{site_a}_{site_b}__dft_neb_first_bite"
    clean_suffix = sanitize_name(suffix.strip())
    if clean_suffix:
        return f"{base}__{clean_suffix}"
    return base


def run_checked(cmd: list[str], dry_run: bool) -> None:
    print("[RUN]", " ".join(str(part) for part in cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    scripts_dir = repo_root / "scripts"
    shortlist_rows = read_csv_rows(args.phase1_shortlist_csv)
    if args.limit is not None:
        shortlist_rows = shortlist_rows[: args.limit]
    if not shortlist_rows:
        raise ValueError("No shortlist rows selected")

    out_root = args.out_root.resolve()
    manifest_csv = args.manifest_csv.resolve()
    site_map_csv = args.site_map_csv.resolve()
    job_json_ref = args.job_json_ref.resolve()
    scaffold_root = out_root / "scaffolds"
    final_relax_root = out_root / "final_relax_jobs"
    batch_info_path = out_root / "batch_prepare_info.json"

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

    if not args.dry_run:
        scaffold_root.mkdir(parents=True, exist_ok=True)
        final_relax_root.mkdir(parents=True, exist_ok=True)

    batch_rows: list[dict[str, Any]] = []

    for idx, row in enumerate(shortlist_rows, start=1):
        system_id = row["system_id"]
        site_a = int(row["site_a"])
        site_b = int(row["site_b"])
        scaffold_name = build_scaffold_job_name(system_id, site_a, site_b, args.job_name_suffix)
        scaffold_dir = scaffold_root / scaffold_name
        final_relax_dir = final_relax_root / f"{scaffold_name}__final_relax_hpc"

        scaffold_cmd = [
            sys.executable,
            str(scripts_dir / "step4.9-prepare-single-dft-neb-first-bite.py"),
            "--manifest",
            str(manifest_csv),
            "--site-map",
            str(site_map_csv),
            "--job-json-ref",
            str(job_json_ref),
            "--system-id",
            system_id,
            "--site-a",
            str(site_a),
            "--site-b",
            str(site_b),
            "--job-name-suffix",
            args.job_name_suffix,
            "--out-root",
            str(scaffold_root),
            "--images-total",
            str(args.images_total),
        ]
        if args.force_overwrite:
            scaffold_cmd.append("--force-overwrite")

        final_relax_cmd = [
            sys.executable,
            str(scripts_dir / "step4.9.1-prepare-dft-final-endpoint-relax.py"),
            "--scaffold-dir",
            str(scaffold_dir),
            "--out-dir",
            str(final_relax_dir),
            "--job-name-suffix",
            args.final_relax_job_name_suffix,
            "--nsw",
            str(args.nsw),
            "--ediffg",
            str(args.ediffg),
            "--ibrion",
            str(args.ibrion),
            "--potim",
            str(args.potim),
        ]
        if args.nbands is not None:
            final_relax_cmd.extend(["--nbands", str(args.nbands)])
        if args.ncore is not None:
            final_relax_cmd.extend(["--ncore", str(args.ncore)])
        if args.drop_performance_tags:
            final_relax_cmd.append("--drop-performance-tags")
        if args.force_overwrite:
            final_relax_cmd.append("--force-overwrite")

        print(
            f"[{idx}/{len(shortlist_rows)}] prepare Phase-1 final relax for "
            f"{system_id} event {site_a}->{site_b}"
        )
        run_checked(scaffold_cmd, args.dry_run)
        run_checked(final_relax_cmd, args.dry_run)

        batch_row: dict[str, Any] = {
            "phase": row["phase"],
            "submission_rank": row["submission_rank"],
            "submission_project_tag": args.submission_project_tag,
            "event_pair_key": row["event_pair_key"],
            "system_id": system_id,
            "formula": row["formula"],
            "event_type": row["event_type"],
            "event_family": row["event_family"],
            "story_role": row["story_role"],
            "site_a": site_a,
            "site_b": site_b,
            "dft_action": row["dft_action"],
            "shortlist_bucket": row.get("shortlist_bucket", ""),
            "shortlist_reason": row.get("shortlist_reason", ""),
            "selection_reason": row.get("selection_reason", ""),
            "recommended_priority_tier": row["recommended_priority_tier"],
            "rank_within_action": row["rank_within_action"],
            "rank_within_action_event_type": row["rank_within_action_event_type"],
            "selection_score": row["selection_score"],
            "phase1_goal": row["phase1_goal"],
            "phase2_followup": row["phase2_followup"],
            "preferred_endpoint_guess_model": row["preferred_endpoint_guess_model"],
            "images_total": args.images_total,
            "final_relax_nsw": args.nsw,
            "final_relax_ediffg": args.ediffg,
            "final_relax_ibrion": args.ibrion,
            "final_relax_potim": args.potim,
            "drop_performance_tags": args.drop_performance_tags,
            "nbands": "" if args.nbands is None else args.nbands,
            "ncore": "" if args.ncore is None else args.ncore,
            "scaffold_dir": str(scaffold_dir),
            "final_relax_dir": str(final_relax_dir),
            "final_relax_job_json": str(final_relax_dir / "job.json"),
            "final_relax_run_sh": str(final_relax_dir / "run.sh"),
            "status": "prepared(dry-run)" if args.dry_run else "prepared",
        }
        batch_rows.append(batch_row)

    if not args.dry_run:
        write_csv(args.batch_manifest_csv, batch_rows)
        write_json(
            batch_info_path,
            {
                "task_class": "phase1_dft_final_relax_batch",
                "phase": "phase1",
                "phase_scope": "first_wave_shortlist",
                "submission_project_tag": args.submission_project_tag,
                "source_shortlist_csv": str(args.phase1_shortlist_csv.resolve()),
                "generated_rows": len(batch_rows),
                "out_root": str(out_root),
                "scaffold_root": str(scaffold_root),
                "final_relax_root": str(final_relax_root),
                "batch_manifest_csv": str(args.batch_manifest_csv.resolve()),
                "note": (
                    "This batch contains only Phase-1 final-endpoint relaxation tasks. "
                    "Formal DFT-NEB must be generated later, only after final-relax results pass basin checks."
                ),
            },
        )

    print(f"[OK] Phase-1 tasks listed: {len(batch_rows)}")
    if args.dry_run:
        print("[DRY-RUN] no directories or manifest were written")
    else:
        print(f"[OK] wrote batch manifest: {args.batch_manifest_csv.resolve()}")
        print(f"[OK] wrote batch info: {batch_info_path}")


if __name__ == "__main__":
    main()
