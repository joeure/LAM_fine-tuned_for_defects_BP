#!/usr/bin/env python3
"""
Batch basin validation for archived Stage 4.8 endpoint-relaxation shards.

This script bridges the archived shard layout back to the single-job
Step 4.5 / 4.6 workflow:

- it reuses the live shard job skeletons under batch_run/<shard>/jobs/
- it reuses archived final_relaxed.data files under archived_results/<shard>__<jobid>/results/
- it optionally re-downloads a shard payload from Bohrium and extracts only
  trajectory files when Step 4.6 data are missing
- it writes basin_check / basin_trace outputs back into the live job meta/
  directories so later batch steps can read them directly
- it mirrors those basin outputs into the archive under jobs/<job_name>/meta/
- it writes a shard-wide CSV summary for downstream filtering

The archive policy remains compact:
- no model file is copied into the archive
- Bohrium payload downloads are removed immediately after needed trajectories
  are extracted
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--batch-root",
        type=Path,
        default=Path("lammps_neb_workspace/batch_run"),
        help="Root directory containing live shard folders and archived_results/.",
    )
    p.add_argument(
        "--archive-root",
        type=Path,
        default=None,
        help="Override archive root. Defaults to <batch-root>/archived_results.",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data_center/step4.8_endpoint_basin_summary_formal_v1.csv"),
        help="Summary CSV for downstream batch NEB preparation.",
    )
    p.add_argument(
        "--tol",
        type=float,
        default=0.8,
        help="Distance tolerance in Angstrom for Step 4.5 / 4.6 basin labeling.",
    )
    p.add_argument(
        "--tail-window",
        type=int,
        default=8,
        help="Tail-window passed to Step 4.6 trajectory classification.",
    )
    p.add_argument(
        "--include-traj-check",
        action="store_true",
        help="Run Step 4.6 as well as Step 4.5. Missing trajectories can be "
        "re-downloaded one shard at a time if --download-missing-traj is set.",
    )
    p.add_argument(
        "--download-missing-traj",
        action="store_true",
        help="If Step 4.6 is enabled and archive trajectories are missing, "
        "re-download each shard payload, extract only traj.lammpstrj files, "
        "store them under the archive, then delete the payload.",
    )
    p.add_argument(
        "--force-redo",
        action="store_true",
        help="Re-run basin checks even if live meta outputs already exist.",
    )
    p.add_argument(
        "--keep-live-links",
        action="store_true",
        help="Keep temporary result/traj symlinks in the live jobs after analysis. "
        "Default is to clean them immediately.",
    )
    p.add_argument(
        "--exclude-archive-job-ids",
        nargs="*",
        default=[],
        help="Optional Bohrium archive job ids to skip when auditing historical shard archives.",
    )
    return p.parse_args()


def run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_pairwise_columns(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_event: Dict[Tuple[str, str, str], Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["system_id"]),
            str(row["vac_site_144"]),
            str(row["neighbor_site_144"]),
        )
        by_event.setdefault(key, {})[str(row["model_tag"])] = row

    for row in rows:
        key = (
            str(row["system_id"]),
            str(row["vac_site_144"]),
            str(row["neighbor_site_144"]),
        )
        event_group = by_event[key]
        other_model = "finetuned" if row["model_tag"] == "foundation" else "foundation"
        peer = event_group.get(other_model)

        row["event_pair_key"] = f"{key[0]}__{key[1]}__{key[2]}"
        row["has_model_pair"] = len(event_group) == 2
        row["paired_model_tag"] = other_model if peer is not None else None
        row["paired_job_name"] = peer.get("job_name") if peer is not None else None
        row["paired_archive_dir"] = peer.get("archive_dir") if peer is not None else None
        row["paired_preferred_classification"] = (
            peer.get("preferred_classification") if peer is not None else None
        )
        row["paired_ready_for_neb"] = peer.get("ready_for_neb") if peer is not None else None
        row["both_models_ready_for_neb"] = bool(row["ready_for_neb"]) and bool(
            peer.get("ready_for_neb") if peer is not None else False
        )
        row["either_model_ready_for_neb"] = bool(row["ready_for_neb"]) or bool(
            peer.get("ready_for_neb") if peer is not None else False
        )
        row["both_models_different_basin"] = (
            row["preferred_classification"] == "different_basin"
            and (peer is not None and peer.get("preferred_classification") == "different_basin")
        )
    return rows


def parse_archive_dir_name(name: str) -> Tuple[str, str]:
    base, sep, job_id = name.rpartition("__")
    if not sep or not job_id.isdigit():
        raise ValueError(f"Cannot parse archived shard name: {name}")
    return base, job_id


def discover_archived_shards(archive_root: Path, exclude_job_ids: Iterable[str] = ()) -> List[Path]:
    exclude = {str(x) for x in exclude_job_ids}
    out = []
    for path in sorted(archive_root.iterdir()):
        if not path.is_dir():
            continue
        name = path.name
        if "__shard_" not in name:
            continue
        if "formal_v1" not in name:
            continue
        _, job_id = parse_archive_dir_name(name)
        if job_id in exclude:
            continue
        out.append(path)
    return out


def load_events_csv(path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows[row["job_name"]] = row
    return rows


def result_job_name(result_path: Path) -> str:
    suffix = "__final_relaxed.data"
    if not result_path.name.endswith(suffix):
        raise ValueError(f"Unexpected result filename: {result_path.name}")
    return result_path.name[: -len(suffix)]


def archive_job_root(archive_dir: Path, job_name: str) -> Path:
    return archive_dir / "jobs" / job_name


def archive_traj_path(archive_dir: Path, job_name: str) -> Path:
    return archive_job_root(archive_dir, job_name) / "dumps" / "relax_final" / "traj.lammpstrj"


def archive_meta_dir(archive_dir: Path, job_name: str) -> Path:
    return archive_job_root(archive_dir, job_name) / "meta"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def extract_missing_trajs(batch_root: Path, archive_dir: Path, shard_live_name: str, shard_job_id: str) -> None:
    wanted = [job_root.parent.name for job_root in archive_dir.glob("jobs/*/dumps/relax_final/traj.lammpstrj")]
    _ = wanted  # quiet lint when nothing is missing
    result_job_names = sorted(result_job_name(p) for p in (archive_dir / "results").glob("*__final_relaxed.data"))
    missing = [job for job in result_job_names if not archive_traj_path(archive_dir, job).exists()]
    if not missing:
        return

    with tempfile.TemporaryDirectory(prefix=f"{shard_live_name}__", dir=str(batch_root)) as tmpdir:
        tmp_root = Path(tmpdir)
        payload_root = tmp_root / "payload"
        payload_root.mkdir(parents=True, exist_ok=True)
        run_cmd(["bohr", "job", "download", "-j", shard_job_id, "-o", str(payload_root)])
        out_zip = payload_root / shard_job_id / "out.zip"
        if not out_zip.exists():
            raise FileNotFoundError(f"Missing downloaded out.zip for shard {shard_live_name}: {out_zip}")

        extract_root = tmp_root / "extract"
        extract_root.mkdir(parents=True, exist_ok=True)
        run_cmd(
            [
                "unzip",
                "-q",
                str(out_zip),
                "jobs/*/dumps/relax_final/traj.lammpstrj",
                "-d",
                str(extract_root),
            ]
        )

        for traj in extract_root.glob("jobs/*/dumps/relax_final/traj.lammpstrj"):
            job_name = traj.parents[2].name
            dest = archive_traj_path(archive_dir, job_name)
            ensure_parent(dest)
            shutil.copy2(traj, dest)


def ensure_symlink(src: Path, dest: Path) -> None:
    ensure_parent(dest)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    dest.symlink_to(src)


def cleanup_empty_dirs(path: Path, stop_at: Path) -> None:
    cur = path
    while cur != stop_at and cur.exists():
        try:
            cur.rmdir()
        except OSError:
            break
        cur = cur.parent


def run_single_job_checks(
    scripts_dir: Path,
    live_job_dir: Path,
    result_path: Path,
    traj_path: Optional[Path],
    tol: float,
    tail_window: int,
    include_traj_check: bool,
    keep_live_links: bool,
) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    live_result = live_job_dir / "results" / "final_relaxed.data"
    live_traj = live_job_dir / "dumps" / "relax_final" / "traj.lammpstrj"

    ensure_symlink(result_path, live_result)
    if include_traj_check and traj_path is not None:
        ensure_symlink(traj_path, live_traj)

    basin_check = live_job_dir / "meta" / "basin_check.json"
    basin_trace_csv = live_job_dir / "meta" / "basin_trace.csv"
    basin_trace_summary = live_job_dir / "meta" / "basin_trace_summary.json"

    run_cmd(
        [
            sys.executable,
            str(scripts_dir / "step4.5-check_relaxed_endpoint_basin.py"),
            "--job-dir",
            str(live_job_dir),
            "--tol",
            str(tol),
        ]
    )

    if include_traj_check and traj_path is not None:
        run_cmd(
            [
                sys.executable,
                str(scripts_dir / "step4.6-check_relaxed_endpoint_basin_from_traj.py"),
                "--job-dir",
                str(live_job_dir),
                "--tol",
                str(tol),
                "--tail-window",
                str(tail_window),
                "--check-final-data",
            ]
        )

    if not keep_live_links:
        if live_result.exists() or live_result.is_symlink():
            live_result.unlink()
        cleanup_empty_dirs(live_result.parent, live_job_dir)
        if live_traj.exists() or live_traj.is_symlink():
            live_traj.unlink()
        cleanup_empty_dirs(live_traj.parent, live_job_dir)

    return (
        basin_check if basin_check.exists() else None,
        basin_trace_csv if basin_trace_csv.exists() else None,
        basin_trace_summary if basin_trace_summary.exists() else None,
    )


def mirror_job_outputs_to_archive(
    archive_dir: Path,
    job_name: str,
    basin_check: Optional[Path],
    basin_trace_csv: Optional[Path],
    basin_trace_summary: Optional[Path],
) -> None:
    meta_dir = archive_meta_dir(archive_dir, job_name)
    meta_dir.mkdir(parents=True, exist_ok=True)
    if basin_check is not None:
        shutil.copy2(basin_check, meta_dir / basin_check.name)
    if basin_trace_csv is not None:
        shutil.copy2(basin_trace_csv, meta_dir / basin_trace_csv.name)
    if basin_trace_summary is not None:
        shutil.copy2(basin_trace_summary, meta_dir / basin_trace_summary.name)


def build_summary_row(
    archive_dir: Path,
    shard_live_name: str,
    shard_job_id: str,
    job_name: str,
    event_row: Dict[str, Any],
    live_job_dir: Path,
) -> Dict[str, Any]:
    meta_json = read_json(live_job_dir / "meta" / "meta.json")
    job_info = meta_json["job_info"]

    basin_check_path = live_job_dir / "meta" / "basin_check.json"
    basin_trace_path = live_job_dir / "meta" / "basin_trace_summary.json"
    basin_check = read_json(basin_check_path) if basin_check_path.exists() else None
    basin_trace = read_json(basin_trace_path) if basin_trace_path.exists() else None

    check_label = None
    check_reason = None
    if basin_check is not None:
        check_label = basin_check.get("result", {}).get("classification")
        check_reason = basin_check.get("result", {}).get("reason")

    trace_label = None
    trace_reason = None
    tail_window_used = None
    nframes = None
    if basin_trace is not None:
        trace_label = basin_trace.get("classification")
        trace_reason = basin_trace.get("reason")
        tail_window_used = basin_trace.get("tail_window_used")
        nframes = basin_trace.get("nframes")

    preferred = trace_label or check_label
    return {
        "archive_dir": str(archive_dir),
        "archive_job_id": shard_job_id,
        "live_shard_dir": str((live_job_dir.parents[2])),
        "shard_name": shard_live_name,
        "job_name": job_name,
        "system_id": job_info.get("system_id"),
        "model_tag": job_info.get("model_tag"),
        "structure": event_row.get("structure"),
        "split": event_row.get("split"),
        "event_type": event_row.get("event_type"),
        "vac_site_144": event_row.get("vac_site_144"),
        "neighbor_site_144": event_row.get("neighbor_site_144"),
        "site_a": job_info.get("site_a"),
        "site_b": job_info.get("site_b"),
        "moving_atom_index_0based": job_info.get("moving_atom_index_0based"),
        "moving_atom_element": job_info.get("moving_atom_element"),
        "basin_check_classification": check_label,
        "basin_check_reason": check_reason,
        "basin_trace_classification": trace_label,
        "basin_trace_reason": trace_reason,
        "preferred_classification": preferred,
        "ready_for_neb": preferred == "different_basin",
        "traj_available": archive_traj_path(archive_dir, job_name).exists(),
        "nframes": nframes,
        "tail_window_used": tail_window_used,
        "live_basin_check_path": str(basin_check_path) if basin_check_path.exists() else None,
        "live_basin_trace_summary_path": str(basin_trace_path) if basin_trace_path.exists() else None,
        "archive_basin_meta_dir": str(archive_meta_dir(archive_dir, job_name)),
    }


def main() -> None:
    args = parse_args()
    batch_root = args.batch_root.resolve()
    archive_root = (args.archive_root or (batch_root / "archived_results")).resolve()
    scripts_dir = Path(__file__).resolve().parent

    archived_shards = discover_archived_shards(archive_root, args.exclude_archive_job_ids)
    if not archived_shards:
        raise FileNotFoundError(f"No formal archived shards found under {archive_root}")

    summary_rows: List[Dict[str, Any]] = []

    for archive_dir in archived_shards:
        shard_live_name, shard_job_id = parse_archive_dir_name(archive_dir.name)
        live_shard_dir = batch_root / shard_live_name
        if not live_shard_dir.exists():
            raise FileNotFoundError(f"Missing live shard directory for {archive_dir.name}: {live_shard_dir}")

        events_map = load_events_csv(archive_dir / "meta" / "events.csv")
        result_files = sorted((archive_dir / "results").glob("*__final_relaxed.data"))
        if not result_files:
            raise FileNotFoundError(f"No archived result files found in {archive_dir / 'results'}")

        if args.include_traj_check and args.download_missing_traj:
            extract_missing_trajs(batch_root, archive_dir, shard_live_name, shard_job_id)

        for result_path in result_files:
            job_name = result_job_name(result_path)
            live_job_dir = live_shard_dir / "jobs" / job_name
            if not live_job_dir.exists():
                raise FileNotFoundError(f"Missing live job skeleton: {live_job_dir}")

            event_row = events_map.get(job_name, {})
            traj_path = archive_traj_path(archive_dir, job_name)
            traj_available = traj_path.exists()

            have_outputs = (
                (live_job_dir / "meta" / "basin_check.json").exists()
                and (
                    not args.include_traj_check
                    or not traj_available
                    or (live_job_dir / "meta" / "basin_trace_summary.json").exists()
                )
            )

            if args.force_redo or not have_outputs:
                basin_check, basin_trace_csv, basin_trace_summary = run_single_job_checks(
                    scripts_dir=scripts_dir,
                    live_job_dir=live_job_dir,
                    result_path=result_path,
                    traj_path=traj_path if traj_available else None,
                    tol=args.tol,
                    tail_window=args.tail_window,
                    include_traj_check=args.include_traj_check,
                    keep_live_links=args.keep_live_links,
                )
                mirror_job_outputs_to_archive(
                    archive_dir=archive_dir,
                    job_name=job_name,
                    basin_check=basin_check,
                    basin_trace_csv=basin_trace_csv,
                    basin_trace_summary=basin_trace_summary,
                )

            summary_rows.append(
                build_summary_row(
                    archive_dir=archive_dir,
                    shard_live_name=shard_live_name,
                    shard_job_id=shard_job_id,
                    job_name=job_name,
                    event_row=event_row,
                    live_job_dir=live_job_dir,
                )
            )

    fieldnames = [
        "archive_dir",
        "archive_job_id",
        "live_shard_dir",
        "shard_name",
        "job_name",
        "event_pair_key",
        "system_id",
        "model_tag",
        "structure",
        "split",
        "event_type",
        "vac_site_144",
        "neighbor_site_144",
        "site_a",
        "site_b",
        "moving_atom_index_0based",
        "moving_atom_element",
        "basin_check_classification",
        "basin_check_reason",
        "basin_trace_classification",
        "basin_trace_reason",
        "preferred_classification",
        "ready_for_neb",
        "has_model_pair",
        "paired_model_tag",
        "paired_job_name",
        "paired_archive_dir",
        "paired_preferred_classification",
        "paired_ready_for_neb",
        "both_models_ready_for_neb",
        "either_model_ready_for_neb",
        "both_models_different_basin",
        "traj_available",
        "nframes",
        "tail_window_used",
        "live_basin_check_path",
        "live_basin_trace_summary_path",
        "archive_basin_meta_dir",
    ]
    summary_rows = add_pairwise_columns(summary_rows)
    write_csv(args.output_csv.resolve(), summary_rows, fieldnames)
    print(f"[STEP4.8 BASIN] wrote {len(summary_rows)} rows to {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
