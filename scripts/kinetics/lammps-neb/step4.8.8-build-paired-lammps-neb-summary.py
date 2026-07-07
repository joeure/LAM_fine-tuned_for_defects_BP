#!/usr/bin/env python3
"""Build one paired foundation-vs-finetuned CI-NEB summary CSV.

This script consolidates the manifest-driven live shard results produced after
Stage 4.8 shard postprocessing. It uses the event-pair transition table as the
canonical event universe, then joins in per-model NEB outcomes from the live
shard `partial_result_status_manifest.json` files.

The output is intended to support the next scientific step:
- identify where finetuning changes usable barrier predictions relative to the
  foundation model,
- compare coverage across the four representative event classes,
- and rank candidates for the later DFT calibration subset.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


TASK_NAME_RE = re.compile(
    r"^job_(?P<system_id>.+)__event_(?P<site_a>\d+)_(?P<site_b>\d+)__(?P<model_tag>foundation|finetuned)__cineb$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-run-root",
        type=Path,
        default=Path("lammps_neb_workspace/batch_run"),
        help="Root directory that contains the live CI-NEB shard directories.",
    )
    parser.add_argument(
        "--transition-table",
        type=Path,
        default=Path("data_center/step4.8.2_foundation_finetuned_transition_table_formal_v1.csv"),
        help="Canonical event-pair table used as the base event universe.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data_center/step4.8.8_paired_lammps_neb_summary_formal_v1.csv"),
        help="Where to write the paired event-level CI-NEB summary CSV.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_task_name(task_name: str) -> dict[str, Any]:
    match = TASK_NAME_RE.match(task_name)
    if match is None:
        raise ValueError(f"Unrecognized task name format: {task_name}")
    return {
        "system_id": match.group("system_id"),
        "site_a": int(match.group("site_a")),
        "site_b": int(match.group("site_b")),
        "model_tag": match.group("model_tag"),
        "event_pair_key": (
            f"{match.group('system_id')}__{int(match.group('site_a'))}__{int(match.group('site_b'))}"
        ),
    }


def normalize_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    return value


def collect_model_records(batch_run_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    manifest_paths = sorted(
        batch_run_root.glob(
            "cineb__*__tier1_event_list_graph_topo_n2nn_formal_v1__shard_*_of_006/meta/partial_result_status_manifest.json"
        )
    )
    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        shard_dir = Path(manifest["shard_dir"])
        archive_dir = manifest.get("archive_dir", "")
        for task in manifest["tasks"]:
            parsed = parse_task_name(task["task_name"])
            key = (parsed["event_pair_key"], parsed["model_tag"])
            if key in records:
                raise ValueError(f"Duplicate record for {key} from {manifest_path}")

            row: dict[str, Any] = {
                "event_pair_key": parsed["event_pair_key"],
                "system_id": parsed["system_id"],
                "site_a": parsed["site_a"],
                "site_b": parsed["site_b"],
                "model_tag": parsed["model_tag"],
                "task_name": task["task_name"],
                "task_dir": str(shard_dir / task["task_dir"]),
                "status": task.get("status", ""),
                "usable_complete": task.get("usable_complete", False),
                "suspicious": task.get("suspicious", False),
                "has_neb_finished": task.get("has_neb_finished", False),
                "log_neb_parseable": task.get("log_neb_parseable", False),
                "endpoint_files_ok": task.get("endpoint_files_ok", False),
                "replica_data_ok": task.get("replica_data_ok", False),
                "replica_traj_ok": task.get("replica_traj_ok", False),
                "grad_error": task.get("grad_error", False),
                "mpi_abort": task.get("mpi_abort", False),
                "segfault": task.get("segfault", False),
                "n_images": task.get("n_images", ""),
                "final_step": task.get("final_step", ""),
                "final_max_replica_force_eVA": task.get("final_max_replica_force_eVA", ""),
                "forward_barrier_eV": task.get("forward_barrier_eV", ""),
                "reverse_barrier_eV": task.get("reverse_barrier_eV", ""),
                "saddle_image_index": task.get("saddle_image_index", ""),
                "path_shape": task.get("path_shape", ""),
                "notes": task.get("notes", ""),
                "live_shard_dir": str(shard_dir),
                "archive_dir": archive_dir,
            }
            records[key] = {k: normalize_value(v) for k, v in row.items()}
    return records


def to_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def to_float_or_blank(value: Any) -> Any:
    if value in ("", None):
        return ""
    return float(value)


def build_rows(
    base_rows: list[dict[str, str]],
    model_records: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    for base in base_rows:
        event_pair_key = base["event_pair_key"]
        foundation = model_records.get((event_pair_key, "foundation"))
        finetuned = model_records.get((event_pair_key, "finetuned"))

        fdn_forward = to_float_or_blank(foundation["forward_barrier_eV"]) if foundation else ""
        ftn_forward = to_float_or_blank(finetuned["forward_barrier_eV"]) if finetuned else ""
        fdn_reverse = to_float_or_blank(foundation["reverse_barrier_eV"]) if foundation else ""
        ftn_reverse = to_float_or_blank(finetuned["reverse_barrier_eV"]) if finetuned else ""

        both_usable = bool(
            foundation
            and finetuned
            and foundation["usable_complete"]
            and finetuned["usable_complete"]
        )
        both_status_usable = bool(
            foundation
            and finetuned
            and foundation["status"] == "usable"
            and finetuned["status"] == "usable"
        )
        both_parseable = bool(
            foundation
            and finetuned
            and foundation["log_neb_parseable"]
            and finetuned["log_neb_parseable"]
        )

        row: dict[str, Any] = {
            "event_pair_key": event_pair_key,
            "system_id": base["system_id"],
            "split": base["split"],
            "event_type": base["event_type"],
            "event_family": base["event_family"],
            "family_aczz": base["family_aczz"],
            "is_near_dopant": to_bool(base["is_near_dopant"]),
            "is_far_from_dopant": to_bool(base["is_far_from_dopant"]),
            "is_escaping": to_bool(base["is_escaping"]),
            "is_non_escaping": to_bool(base["is_non_escaping"]),
            "vac_site_144": int(base["vac_site_144"]),
            "neighbor_site_144": int(base["neighbor_site_144"]),
            "site_a": int(base["site_a"]),
            "site_b": int(base["site_b"]),
            "foundation_endpoint_label": base["foundation_label"],
            "finetuned_endpoint_label": base["finetuned_label"],
            "transition_label": base["transition_label"],
            "endpoint_is_disagree": to_bool(base["is_disagree"]),
            "foundation_ready_for_neb": to_bool(base["foundation_ready_for_neb"]),
            "finetuned_ready_for_neb": to_bool(base["finetuned_ready_for_neb"]),
            "both_models_ready_for_neb": to_bool(base["both_models_ready_for_neb"]),
            "foundation_neb_present": foundation is not None,
            "finetuned_neb_present": finetuned is not None,
            "foundation_neb_status": foundation["status"] if foundation else "missing",
            "finetuned_neb_status": finetuned["status"] if finetuned else "missing",
            "foundation_usable_complete": foundation["usable_complete"] if foundation else False,
            "finetuned_usable_complete": finetuned["usable_complete"] if finetuned else False,
            "both_models_usable_complete": both_usable,
            "foundation_status_is_usable": foundation["status"] == "usable" if foundation else False,
            "finetuned_status_is_usable": finetuned["status"] == "usable" if finetuned else False,
            "both_models_status_usable": both_status_usable,
            "both_models_log_neb_parseable": both_parseable,
            "foundation_forward_barrier_eV": fdn_forward,
            "finetuned_forward_barrier_eV": ftn_forward,
            "foundation_reverse_barrier_eV": fdn_reverse,
            "finetuned_reverse_barrier_eV": ftn_reverse,
            "forward_barrier_delta_ft_minus_foundation_eV": (
                ftn_forward - fdn_forward if both_parseable else ""
            ),
            "reverse_barrier_delta_ft_minus_foundation_eV": (
                ftn_reverse - fdn_reverse if both_parseable else ""
            ),
            "abs_forward_barrier_delta_eV": (
                abs(ftn_forward - fdn_forward) if both_parseable else ""
            ),
            "abs_reverse_barrier_delta_eV": (
                abs(ftn_reverse - fdn_reverse) if both_parseable else ""
            ),
            "strict_forward_barrier_delta_ft_minus_foundation_eV": (
                ftn_forward - fdn_forward if both_status_usable else ""
            ),
            "strict_reverse_barrier_delta_ft_minus_foundation_eV": (
                ftn_reverse - fdn_reverse if both_status_usable else ""
            ),
            "strict_abs_forward_barrier_delta_eV": (
                abs(ftn_forward - fdn_forward) if both_status_usable else ""
            ),
            "strict_abs_reverse_barrier_delta_eV": (
                abs(ftn_reverse - fdn_reverse) if both_status_usable else ""
            ),
            "foundation_path_shape": foundation["path_shape"] if foundation else "",
            "finetuned_path_shape": finetuned["path_shape"] if finetuned else "",
            "foundation_final_step": foundation["final_step"] if foundation else "",
            "finetuned_final_step": finetuned["final_step"] if finetuned else "",
            "foundation_final_max_replica_force_eVA": (
                foundation["final_max_replica_force_eVA"] if foundation else ""
            ),
            "finetuned_final_max_replica_force_eVA": (
                finetuned["final_max_replica_force_eVA"] if finetuned else ""
            ),
            "foundation_notes": foundation["notes"] if foundation else "",
            "finetuned_notes": finetuned["notes"] if finetuned else "",
            "foundation_live_shard_dir": foundation["live_shard_dir"] if foundation else "",
            "finetuned_live_shard_dir": finetuned["live_shard_dir"] if finetuned else "",
            "foundation_archive_dir": foundation["archive_dir"] if foundation else "",
            "finetuned_archive_dir": finetuned["archive_dir"] if finetuned else "",
        }
        out_rows.append(row)

    out_rows.sort(
        key=lambda r: (
            r["system_id"],
            int(r["site_a"]),
            int(r["site_b"]),
        )
    )
    return out_rows


def main() -> None:
    args = parse_args()
    base_rows = read_csv_rows(args.transition_table)
    model_records = collect_model_records(args.batch_run_root)
    rows = build_rows(base_rows, model_records)
    write_csv(args.output_csv, rows)

    both_usable = sum(1 for row in rows if row["both_models_usable_complete"])
    both_status_usable = sum(1 for row in rows if row["both_models_status_usable"])
    foundation_usable = sum(1 for row in rows if row["foundation_usable_complete"])
    finetuned_usable = sum(1 for row in rows if row["finetuned_usable_complete"])

    print(f"[OK] wrote paired CI-NEB summary: {args.output_csv.resolve()}")
    print(f"[Rows] {len(rows)}")
    print(f"[Foundation usable] {foundation_usable}")
    print(f"[Finetuned usable] {finetuned_usable}")
    print(f"[Both usable] {both_usable}")
    print(f"[Both status=usable] {both_status_usable}")


if __name__ == "__main__":
    main()
