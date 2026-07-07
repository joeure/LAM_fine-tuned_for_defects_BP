#!/usr/bin/env python3
"""Select DFT calibration candidates from the paired LAMMPS CI-NEB summary.

This script separates the next DFT work into two scientifically distinct groups:

1. `final_relax_only`
   For events where foundation and finetuned disagree on the final basin.
   These are used to adjudicate endpoint / topology correctness before spending
   full DFT-NEB cost.

2. `full_neb`
   For events where the final basin is already aligned well enough that a DFT
   barrier comparison is meaningful for the paper's barrier/rate story.

The selection logic is intentionally story-driven rather than purely numerical:
- preserve coverage across the four representative event classes,
- prefer events where fine-tuning changes a kinetics-relevant quantity,
- avoid obviously pathological NEB rows when building the full-NEB list,
- and keep cost-related metadata visible in the output.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


EVENT_TYPE_PRIORITY = {
    "trap_escape": 1.50,
    "nearN_hop": 1.35,
    "AC_hop": 1.00,
    "ZZ_hop": 1.00,
}

ENDPOINT_TRANSITION_SEVERITY = {
    "different_basin -> same_basin": 4.0,
    "same_basin -> different_basin": 4.0,
    "different_basin -> other_basin": 4.0,
    "other_basin -> different_basin": 4.0,
    "same_basin -> other_basin": 2.0,
    "other_basin -> same_basin": 2.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paired-summary-csv",
        type=Path,
        default=Path("data_center/step4.8.8_paired_lammps_neb_summary_formal_v1.csv"),
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=Path("data_center/manifest_test.csv"),
    )
    parser.add_argument(
        "--neb-batch-manifest-csv",
        type=Path,
        default=Path("data_center/step4.8.1_lammps_mace_neb_batch_manifest_formal_v1.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data_center/step4.8.9_dft_calibration_candidates_formal_v1.csv"),
    )
    parser.add_argument(
        "--final-relax-only-csv",
        type=Path,
        default=Path("data_center/step4.8.9_dft_final_relax_only_candidates_formal_v1.csv"),
    )
    parser.add_argument(
        "--full-neb-csv",
        type=Path,
        default=Path("data_center/step4.8.9_dft_full_neb_candidates_formal_v1.csv"),
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


def as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def as_float(value: str) -> float | None:
    text = value.strip()
    if not text:
        return None
    return float(text)


def build_manifest_index(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["system_id"]: row for row in rows}


def build_neb_batch_index(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["event_pair_key"], row["model_tag"]): row for row in rows}


def full_neb_eligible(row: dict[str, str]) -> bool:
    if not as_bool(row["both_models_status_usable"]):
        return False
    if as_bool(row["endpoint_is_disagree"]):
        return False
    foundation_forward = as_float(row["foundation_forward_barrier_eV"])
    finetuned_forward = as_float(row["finetuned_forward_barrier_eV"])
    if foundation_forward is None or finetuned_forward is None:
        return False
    if foundation_forward < -0.05 or finetuned_forward < -0.05:
        return False
    if foundation_forward > 2.5 or finetuned_forward > 2.5:
        return False
    if row["foundation_path_shape"] != "interior_saddle":
        return False
    if row["finetuned_path_shape"] != "interior_saddle":
        return False
    return True


def final_relax_only_eligible(row: dict[str, str]) -> bool:
    return as_bool(row["endpoint_is_disagree"])


def classify_story_role(event_type: str) -> str:
    if event_type == "trap_escape":
        return "trap/detrap bottleneck"
    if event_type == "nearN_hop":
        return "near-dopant environment sensitivity"
    if event_type == "AC_hop":
        return "far-from-N armchair baseline"
    if event_type == "ZZ_hop":
        return "far-from-N zigzag baseline"
    return "event-level comparison"


def build_selection_reason(action: str, row: dict[str, str]) -> str:
    if action == "final_relax_only":
        return (
            "Foundation and finetuned disagree on final basin identity; use DFT final-state "
            "relaxation first to adjudicate endpoint/topology before spending full NEB cost."
        )
    return (
        "Foundation and finetuned share a comparable final basin and both have usable CI-NEB "
        "results; this is suitable for direct DFT barrier / hopping-rate calibration."
    )


def score_final_relax_only(row: dict[str, str], manifest_row: dict[str, str]) -> tuple[float, dict[str, Any]]:
    event_type = row["event_type"]
    story_weight = EVENT_TYPE_PRIORITY.get(event_type, 1.0)
    transition_severity = ENDPOINT_TRANSITION_SEVERITY.get(row["transition_label"], 1.0)
    readiness_bonus = 0.0
    if as_bool(row["foundation_ready_for_neb"]) ^ as_bool(row["finetuned_ready_for_neb"]):
        readiness_bonus = 0.50
    elif as_bool(row["foundation_ready_for_neb"]) and as_bool(row["finetuned_ready_for_neb"]):
        readiness_bonus = 0.20
    defect_count = int(manifest_row["nsubs"]) + int(manifest_row["nvacs"])
    cost_penalty = 0.03 * defect_count
    score = 10.0 * transition_severity + 4.0 * story_weight + readiness_bonus - cost_penalty
    components = {
        "story_weight": story_weight,
        "transition_severity": transition_severity,
        "readiness_bonus": readiness_bonus,
        "cost_penalty": cost_penalty,
    }
    return score, components


def score_full_neb(row: dict[str, str], manifest_row: dict[str, str]) -> tuple[float, dict[str, Any]]:
    event_type = row["event_type"]
    story_weight = EVENT_TYPE_PRIORITY.get(event_type, 1.0)
    foundation_forward = as_float(row["foundation_forward_barrier_eV"])
    finetuned_forward = as_float(row["finetuned_forward_barrier_eV"])
    strict_abs_delta = as_float(row["strict_abs_forward_barrier_delta_eV"])
    if foundation_forward is None or finetuned_forward is None or strict_abs_delta is None:
        raise ValueError("Full-NEB scoring requires parsed forward barriers and strict delta")
    barrier_disagreement_term = min(strict_abs_delta, 1.5)
    kinetics_importance = 1.0 / (0.05 + max(min(foundation_forward, finetuned_forward), 0.0))
    kinetics_term = min(kinetics_importance, 10.0)
    defect_count = int(manifest_row["nsubs"]) + int(manifest_row["nvacs"])
    cost_penalty = 0.03 * defect_count
    score = 8.0 * barrier_disagreement_term + 4.0 * story_weight + 2.0 * kinetics_term - cost_penalty
    components = {
        "story_weight": story_weight,
        "barrier_disagreement_term": barrier_disagreement_term,
        "kinetics_term": kinetics_term,
        "cost_penalty": cost_penalty,
    }
    return score, components


def build_candidate_rows(
    paired_rows: list[dict[str, str]],
    manifest_index: dict[str, dict[str, str]],
    neb_batch_index: dict[tuple[str, str], dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in paired_rows:
        system_id = row["system_id"]
        manifest_row = manifest_index[system_id]
        action = ""
        if final_relax_only_eligible(row):
            action = "final_relax_only"
            score, components = score_final_relax_only(row, manifest_row)
        elif full_neb_eligible(row):
            action = "full_neb"
            score, components = score_full_neb(row, manifest_row)
        else:
            continue

        foundation_batch = neb_batch_index.get((row["event_pair_key"], "foundation"), {})
        finetuned_batch = neb_batch_index.get((row["event_pair_key"], "finetuned"), {})

        candidate_row: dict[str, Any] = {
            "dft_action": action,
            "event_pair_key": row["event_pair_key"],
            "system_id": row["system_id"],
            "formula": manifest_row["formula"],
            "natoms": int(manifest_row["natoms"]),
            "nsubs": int(manifest_row["nsubs"]),
            "nvacs": int(manifest_row["nvacs"]),
            "split": row["split"],
            "event_type": row["event_type"],
            "event_family": row["event_family"],
            "story_role": classify_story_role(row["event_type"]),
            "site_a": int(row["site_a"]),
            "site_b": int(row["site_b"]),
            "foundation_endpoint_label": row["foundation_endpoint_label"],
            "finetuned_endpoint_label": row["finetuned_endpoint_label"],
            "transition_label": row["transition_label"],
            "foundation_neb_status": row["foundation_neb_status"],
            "finetuned_neb_status": row["finetuned_neb_status"],
            "foundation_ready_for_neb": as_bool(row["foundation_ready_for_neb"]),
            "finetuned_ready_for_neb": as_bool(row["finetuned_ready_for_neb"]),
            "both_models_ready_for_neb": as_bool(row["both_models_ready_for_neb"]),
            "both_models_status_usable": as_bool(row["both_models_status_usable"]),
            "foundation_forward_barrier_eV": row["foundation_forward_barrier_eV"],
            "finetuned_forward_barrier_eV": row["finetuned_forward_barrier_eV"],
            "foundation_reverse_barrier_eV": row["foundation_reverse_barrier_eV"],
            "finetuned_reverse_barrier_eV": row["finetuned_reverse_barrier_eV"],
            "strict_abs_forward_barrier_delta_eV": row["strict_abs_forward_barrier_delta_eV"],
            "strict_abs_reverse_barrier_delta_eV": row["strict_abs_reverse_barrier_delta_eV"],
            "foundation_path_shape": row["foundation_path_shape"],
            "finetuned_path_shape": row["finetuned_path_shape"],
            "foundation_notes": row["foundation_notes"],
            "finetuned_notes": row["finetuned_notes"],
            "selection_score": round(score, 6),
            "score_story_weight": components["story_weight"],
            "score_cost_penalty": round(components["cost_penalty"], 6),
            "selection_reason": build_selection_reason(action, row),
            "foundation_source_init_data": foundation_batch.get("source_init_data", ""),
            "finetuned_source_init_data": finetuned_batch.get("source_init_data", ""),
            "foundation_source_final_relaxed_data": foundation_batch.get("source_final_relaxed_data", ""),
            "finetuned_source_final_relaxed_data": finetuned_batch.get("source_final_relaxed_data", ""),
            "foundation_live_job_dir": foundation_batch.get("live_job_dir", ""),
            "finetuned_live_job_dir": finetuned_batch.get("live_job_dir", ""),
        }

        if action == "final_relax_only":
            candidate_row["score_transition_severity"] = components["transition_severity"]
            candidate_row["score_readiness_bonus"] = components["readiness_bonus"]
            candidate_row["score_barrier_disagreement_term"] = ""
            candidate_row["score_kinetics_term"] = ""
        else:
            candidate_row["score_transition_severity"] = ""
            candidate_row["score_readiness_bonus"] = ""
            candidate_row["score_barrier_disagreement_term"] = round(
                components["barrier_disagreement_term"], 6
            )
            candidate_row["score_kinetics_term"] = round(components["kinetics_term"], 6)

        rows.append(candidate_row)

    return rows


def assign_ranks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_rows = sorted(
        rows,
        key=lambda r: (
            r["dft_action"],
            -float(r["selection_score"]),
            r["event_type"],
            r["system_id"],
            int(r["site_a"]),
            int(r["site_b"]),
        ),
    )
    action_rank = defaultdict(int)
    event_type_rank = defaultdict(int)
    for row in sorted_rows:
        action = str(row["dft_action"])
        event_type = str(row["event_type"])
        action_rank[action] += 1
        event_type_rank[(action, event_type)] += 1
        row["rank_within_action"] = action_rank[action]
        row["rank_within_action_event_type"] = event_type_rank[(action, event_type)]
        row["recommended_priority_tier"] = (
            "core" if event_type_rank[(action, event_type)] <= 2 else "extended"
        )
    return sorted_rows


def main() -> None:
    args = parse_args()
    paired_rows = read_csv_rows(args.paired_summary_csv)
    manifest_index = build_manifest_index(read_csv_rows(args.manifest_csv))
    neb_batch_index = build_neb_batch_index(read_csv_rows(args.neb_batch_manifest_csv))

    candidate_rows = build_candidate_rows(paired_rows, manifest_index, neb_batch_index)
    ranked_rows = assign_ranks(candidate_rows)
    final_relax_rows = [row for row in ranked_rows if row["dft_action"] == "final_relax_only"]
    full_neb_rows = [row for row in ranked_rows if row["dft_action"] == "full_neb"]

    write_csv(args.output_csv, ranked_rows)
    write_csv(args.final_relax_only_csv, final_relax_rows)
    write_csv(args.full_neb_csv, full_neb_rows)

    print(f"[OK] wrote combined candidate table: {args.output_csv.resolve()}")
    print(f"[OK] wrote final-relax-only candidates: {args.final_relax_only_csv.resolve()}")
    print(f"[OK] wrote full-NEB candidates: {args.full_neb_csv.resolve()}")
    print(f"[Candidates total] {len(ranked_rows)}")
    print(f"[Final relax only] {len(final_relax_rows)}")
    print(f"[Full NEB] {len(full_neb_rows)}")


if __name__ == "__main__":
    main()
