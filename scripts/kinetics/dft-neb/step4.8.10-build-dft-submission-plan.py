#!/usr/bin/env python3
"""Build execution-oriented DFT submission plan tables.

This converts the story-driven DFT candidate tables into two practical planning
tables:

1. Phase 1: all selected final-state relaxations
2. Phase 2: conditional full-NEB seeds that should only be submitted after the
   corresponding Phase-1 relaxed final endpoint is validated
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-csv",
        type=Path,
        default=Path("data_center/step4.8.9_dft_calibration_candidates_formal_v1.csv"),
    )
    parser.add_argument(
        "--phase1-csv",
        type=Path,
        default=Path("data_center/step4.8.10_phase1_all_final_relax_submission_plan_formal_v1.csv"),
    )
    parser.add_argument(
        "--phase2-csv",
        type=Path,
        default=Path("data_center/step4.8.10_phase2_conditional_full_neb_submission_plan_formal_v1.csv"),
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


def choose_preferred_endpoint_guess_model(row: dict[str, str]) -> str:
    fin_ready = row["finetuned_ready_for_neb"] == "True"
    fdn_ready = row["foundation_ready_for_neb"] == "True"
    if fin_ready and not fdn_ready:
        return "finetuned"
    if fdn_ready and not fin_ready:
        return "foundation"
    if row["finetuned_endpoint_label"] == "different_basin" and row["foundation_endpoint_label"] != "different_basin":
        return "finetuned"
    if row["foundation_endpoint_label"] == "different_basin" and row["finetuned_endpoint_label"] != "different_basin":
        return "foundation"
    return "finetuned"


def choose_source_path(row: dict[str, str], model_tag: str, kind: str) -> str:
    if model_tag == "foundation":
        return row[f"foundation_source_{kind}"]
    return row[f"finetuned_source_{kind}"]


def build_phase1_rows(candidate_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(candidate_rows, start=1):
        preferred_model = choose_preferred_endpoint_guess_model(row)
        source_init = choose_source_path(row, preferred_model, "init_data")
        source_final = choose_source_path(row, preferred_model, "final_relaxed_data")
        rows.append(
            {
                "phase": "phase1_final_relax",
                "submission_rank": idx,
                "dft_action": row["dft_action"],
                "event_pair_key": row["event_pair_key"],
                "system_id": row["system_id"],
                "formula": row["formula"],
                "event_type": row["event_type"],
                "event_family": row["event_family"],
                "story_role": row["story_role"],
                "site_a": row["site_a"],
                "site_b": row["site_b"],
                "recommended_priority_tier": row["recommended_priority_tier"],
                "rank_within_action": row["rank_within_action"],
                "rank_within_action_event_type": row["rank_within_action_event_type"],
                "selection_score": row["selection_score"],
                "preferred_endpoint_guess_model": preferred_model,
                "preferred_source_init_data": source_init,
                "preferred_source_final_relaxed_data": source_final,
                "foundation_endpoint_label": row["foundation_endpoint_label"],
                "finetuned_endpoint_label": row["finetuned_endpoint_label"],
                "transition_label": row["transition_label"],
                "phase1_goal": (
                    "Adjudicate final basin / topology only"
                    if row["dft_action"] == "final_relax_only"
                    else "Relax final endpoint for later formal DFT-NEB"
                ),
                "phase2_followup": (
                    "stop_after_phase1"
                    if row["dft_action"] == "final_relax_only"
                    else "eligible_for_phase2_if_relaxed_final_basin_is_valid"
                ),
                "selection_reason": row["selection_reason"],
            }
        )
    return rows


def build_phase2_rows(candidate_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    full_neb_rows = [row for row in candidate_rows if row["dft_action"] == "full_neb"]
    phase2_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(full_neb_rows, start=1):
        preferred_model = choose_preferred_endpoint_guess_model(row)
        phase2_rows.append(
            {
                "phase": "phase2_formal_dft_neb",
                "submission_rank": idx,
                "event_pair_key": row["event_pair_key"],
                "system_id": row["system_id"],
                "formula": row["formula"],
                "event_type": row["event_type"],
                "event_family": row["event_family"],
                "story_role": row["story_role"],
                "site_a": row["site_a"],
                "site_b": row["site_b"],
                "recommended_priority_tier": row["recommended_priority_tier"],
                "rank_within_action": row["rank_within_action"],
                "rank_within_action_event_type": row["rank_within_action_event_type"],
                "selection_score": row["selection_score"],
                "preferred_endpoint_guess_model": preferred_model,
                "phase1_dependency": "Must finish corresponding Phase-1 final relaxation first",
                "phase2_submit_if": (
                    "DFT final relax converges to the intended distinct final basin and does not collapse to the initial basin"
                ),
                "strict_abs_forward_barrier_delta_eV": row["strict_abs_forward_barrier_delta_eV"],
                "strict_abs_reverse_barrier_delta_eV": row["strict_abs_reverse_barrier_delta_eV"],
                "foundation_forward_barrier_eV": row["foundation_forward_barrier_eV"],
                "finetuned_forward_barrier_eV": row["finetuned_forward_barrier_eV"],
                "foundation_path_shape": row["foundation_path_shape"],
                "finetuned_path_shape": row["finetuned_path_shape"],
                "selection_reason": row["selection_reason"],
            }
        )
    return phase2_rows


def main() -> None:
    args = parse_args()
    candidate_rows = read_csv_rows(args.candidate_csv)
    phase1_rows = build_phase1_rows(candidate_rows)
    phase2_rows = build_phase2_rows(candidate_rows)
    write_csv(args.phase1_csv, phase1_rows)
    write_csv(args.phase2_csv, phase2_rows)

    print(f"[OK] wrote phase-1 final-relax plan: {args.phase1_csv.resolve()}")
    print(f"[OK] wrote phase-2 full-NEB plan: {args.phase2_csv.resolve()}")
    print(f"[Phase 1 rows] {len(phase1_rows)}")
    print(f"[Phase 2 rows] {len(phase2_rows)}")


if __name__ == "__main__":
    main()
