#!/usr/bin/env python3
"""Build a compact high-value DFT shortlist from the execution plan tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


EVENT_TYPES = ["AC_hop", "ZZ_hop", "nearN_hop", "trap_escape"]


def parse_quota_map(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        key, value = item.split("=", 1)
        out[key.strip()] = int(value.strip())
    missing = [event_type for event_type in EVENT_TYPES if event_type not in out]
    if missing:
        raise ValueError(f"Missing quota for event types: {missing}")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase1-plan-csv",
        type=Path,
        default=Path("data_center/step4.8.10_phase1_all_final_relax_submission_plan_formal_v1.csv"),
    )
    parser.add_argument(
        "--phase2-plan-csv",
        type=Path,
        default=Path("data_center/step4.8.10_phase2_conditional_full_neb_submission_plan_formal_v1.csv"),
    )
    parser.add_argument(
        "--phase2-quotas",
        default="AC_hop=2,ZZ_hop=2,nearN_hop=2,trap_escape=2",
        help="Per-event-type quotas for the first-wave full-NEB shortlist.",
    )
    parser.add_argument(
        "--relax-only-quotas",
        default="AC_hop=1,ZZ_hop=1,nearN_hop=2,trap_escape=2",
        help="Per-event-type quotas for additional phase-1-only endpoint-adjudication tasks.",
    )
    parser.add_argument(
        "--phase1-shortlist-csv",
        type=Path,
        default=Path("data_center/step4.8.11_phase1_first_wave_dft_shortlist_formal_v1.csv"),
    )
    parser.add_argument(
        "--phase2-shortlist-csv",
        type=Path,
        default=Path("data_center/step4.8.11_phase2_first_wave_dft_shortlist_formal_v1.csv"),
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


def select_by_quota(rows: list[dict[str, str]], quotas: dict[str, int]) -> list[dict[str, str]]:
    picked: list[dict[str, str]] = []
    counts = {event_type: 0 for event_type in EVENT_TYPES}
    for row in rows:
        event_type = row["event_type"]
        if counts[event_type] >= quotas[event_type]:
            continue
        picked.append(row)
        counts[event_type] += 1
        if all(counts[et] >= quotas[et] for et in EVENT_TYPES):
            break
    return picked


def main() -> None:
    args = parse_args()
    phase2_quotas = parse_quota_map(args.phase2_quotas)
    relax_only_quotas = parse_quota_map(args.relax_only_quotas)

    phase1_rows = read_csv_rows(args.phase1_plan_csv)
    phase2_rows = read_csv_rows(args.phase2_plan_csv)

    phase2_shortlist = select_by_quota(phase2_rows, phase2_quotas)
    phase2_event_keys = {row["event_pair_key"] for row in phase2_shortlist}

    relax_only_rows = [row for row in phase1_rows if row["dft_action"] == "final_relax_only"]
    relax_only_shortlist = select_by_quota(relax_only_rows, relax_only_quotas)

    phase1_index = {row["event_pair_key"]: row for row in phase1_rows}
    phase1_shortlist: list[dict[str, Any]] = []

    for row in phase2_shortlist:
        phase1_row = dict(phase1_index[row["event_pair_key"]])
        phase1_row["shortlist_bucket"] = "phase1_support_for_phase2_full_neb"
        phase1_row["shortlist_reason"] = (
            "First-wave final relaxation for a top-priority DFT-NEB seed event."
        )
        phase1_shortlist.append(phase1_row)

    for row in relax_only_shortlist:
        if row["event_pair_key"] in phase2_event_keys:
            continue
        phase1_row = dict(row)
        phase1_row["shortlist_bucket"] = "phase1_endpoint_adjudication_only"
        phase1_row["shortlist_reason"] = (
            "High-value endpoint disagreement chosen to clarify basin/topology with DFT final relaxation only."
        )
        phase1_shortlist.append(phase1_row)

    phase1_shortlist.sort(
        key=lambda r: (
            0 if r["shortlist_bucket"] == "phase1_support_for_phase2_full_neb" else 1,
            r["event_type"],
            int(r["rank_within_action_event_type"]),
            int(r["rank_within_action"]),
        )
    )

    phase2_shortlist_out: list[dict[str, Any]] = []
    for row in phase2_shortlist:
        out = dict(row)
        out["shortlist_bucket"] = "phase2_formal_neb_first_wave"
        out["shortlist_reason"] = (
            "Top-priority full DFT-NEB candidate after phase-1 final relaxation passes basin validation."
        )
        phase2_shortlist_out.append(out)

    write_csv(args.phase1_shortlist_csv, phase1_shortlist)
    write_csv(args.phase2_shortlist_csv, phase2_shortlist_out)

    print(f"[OK] wrote phase-1 shortlist: {args.phase1_shortlist_csv.resolve()}")
    print(f"[OK] wrote phase-2 shortlist: {args.phase2_shortlist_csv.resolve()}")
    print(f"[Phase 1 shortlist rows] {len(phase1_shortlist)}")
    print(f"[Phase 2 shortlist rows] {len(phase2_shortlist_out)}")


if __name__ == "__main__":
    main()
