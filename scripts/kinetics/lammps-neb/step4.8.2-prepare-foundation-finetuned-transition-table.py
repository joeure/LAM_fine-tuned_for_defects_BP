#!/usr/bin/env python3
"""
Prepare an event-level foundation-vs-finetuned transition table for the
Panel-(c) heatmap data source described in BACKGROUND.md.

Input
-----
- data_center/step4.8_endpoint_basin_summary_formal_v1.csv

Output
------
- one row per unique event pair
- explicit foundation / finetuned classification labels
- disagreement flag
- lightweight event-family / motif tags for later heatmap-inset statistics

This script intentionally does not draw figures. It only writes a stable CSV
that later plotting scripts can consume directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


CLASSIFICATION_ORDER = [
    "same_basin",
    "likely_same_basin",
    "ambiguous",
    "other_basin",
    "different_basin",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data_center/step4.8_endpoint_basin_summary_formal_v1.csv"),
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data_center/step4.8.2_foundation_finetuned_transition_table_formal_v1.csv"),
    )
    return p.parse_args()


def require_columns(df: pd.DataFrame, required: List[str], path: Path) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")


def bool_from_row(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"Cannot parse boolean-like value: {value!r}")


def family_aczz(event_type: str) -> str:
    if event_type == "AC_hop":
        return "AC"
    if event_type == "ZZ_hop":
        return "ZZ"
    return "non_ACZZ"


def canonical_event_family(event_type: str) -> str:
    mapping = {
        "AC_hop": "far_AC",
        "ZZ_hop": "far_ZZ",
        "nearN_hop": "near_dopant",
        "trap_escape": "escape",
    }
    return mapping.get(event_type, "other")


def classification_rank(label: str) -> int:
    try:
        return CLASSIFICATION_ORDER.index(label)
    except ValueError:
        return len(CLASSIFICATION_ORDER)


def build_event_level_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    grouped = df.sort_values(["event_pair_key", "model_tag"]).groupby("event_pair_key", sort=True)

    for event_pair_key, group in grouped:
        model_rows = {str(row["model_tag"]): row for _, row in group.iterrows()}
        foundation = model_rows.get("foundation")
        finetuned = model_rows.get("finetuned")
        if foundation is None or finetuned is None:
            raise ValueError(f"Expected both foundation and finetuned rows for {event_pair_key}")

        foundation_label = str(foundation["preferred_classification"])
        finetuned_label = str(finetuned["preferred_classification"])
        event_type = str(foundation["event_type"])
        split = str(foundation["split"])
        system_id = str(foundation["system_id"])
        vac_site = int(foundation["vac_site_144"])
        neighbor_site = int(foundation["neighbor_site_144"])

        foundation_ready = bool_from_row(foundation["ready_for_neb"])
        finetuned_ready = bool_from_row(finetuned["ready_for_neb"])
        both_ready = bool_from_row(foundation["both_models_ready_for_neb"])
        either_ready = bool_from_row(foundation["either_model_ready_for_neb"])

        near_dopant = event_type in {"nearN_hop", "trap_escape"}
        escaping = event_type == "trap_escape"

        rows.append(
            {
                "event_pair_key": event_pair_key,
                "system_id": system_id,
                "split": split,
                "event_type": event_type,
                "event_family": canonical_event_family(event_type),
                "family_aczz": family_aczz(event_type),
                "is_near_dopant": near_dopant,
                "is_far_from_dopant": not near_dopant,
                "is_escaping": escaping,
                "is_non_escaping": not escaping,
                "vac_site_144": vac_site,
                "neighbor_site_144": neighbor_site,
                "site_a": int(foundation["site_a"]),
                "site_b": int(foundation["site_b"]),
                "foundation_label": foundation_label,
                "finetuned_label": finetuned_label,
                "foundation_label_rank": classification_rank(foundation_label),
                "finetuned_label_rank": classification_rank(finetuned_label),
                "is_disagree": foundation_label != finetuned_label,
                "transition_label": f"{foundation_label} -> {finetuned_label}",
                "foundation_ready_for_neb": foundation_ready,
                "finetuned_ready_for_neb": finetuned_ready,
                "both_models_ready_for_neb": both_ready,
                "either_model_ready_for_neb": either_ready,
                "foundation_archive_dir": str(foundation["archive_dir"]),
                "finetuned_archive_dir": str(finetuned["archive_dir"]),
                "foundation_job_name": str(foundation["job_name"]),
                "finetuned_job_name": str(finetuned["job_name"]),
                "foundation_archive_job_id": str(foundation["archive_job_id"]),
                "finetuned_archive_job_id": str(finetuned["archive_job_id"]),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["system_id", "vac_site_144", "neighbor_site_144"]
    ).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    require_columns(
        df,
        [
            "event_pair_key",
            "system_id",
            "model_tag",
            "split",
            "event_type",
            "vac_site_144",
            "neighbor_site_144",
            "site_a",
            "site_b",
            "preferred_classification",
            "ready_for_neb",
            "both_models_ready_for_neb",
            "either_model_ready_for_neb",
            "archive_dir",
            "archive_job_id",
            "job_name",
        ],
        args.input_csv,
    )

    event_df = build_event_level_table(df)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    event_df.to_csv(args.output_csv, index=False)

    print(f"[OK] wrote event-level transition table: {args.output_csv.resolve()}")
    print(f"[Summary] rows: {len(event_df)}")
    print(f"[Summary] disagree rows: {int(event_df['is_disagree'].sum())}")
    print(f"[Summary] unique transitions: {event_df['transition_label'].nunique()}")


if __name__ == "__main__":
    main()
