#!/usr/bin/env python3
"""
Prepare Stage 4.8 shard-based LAMMPS-MACE NEB submission roots from validated
endpoint-relaxation results.

This script reads the Stage 4.8 basin-summary CSV, selects only events whose
foundation and finetuned endpoint relaxations are both classified as
`different_basin`, then builds Bohrium-ready shard directories with:

- many independent per-event NEB jobs under `jobs/`
- one shared model file per shard under `models/`
- a top-level `run.sh` that dispatches the per-event jobs
- shard and event manifests for later submission / archival / analysis

The implementation intentionally follows the successful Stage 4.7 single-job
NEB layout for each event, but uses the Stage 4.8 shard layout instead of the
older in-bundle multi-NEB mode.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd


DEFAULT_BATCH_JOB_JSON = {
    "job_type": "container",
    "command": "chmod +x ./run.sh && ./run.sh",
    "backward_files": [
        "jobs",
        "logs",
        "meta",
        "run.sh",
        "job.json",
        "submit.log",
    ],
    "project_id": 0,
    "platform": "<CLOUD_PLATFORM>",
    "machine_type": "<EXAMPLE_MACHINE_TYPE>",
    "image_address": "<LAMMPS_MACE_IMAGE_ADDRESS>",
    "max_reschedule_times": 2,
}


@dataclass
class BatchShardInfo:
    phase: str
    model_tag: str
    batch_label: str
    selection_rule: str
    shard_index_1based: int
    nshards_total: int
    njobs_in_shard: int
    npairs_represented: int
    jobs_root: str
    run_sh_path: str
    job_json_path: str
    gpus_total: int
    jobs_per_gpu: int
    max_parallel_jobs: int
    images: int
    total_ranks_per_job: int
    ranks_per_replica: int
    lmp_bin: str
    kokkos: str
    note: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--basin-summary-csv",
        type=Path,
        default=Path("data_center/step4.8_endpoint_basin_summary_formal_v1.csv"),
    )
    p.add_argument(
        "--selection-column",
        default="both_models_ready_for_neb",
        help="Boolean column used to select event rows from the summary CSV.",
    )
    p.add_argument(
        "--selection-value",
        default="True",
        help="String value matched after stringification. Default selects rows with both_models_ready_for_neb == True.",
    )
    p.add_argument(
        "--model-tags",
        nargs="+",
        choices=["foundation", "finetuned"],
        default=["foundation", "finetuned"],
    )
    p.add_argument("--batch-label", default="tier1_event_list_graph_topo_n2nn_formal_v1")
    p.add_argument("--phase", default="cineb")
    p.add_argument("--out-root", type=Path, default=Path("lammps_neb_workspace/batch_run"))
    p.add_argument("--jobs-per-shard", type=int, default=56)
    p.add_argument(
        "--shard-indices",
        type=int,
        nargs="*",
        default=None,
        help="Optional 1-based shard indices per model group to materialize.",
    )
    p.add_argument("--limit-pairs", type=int, default=None)
    p.add_argument("--gpus-total", type=int, default=1)
    p.add_argument("--jobs-per-gpu", type=int, default=1)
    p.add_argument("--machine-type", default=DEFAULT_BATCH_JOB_JSON["machine_type"])
    p.add_argument("--image-address", default=DEFAULT_BATCH_JOB_JSON["image_address"])
    p.add_argument("--project-id", type=int, default=DEFAULT_BATCH_JOB_JSON["project_id"])
    p.add_argument("--platform", default=DEFAULT_BATCH_JOB_JSON["platform"])
    p.add_argument("--max-reschedule-times", type=int, default=DEFAULT_BATCH_JOB_JSON["max_reschedule_times"])
    p.add_argument("--image-mode", choices=["lammps_interpolate", "explicit_images"], default="explicit_images")
    p.add_argument("--images", type=int, default=7)
    p.add_argument("--spring-constant", type=float, default=1.0)
    p.add_argument("--parallel-mode", choices=["neigh"], default="neigh")
    p.add_argument("--perp-spring", type=float, default=0.0)
    p.add_argument("--neb-type", choices=["ci-neb", "plain-neb"], default="ci-neb")
    p.add_argument("--etol", type=float, default=0.0)
    p.add_argument("--ftol", type=float, default=0.02)
    p.add_argument("--n1-steps", type=int, default=3000)
    p.add_argument("--n2-steps", type=int, default=3000)
    p.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="Shared NEB cadence parameter used for thermo, dump, and the 5th integer argument of the LAMMPS neb command. Current finetuned-safe default is 1.",
    )
    p.add_argument("--min-style", default="fire")
    p.add_argument("--timestep", type=float, default=0.01)
    p.add_argument("--mpirun-bin", default="mpirun")
    p.add_argument("--lmp-bin", default="lmp")
    p.add_argument("--allow-mpirun-as-root", action="store_true")
    p.add_argument("--ranks-per-replica", type=int, default=1)
    p.add_argument("--total-ranks", type=int, default=None)
    p.add_argument("--kokkos", choices=["on", "off"], default="off")
    p.add_argument("--gpus", type=int, default=0)
    p.add_argument("--suffix", default="kk")
    p.add_argument("--env-script", default="")
    p.add_argument("--venv-activate", default="")
    p.add_argument("--manifest-out", type=Path, default=Path("data_center/step4.8.1_lammps_mace_neb_batch_manifest_formal_v1.csv"))
    p.add_argument("--force-overwrite-shard", action="store_true")
    return p.parse_args()


def sanitize_name(text: str) -> str:
    return text.replace("/", "_").replace(" ", "_").replace(":", "_")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_step47_module(script_dir: Path) -> ModuleType:
    path = script_dir / "step4.7-prepare-lammps-mace-neb.py"
    spec = importlib.util.spec_from_file_location("step47_prepare_lammps_mace_neb", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def chunk_dataframe(df: pd.DataFrame, chunk_size: int) -> List[pd.DataFrame]:
    if chunk_size <= 0:
        raise ValueError("--jobs-per-shard must be > 0")
    return [df.iloc[i : i + chunk_size].copy() for i in range(0, len(df), chunk_size)]


def build_shard_name(phase: str, model_tag: str, batch_label: str, shard_idx: int, nshards: int) -> str:
    return sanitize_name(
        f"{phase}__{model_tag}__{batch_label}__shard_{shard_idx:03d}_of_{nshards:03d}"
    )


def batch_job_json(args: argparse.Namespace, shard_name: str) -> Dict[str, Any]:
    payload = dict(DEFAULT_BATCH_JOB_JSON)
    payload["job_name"] = shard_name
    payload["project_id"] = args.project_id
    payload["platform"] = args.platform
    payload["machine_type"] = args.machine_type
    payload["image_address"] = args.image_address
    payload["max_reschedule_times"] = args.max_reschedule_times
    return payload


def build_dispatch_run_sh(job_names: Sequence[str], gpus_total: int, jobs_per_gpu: int) -> str:
    max_parallel = max(1, gpus_total * jobs_per_gpu)
    gpu_count = max(1, gpus_total)
    job_entries = "\n".join(f'  "{name}"' for name in job_names)
    return f"""#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"
COMPLETED_LIST="$LOG_DIR/completed_jobs.txt"
FAILED_LIST="$LOG_DIR/failed_jobs.txt"
SKIPPED_LIST="$LOG_DIR/skipped_jobs.txt"
: > "$COMPLETED_LIST"
: > "$FAILED_LIST"
: > "$SKIPPED_LIST"

GPU_COUNT="${{GPU_COUNT:-{gpu_count}}}"
JOBS_PER_GPU="${{JOBS_PER_GPU:-{jobs_per_gpu}}}"
MAX_PARALLEL_JOBS="${{MAX_PARALLEL_JOBS:-{max_parallel}}}"
SLEEP_SECS="${{SLEEP_SECS:-5}}"
USABLE_COMPLETE_LIST="$ROOT_DIR/meta/usable_complete_tasks.txt"

JOB_NAMES=(
{job_entries}
)

launch_job() {{
  local index="$1"
  local job_name="${{JOB_NAMES[$index]}}"
  local gpu_id=$(( index % GPU_COUNT ))
  (
    export CUDA_VISIBLE_DEVICES="$gpu_id"
    cd "$ROOT_DIR/jobs/$job_name"
    if [ -s "$USABLE_COMPLETE_LIST" ] && grep -Fxq "$job_name" "$USABLE_COMPLETE_LIST"; then
      echo "$job_name" >> "$SKIPPED_LIST"
      exit 0
    fi
    bash run.sh > "$LOG_DIR/${{job_name}}.log" 2>&1
  ) &
  PIDS+=("$!")
  PID_TO_JOB["$!"]="$job_name"
}}

declare -a PIDS=()
declare -A PID_TO_JOB=()

for idx in "${{!JOB_NAMES[@]}}"; do
  while (( $(jobs -pr | wc -l | tr -d ' ') >= MAX_PARALLEL_JOBS )); do
    sleep "$SLEEP_SECS"
  done
  launch_job "$idx"
done

overall_rc=0
for pid in "${{PIDS[@]}}"; do
  job_name="${{PID_TO_JOB[$pid]}}"
  if wait "$pid"; then
    if [ -s "$USABLE_COMPLETE_LIST" ] && grep -Fxq "$job_name" "$USABLE_COMPLETE_LIST"; then
      echo "$job_name" >> "$COMPLETED_LIST"
    elif [ -s "$ROOT_DIR/jobs/$job_name/log.neb" ] && grep -q "NEB finished" "$ROOT_DIR/jobs/$job_name/log.neb"; then
      echo "$job_name" >> "$COMPLETED_LIST"
    else
      echo "$job_name" >> "$SKIPPED_LIST"
    fi
  else
    echo "$job_name" >> "$FAILED_LIST"
    overall_rc=1
  fi
done

exit "$overall_rc"
"""


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def bool_string(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def source_live_job_dir(row: pd.Series) -> Path:
    return Path(row["live_shard_dir"]) / str(row["shard_name"]) / "jobs" / str(row["job_name"])


def source_archive_final_data(row: pd.Series) -> Path:
    return Path(row["archive_dir"]) / "results" / f"{row['job_name']}__final_relaxed.data"


def resolve_model_src(meta: Dict[str, Any]) -> Path:
    job_info = meta.get("job_info", {})
    model_file_input = job_info.get("model_file_input")
    if not model_file_input:
        raise ValueError("source meta.json missing job_info.model_file_input")
    path = Path(model_file_input)
    if not path.exists():
        raise FileNotFoundError(f"Referenced model file does not exist: {path}")
    return path


def select_rows(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if args.selection_column not in df.columns:
        raise ValueError(f"Missing selection column in {args.basin_summary_csv}: {args.selection_column}")
    out = df[df[args.selection_column].astype(str) == args.selection_value].copy()
    out = out[out["model_tag"].isin(args.model_tags)].copy()
    out = out[out["ready_for_neb"].astype(bool)].copy()
    out = out.sort_values(["model_tag", "system_id", "site_a", "site_b"]).reset_index(drop=True)
    if args.limit_pairs is not None:
        pair_keys = sorted(out["event_pair_key"].drop_duplicates())[: args.limit_pairs]
        out = out[out["event_pair_key"].isin(pair_keys)].copy().reset_index(drop=True)
    return out


def validate_sources(df: pd.DataFrame) -> None:
    missing: List[str] = []
    for _, row in df.iterrows():
        live_job_dir = source_live_job_dir(row)
        final_data = source_archive_final_data(row)
        meta_path = live_job_dir / "meta" / "meta.json"
        init_path = live_job_dir / "data" / "init.data"
        if not live_job_dir.exists():
            missing.append(str(live_job_dir))
        if not meta_path.exists():
            missing.append(str(meta_path))
        if not init_path.exists():
            missing.append(str(init_path))
        if not final_data.exists():
            missing.append(str(final_data))
    if missing:
        raise FileNotFoundError("Missing required source files:\n" + "\n".join(missing[:20]))


def build_prepare_status(
    source_job_dir: Path,
    source_archive_dir: Path,
    source_final_relaxed: Path,
    source_job_info: Dict[str, Any],
    args: argparse.Namespace,
    effective_gpus: int,
    basin_status: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    return {
        "source_job_dir": str(source_job_dir),
        "source_archive_dir": str(source_archive_dir),
        "source_final_relaxed_data": str(source_final_relaxed),
        "source_stage": "stage4.8_endpoint_relaxation",
        "system_id": source_job_info.get("system_id"),
        "model_tag": source_job_info.get("model_tag"),
        "site_a": source_job_info.get("site_a"),
        "site_b": source_job_info.get("site_b"),
        "image_mode": args.image_mode,
        "package_mode": "batch_shared_model",
        "neb_type": args.neb_type,
        "nreplicas_total": args.images,
        "nimages_intermediate": args.images - 2,
        "kokkos": args.kokkos,
        "gpus": effective_gpus,
        "allow_mpirun_as_root": bool(args.allow_mpirun_as_root),
        "prepared": True,
        "output_dir": str(output_dir),
        "reason": "Prepared NEB job under shared Stage 4.8 shard root.",
        "allow_unvalidated_final": False,
        "readiness": basin_status,
    }


def prepare_single_neb_job(
    step47: ModuleType,
    row: pd.Series,
    shard_root: Path,
    args: argparse.Namespace,
    total_ranks: int,
    requested_gpus: int,
    effective_min_style: str,
    model_relpath: str,
) -> Dict[str, Any]:
    live_job_dir = source_live_job_dir(row)
    archive_dir = Path(row["archive_dir"])
    final_relaxed_path = source_archive_final_data(row)
    meta = step47.read_json(live_job_dir / "meta" / "meta.json")
    source_job_info = meta["job_info"]
    elements = meta.get("elements_specorder", ["N", "P"])
    basin_status = step47.resolve_basin_status(live_job_dir)

    init_atoms = step47.read_lammps_atomic_data(live_job_dir / "data" / "init.data")
    final_atoms = step47.read_lammps_atomic_data(final_relaxed_path)
    step47.apply_elements_specorder(init_atoms, elements)
    step47.apply_elements_specorder(final_atoms, elements)
    step47.validate_finite_positions(init_atoms, "init endpoint")
    step47.validate_finite_positions(final_atoms, "final relaxed endpoint")
    aligned_final_atoms, mic_displacements = step47.align_final_endpoint_to_init(init_atoms, final_atoms)

    neb_job_name = step47.build_neb_job_name(
        source_job_info["system_id"],
        int(source_job_info["site_a"]),
        int(source_job_info["site_b"]),
        source_job_info["model_tag"],
        args.neb_type,
        "",
    )

    job_dir = ensure_dir(shard_root / "jobs" / neb_job_name)
    ensure_dir(job_dir / "data")
    ensure_dir(job_dir / "results")
    ensure_dir(job_dir / "dumps")
    meta_dir = ensure_dir(job_dir / "meta")

    shutil.copy2(live_job_dir / "data" / "init.data", job_dir / "data" / "init.data")
    step47.write_lammps_data_with_specorder(
        aligned_final_atoms,
        job_dir / "data" / "final_relaxed.data",
        specorder=elements,
    )

    replica_prefix_relpath = "data/replicas/replica_"
    if args.image_mode == "explicit_images":
        replicas = step47.interpolate_replicas(init_atoms, aligned_final_atoms, args.images)
        for idx, replica in enumerate(replicas):
            outpath = job_dir / "data" / "replicas" / f"replica_{idx:02d}.data"
            step47.write_lammps_data_with_specorder(replica, outpath, specorder=elements)

    in_file = job_dir / "in.neb.lammps"
    in_file.write_text(
        step47.render_single_input(
            image_mode=args.image_mode,
            nreplicas=args.images,
            final_data_relpath="data/final_relaxed.data",
            replica_prefix_relpath=replica_prefix_relpath,
            model_relpath=model_relpath,
            kokkos=args.kokkos,
            elements=elements,
            spring_constant=args.spring_constant,
            parallel_mode=args.parallel_mode,
            perp_spring=args.perp_spring,
            etol=args.etol,
            ftol=args.ftol,
            n1_steps=args.n1_steps,
            n2_steps=args.n2_steps if args.neb_type == "ci-neb" else 0,
            print_every=args.print_every,
            min_style=effective_min_style,
            timestep=args.timestep,
        ),
        encoding="utf-8",
    )

    step47.write_single_run_script(
        run_path=job_dir / "run.sh",
        mpirun_bin=args.mpirun_bin,
        lmp_bin=args.lmp_bin,
        input_relpath="in.neb.lammps",
        total_ranks=total_ranks,
        nreplicas=args.images,
        ranks_per_replica=args.ranks_per_replica,
        kokkos=args.kokkos,
        gpus=requested_gpus,
        allow_mpirun_as_root=args.allow_mpirun_as_root,
        suffix=args.suffix,
        env_script=args.env_script,
        venv_activate=args.venv_activate,
    )

    prepare_status = build_prepare_status(
        source_job_dir=live_job_dir,
        source_archive_dir=archive_dir,
        source_final_relaxed=final_relaxed_path,
        source_job_info=source_job_info,
        args=args,
        effective_gpus=requested_gpus,
        basin_status=basin_status,
        output_dir=job_dir,
    )

    step47.write_json(
        meta_dir / "meta.json",
        {
            "neb_job_info": {
                "source_job_dir": str(live_job_dir),
                "source_archive_dir": str(archive_dir),
                "source_final_relaxed_data": str(final_relaxed_path),
                "source_stage": "stage4.8_endpoint_relaxation",
                "system_id": source_job_info["system_id"],
                "model_tag": source_job_info["model_tag"],
                "package_mode": "batch_shared_model",
                "batch_name": shard_root.name,
                "image_mode": args.image_mode,
                "neb_job_name": neb_job_name,
                "site_a": int(source_job_info["site_a"]),
                "site_b": int(source_job_info["site_b"]),
                "moving_atom_index_0based": int(source_job_info["moving_atom_index_0based"]),
                "moving_atom_element": str(source_job_info["moving_atom_element"]),
                "init_data": "data/init.data",
                "final_relaxed_data": "data/final_relaxed.data",
                "model_file_input": str(resolve_model_src(meta)),
                "model_file_written_path": model_relpath,
                "output_dir": str(job_dir),
                "nreplicas_total": args.images,
                "nimages_intermediate": args.images - 2,
                "neb_type": args.neb_type,
                "spring_constant": args.spring_constant,
                "parallel_mode": args.parallel_mode,
                "perp_spring": args.perp_spring,
                "etol": args.etol,
                "ftol": args.ftol,
                "n1_steps": args.n1_steps,
                "n2_steps": args.n2_steps if args.neb_type == "ci-neb" else 0,
                "print_every": args.print_every,
                "min_style": effective_min_style,
                "timestep": args.timestep,
                "kokkos": args.kokkos,
                "gpus": requested_gpus,
                "allow_mpirun_as_root": args.allow_mpirun_as_root,
                "total_ranks": total_ranks,
                "ranks_per_replica": args.ranks_per_replica,
                "note": "Stage 4.8 shard NEB job with one shared model per shard.",
            },
            "source_job_info": source_job_info,
            "source_move_meta": meta.get("move_meta", {}),
            "source_manifest_row": meta.get("manifest_row", {}),
            "elements_specorder": elements,
            "pbc_alignment": {
                "aligned_final_written": "data/final_relaxed.data",
                "alignment_rule": "per-atom minimum-image displacement relative to init endpoint",
                "max_abs_mic_displacement_A": float(step47.np.max(step47.np.abs(mic_displacements))),
            },
            "readiness": basin_status,
        },
    )
    step47.write_json(meta_dir / "neb_prepare_status.json", prepare_status)

    return {
        "job_name": str(row["job_name"]),
        "neb_job_name": neb_job_name,
        "event_pair_key": str(row["event_pair_key"]),
        "system_id": str(row["system_id"]),
        "model_tag": str(row["model_tag"]),
        "structure": str(row["structure"]),
        "split": str(row["split"]),
        "event_type": str(row["event_type"]),
        "vac_site_144": int(row["vac_site_144"]),
        "neighbor_site_144": int(row["neighbor_site_144"]),
        "site_a": int(row["site_a"]),
        "site_b": int(row["site_b"]),
        "moving_atom_index_0based": int(row["moving_atom_index_0based"]),
        "moving_atom_element": str(row["moving_atom_element"]),
        "archive_dir": str(archive_dir),
        "live_job_dir": str(live_job_dir),
        "source_final_relaxed_data": str(final_relaxed_path),
        "source_init_data": str(live_job_dir / "data" / "init.data"),
        "shard_name": shard_root.name,
        "shard_dir": str(shard_root.resolve()),
        "job_dir": str(job_dir.resolve()),
        "model_relpath": model_relpath,
        "preferred_classification": str(row["preferred_classification"]),
        "paired_model_tag": str(row["paired_model_tag"]),
        "paired_job_name": str(row["paired_job_name"]),
        "paired_archive_dir": str(row["paired_archive_dir"]),
        "paired_preferred_classification": str(row["paired_preferred_classification"]),
        "both_models_ready_for_neb": bool(row["both_models_ready_for_neb"]),
    }


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    step47 = load_step47_module(script_dir)

    if args.images < 2:
        raise ValueError("--images must be at least 2")

    total_ranks = args.total_ranks
    if total_ranks is None:
        total_ranks = args.images * args.ranks_per_replica
    step47.validate_total_ranks(total_ranks, args.images, args.ranks_per_replica)
    requested_gpus = step47.resolve_requested_gpus(args.kokkos, args.gpus)
    effective_min_style = step47.resolve_effective_min_style(args.kokkos, args.min_style)

    summary_df = pd.read_csv(args.basin_summary_csv)
    selected_df = select_rows(summary_df, args)
    if selected_df.empty:
        raise ValueError("No rows selected for NEB preparation")
    validate_sources(selected_df)

    manifest_rows: List[Dict[str, Any]] = []

    for model_tag in args.model_tags:
        model_df = selected_df[selected_df["model_tag"] == model_tag].copy().reset_index(drop=True)
        if model_df.empty:
            continue

        chunks = chunk_dataframe(model_df, args.jobs_per_shard)
        nshards = len(chunks)
        requested_shards = None
        if args.shard_indices:
            requested_shards = sorted(set(args.shard_indices))
            bad = [idx for idx in requested_shards if idx < 1 or idx > nshards]
            if bad:
                raise ValueError(f"--shard-indices out of range for model={model_tag}, nshards={nshards}: {bad}")

        first_meta = step47.read_json(source_live_job_dir(model_df.iloc[0]) / "meta" / "meta.json")
        model_src = resolve_model_src(first_meta).resolve()

        for shard_idx, chunk in enumerate(chunks, start=1):
            if requested_shards is not None and shard_idx not in requested_shards:
                continue

            shard_name = build_shard_name(args.phase, model_tag, args.batch_label, shard_idx, nshards)
            shard_root = args.out_root / shard_name
            if shard_root.exists() and args.force_overwrite_shard:
                shutil.rmtree(shard_root)
            if shard_root.exists():
                raise FileExistsError(
                    f"Shard directory already exists: {shard_root}. Use --force-overwrite-shard to replace it."
                )

            ensure_dir(shard_root / "jobs")
            ensure_dir(shard_root / "logs")
            ensure_dir(shard_root / "meta")
            ensure_dir(shard_root / "results")
            ensure_dir(shard_root / "dumps")
            models_dir = ensure_dir(shard_root / "models")
            shared_model_path = models_dir / model_src.name
            shutil.copy2(model_src, shared_model_path)
            model_relpath = f"../../models/{model_src.name}"

            shard_manifest_rows: List[Dict[str, Any]] = []
            job_names: List[str] = []
            for _, row in chunk.iterrows():
                prepared = prepare_single_neb_job(
                    step47=step47,
                    row=row,
                    shard_root=shard_root,
                    args=args,
                    total_ranks=total_ranks,
                    requested_gpus=requested_gpus,
                    effective_min_style=effective_min_style,
                    model_relpath=model_relpath,
                )
                shard_manifest_rows.append(prepared)
                manifest_rows.append(prepared)
                job_names.append(prepared["neb_job_name"])

            (shard_root / "run.sh").write_text(
                build_dispatch_run_sh(
                    job_names=job_names,
                    gpus_total=args.gpus_total,
                    jobs_per_gpu=args.jobs_per_gpu,
                ),
                encoding="utf-8",
            )
            (shard_root / "run.sh").chmod(0o755)
            write_json(shard_root / "job.json", batch_job_json(args, shard_name))

            shard_info = BatchShardInfo(
                phase=args.phase,
                model_tag=model_tag,
                batch_label=args.batch_label,
                selection_rule=f"{args.selection_column} == {args.selection_value}",
                shard_index_1based=shard_idx,
                nshards_total=nshards,
                njobs_in_shard=len(job_names),
                npairs_represented=len(set(row["event_pair_key"] for row in shard_manifest_rows)),
                jobs_root=str((shard_root / "jobs").resolve()),
                run_sh_path=str((shard_root / "run.sh").resolve()),
                job_json_path=str((shard_root / "job.json").resolve()),
                gpus_total=args.gpus_total,
                jobs_per_gpu=args.jobs_per_gpu,
                max_parallel_jobs=max(1, args.gpus_total * args.jobs_per_gpu),
                images=args.images,
                total_ranks_per_job=total_ranks,
                ranks_per_replica=args.ranks_per_replica,
                lmp_bin=args.lmp_bin,
                kokkos=args.kokkos,
                note="Stage 4.8 shard built from archived endpoint-relax results with one shared model file per shard.",
            )
            write_json(shard_root / "meta" / "shard_info.json", asdict(shard_info))
            write_csv(shard_root / "meta" / "events.csv", shard_manifest_rows)

            print(f"[OK] Prepared shard: {shard_root}")
            print(f"  jobs: {len(job_names)}")
            print(f"  pairs represented: {len(set(row['event_pair_key'] for row in shard_manifest_rows))}")
            print(f"  max_parallel_jobs: {max(1, args.gpus_total * args.jobs_per_gpu)}")

    manifest_rows = sorted(manifest_rows, key=lambda row: (row["model_tag"], row["system_id"], row["site_a"], row["site_b"]))
    write_csv(args.manifest_out, manifest_rows)
    print(f"[Summary] wrote manifest: {args.manifest_out.resolve()}")
    print(f"[Summary] prepared jobs: {len(manifest_rows)}")
    print(f"[Summary] prepared pairs: {len({row['event_pair_key'] for row in manifest_rows})}")


if __name__ == "__main__":
    main()
