#!/usr/bin/env python3
"""Summarize a rough VASP DFT-NEB result from an out.zip or extracted root.

The script intentionally reads only INCAR, OSZICAR, and OUTCAR-style text files.
It does not inspect or print POTCAR content.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REACHED_ACCURACY_MARKER = "reached required accuracy - stopping structural energy minimisation"
NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
OSZICAR_F_RE = re.compile(
    rf"^\s*(\d+)\s+F=\s*({NUMBER_RE})\s+E0=\s*({NUMBER_RE})\s+d E\s*=\s*({NUMBER_RE})",
    flags=re.MULTILINE,
)
TOTEN_RE = re.compile(rf"free\s+energy\s+TOTEN\s+=\s*({NUMBER_RE})\s+eV")


@dataclass
class OszicarSummary:
    ionic_steps: int | None
    final_free_energy_eV: float | None
    final_energy_e0_eV: float | None
    final_dE_eV: float | None
    energies_eV: list[float]
    parse_note: str


@dataclass
class OutcarSummary:
    exists: bool
    final_toten_eV: float | None
    final_max_force_norm_eVA: float | None
    final_max_force_component_eVA: float | None
    reached_required_accuracy: bool
    has_edddav_error: bool
    has_eddrmm_error: bool
    has_zhegv_error: bool
    has_subspace_rotation_error: bool
    has_brmix_warning: bool
    has_cnormn_warning: bool
    force_parse_note: str

    @property
    def has_electronic_hard_error(self) -> bool:
        return (
            self.has_edddav_error
            or self.has_eddrmm_error
            or self.has_zhegv_error
            or self.has_subspace_rotation_error
        )

    @property
    def hard_error_flags(self) -> str:
        flags: list[str] = []
        if self.has_edddav_error:
            flags.append("EDDDAV")
        if self.has_eddrmm_error:
            flags.append("EDDRMM")
        if self.has_zhegv_error:
            flags.append("ZHEGV")
        if self.has_subspace_rotation_error:
            flags.append("SUBSPACE_ROTATION")
        return "|".join(flags)


@dataclass
class ImageMetric:
    image_index: int
    has_oszicar: bool
    has_outcar: bool
    oszicar: OszicarSummary
    outcar: OutcarSummary
    hit_nsw_limit: bool


class ResultReader:
    def __init__(self, result: Path) -> None:
        self.result = result
        self._zip: zipfile.ZipFile | None = None
        self._root: Path | None = None
        self._zip_prefix = ""

        if result.is_file():
            if not zipfile.is_zipfile(result):
                raise ValueError(f"--result must be an out.zip or extracted root: {result}")
            self._zip = zipfile.ZipFile(result)
            self._zip_prefix = self._find_zip_prefix()
        elif result.is_dir():
            self._root = self._find_extracted_root(result)
        else:
            raise FileNotFoundError(f"Result path not found: {result}")

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()

    def exists(self, relative: str) -> bool:
        relative = relative.strip("/")
        if self._zip is not None:
            return self._zip_name(relative) in self._zip.namelist()
        assert self._root is not None
        return (self._root / relative).exists()

    def read_text(self, relative: str) -> str:
        relative = relative.strip("/")
        if self._zip is not None:
            name = self._zip_name(relative)
            with self._zip.open(name) as handle:
                return handle.read().decode("utf-8", errors="replace")
        assert self._root is not None
        return (self._root / relative).read_text(encoding="utf-8", errors="replace")

    def display_path(self, relative: str) -> str:
        relative = relative.strip("/")
        if self._zip is not None:
            return f"{self.result}!/{self._zip_name(relative)}"
        assert self._root is not None
        return str(self._root / relative)

    def _zip_name(self, relative: str) -> str:
        return f"{self._zip_prefix}{relative}"

    def _find_zip_prefix(self) -> str:
        assert self._zip is not None
        candidates = [
            name
            for name in self._zip.namelist()
            if name.rstrip("/").endswith("/INCAR") or name.rstrip("/") == "INCAR"
        ]
        if not candidates:
            raise FileNotFoundError(f"No INCAR found in {self.result}")
        candidates.sort(key=lambda name: (name.count("/"), len(name)))
        incar_name = candidates[0].rstrip("/")
        return incar_name[: -len("INCAR")]

    @staticmethod
    def _find_extracted_root(result: Path) -> Path:
        if (result / "INCAR").exists():
            return result
        candidates = sorted(result.rglob("INCAR"), key=lambda path: (len(path.relative_to(result).parts), len(str(path))))
        if not candidates:
            raise FileNotFoundError(f"No INCAR found under extracted root: {result}")
        return candidates[0].parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path, help="VASP DFT-NEB out.zip or extracted result root.")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--system-id", required=True)
    parser.add_argument("--site-a", required=True, type=int)
    parser.add_argument("--site-b", required=True, type=int)
    parser.add_argument("--initial-outcar", required=True, type=Path)
    parser.add_argument("--final-outcar", required=True, type=Path)
    parser.add_argument("--summary-out", required=True, type=Path)
    parser.add_argument("--image-metrics-out", type=Path, default=None)
    return parser.parse_args()


def parse_incar(text: str) -> dict[str, str]:
    tags: dict[str, str] = {}
    for line in text.splitlines():
        body = line.split("#", 1)[0].split("!", 1)[0].strip()
        if not body or "=" not in body:
            continue
        raw_key, raw_value = body.split("=", 1)
        key = raw_key.strip().upper()
        value = raw_value.strip().split()[0] if raw_value.strip() else ""
        tags[key] = value
    return tags


def tag_int(tags: dict[str, str], key: str) -> int | None:
    value = tags.get(key.upper())
    if value in {None, ""}:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def tag_float(tags: dict[str, str], key: str) -> float | None:
    value = tags.get(key.upper())
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_oszicar_text(text: str) -> OszicarSummary:
    matches = OSZICAR_F_RE.findall(text)
    if not matches:
        return OszicarSummary(
            ionic_steps=None,
            final_free_energy_eV=None,
            final_energy_e0_eV=None,
            final_dE_eV=None,
            energies_eV=[],
            parse_note="missing_F_lines",
        )
    energies = [float(match[1]) for match in matches]
    step, free_energy, e0, de = matches[-1]
    return OszicarSummary(
        ionic_steps=int(step),
        final_free_energy_eV=float(free_energy),
        final_energy_e0_eV=float(e0),
        final_dE_eV=float(de),
        energies_eV=energies,
        parse_note="ok",
    )


def parse_oszicar_reader(reader: ResultReader, relative: str) -> tuple[bool, OszicarSummary]:
    if not reader.exists(relative):
        return (
            False,
            OszicarSummary(
                ionic_steps=None,
                final_free_energy_eV=None,
                final_energy_e0_eV=None,
                final_dE_eV=None,
                energies_eV=[],
                parse_note="missing_oszicar",
            ),
        )
    return True, parse_oszicar_text(reader.read_text(relative))


def parse_outcar_text(text: str, exists: bool = True) -> OutcarSummary:
    lower = text.lower()
    force_norm, force_component, force_note = parse_last_force_block(text)
    toten_matches = TOTEN_RE.findall(text)
    final_toten = float(toten_matches[-1]) if toten_matches else None
    return OutcarSummary(
        exists=exists,
        final_toten_eV=final_toten,
        final_max_force_norm_eVA=force_norm,
        final_max_force_component_eVA=force_component,
        reached_required_accuracy=REACHED_ACCURACY_MARKER in lower,
        has_edddav_error="edddav" in lower,
        has_eddrmm_error="eddrmm" in lower,
        has_zhegv_error="zhegv" in lower,
        has_subspace_rotation_error="sub-space-matrix is not hermitian" in lower
        or "error in subspace rotation" in lower,
        has_brmix_warning="brmix" in lower,
        has_cnormn_warning="cnormn" in lower,
        force_parse_note=force_note,
    )


def missing_outcar(note: str = "missing_outcar") -> OutcarSummary:
    return OutcarSummary(
        exists=False,
        final_toten_eV=None,
        final_max_force_norm_eVA=None,
        final_max_force_component_eVA=None,
        reached_required_accuracy=False,
        has_edddav_error=False,
        has_eddrmm_error=False,
        has_zhegv_error=False,
        has_subspace_rotation_error=False,
        has_brmix_warning=False,
        has_cnormn_warning=False,
        force_parse_note=note,
    )


def parse_outcar_path(path: Path) -> OutcarSummary:
    if not path.exists():
        return missing_outcar("missing_endpoint_outcar")
    return parse_outcar_text(path.read_text(encoding="utf-8", errors="replace"), exists=True)


def parse_outcar_reader(reader: ResultReader, relative: str) -> tuple[bool, OutcarSummary]:
    if not reader.exists(relative):
        return False, missing_outcar()
    return True, parse_outcar_text(reader.read_text(relative), exists=True)


def parse_last_force_block(text: str) -> tuple[float | None, float | None, str]:
    lines = text.splitlines()
    header_idx: int | None = None
    for idx, line in enumerate(lines):
        if "TOTAL-FORCE" in line:
            header_idx = idx
    if header_idx is None:
        return None, None, "missing_force_block"

    forces: list[tuple[float, float, float]] = []
    for line in lines[header_idx + 1 :]:
        stripped = line.strip()
        if not stripped:
            if forces:
                break
            continue
        if set(stripped) <= {"-"}:
            continue
        parts = line.split()
        if len(parts) < 6:
            if forces:
                break
            continue
        try:
            fx, fy, fz = float(parts[3]), float(parts[4]), float(parts[5])
        except ValueError:
            if forces:
                break
            continue
        forces.append((fx, fy, fz))

    if not forces:
        return None, None, "empty_force_block"
    max_norm = max(math.sqrt(fx * fx + fy * fy + fz * fz) for fx, fy, fz in forces)
    max_component = max(max(abs(fx), abs(fy), abs(fz)) for fx, fy, fz in forces)
    return max_norm, max_component, "ok"


def energy_drift(energies: list[float], window: int) -> float | None:
    if len(energies) < 2:
        return None
    start_idx = max(0, len(energies) - window)
    return energies[-1] - energies[start_idx]


def as_csv(value: Any) -> Any:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "NA"
        return f"{value:.10g}"
    return value


def read_image_metrics(reader: ResultReader, images: int | None, nsw: int | None) -> list[ImageMetric]:
    if images is None:
        images = infer_image_count(reader)
    metrics: list[ImageMetric] = []
    for image in range(1, images + 1):
        label = f"{image:02d}"
        has_oszicar, oszicar = parse_oszicar_reader(reader, f"{label}/OSZICAR")
        has_outcar, outcar = parse_outcar_reader(reader, f"{label}/OUTCAR")
        hit_nsw = bool(nsw is not None and oszicar.ionic_steps is not None and oszicar.ionic_steps >= nsw)
        metrics.append(
            ImageMetric(
                image_index=image,
                has_oszicar=has_oszicar,
                has_outcar=has_outcar,
                oszicar=oszicar,
                outcar=outcar,
                hit_nsw_limit=hit_nsw,
            )
        )
    return metrics


def infer_image_count(reader: ResultReader) -> int:
    image = 1
    while reader.exists(f"{image:02d}/OSZICAR") or reader.exists(f"{image:02d}/OUTCAR"):
        image += 1
    inferred = image - 1
    if inferred <= 0:
        raise ValueError("Could not infer intermediate image count and INCAR lacks IMAGES")
    return inferred


def classify_quality(
    image_metrics: list[ImageMetric],
    initial_endpoint: OutcarSummary,
    final_endpoint: OutcarSummary,
    ediffg: float | None,
    forward_barrier: float | None,
    reverse_barrier: float | None,
) -> tuple[str, str]:
    notes: list[str] = []
    energies_complete = all(metric.oszicar.final_free_energy_eV is not None for metric in image_metrics)
    endpoints_complete = initial_endpoint.final_toten_eV is not None and final_endpoint.final_toten_eV is not None
    any_hard_error = any(metric.outcar.has_electronic_hard_error for metric in image_metrics)
    force_values = [
        metric.outcar.final_max_force_norm_eVA
        for metric in image_metrics
        if metric.outcar.final_max_force_norm_eVA is not None
    ]
    max_force = max(force_values) if force_values else None
    hit_nsw_count = sum(1 for metric in image_metrics if metric.hit_nsw_limit)
    reached_count = sum(1 for metric in image_metrics if metric.outcar.reached_required_accuracy)
    max_abs_drift20 = max_abs_drift(image_metrics, 20)
    target_force = abs(ediffg) if ediffg is not None and ediffg < 0 else None

    if not energies_complete:
        notes.append("one_or_more_intermediate_images_lack_final_F_energy")
    if not endpoints_complete:
        notes.append("one_or_both_endpoint_OUTCAR_energies_missing")
    if any_hard_error:
        notes.append("electronic_hard_error_flag_found_in_image_OUTCAR")
    if force_values and len(force_values) != len(image_metrics):
        notes.append("one_or_more_image_force_blocks_missing")
    if hit_nsw_count:
        notes.append(f"{hit_nsw_count}_images_hit_NSW_limit")
    if reached_count != len(image_metrics):
        notes.append(f"{reached_count}_of_{len(image_metrics)}_images_reached_required_accuracy")
    if max_force is not None:
        notes.append(f"max_image_force_norm={max_force:.4g}_eV_per_A")
    if max_abs_drift20 is not None:
        notes.append(f"max_abs_last20_energy_drift={max_abs_drift20:.4g}_eV")

    if (
        not energies_complete
        or not endpoints_complete
        or any_hard_error
        or forward_barrier is None
        or reverse_barrier is None
    ):
        return "failed_or_uninterpretable", "; ".join(notes)

    all_reached = reached_count == len(image_metrics)
    if target_force is not None and max_force is not None:
        all_below_target = max_force <= target_force
    else:
        all_below_target = all_reached

    if all_reached and all_below_target:
        return "converged", "; ".join(notes + ["all_images_satisfy_force_stop"])

    if max_force is not None and max_abs_drift20 is not None:
        if max_force <= 0.30 and max_abs_drift20 <= 0.05:
            return "usable_low_precision", "; ".join(
                notes
                + [
                    "rough_barrier_available",
                    "not_force_converged_but_force_and_last20_energy_drift_are_moderate",
                ]
            )
        if max_force <= 1.00 and max_abs_drift20 <= 0.20:
            return "borderline", "; ".join(
                notes
                + [
                    "rough_barrier_available",
                    "use_for_triage_only_until_continuation_or_converged_NEB_confirms",
                ]
            )

    return "failed_or_uninterpretable", "; ".join(
        notes + ["rough_barrier_available_but_force_or_energy_stability_thresholds_failed"]
    )


def max_abs_drift(image_metrics: list[ImageMetric], window: int) -> float | None:
    values = []
    for metric in image_metrics:
        drift = energy_drift(metric.oszicar.energies_eV, window)
        if drift is not None:
            values.append(abs(drift))
    return max(values) if values else None


def build_image_rows(
    job_id: str,
    system_id: str,
    site_a: int,
    site_b: int,
    image_metrics: list[ImageMetric],
    nsw: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in image_metrics:
        drift10 = energy_drift(metric.oszicar.energies_eV, 10)
        drift20 = energy_drift(metric.oszicar.energies_eV, 20)
        drift50 = energy_drift(metric.oszicar.energies_eV, 50)
        parse_notes = [
            note
            for note in [metric.oszicar.parse_note, metric.outcar.force_parse_note]
            if note and note != "ok"
        ]
        rows.append(
            {
                "job_id": job_id,
                "system_id": system_id,
                "site_a": site_a,
                "site_b": site_b,
                "image_index": metric.image_index,
                "image_label": f"{metric.image_index:02d}",
                "incar_nsw_steps": nsw,
                "has_oszicar": metric.has_oszicar,
                "has_outcar": metric.has_outcar,
                "ionic_steps": metric.oszicar.ionic_steps,
                "hit_nsw_limit": metric.hit_nsw_limit,
                "final_free_energy_eV": metric.oszicar.final_free_energy_eV,
                "final_energy_e0_eV": metric.oszicar.final_energy_e0_eV,
                "final_oszicar_dE_eV": metric.oszicar.final_dE_eV,
                "final_toten_outcar_eV": metric.outcar.final_toten_eV,
                "final_max_force_norm_eVA": metric.outcar.final_max_force_norm_eVA,
                "final_max_force_component_eVA": metric.outcar.final_max_force_component_eVA,
                "last10_energy_drift_eV": drift10,
                "last10_abs_energy_drift_eV": abs(drift10) if drift10 is not None else None,
                "last20_energy_drift_eV": drift20,
                "last20_abs_energy_drift_eV": abs(drift20) if drift20 is not None else None,
                "last50_energy_drift_eV": drift50,
                "last50_abs_energy_drift_eV": abs(drift50) if drift50 is not None else None,
                "reached_required_accuracy": metric.outcar.reached_required_accuracy,
                "has_edddav_error": metric.outcar.has_edddav_error,
                "has_eddrmm_error": metric.outcar.has_eddrmm_error,
                "has_zhegv_error": metric.outcar.has_zhegv_error,
                "has_subspace_rotation_error": metric.outcar.has_subspace_rotation_error,
                "has_brmix_warning": metric.outcar.has_brmix_warning,
                "has_cnormn_warning": metric.outcar.has_cnormn_warning,
                "has_electronic_hard_error": metric.outcar.has_electronic_hard_error,
                "hard_error_flags": metric.outcar.hard_error_flags or "NA",
                "parse_notes": "|".join(parse_notes) if parse_notes else "NA",
            }
        )
    return [{key: as_csv(value) for key, value in row.items()} for row in rows]


def build_summary_row(
    args: argparse.Namespace,
    reader: ResultReader,
    tags: dict[str, str],
    image_metrics: list[ImageMetric],
    initial_endpoint: OutcarSummary,
    final_endpoint: OutcarSummary,
) -> dict[str, Any]:
    nsw = tag_int(tags, "NSW")
    ediffg = tag_float(tags, "EDIFFG")
    images = tag_int(tags, "IMAGES")
    energies = [
        metric.oszicar.final_free_energy_eV
        for metric in image_metrics
        if metric.oszicar.final_free_energy_eV is not None
    ]
    saddle_energy = max(energies) if energies else None
    saddle_image: int | None = None
    if saddle_energy is not None:
        for metric in image_metrics:
            if metric.oszicar.final_free_energy_eV == saddle_energy:
                saddle_image = metric.image_index
                break

    forward_barrier = (
        saddle_energy - initial_endpoint.final_toten_eV
        if saddle_energy is not None and initial_endpoint.final_toten_eV is not None
        else None
    )
    reverse_barrier = (
        saddle_energy - final_endpoint.final_toten_eV
        if saddle_energy is not None and final_endpoint.final_toten_eV is not None
        else None
    )
    quality_label, notes = classify_quality(
        image_metrics=image_metrics,
        initial_endpoint=initial_endpoint,
        final_endpoint=final_endpoint,
        ediffg=ediffg,
        forward_barrier=forward_barrier,
        reverse_barrier=reverse_barrier,
    )

    max_force = max_optional(metric.outcar.final_max_force_norm_eVA for metric in image_metrics)
    max_component = max_optional(metric.outcar.final_max_force_component_eVA for metric in image_metrics)
    rows_with_energy = sum(1 for metric in image_metrics if metric.oszicar.final_free_energy_eV is not None)
    rows_with_outcar = sum(1 for metric in image_metrics if metric.has_outcar)
    reached_count = sum(1 for metric in image_metrics if metric.outcar.reached_required_accuracy)
    hit_nsw_count = sum(1 for metric in image_metrics if metric.hit_nsw_limit)
    any_hard_error = any(metric.outcar.has_electronic_hard_error for metric in image_metrics)

    row = {
        "job_id": args.job_id,
        "system_id": args.system_id,
        "site_a": args.site_a,
        "site_b": args.site_b,
        "result_path": str(args.result),
        "incar_path": reader.display_path("INCAR"),
        "incar_nsw_steps": nsw,
        "incar_ediffg_eVA": ediffg,
        "incar_lclimb": tags.get("LCLIMB", "NA"),
        "incar_potim": tag_float(tags, "POTIM"),
        "incar_ibrion": tag_int(tags, "IBRION"),
        "incar_images": images,
        "initial_outcar": str(args.initial_outcar),
        "final_outcar": str(args.final_outcar),
        "initial_endpoint_energy_eV": initial_endpoint.final_toten_eV,
        "final_endpoint_energy_eV": final_endpoint.final_toten_eV,
        "initial_endpoint_max_force_norm_eVA": initial_endpoint.final_max_force_norm_eVA,
        "initial_endpoint_max_force_component_eVA": initial_endpoint.final_max_force_component_eVA,
        "final_endpoint_max_force_norm_eVA": final_endpoint.final_max_force_norm_eVA,
        "final_endpoint_max_force_component_eVA": final_endpoint.final_max_force_component_eVA,
        "initial_endpoint_reached_required_accuracy": initial_endpoint.reached_required_accuracy,
        "final_endpoint_reached_required_accuracy": final_endpoint.reached_required_accuracy,
        "saddle_image": saddle_image,
        "saddle_energy_eV": saddle_energy,
        "forward_barrier_eV": forward_barrier,
        "reverse_barrier_eV": reverse_barrier,
        "image_count_expected": images,
        "image_count_parsed": len(image_metrics),
        "images_with_energy": rows_with_energy,
        "images_with_outcar": rows_with_outcar,
        "images_reached_required_accuracy": reached_count,
        "images_hit_nsw_limit": hit_nsw_count,
        "any_electronic_hard_error": any_hard_error,
        "max_image_force_norm_eVA": max_force,
        "max_image_force_component_eVA": max_component,
        "max_abs_last10_energy_drift_eV": max_abs_drift(image_metrics, 10),
        "max_abs_last20_energy_drift_eV": max_abs_drift(image_metrics, 20),
        "max_abs_last50_energy_drift_eV": max_abs_drift(image_metrics, 50),
        "quality_label": quality_label,
        "notes": notes if notes else "NA",
    }
    return {key: as_csv(value) for key, value in row.items()}


def max_optional(values: Any) -> float | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    reader = ResultReader(args.result)
    try:
        tags = parse_incar(reader.read_text("INCAR"))
        nsw = tag_int(tags, "NSW")
        images = tag_int(tags, "IMAGES")
        image_metrics = read_image_metrics(reader, images, nsw)
        initial_endpoint = parse_outcar_path(args.initial_outcar)
        final_endpoint = parse_outcar_path(args.final_outcar)
        summary_row = build_summary_row(args, reader, tags, image_metrics, initial_endpoint, final_endpoint)
        image_rows = build_image_rows(args.job_id, args.system_id, args.site_a, args.site_b, image_metrics, nsw)

        write_csv(args.summary_out, [summary_row])
        if args.image_metrics_out is not None:
            write_csv(args.image_metrics_out, image_rows)
    finally:
        reader.close()

    print(f"[OK] wrote summary: {args.summary_out.resolve()}")
    if args.image_metrics_out is not None:
        print(f"[OK] wrote image metrics: {args.image_metrics_out.resolve()}")
    print(
        "[ROUGH DFT-NEB] "
        f"job_id={summary_row['job_id']} "
        f"forward_barrier_eV={summary_row['forward_barrier_eV']} "
        f"reverse_barrier_eV={summary_row['reverse_barrier_eV']} "
        f"quality={summary_row['quality_label']}"
    )


if __name__ == "__main__":
    main()
