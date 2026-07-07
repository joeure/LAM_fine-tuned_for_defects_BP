#!/usr/bin/env python3
"""
Assemble a minimal runnable DFT-NEB smoke-test directory from a first-bite scaffold.

This is intentionally not a production DFT-NEB packager. Its job is only to turn an
existing scaffold such as:

  dft_neb_workspace/first-bite/job_*__dft_neb_first_bite__incar-tuned-v2

into a single submission root that VASP can start as a NEB calculation on Bohrium.

Design goals:
- preserve the existing first-bite geometry scaffold
- avoid waiting for final_endpoint_relax when the goal is only a smoke test
- make the runnable directory explicit and disposable
- keep the generated package self-contained for remote submission
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scaffold-dir", required=True, type=Path)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional explicit output directory. Defaults to <scaffold-dir>__neb_smoke.",
    )
    p.add_argument(
        "--nsw",
        type=int,
        default=12,
        help="Short ionic step count for a cheap smoke test.",
    )
    p.add_argument(
        "--ediffg",
        type=float,
        default=-0.05,
        help="Looser force target for smoke testing only.",
    )
    p.add_argument(
        "--climb",
        choices=["on", "off"],
        default="off",
        help="Default off for smoke testing to reduce moving parts.",
    )
    p.add_argument(
        "--force-overwrite",
        action="store_true",
        help="Overwrite an existing smoke directory.",
    )
    return p.parse_args()


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines()


def update_incar_for_smoke(lines: List[str], nsw: int, ediffg: float, climb: str, system_label: str) -> str:
    replacements = {
        "SYSTEM": f"SYSTEM = {system_label}",
        "NSW": f"NSW = {nsw}",
        "EDIFFG": f"EDIFFG = {ediffg}",
        "LCLIMB": f"LCLIMB = {'.TRUE.' if climb == 'on' else '.FALSE.'}",
    }

    seen = set()
    out: List[str] = []
    for raw in lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            out.append(raw)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in replacements:
            out.append(replacements[key])
            seen.add(key)
        else:
            out.append(raw)

    for key, value in replacements.items():
        if key not in seen:
            out.append(value)

    out.append("")
    out.append("# Smoke-test note: this directory is only for checking that")
    out.append("# the server-side VASP NEB setup starts and runs, not for")
    out.append("# production-quality barriers.")
    return "\n".join(out) + "\n"


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> None:
    args = parse_args()
    scaffold_dir = args.scaffold_dir.resolve()
    out_dir = (args.out_dir.resolve() if args.out_dir else Path(f"{scaffold_dir}__neb_smoke").resolve())

    if not scaffold_dir.exists():
        raise FileNotFoundError(f"Missing scaffold directory: {scaffold_dir}")

    neb_scaffold = scaffold_dir / "neb_image_scaffold"
    incar_neb = scaffold_dir / "INCAR.neb"
    kpoints = scaffold_dir / "KPOINTS"
    potcar = scaffold_dir / "POTCAR"
    job_json = scaffold_dir / "job.json"
    run_sh = scaffold_dir / "run.sh"
    meta_json = scaffold_dir / "meta" / "meta.json"

    required = [neb_scaffold, incar_neb, kpoints, potcar, job_json, run_sh, meta_json]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Scaffold is incomplete. Missing: {missing}")

    if out_dir.exists():
        if not args.force_overwrite:
            raise FileExistsError(f"Output directory exists: {out_dir}")
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    meta = read_json(meta_json)
    job_name = f"{meta['job_info']['job_name']}__neb_smoke"

    # Root-level files expected by a native VASP NEB run.
    shutil.copy2(kpoints, out_dir / "KPOINTS")
    shutil.copy2(potcar, out_dir / "POTCAR")
    shutil.copy2(run_sh, out_dir / "run.sh")

    incar_text = update_incar_for_smoke(
        read_lines(incar_neb),
        nsw=args.nsw,
        ediffg=args.ediffg,
        climb=args.climb,
        system_label=f"{job_name} dft-neb-smoke",
    )
    (out_dir / "INCAR").write_text(incar_text, encoding="utf-8")

    for image_dir in sorted(p for p in neb_scaffold.iterdir() if p.is_dir()):
        copy_tree(image_dir, out_dir / image_dir.name)

    payload = read_json(job_json)
    payload["job_name"] = job_name
    payload["backward_files"] = sorted(
        [p.name for p in out_dir.iterdir() if p.name.isdigit()] + ["INCAR", "KPOINTS", "OUTCAR", "OSZICAR", "STDOUTERR", "vasprun.xml"]
    )
    write_json(out_dir / "job.json", payload)

    smoke_meta = {
        "task_class": "dft_neb_smoke_test",
        "source_scaffold_dir": str(scaffold_dir),
        "job_name": job_name,
        "uses_final_endpoint_relaxed": False,
        "final_endpoint_source": "final_endpoint_guess",
        "images_total": meta["job_info"]["images_total"],
        "images_intermediate": meta["job_info"]["images_intermediate"],
        "smoke_incar_settings": {
            "NSW": args.nsw,
            "EDIFFG": args.ediffg,
            "LCLIMB": ".TRUE." if args.climb == "on" else ".FALSE.",
        },
        "note": (
            "Minimal runnable Bohrium smoke-test package for DFT-NEB. "
            "This package intentionally reuses the guessed final endpoint to validate "
            "server-side NEB startup before waiting for the DFT-relaxed final endpoint."
        ),
    }
    write_json(out_dir / "smoke_test_info.json", smoke_meta)
    (out_dir / "README.smoke.md").write_text(
        "\n".join(
            [
                "# DFT NEB Smoke Test",
                "",
                f"Source scaffold: `{scaffold_dir}`",
                "",
                "This directory is intended only to verify that the Bohrium-side VASP",
                "NEB configuration starts and runs with the expected multi-image layout.",
                "",
                "Important limits:",
                "- The final endpoint is still the constructed guess, not the DFT-relaxed final endpoint.",
                "- The INCAR settings are intentionally loosened for a cheap smoke test.",
                "- Do not use this run for paper-quality barriers.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[DFT NEB SMOKE] created {out_dir}")


if __name__ == "__main__":
    main()
