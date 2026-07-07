# Scripts

This directory contains cleaned research scripts that expose selected workflow
details used in the manuscript. The actual calculations were run across several
local and cloud environments and also included many exploratory branches whose
results are not part of the manuscript's main logic. For that reason, the files
here are reference snippets for inspecting implementation details described in
the paper; they are not intended to run directly as a complete workflow.

## Scope

The retained scripts cover:

- conversion of VASP relaxation outputs to DeePMD/DPA and MACE training formats;
- DPA-3 and MACE fine-tuning configuration examples;
- LAMMPS relaxation and single-point evaluation setup;
- VASP input preparation and convergence checks;
- LAMMPS-NEB and DFT-NEB event preparation, filtering, and summary tables;
- model export/freeze helpers for MACE and DeepMD-compatible checkpoints.

Large training data, model checkpoints, calculation outputs, submission logs,
VASP proprietary inputs, and cloud-runtime workspaces are intentionally not
duplicated here. Use `../data/`, `../models/`, and `../results/` for the public
artifacts.

## Layout

```text
scripts/
├── prepare-datasets/       # VASP -> DPA/MACE data conversion examples
├── train/                  # DPA-3 and MACE fine-tuning configs/commands
├── lammps-relaxation/      # LAMMPS relaxation/evaluation workflow helpers
├── single-point-eval/      # DFT-final single-point evaluation bundle helpers
├── dft-relax-prepare/      # VASP input preparation and EDIFFG adjustment
├── dft-relax-analysis/     # VASP convergence/OOD analysis helpers
├── kinetics/               # event preprocessing, LAMMPS-NEB, and DFT-NEB helpers
├── transfer-model/         # model conversion/freeze examples
├── batch-export-models/    # MACE checkpoint export helper
├── extract-pt-data/        # checkpoint/data extraction helper
└── quantization/           # extxyz conversion helper
```

## Directory Guide

The table below describes the leaf directories as reference implementations for
specific manuscript workflow components.

| Directory | Reference role |
| --- | --- |
| `prepare-datasets/` | VASP relaxation output conversion to DeePMD/DPA arrays and MACE `extxyz`, including public train/validation/test split handling. |
| `train/dpa3-finetuning/check-envs/` | Example environment/version probes used before DPA-3 training jobs. |
| `train/dpa3-finetuning/data-transfer/` | Example data staging and preprocessing helpers for DPA-3 training inputs. |
| `train/dpa3-finetuning/replay-data/` | Replay-data counting and JARVIS 2D download helper used around the DPA-3 replay-data workflow. |
| `train/dpa3-finetuning/submit-example/` | Historical DPA-3 training-chain and ablation configuration examples. Most files are split lists or ablation variants; for a compact release, this directory can be reduced to representative examples. |
| `train/dpa3-finetuning/train-example/` | Minimal DPA-3 training input/config construction and comparison helper. |
| `train/mace-finetuning-fp32/` | MACE fine-tuning, smoke-test, SWA, and element-energy configuration examples. |
| `lammps-relaxation/dpa3-mixu/` | DPA/DeepMD LAMMPS relaxation and benchmark workflow example. |
| `lammps-relaxation/mace-relaxation/` | MACE LAMMPS relaxation and benchmark workflow example. |
| `lammps-relaxation/hpc-workflow/` | HPC-style LAMMPS evaluation and postprocessing workflow snippets. |
| `single-point-eval/single-point-calculation/` | Single-point evaluation bundle preparation and DPA/MACE run wrappers for DFT-final structures. |
| `dft-relax-prepare/templates/` | VASP INCAR/KPOINTS/job template fragments used by the DFT relaxation preparation scripts. |
| `dft-relax-prepare/one_process_example/` | Single-process VASP analysis example. |
| `dft-relax-analysis/` | VASP convergence, OOD/fmax scan, and unconverged-run analysis helpers. |
| `kinetics/preprocess/` | Kinetic-event preprocessing: manifest construction, neighboring-site search, orientation checks, and tier-1 event-list generation. |
| `kinetics/lammps-neb/` | LAMMPS/MACE endpoint relaxation, basin checking, NEB package preparation, paired model transition summaries, and DFT-calibration candidate selection. |
| `kinetics/dft-neb/` | DFT-NEB submission planning, shortlist generation, first-bite scaffold construction, final-endpoint relaxation, formal NEB packaging, and rough-result summarization. |
| `transfer-model/dpa3-transfer/freeze-example/` | DeepMD/DPA model download/freeze job example. |
| `transfer-model/mace-transfer/` | MACE model transfer/export job example. |
| `batch-export-models/` | MACE checkpoint inspection and batch export helper scripts. |
| `extract-pt-data/` | Replay `extxyz` extraction/remapping helper around foundation-model training data. |
| `quantization/` | `extxyz` data-file conversion helper used for precision/format adjustment. |

## Sanitized Runtime Configuration

Private runtime values were removed. Before adapting these scripts, provide
your own values through the relevant CLI arguments, job JSON files, or
environment variables:

| Value | Meaning |
| --- | --- |
| `BOHRIUM_PROJECT_ID` or `project_id` | Bohrium project identifier for remote submissions |
| `<CLOUD_PLATFORM>` | Placeholder for the cloud/HPC backend name used by example `job.json` files |
| `<EXAMPLE_MACHINE_TYPE>` | Placeholder for scheduler or cloud machine/resource type |
| `<DEEPMD_IMAGE_ADDRESS>` | DeepMD/DPA container image |
| `<DEEPMD_LAMMPS_IMAGE_ADDRESS>` | LAMMPS+DeepMD/DPA evaluation container image |
| `<MACE_IMAGE_ADDRESS>` | MACE training/export container image |
| `<MACE_LAMMPS_IMAGE_ADDRESS>` | LAMMPS+MACE evaluation container image |
| `<LAMMPS_MACE_IMAGE_ADDRESS>` | LAMMPS-NEB+MACE container image |
| `<VASP_IMAGE_ADDRESS>` | Licensed VASP runtime image |
| `VASP_ENV_SCRIPT`, `VASP_MPI_RANKS`, `VASP_STD_BIN` | User-provided VASP runtime setup, MPI rank count, and executable name for generated DFT-NEB examples |
| `OMP_NUM_THREADS`, `DP_INTRA_OP_PARALLELISM_THREADS`, `DP_INTER_OP_PARALLELISM_THREADS` | User-tunable threading variables; public examples default to conservative values |
| `DFT_ROOT` | Directory containing VASP outputs such as `vasprun.xml`, `OUTCAR`, and `CONTCAR` |
| `META_CSV`, `OOD_CSV`, `OUT_CSV` | Metadata and output CSV paths used by helper shell scripts |

The `job.json` files are examples. Their `project_id` fields are set to `0`,
their `platform` and `machine_type` fields are placeholders, and their
`image_address` fields do not contain real container registry addresses.

## VASP Notes

No `POTCAR`, `WAVECAR`, `CHGCAR`, `OUTCAR`, or `vasprun.xml` files are stored in
`scripts/`. VASP runs require a valid VASP installation and licensed
pseudopotentials. If the preparation scripts call pymatgen POTCAR generation,
set `PMG_VASP_PSP_DIR` in your environment.

## Example Entry Points

Dataset conversion examples:

```bash
python scripts/prepare-datasets/prepare_dpa-3.py --config scripts/prepare-datasets/config_dpa-3.json
python scripts/prepare-datasets/parse_mace.py --config scripts/prepare-datasets/config_mace.json
```

VASP input preparation examples:

```bash
python scripts/dft-relax-prepare/batch_workflow.py \
  --system BP \
  --parent-dir data/src \
  --prepared-path scratch/dft-relax-prepare

python scripts/dft-relax-prepare/replace_ediffg.py \
  --vasp-root scratch/dft-relax-prepare/BP \
  --fmax-csv results/generated/vasp_scan.csv
```

Single-point evaluation bundle example:

```bash
REPO_ROOT="$(pwd)" \
DFT_ROOT="data/raw-vasp/BP_1014" \
META_CSV="metadata/BP_defects.csv" \
bash scripts/single-point-eval/single-point-calculation/single_point.sh
```

NEB scripts are organized as staged preprocessing, LAMMPS-side screening, and
DFT-side calibration under `scripts/kinetics/`. The corresponding intermediate
CSV tables are available in `../results/kinetics/`.

## Reproducibility Status

These scripts expose the core decisions needed to inspect the study workflow:
split construction, OOD-quintile selection, DPA/MACE data formats, fine-tuning
configuration, LAMMPS relaxation/NEB settings, VASP input construction, and
result summarization. They do not include a complete managed environment, cloud
submission automation, licensed VASP assets, or all original raw calculation
outputs.
