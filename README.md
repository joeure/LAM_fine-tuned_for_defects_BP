# LAM Fine-Tuning for Defective Black Phosphorus

This repository is a pre-publication artifact snapshot for the manuscript
**"Fine-tuning Foundation Atomic Models for Point Defect Properties"**.

The current release focuses on public-facing research artifacts that are useful
for inspection and reuse:

- training, validation, and test data used for black-phosphorus fine-tuning;
- foundation and fine-tuned model checkpoints;
- intermediate CSV result tables used to support the manuscript analyses;
- cleaned workflow scripts that document the main data preparation, training
  configuration, relaxation/evaluation, and kinetics-analysis steps.

The scripts are included to expose the main methodological details, but this is
not an end-to-end push-button reproducibility package. The actual calculations
used multiple local and cloud environments and many exploratory branches whose
results are not part of the manuscript's main logic. The public scripts should
therefore be read as reference snippets for implementation details described in
the paper, not as files that can be run directly to reproduce the full workflow.

## Repository Layout

```text
.
├── data/
│   ├── DPA3-data/          # DeePMD/DPA-style train/val/test arrays
│   ├── MACE-data/          # MACE-compatible extxyz datasets
│   └── src/                # Source CIF structures from defect datasets
├── models/
│   ├── foundation/         # Upstream foundation model files used as starts
│   └── fine-tuned/         # Fine-tuned model checkpoints for this study
├── results/
│   ├── ablations/          # Fine-tuning ablation coordinate-error tables
│   ├── benchamrks/         # Baseline benchmark tables; directory name kept as-is
│   ├── evaluations/        # Post-fine-tuning evaluation tables
│   ├── extrapolations/     # Cross-material extrapolation coordinate metrics
│   └── kinetics/           # Kinetic-event and DFT-NEB screening tables
├── scripts/                # Cleaned workflow scripts and example job configs
├── LICENSE
└── README.md
```

## Data

### `data/DPA3-data/`

DeePMD/DPA-style arrays are organized by split:

| Split | Systems | Frames | Atom records |
| --- | ---: | ---: | ---: |
| `train` | 401 | 401 | 55,652 |
| `val` | 401 | 401 | 55,652 |
| `test` | 51 | 51 | 7,098 |

Each system directory follows the standard DeePMD layout, with `type.raw`,
`type_map.raw`, and `set.000/{box,coord,energy,force,virial}.npy`.

### `data/MACE-data/`

MACE-compatible `extxyz` files:

| File | Frames |
| --- | ---: |
| `MergeBPTrain_fp32.xyz` | 63,349 |
| `MergeBPVal_fp32.xyz` | 7,738 |
| `MergeBPTest_fp32.xyz` | 7,501 |
| `replay_omat_np_30k_remap_fp32.xyz` | 30,000 |

These files include energy, force, stress/virial, and periodic-cell metadata in
the `extxyz` header. Because the full `extxyz` files are 160 MB to 1.3 GB each,
they are not tracked in the GitHub repository. They are provided separately via
the Zenodo data record below; expected filenames and local checksums are listed
in `data/MACE-data/`.

- Zenodo DOI for MACE `extxyz` files:
  <https://doi.org/10.5281/zenodo.21253393>

### `data/src/`

Source CIF structures are grouped by defect density and material family:

| Directory | CIF files |
| --- | ---: |
| `high_density_defects/BP` | 1,001 |
| `high_density_defects/GaSe` | 1,001 |
| `high_density_defects/InSe` | 1,001 |
| `high_density_defects/MoS2` | 1,002 |
| `high_density_defects/WSe2` | 1,002 |
| `high_density_defects/hBN` | 1,001 |
| `low_density_defects/MoS2` | 11,867 |
| `low_density_defects/WSe2` | 11,867 |

The scientific structure identifiers in the result tables are intended to
remain aligned with these CIF filenames.

### Upstream Dataset Source

The source defect structures are derived from the DefiNet Dataset on Zenodo:

- Zenodo record: <https://zenodo.org/records/14027373>
- DOI: <https://doi.org/10.5281/zenodo.14027373>
- Dataset title: `DefiNet Dataset`
- Creators listed by Zenodo: Ziduo Yang and Lei Shen
- License listed by Zenodo: Creative Commons Attribution 4.0 International
  (CC BY 4.0)

The corresponding DefiNet manuscript states that the supporting data are
available from the same Zenodo record and that the DefiNet source code is
available at <https://github.com/Shen-Group/DefiNet>.

## Models

Foundation model files:

- `models/foundation/DPA-3.1-3M.pt`
- `models/foundation/mace-omat-0-medium.model`

These upstream foundation weights are not tracked in GitHub. Retrieve the
corresponding model implementations and release information from the upstream
GitHub repositories listed below, and place local weight files under
`models/foundation/` only for local reuse.

Fine-tuned model files:

- `models/fine-tuned/DPA3-MPtrj-ckpt.pt`
- `models/fine-tuned/DPA3-OMat24-ckpt.pt`
- `models/fine-tuned/frozen-ckpt_DPA3-MPtrj.pth`
- `models/fine-tuned/frozen-ckpt_DPA3-OMat24.pth`
- `models/fine-tuned/mace-omat_finetuned_BP_compiled.model`

### Upstream Model Sources

| Local file | Upstream model/source | Public source |
| --- | --- | --- |
| `models/foundation/mace-omat-0-medium.model` | MACE-OMAT-0 medium foundation model. The MACE repositories list MACE-OMAT-0 as a materials foundation model trained on OMAT with DFT PBE+U/VASP labels. | MACE code: <https://github.com/ACEsuit/mace>; MACE foundation models: <https://github.com/ACEsuit/mace-foundations> |
| `models/foundation/DPA-3.1-3M.pt` | DPA-3.1-3M large atomistic model, implemented in the DeePMD-kit/DPA3 model family. The DPA3 paper describes DPA-3.1-3M as trained on the OpenLAM-v1 collection. | DeePMD-kit code: <https://github.com/deepmodeling/deepmd-kit> |

The MACE-OMAT-0 release is distributed under the license stated by the MACE
foundation-model repository. DeePMD-kit is distributed under LGPL-3.0; check the
upstream model release information and paper for weight-specific terms.

## Results

The `results/` directory contains intermediate CSV tables supporting the
manuscript figures and analyses:

| Directory | CSV files | Contents |
| --- | ---: | --- |
| `results/ablations/` | 28 | Ablation coordinate-error tables |
| `results/benchamrks/` | 7 | Baseline benchmark metrics |
| `results/evaluations/` | 6 | Post-fine-tuning evaluation metrics |
| `results/extrapolations/` | 10 | Cross-material coordinate metrics |
| `results/kinetics/` | 10 | Kinetic-event screening and DFT-NEB summaries |

Columns containing local absolute paths and cloud-runtime job IDs/names have
been removed from the public CSV tables. Scientific identifiers such as
`cif_id`, `system_id`, `event_pair_key`, and event site indices are retained.

## Scripts

The `scripts/` directory contains cleaned scripts and example job/config files
covering dataset preparation, DPA-3/MACE fine-tuning configs, LAMMPS
relaxation/single-point evaluation, VASP input preparation, and LAMMPS/DFT NEB
analysis. Private project IDs, Bohrium image addresses, local absolute paths,
runtime job IDs, VASP proprietary inputs, and duplicated data/model files have
been removed or replaced by placeholders. These scripts are retained to show
how the manuscript-level workflow details were implemented, not to provide a
directly runnable calculation workspace.

HPC/cloud submission fields are intentionally example-only: `project_id` is set
to `0`, `platform`, `machine_type`, and container `image_address` values are
placeholders, and MPI/thread counts are supplied through user-controlled
environment variables or CLI arguments rather than fixed to the original
runtime environment.

### Scripts Directory Guide

| Directory | Reference role |
| --- | --- |
| `scripts/prepare-datasets/` | VASP relaxation output conversion to DeePMD/DPA arrays and MACE `extxyz`, including train/validation/test split handling. |
| `scripts/train/` | Fine-tuning examples for DPA-3 and MACE. |
| `scripts/train/dpa3-finetuning/` | DPA-3 fine-tuning workspace examples, environment notes, split lists, and submission/config snippets. |
| `scripts/train/dpa3-finetuning/check-envs/` | Example environment/version probes used before DPA-3 training jobs. |
| `scripts/train/dpa3-finetuning/data-transfer/` | Example data staging and preprocessing helpers for DPA-3 training inputs. |
| `scripts/train/dpa3-finetuning/replay-data/` | Replay-data counting and JARVIS 2D download helper used around the DPA-3 replay-data workflow. |
| `scripts/train/dpa3-finetuning/submit-example/` | Historical DPA-3 training-chain and ablation configuration examples; most files are split lists or ablation variants. |
| `scripts/train/dpa3-finetuning/train-example/` | Minimal DPA-3 training input/config construction and comparison helper. |
| `scripts/train/mace-finetuning-fp32/` | MACE fine-tuning, smoke-test, SWA, and element-energy configuration examples. |
| `scripts/lammps-relaxation/` | LAMMPS relaxation and benchmark workflow examples for DPA/DeepMD and MACE models. |
| `scripts/lammps-relaxation/dpa3-mixu/` | DPA/DeepMD LAMMPS relaxation and benchmark workflow example. |
| `scripts/lammps-relaxation/dpa3-mixu/templates/` | LAMMPS relaxation input template used by the DPA/DeepMD workflow. |
| `scripts/lammps-relaxation/mace-relaxation/` | MACE LAMMPS relaxation and benchmark workflow example. |
| `scripts/lammps-relaxation/mace-relaxation/templates/` | LAMMPS relaxation input template used by the MACE workflow. |
| `scripts/lammps-relaxation/hpc-workflow/` | HPC-style LAMMPS evaluation and postprocessing workflow snippets. |
| `scripts/lammps-relaxation/hpc-workflow/templates/` | LAMMPS static, relaxation, and benchmarking input templates for the HPC workflow. |
| `scripts/single-point-eval/` | Single-point evaluation examples for DFT-final structures. |
| `scripts/single-point-eval/single-point-calculation/` | Single-point evaluation bundle preparation and DPA/MACE run wrappers. |
| `scripts/dft-relax-prepare/` | VASP input preparation, EDIFFG adjustment, and related DFT relaxation setup helpers. |
| `scripts/dft-relax-prepare/templates/` | VASP INCAR/KPOINTS/job template fragments used by the DFT relaxation preparation scripts. |
| `scripts/dft-relax-prepare/one_process_example/` | Single-process VASP analysis example. |
| `scripts/dft-relax-analysis/` | VASP convergence, OOD/fmax scan, and unconverged-run analysis helpers. |
| `scripts/kinetics/` | Kinetic-event preprocessing, LAMMPS-NEB screening, and DFT-NEB calibration helpers. |
| `scripts/kinetics/preprocess/` | Manifest construction, neighboring-site search, orientation checks, and tier-1 event-list generation. |
| `scripts/kinetics/lammps-neb/` | LAMMPS/MACE endpoint relaxation, basin checking, NEB package preparation, paired model transition summaries, and DFT-calibration candidate selection. |
| `scripts/kinetics/dft-neb/` | DFT-NEB submission planning, shortlist generation, first-bite scaffold construction, final-endpoint relaxation, formal NEB packaging, and rough-result summarization. |
| `scripts/transfer-model/` | Model transfer, download, freeze, and export examples. |
| `scripts/transfer-model/dpa3-transfer/` | DeepMD/DPA model transfer workspace examples. |
| `scripts/transfer-model/dpa3-transfer/freeze-example/` | DeepMD/DPA model download/freeze job example. |
| `scripts/transfer-model/mace-transfer/` | MACE model transfer/export job example. |
| `scripts/batch-export-models/` | MACE checkpoint inspection and batch export helper scripts. |
| `scripts/extract-pt-data/` | Replay `extxyz` extraction/remapping helper around foundation-model training data. |
| `scripts/quantization/` | `extxyz` data-file conversion helper used for precision/format adjustment. |

See `scripts/README.md` for entry points and required user-provided runtime
configuration.

## Current Limitations

This pre-publication snapshot does not yet include:

- environment files such as `pyproject.toml`, `requirements.txt`, or
  `environment.yml`;
- licensed VASP pseudopotentials or raw VASP/LAMMPS output workspaces;
- private Bohrium project/image configuration;
- a permanent data/model DOI.

For a fully reproducible release, these items should be added once the workflow
environment and release metadata have been finalized.

## Large Files

Several data files are larger than GitHub's normal single-file limit. For this
GitHub snapshot, the large MACE `extxyz` files under `data/MACE-data/` are
excluded from Git and represented by `data/MACE-data/README.md` plus
`data/MACE-data/SHA256SUMS`. The full MACE files are archived separately at:

- <https://doi.org/10.5281/zenodo.21253393>

Fine-tuned model files are currently below GitHub's hard 100 MiB per-file limit
and are tracked directly in this repository.

## Citation

Citation information will be updated after the manuscript is accepted or posted
publicly. For now, please cite the repository and manuscript title if you use
these artifacts in a pre-publication collaboration.

## License

The repository currently uses the MIT License. That license covers repository
text and code added here. Dataset and model files may also be subject to
upstream dataset/model licenses; check the original sources before reuse or
redistribution.
