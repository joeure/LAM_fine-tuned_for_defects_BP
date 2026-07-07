# MACE Data Files

The full MACE-compatible `extxyz` datasets are intentionally not tracked in
Git because several files exceed GitHub's normal file-size limits.

Expected files:

| File | Frames | Public Git status |
| --- | ---: | --- |
| `MergeBPTrain_fp32.xyz` | 63,349 | Not tracked; distribute via DOI-backed data archive |
| `MergeBPVal_fp32.xyz` | 7,738 | Not tracked; distribute via DOI-backed data archive |
| `MergeBPTest_fp32.xyz` | 7,501 | Not tracked; distribute via DOI-backed data archive |
| `replay_omat_np_30k_remap_fp32.xyz` | 30,000 | Not tracked; distribute via DOI-backed data archive |

`SHA256SUMS` records checksums for the local files used when preparing this
repository snapshot. Update the README and checksums after depositing the
public archive, and cite the archive DOI from the root `README.md`.
