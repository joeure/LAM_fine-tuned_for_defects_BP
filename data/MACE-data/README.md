# MACE Data Files

The full MACE-compatible `extxyz` datasets are intentionally not tracked in
Git because several files exceed GitHub's normal file-size limits. They are
provided through the Zenodo data record:

- <https://doi.org/10.5281/zenodo.21253393>

Expected files:

| File | Frames | Public Git status |
| --- | ---: | --- |
| `MergeBPTrain_fp32.xyz` | 63,349 | Not tracked in Git; available from Zenodo |
| `MergeBPVal_fp32.xyz` | 7,738 | Not tracked in Git; available from Zenodo |
| `MergeBPTest_fp32.xyz` | 7,501 | Not tracked in Git; available from Zenodo |
| `replay_omat_np_30k_remap_fp32.xyz` | 30,000 | Not tracked in Git; available from Zenodo |

`SHA256SUMS` records checksums for the local files used when preparing this
repository snapshot. Use these checksums to verify downloaded files from the
Zenodo record.
