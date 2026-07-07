#!/usr/bin/env python3
import os, sys, argparse
import numpy as np
from typing import Iterable
from ase import Atoms
from ase.io import iread, write

# Keys you commonly use; add more if needed
NUM_ARRAY_KEYS = {
    "REF_forces", "REF_forces",
    "REF_stress", "REF_stress",
    "REF_virials", "REF_virials",
    "REF_charges", "REF_charges",
}
NUM_INFO_KEYS = {
    "REF_energy", "energy", "corrected_total_energy",
    "dipole",
}

def _round_inplace(a: np.ndarray, decimals: int) -> np.ndarray:
    # np.round returns a new array; keep it explicit
    return np.round(a, decimals=decimals, out=None)

def cast_atoms_fp32(at: Atoms, decimals: int = 7) -> Atoms:
    # positions & cell
    pos = np.asarray(at.get_positions(), dtype=np.float32)
    pos = _round_inplace(pos, decimals)
    at.set_positions(pos)

    cell = np.asarray(at.cell.array, dtype=np.float32)
    cell = _round_inplace(cell, decimals)
    at.set_cell(cell, scale_atoms=False)

    # arrays (per-atom)
    for k, v in list(at.arrays.items()):
        if k in NUM_ARRAY_KEYS and isinstance(v, np.ndarray):
            vv = np.asarray(v, dtype=np.float32)
            vv = _round_inplace(vv, decimals)
            at.arrays[k] = vv

    # info (per-frame scalars / vectors)
    for k in list(at.info.keys()):
        v = at.info[k]
        if k in NUM_INFO_KEYS:
            if isinstance(v, (int, float, np.floating)):
                at.info[k] = np.float32(v).item()
            elif isinstance(v, (list, tuple, np.ndarray)):
                at.info[k] = _round_inplace(np.asarray(v, dtype=np.float32), decimals)
    return at

def out_path_with_fp32_suffix(p: str) -> str:
    root, ext = os.path.splitext(p)
    if ext.lower() in (".xyz", ".extxyz"):
        return f"{root}_fp32{ext}"
    return f"{p}_fp32"

def convert_xyz_to_fp32(infile: str, outfile: str, decimals: int = 7) -> None:
    first = True
    count = 0
    for at in iread(infile, format="extxyz", index=":"):
        at32 = cast_atoms_fp32(at, decimals=decimals)
        write(
            outfile,
            at32,
            format="extxyz",
            append=not first,
            write_info=True,
        )
        first = False
        count += 1
    print(f"[OK] {infile} -> {outfile}  (frames: {count})")

def main():
    ap = argparse.ArgumentParser(
        description="Downcast EXTXYZ numeric data to float32 and write *_fp32.xyz files."
    )
    ap.add_argument("inputs", nargs="+", help="Input .xyz/.extxyz files")
    ap.add_argument("--decimals", type=int, default=7,
                    help="Decimal places to round to (default: 7 ~ float32 precision)")
    args = ap.parse_args()

    for inp in args.inputs:
        if not os.path.isfile(inp):
            print(f"[skip] not a file: {inp}", file=sys.stderr)
            continue
        outp = out_path_with_fp32_suffix(inp)
        convert_xyz_to_fp32(inp, outp, decimals=args.decimals)

if __name__ == "__main__":
    main()
