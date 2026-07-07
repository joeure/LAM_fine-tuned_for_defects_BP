from pathlib import Path
import numpy as np
import pandas as pd
import dpdata

def keep_last_frame(ls: dpdata.LabeledSystem):
    """Trim a LabeledSystem to its last ionic step (kept as a 1-frame batch)."""
    for k in ["energies", "forces", "virials", "coords", "cells"]:
        arr = ls.data.get(k, None)
        if isinstance(arr, np.ndarray) and arr.ndim >= 1 and arr.shape[0] > 1:
            ls.data[k] = arr[-1:,...]   # keep last; keep leading frame axis
    return ls

def load_cif_system(cif_path: Path):
    """Read CIF -> dpdata.System, trying ASE first then pymatgen."""
    # --- ASE route ---
    try:
        from ase.io import read as ase_read
        atoms = ase_read(str(cif_path))
        # ase_read can return a list if CIF has multiple frames
        if isinstance(atoms, (list, tuple)):
            if not atoms:
                raise ValueError("CIF contains no structures")
            atoms = atoms[-1]  # choose last structure
        return dpdata.System(atoms, fmt="ase/structure")
    except Exception as e_ase:
        # --- pymatgen route ---
        try:
            from pymatgen.core import Structure
            st = Structure.from_file(str(cif_path))
            return dpdata.System(st, fmt="pymatgen/structure")
        except Exception as e_pm:
            raise RuntimeError(
                f"Failed to read CIF via ASE and pymatgen:\nASE: {e_ase}\npymatgen: {e_pm}"
            )

def transfer_one_system(cif_path: Path, vasp_dir: Path):
    """Load labels from vasprun.xml/OUTCAR and overwrite geometry with CIF."""
    vasprun = vasp_dir / "vasprun.xml"
    outcar  = vasp_dir / "OUTCAR"
    if outcar.exists():
        lab = dpdata.LabeledSystem(str(outcar), fmt="vasp/outcar")
    elif vasprun.exists():
        lab = dpdata.LabeledSystem(str(vasprun), fmt="vasp/xml")
    else:
        raise FileNotFoundError(f"No vasprun.xml or OUTCAR under {vasp_dir}")

    stru = load_cif_system(cif_path=cif_path)

    # Basic consistency checks
    if lab.get_natoms() != stru.get_natoms():
        raise ValueError(f"Natoms mismatch: VASP {lab.get_natoms()} vs CIF {stru.get_natoms()}")

    names_l = getattr(lab, "atom_names", None) or lab.data.get("atom_names")
    names_s = getattr(stru, "atom_names", None) or stru.data.get("atom_names")
    if names_l and names_s and names_l != names_s:
        raise ValueError(f"Element ordering differs:\nVASP {names_l}\nCIF  {names_s}")

    # Keep only last ionic step from VASP
    keep_last_frame(lab)  # now typically n_frames == 1
    
    assert lab.data["coords"].shape[:2] == (lab.get_nframes(), lab.get_natoms())
    assert lab.data["cells"].shape[0]   == lab.get_nframes()
    # If you kept >1 frame by mistake, stop:
    assert lab.get_nframes() == 1, "You overwrote geometry but still have multiple frames!"

    # Overwrite geometry with CIF's coords + cell
    coords0 = stru.data["coords"]  # (natoms, 3) or (1, natoms, 3)
    cell0   = stru.data["cells"]   # (3,3) or (1,3,3)
    if coords0.ndim == 2: coords0 = coords0[None, ...]
    if cell0.ndim   == 2: cell0   = cell0[None,   ...]
    nframes = lab.get_nframes()
    lab.data["coords"] = np.repeat(coords0, nframes, axis=0)
    lab.data["cells"]  = np.repeat(cell0,   nframes, axis=0)

    return lab  # LabeledSystem ready for .to("deepmd/npy", ...)

def transfer_batch(DFTPath: str, DefinetPath: str, OutputPath: str, systemName: str, highOrLow: bool, refSys:str):
    """
    Convert many IDs listed in train/val/test CSVs:
      CSV schema must contain column 'atoms_id'.
      CIF path pattern:   {DefinetPath}/{density}/{systemName}/CIF/{id}_unrelaxed.cif
      VASP output folder: {DFTPath}/{id}_relaxed/{vasprun.xml|OUTCAR}
    Writes deepmd/npy systems to:
      {OutputPath}/{split}/system.000000/, system.000001/, ...
    """
    density = "high_density_defects" if highOrLow else "low_density_defects"
    base = Path(DefinetPath) / density / systemName
    out_root = Path(OutputPath)

    for split in ["train", "val", "test"]:
        csv = base / f"{split}.csv"
        if not csv.exists():
            print(f"[skip] {csv} not found")
            continue

        ids = pd.read_csv(csv)["atoms_id"].astype(str).tolist()
        out_split = out_root / split
        out_split.mkdir(parents=True, exist_ok=True)
        ref_vdir = Path(DFTPath) / refSys
        if not ref_vdir.exists():
            raise FileNotFoundError(f"{ref_vdir} not existed!")
        ref_outcar = ref_vdir / "OUTCAR"
        if ref_outcar.exists():
            ref_lab = dpdata.LabeledSystem(str(ref_outcar), fmt="vasp/outcar")
        keep_last_frame(ref_lab)
        numberref = 0
        out_sysRef = out_split / f"system.{numberref:06d}"
        ref_lab.to("deepmd/npy", str(out_sysRef), set_size=ref_lab.get_nframes())
        
        print(f"[ok] {split} {refSys} -> {out_sysRef}")

        for idx, sid in enumerate(ids):
            cif  = base / "CIF" / f"{sid}_unrelaxed.cif"
            vdir = Path(DFTPath) / f"{sid}_relaxed"

            if not cif.exists():
                print(f"[skip] {sid}: missing CIF {cif}")
                continue
            try:
                lab = transfer_one_system(cif, vdir)
            except Exception as e:
                print(f"[skip] {sid}: {e}")
                continue
            
            number = idx + 1

            # Each ID -> one system folder. Since we kept 1 frame, set_size = 1 is fine.
            out_sys = out_split / f"system.{number:06d}"
            lab.to("deepmd/npy", str(out_sys), set_size=lab.get_nframes())
            print(f"[ok] {split} {sid} -> {out_sys}")

transfer_batch(
    DFTPath="./BP_done",          # where {id}_relaxed/ lives
    DefinetPath="./DefiNet",     # where high_density_defects/… lives
    OutputPath="./data",              # will write data/{train,val,test}/system.xxxxxx/
    systemName="BP_spin_500",                      # your project/system group name
    highOrLow=True,                           # or False
    refSys="P"
)