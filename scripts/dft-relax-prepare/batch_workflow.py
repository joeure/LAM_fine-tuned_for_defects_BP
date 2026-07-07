from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Potcar
from ase.io import read, write
import os
import argparse
import logging
import jinja2
import dataclasses
# from pymatgen import  SETTINGS
# Set PMG_VASP_PSP_DIR in the environment before running if POTCAR files are required.
# import shutil
# import json
import math
# import time
# import zipfile
# from pymatgen.io.vasp import Vasprun
# from pymatgen.io.vasp import Outcar
import re
# import itertools as it
from collections import OrderedDict
import pandas as pd

@dataclasses.dataclass
class VaspIncarInfo:
    systemName: str
    encut: int
    nbands: int
    magmom: str
    ncore: int = 2
    lreal: str = "Auto"
    nelm: int = 120
    algo: str = "Fast"
    prec: str = "Normal"
    
@dataclasses.dataclass
class VaspKpointsInfo:
    """
    Data class to hold VASP KPOINTS information.
    """
    gamma_centered: bool = True
    kx: int = 1
    ky: int = 1

NxNyLookUpDict = {
    "MoS2_1x1": (15, 15),
    "MoS2_8x8": (1, 1),
    "WSe2_1x1": (15, 15),
    "WSe2_8x8": (1, 1),
    "hBN_1x1": (18, 18),
    "hBN_8x8": (1, 1),
    "GaSe_1x1": (12, 12),
    "GaSe_6x6": (1, 1),
    "InSe_1x1": (12, 12),
    "InSe_6x6": (1, 1),
    "BP_1x1": (15, 9),
    "BP_6x6": (2, 3),
}

supercellSizeLookUpDict = {
    "MoS2": (8, 8),
    "WSe2": (8, 8),
    "BN": (8, 8),
    "GaSe": (6, 6),
    "InSe": (6, 6),
    "BP": (6, 6),
    "MoS2_500": (1, 1),
    "WSe2_500": (1, 1)
}

masses = {
    "Mo":95.95, 
    "S":32.065,
    "W":183.84,
    "Se":78.96,
    "B":10.81,
    "N":14.007,
    "Ga":69.723,
    "In":114.82,
    "P": 30.973762,
    "C": 12.011
}

safeSets = {
    "MoS2": {"Mo", "S", "W", "S"},
    "WSe2": {"W", "Se", "Mo", "Se"},
    "BN": {"B", "N", "C"},
    "GaSe": {"Ga", "Se", "S", "In"},
    "InSe": {"In", "Se", "S", "Ga"},
    "BP": {"P", "N"},
    "MoS2_500": {"Mo", "S", "W", "S"},
    "WSe2_500": {"W", "Se", "Mo", "Se"},
}

idealAtomsNumDict = {
    "MoS2": 192, # 8x8 supercell
    "WSe2": 192, # 8x8 supercell
    "BN": 192, # 8x8 supercell
    "GaSe": 72, # 6x6 supercell
    "InSe": 72, # 6x6 supercell
    "BP": 54, # 6x6 supercell?
    "P": 144, # 6x6 supercell?, 36 * 4 = 144
    "MoS2_500": 192, # 8x8 supercell
    "WSe2_500": 192, # 8x8 supercell
}

valenceElectronsDict = {
    "Mo": 6,
    "S": 6,
    "W": 6,
    "Se": 6,
    "B": 3,
    "N": 5,
    "Ga": 3,
    "In": 3,
    "P": 5,
    "C": 4
}

magmoms = {
    "H": 1,
    "He": 0,
    "Li": 1,
    "Be": 0,
    "B": 1,
    "C": 2,
    "N": 1.0,
    "O": 2,
    "F": 1,
    "Ne": 0,
    "Na": 1,
    "Mg": 0,
    "Al": 1,
    "Si": 2,
    "P": 0.1,
    "S": 2,
    "Cl": 1,
    "Ar": 0,
    "K": 1,
    "Ca": 0,
    "Sc": 2,
    "Ti": 2,
    "V": 3,
    "Cr": 6,
    "Mn": 5,
    "Fe": 4,
    "Co": 3,
    "Ni": 2,
    "Cu": 1,
    "Zn": 0,
    "Ga": 1,
    "Ge": 2,
    "As": 3,
    "Se": 2,
    "Br": 1,
    "Kr": 0,
    "Rb": 1,
    "Sr": 0,
    "Y": 2,
    "Zr": 2,
    "Nb": 5,
    "Mo": 6,
    "Tc": 5,
    "Ru": 4,
    "Rh": 3,
    "Pd": 0,
    "Ag": 1,
    "Cd": 0,
    "In": 1,
    "Sn": 2,
    "Sb": 3,
    "Te": 2,
    "I": 1,
    "Xe": 0,
    "Cs": 1,
    "Ba": 0,
    "La": 2,
    "Ce": 2,
    "Pr": 3,
    "Nd": 4,
    "Pm": 5,
    "Sm": 6,
    "Eu": 7,
    "Gd": 7,
    "Tb": 6,
    "Dy": 5,
    "Ho": 4,
    "Er": 3,
    "Tm": 2,
    "Yb": 1,
    "Lu": 0,
    "Hf": 2,
    "Ta": 3,
    "W": 4,
    "Re": 5,
    "Os": 4,
    "Ir": 3,
    "Pt": 2,
    "Au": 1,
    "Hg": 0,
    "Tl": 1,
    "Pb": 2,
    "Bi": 3,
    "Po": 2,
    "At": 1,
    "Rn": 0,
    "Fr": 1,
    "Ra": 0,
    "Ac": 2,
    "Th": 2,
    "Pa": 3,
    "U": 4,
    "Np": 5,
    "Pu": 6
}


def magmom_line_from_dict(structure, elem_magdict, default=0.6, tol=1e-8):
    """Return a VASP MAGMOM string matching the POSCAR site order.
    elem_magdict: {'Fe':5.0, 'O':0.6, ...} (μB); values may be floats or 3-tuples for noncollinear.
    default: used when an element is missing in the dict."""
    # 1) build per-site list
    per_site = []
    for site in structure:
        sym = getattr(site, "specie", getattr(site, "species", None))
        sym = sym.symbol if hasattr(sym, "symbol") else str(site.specie.symbol)  # robust
        val = elem_magdict.get(sym, default)
        per_site.append(val)

    # 2) collinear (floats): run-length compress into "n*val" chunks
    if isinstance(per_site[0], (int, float)):
        vals = [float(v) for v in per_site]
        parts, count = [], 1
        for i, v in enumerate(vals):
            nxt_same = (i+1 < len(vals) and abs(vals[i+1]-v) <= tol)
            if nxt_same:
                count += 1
            else:
                parts.append(f"{count}*{v:g}" if count > 1 else f"{v:g}")
                count = 1
        return " ".join(parts)

    # 3) noncollinear (3 components per atom): no grouping, one triplet per site
    #    (VASP expects 3*NIONS numbers in order)
    triplets = []
    for v in per_site:
        mx, my, mz = v
        triplets.append(f"{float(mx):g} {float(my):g} {float(mz):g}")
    return "  ".join(triplets)


def get_ratio(fileName):
    ratioName = fileName.split('_')[1]
    # the ratio name is like P126N9 etc. I want to extract [126, 9] two numbers
    pattern = r'\d+'
    matches = re.findall(pattern, ratioName)
    # the numbers might be two or one numbers: P141, or P140N1
    nums = [int(num) for num in matches]
    return nums

def contains_files(directory_path):
    """
    Checks if a given directory contains any files.

    Args:
        directory_path (str): The path to the directory to check.

    Returns:
        bool: True if the directory contains at least one file, False otherwise.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: '{directory_path}' is not a valid directory.")
        return False

    for entry in os.listdir(directory_path):
        full_path = os.path.join(directory_path, entry)
        if os.path.isfile(full_path):
            return True  # Found at least one file

    return False  # No files found in the directory

def mkdir_if_not_exists(directory):
    """
    Creates a directory if it does not already exist.

    Args:
        directory (str): Path to the directory to create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        logging.info(f"Directory {directory} already exists. No need to create it.")
        
def transfer_from_cif_to_poscar(structure, output_file):
    """
    Transfers a crystal structure from a CIF file to a POSCAR file.

    Args:
        input_file (str): Path to the input CIF file.
        output_file (str): Path to the output POSCAR file.
    Returns:
        the symbols of the structure with the same order as in POSCAR
    """
    # Write the structure to a POSCAR file
    structure.to(filename=output_file, fmt='poscar')
    # Return the symbols of the structure in the same order as in POSCAR
    orderedSet = tuple(OrderedDict.fromkeys(
        site.specie.symbol for site in structure
    ))
    # orderedSet = structure.symbol_set  # This will give the unique symbols in the structure
    logging.info(f"Symbols in POSCAR: {orderedSet}")
    return orderedSet

def generate_potcar(symbols, output_file, chemical:str) -> float:
    """
    Generates a POTCAR file for the given symbols.

    Args:
        symbols (list): List of element symbols.
        potcar_dir (str): Directory containing the POTCAR files.
        output_file (str): Path to the output POTCAR file.
    """
    # Create a Potcar object
    potcar = Potcar(symbols=symbols,
                   functional='PBE_54')
    
    expandedSymbols = safeSets.get(chemical, symbols)
    expandPotcar = Potcar(symbols=expandedSymbols,
                          functional='PBE_54')
    max_enmax = 0.0
    for p in expandPotcar:
        enmax = p.ENMAX
        logging.info(f"Element: {p.symbol}, ENMAX: {enmax}")
        if enmax > max_enmax:
            max_enmax = enmax
    
    print(f"Maximum ENMAX value: {max_enmax}")
    
    # Write the POTCAR to the specified output file
    potcar.write_file(output_file)
    return max_enmax

def prepare_one_system(
    relaxedCifPath:str,
    preparedPath:str,
    templatePath:str,
    systemChemicalName:str,
    needExpanded:bool
):
    structure = Structure.from_file(relaxedCifPath, frac_tolerance=0)
    symbols = structure.symbol_set
    
    supercellSize = structure.lattice.abc
    if needExpanded:
        if systemChemicalName in supercellSizeLookUpDict:
            supercellFrame = supercellSizeLookUpDict[systemChemicalName]
            logging.info(f"Expanding the supercell to {supercellSize} for the ideal structure.")
            superCellScalingMatrix = [
                [supercellFrame[0], 0, 0],
                [0, supercellFrame[1], 0],
                [0, 0, 1]  # Assuming the c-axis is not scaled
            ]
            structure.make_supercell(scaling_matrix=superCellScalingMatrix, in_place=True)
            logging.info(f"Supercell size after expansion: {structure.lattice.abc}")
        else:
            logging.warning(f"Supercell size for {systemChemicalName} not found in lookup dictionary. Using original size.")
    systemID = os.path.basename(relaxedCifPath).split('.')[0]
    logging.info(f"Preparing system: {systemID} for the chemical name: {systemChemicalName}")
    mkdir_if_not_exists(os.path.join(preparedPath, systemChemicalName, systemID))
    systemPath = os.path.join(preparedPath, systemChemicalName, systemID)
    # if needExpanded:
    #     ratios = [structure.num_sites]
    # else:
    #     ratios = get_ratio(systemID)
    
    # prepare POSCAR
    poscarPath = os.path.join(systemPath, 'POSCAR')
    symbols = transfer_from_cif_to_poscar(structure, poscarPath)
    # prepare POTCAR, better symbols the same order as in POSCAR
    logging.critical(f"Symbols set in POSCAR: {symbols}")
    
    potcarPath = os.path.join(systemPath, 'POTCAR')
    max_enmax = generate_potcar(symbols, potcarPath, systemChemicalName)
    logging.info(f"Generated POTCAR for {systemChemicalName} with maximum ENMAX: {max_enmax}")
    # prepare KPOINTS
    kpointsPath = os.path.join(systemPath, 'KPOINTS')
    kpointsTemplate = os.path.join(templatePath, 'KPOINTS.jinja')
    kpointsInfo = VaspKpointsInfo(gamma_centered=True, kx=1, ky=1)
    with open(kpointsTemplate, 'r') as f:
        kpointsTemplateContent = f.read()
    templateKpoints = jinja2.Template(kpointsTemplateContent)
    kpointsContent = templateKpoints.render(
        gamma_centered=kpointsInfo.gamma_centered,
        kx=kpointsInfo.kx,
        ky=kpointsInfo.ky
    )
    with open(kpointsPath, 'w') as f:
        f.write(kpointsContent)
    logging.info(f"Generated KPOINTS file at {kpointsPath}")
    # prepare INCAR
    incarPath = os.path.join(systemPath, 'INCAR')
    incarTemplate = os.path.join(templatePath, 'INCAR_BATCH.jinja')
    # total number of electrons for all the atom sites each with its element symbol
    numElectrons = sum(valenceElectronsDict.get(symbol, 0) for symbol in symbols) / len(symbols) * structure.num_sites
    logging.info(f"Number of valence electrons for {systemChemicalName}: {numElectrons}, the symbols: {symbols}")
    magmom = f"{structure.num_sites}*0.6"  # all sites are given the same guess value
    # for BP, with rstio numbers:
    # assert structure.num_sites == sum(ratios), f"Number of sites {structure.num_sites} does not match the sum of ratios {sum(ratios)} for {systemID}."
    # if len(ratios) == 1:
    #     magmom = f"{ratios[0]}*0.01"
    # elif len(ratios) == 2:
    #     magmom = f"{ratios[0]}*0.01 {ratios[1]}*0.5"
    # magmom = f'{structure.num_sites}*0.6'
    magmom = magmom_line_from_dict(structure, magmoms, default=0.6)
    incarInfo = VaspIncarInfo(
        systemName=f"{systemChemicalName}_{systemID}",
        # encut=int(math.ceil(max_enmax * 1.3 / 25) * 25),
        encut=500,
        nbands=int((math.ceil(((len(symbols) / 2 + numElectrons / 2) * 1.3/10))+1)*10),
        ncore=2,
        lreal="Auto",
        nelm=120,
        algo="Fast",
        prec="Normal",
        magmom=magmom
    )
    with open(incarTemplate, 'r') as f:
        incarTemplateContent = f.read()
    templateIncar = jinja2.Template(incarTemplateContent)
    incarContent = templateIncar.render(
        systemName=incarInfo.systemName,
        encut=incarInfo.encut,
        nbands=incarInfo.nbands,
        ncore=incarInfo.ncore,
        lreal=incarInfo.lreal,
        nelem=incarInfo.nelm,
        algo=incarInfo.algo,
        prec=incarInfo.prec,
        magmom=incarInfo.magmom
    )
    with open(incarPath, 'w') as f:
        f.write(incarContent)
    logging.info(f"Generated INCAR file at {incarPath}")
    return systemPath

def prepare_batch_inputs(
    relaxedCifPath:str,
    preparedPath:str,
    templatePath:str,
    systemChemicalName:str,
    mask_csv:str
):
    """
    Prepares the batch inputs for VASP calculations.

    Args:
        relaxedCifPath (str): Path to the relaxed CIF file.
        preparedPath (str): Path to the directory where prepared files will be stored.
        templatePath (str): Path to the directory containing template files.
        systemChemicalName (str): Chemical name of the system.
        needExpanded (bool): Whether to expand the supercell or not.
    """
    logging.info(f"Preparing batch inputs for {systemChemicalName} from {relaxedCifPath}")
    toRemain = None
    if mask_csv:
        maskCsv = pd.read_csv(mask_csv)
        # print(type(maskCsv['converged_electronic'].iloc[5]))
        # print(type(maskCsv['converged'].iloc[5]))
        toRemain = maskCsv[(maskCsv['converged_electronic'] == True) & (maskCsv['converged'] == 1)]['cif_id'].to_list()
    for relaxedCif in os.listdir(relaxedCifPath):
        if relaxedCif.endswith('_unrelaxed.cif') and (toRemain is None or relaxedCif.split('.')[0] not in toRemain):
            relaxedCifFullPath = os.path.join(relaxedCifPath, relaxedCif)
            logging.info(f"Processing CIF file: {relaxedCifFullPath}")
            systemPath = prepare_one_system(
                relaxedCifFullPath,
                preparedPath,
                templatePath,
                systemChemicalName,
                needExpanded=False
            )
            logging.info(f"Prepared system at {systemPath}")
    logging.info(f"Batch inputs preparation completed for {systemChemicalName} in {preparedPath}")

systemFolderLookUpDict = {
    "MoS2": "MoS2",
    "WSe2": "WSe2",
    "BN": "hBN_spin_500",
    "GaSe": "GaSe_spin_500",
    "InSe": "InSe_spin_500",
    "BP": "BP_spin_500",
    "MoS2_500": "MoS2_500",
    "WSe2_500": "WSe2_500"
}

systemRefLookUpDict = {
    "MoS2": "MoS2",
    "WSe2": "WSe2",
    "BN": "BN",
    "GaSe": "GaSe",
    "InSe": "InSe",
    "BP": "P",
    "MoS2_500": "MoS2_8x8",
    "WSe2_500": "WSe2_8x8"
}
  
def prepare_a_batch_system(
    systemChemicalName:str,
    parentDir:str,
    preparedPath:str,
    templatePath:str,
    lowHigh:bool,
    mask_csv:str
):
    # 1. Prepare the reference ideal system
    systemRef = os.path.join(parentDir, 
                             "low_density_defects" if lowHigh else "high_density_defects",
                             f'{systemRefLookUpDict[systemChemicalName]}.cif')
    if not os.path.exists(systemRef):
        logging.error(f"Reference CIF file for {systemChemicalName} not found at {systemRef}.")
        return
    logging.info(f"Preparing reference system for {systemChemicalName} from {systemRef}")
    systemRefPath = prepare_one_system(
        systemRef,
        preparedPath,
        templatePath,
        systemChemicalName,
        needExpanded=True
    )
    logging.info(f"Reference system prepared at {systemRefPath}")
    # 2. Prepare the batch of normal systems
    if lowHigh:
        batchFolder = os.path.join(parentDir, 
                                   "low_density_defects", 
                                   systemFolderLookUpDict[systemChemicalName],
                                   "CIF_POSCAR")
    else:
        batchFolder = os.path.join(parentDir, 
                                   "high_density_defects", 
                                   systemFolderLookUpDict[systemChemicalName],
                                   "CIF")
    if not os.path.exists(batchFolder):
        logging.error(f"Batch folder for {systemChemicalName} not found at {batchFolder}.")
        return
    logging.info(f"Preparing batch systems for {systemChemicalName} from {batchFolder}")
    prepare_batch_inputs(
        batchFolder,
        preparedPath,
        templatePath,
        systemChemicalName,
        mask_csv
    )
    logging.info(f"Batch systems prepared for {systemChemicalName} in {preparedPath}")
    
def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare VASP relaxation input folders from DefiNet-style CIF/POSCAR roots."
    )
    parser.add_argument("--system", default="BP", help="Chemical system key, e.g. BP.")
    parser.add_argument(
        "--parent-dir",
        required=True,
        help="Root containing low_density_defects/ and high_density_defects/ directories.",
    )
    parser.add_argument(
        "--prepared-path",
        required=True,
        help="Output directory for generated VASP input folders.",
    )
    parser.add_argument(
        "--template-path",
        default=os.path.join(os.path.dirname(__file__), "templates"),
        help="Directory containing INCAR/KPOINTS/job templates.",
    )
    parser.add_argument(
        "--low-high",
        action="store_true",
        help="Use the low_density_defects/CIF_POSCAR branch instead of high_density_defects/CIF.",
    )
    parser.add_argument(
        "--mask-csv",
        default=None,
        help="Optional convergence scan CSV used to mask already processed systems.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_a_batch_system(
        systemChemicalName=args.system,
        parentDir=args.parent_dir,
        preparedPath=args.prepared_path,
        templatePath=args.template_path,
        lowHigh=args.low_high,
        mask_csv=args.mask_csv,
    )
    
        
    
