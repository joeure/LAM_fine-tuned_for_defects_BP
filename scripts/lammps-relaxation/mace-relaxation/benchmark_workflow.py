from ase.io import read, write
import logging
import os
# import torch
from lammps import lammps
import pandas as pd
import enum
from dataclasses import dataclass
import jinja2
import argparse
import json
from datetime import datetime
import time
import shutil

chemicalDict = {
    "BP_spin_500": "P",
    "hBN_spin_500": "BN",
    "GaSe_spin_500": "GaSe",
    "InSe_spin_500": "InSe",
    "MoS2_spin_500": "MoS2",
    "WSe2_spin_500": "WSe2",
    "MoS2": "poscar",
    "WSe2": "poscar"
}

@dataclass
class SystemDefinition:
    referenceSystem: list[str]
    lowHighDensity: bool
    systemName: str
    def __str__(self):
        return f""" **************************
    The system: {self.systemName}
    The Reference Perfect Crystal: {self.referenceSystem}
    Is the Density of Defects High? {self.lowHighDensity}
    **************************"""
    
@dataclass
class SystemInfo:
    structure_file: str
    model_file: str
    elements: list[str]
    system_name: str
    output_file: str
    dump_name:str
    ftol:float
    def __str__(self):
        return f""" **************************
    The system: {self.system_name}
    The Structure File: {self.structure_file}
    The Model File: {self.model_file}
    The Elements: {self.elements}
    The Output File: {self.output_file}
    **************************"""

class FileType(enum.Enum):
    CIF = "cif"
    POSCAR = "POSCAR"
    XYZ = "XYZ"
    EXTXYZ = "EXTXYZ"
    DATA = "lammps-data"

    def __str__(self):
        return self.value

    @classmethod
    def from_string(cls, value):
        for item in cls:
            if item.value == value:
                return item
        raise ValueError(f"{value} is not a valid data type.")

class DataType(enum.Enum):
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "val"
    REFERENCE = "reference"

    def __str__(self):
        return self.value

    @classmethod
    def from_string(cls, value):
        for item in cls:
            if item.value == value:
                return item
        raise ValueError(f"{value} is not a valid data type.")

def _compute_target_tol(fmax, mode="relative", floor=0.01, lower=0.005, upper=0.03, factor=0.9):
    """Return a per-system force tolerance in eV/Å (positive number)."""
    if fmax is None or not (fmax == fmax) or fmax <= 0:
        return floor
    if mode == "relative":
        t = factor * float(fmax)           # slightly stricter than last fmax
        t = max(lower, min(t, upper))      # clamp
        t = max(t, floor)                  # never looser than floor
        return t
    elif mode == "constant":
        return floor
    else:
        raise ValueError("mode should be 'relative' or 'constant'")

def mkdir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Creating directory: {directory}")
    else:
        logging.info(f"Directory already exists: {directory}")

def preprocess(csvFile1:str, csvFile2:str, outputFile:str):
    """
    Preprocess two CSV files and save the result to a new CSV file.
    The function assumes that both CSV files have a column named 'atoms_id'.
    """
    df1 = pd.read_csv(csvFile1)
    df2 = pd.read_csv(csvFile2)
    
    # Ensure 'atoms_id' is treated as string
    df1['atoms_id'] = df1['atoms_id'].astype(str)
    df2['atoms_id'] = df2['atoms_id'].astype(str)
    
    # Merge the two DataFrames on 'atoms_id'
    merged_df = pd.merge(df1, df2, on='atoms_id', how='outer')
    
    # Split it into train and val sets of 8:1 ratio and output as two csvs
    # replace the original files but with the original columns with a backup
    if os.path.exists(os.path.join(outputFile, 'train.csv')):
        backupFile = os.path.join(outputFile, 'train_backup.csv')
        if not os.path.exists(backupFile):
            os.rename(os.path.join(outputFile, 'train.csv'), backupFile)
            logging.info(f"Backup of the original file created: {backupFile}")
        else:
            logging.warning(f"Backup file already exists: {backupFile}. Original file will not be overwritten.")
    if os.path.exists(os.path.join(outputFile, 'val.csv')):
        backupFile = os.path.join(outputFile, 'val_backup.csv')
        if not os.path.exists(backupFile):
            os.rename(os.path.join(outputFile, 'val.csv'), backupFile)
            logging.info(f"Backup of the original file created: {backupFile}")
        else:
            logging.warning(f"Backup file already exists: {backupFile}. Original file will not be overwritten.")
    train_df = merged_df.sample(frac=0.8, random_state=42)
    val_df = merged_df.drop(train_df.index)
    train_df.to_csv(os.path.join(outputFile, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(outputFile, 'val.csv'), index=False)

def get_create_log_filename_now(basicName:str):
    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")     # e.g. 2025-06-13-13-17-42
    fileName = f"{basicName}-{stamp}.log"
    mkdir_if_not_exists(os.path.dirname(fileName))
    return os.path.abspath(fileName)

def convert_data_file(input_file, output_directory, output_type:FileType, system:str, specorder=None, isRef:bool=False):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    inputfileName = os.path.basename(input_file)
    if isRef:
        system_id = inputfileName.split('.')[0]
        relaxed = "ref"
    else:
        system_id = inputfileName.split('_')[-2] if len(inputfileName.split('_')) <= 3 else f"{inputfileName.split('_')[-3]}_{inputfileName.split('_')[-2]}"
        relaxed = inputfileName.split('_')[-1].split('.')[0]
    logging.info(f"Converting {input_file} to {output_directory} for system ID: {system_id}")
    atoms = read(input_file)
    if specorder is None:
        uniques = sorted(set(atoms.get_chemical_symbols()))  # α-order
    else:
        uniques = specorder 
    # Convert to the desired output format
    if output_type == FileType.CIF:
        write(os.path.join(output_directory,
            f"{system}_{system_id}_{relaxed}.cif"), atoms, format='cif', specorder=uniques)
    elif output_type == FileType.POSCAR:
        write(os.path.join(output_directory,
            f"{system}_{system_id}_{relaxed}.POSCAR"), atoms, format='vasp', specorder=uniques)
    elif output_type == FileType.XYZ:
        write(os.path.join(output_directory,
            f"{system}_{system_id}_{relaxed}.xyz"), atoms, format='xyz', specorder=uniques)
    elif output_type == FileType.EXTXYZ:
        write(os.path.join(output_directory,
            f"{system}_{system_id}_{relaxed}.extxyz"), atoms, format='extxyz', specorder=uniques)
    elif output_type == FileType.DATA:
        write(os.path.join(output_directory,
            f"{system}_{system_id}_{relaxed}.data"), 
            atoms, 
            format='lammps-data',
            atom_style="atomic", # “full”, “charge”, etc. also possible
            units="metal", # “metal”, “real”, etc. also possible
            masses=True,
            specorder=uniques
            )
    else:
        raise ValueError(f"Unsupported output type: {output_type}")
    return uniques, f"{system}_{system_id}_{relaxed}"

LMP   = shutil.which("lmp")

def calculate(input_script, log_file):
    # device = "cuda" if torch.backends.cuda
    # logging.info(f"Using device: {device}") # No this kind of cmds in Torch 1.13.1
    logging.info(f"Using LAMMPS input script: {input_script}")
    logging.info(f"Using LAMMPS log file: {log_file}")

    cmd = [
        "-log", log_file,
        "-k", "on", "g", "1",       # turn on Kokkos with 1 GPU
        "-sf", "kk",                # activate Kokkos-accelerated styles
    ]

    tries, delay = 5, 15
    for attempt in range(1, tries + 1):
        logging.critical("▶️ attempt %d/%d", attempt, tries)
        try:
            lmp = lammps(cmdargs=cmd)
            lmp.command("info styles minimize")
            lmp.file(input_script)  # run the input script
            logging.info("LAMMPS run completed successfully.")
            lmp.close()
            lmp.finalize()  # close LAMMPS instance
            return 0                                 # success
        except RuntimeError as err:               # LAMMPS threw an error
            if "cudaMalloc" in str(err):          # OOM in Kokkos/CUDA
                logging.warning("⚠️  OOM, sleep %ds …", delay)
            else:
                logging.error("❌  %s retry after %ds", err, delay)
            time.sleep(delay)
    else:
        raise RuntimeError("gave up after %d tries" % tries)


def transfer_data_files(parentDir, lowHigh:bool, system:str, data_type:DataType, outputParent:str, refSystem:str=None):
    defectsDensityMark = "low_density_defects" if lowHigh else "high_density_defects"
    filesFromParent = os.path.join(parentDir, defectsDensityMark)
    filesToParent = os.path.join(outputParent, defectsDensityMark)
    if data_type != DataType.REFERENCE:
        filesTo = os.path.join(filesToParent, system, data_type.value)
        mkdir_if_not_exists(os.path.join(filesTo, "features"))
        mkdir_if_not_exists(os.path.join(filesTo, "labels"))
        mkdir_if_not_exists(os.path.join(filesTo, "results"))
        filesFrom = os.path.join(filesFromParent, system)
        filesDF = pd.read_csv(os.path.join(filesFrom, f"{data_type.value}.csv"))
        filesDF['atoms_id'] = filesDF['atoms_id'].astype(str)
        assert os.path.exists(os.path.join(filesFrom, "CIF_POSCAR")) or os.path.exists(os.path.join(filesFrom, "CIF")), \
            f"Directory {os.path.join(filesFrom, 'CIF_POSCAR')} or {os.path.join(filesFrom, 'CIF')} does not exist."
        subDirName = "CIF_POSCAR" if os.path.exists(os.path.join(filesFrom, "CIF_POSCAR")) else "CIF"
        elementsInputs = [
            convert_data_file(
                input_file=os.path.join(filesFrom, subDirName, f"{atoms_id}_unrelaxed.cif"),
                output_directory=os.path.join(filesTo, "features"),
                output_type=FileType.DATA,
                system=system
            ) for atoms_id in filesDF["atoms_id"]
        ]
        elementsInputsMapping = {element[1] : element[0] for element in elementsInputs}
        elementsLabels = [
            convert_data_file(
                input_file=os.path.join(filesFrom, subDirName, f"{atoms_id}_relaxed.cif"),
                output_directory=os.path.join(filesTo, "labels"),
                output_type=FileType.DATA,
                system=system
            ) for atoms_id in filesDF["atoms_id"]
        ]
        elementsLabelsMapping = {element[1] : element[0] for element in elementsLabels}
        elements = elementsInputsMapping | elementsLabelsMapping
    else:
        assert refSystem is not None, f"Reference system must be provided for DataType.REFERENCE, Now system is {system}."
        mkdir_if_not_exists(os.path.join(filesToParent, system, data_type.value))
        mkdir_if_not_exists(os.path.join(filesToParent, system, data_type.value, "results"))
        elementsList = [
            convert_data_file(
                input_file=os.path.join(filesFromParent, f"{refSystem}.cif"),
                output_directory=os.path.join(filesToParent, system, data_type.value),
                output_type=FileType.DATA,
                system=system,
                isRef=True
            )
        ]
        elements = {element[1]: element[0] for element in elementsList}
    
    return elements

def prepare_scripts(tmplFile, scriptsDir:str, systemsInfo: list[SystemInfo], totalName:str):
    with open(tmplFile, 'r') as file:
        template = jinja2.Template(file.read())
    
    script_content = template.render(system_infos=systemsInfo, total_name=totalName)
    mkdir_if_not_exists(scriptsDir)
    scriptsFile = os.path.join(scriptsDir, f"in.lammps_mace_minimization_{totalName}")
    with open(scriptsFile, 'w') as file:
        file.write(script_content)
    logging.info(f"Script prepared and saved to {scriptsFile}")
    return scriptsFile

def extract_fmax(df:pd.DataFrame, sid:str, mode, floor, lower, upper, factor, systemName:str):
    return 0.010
    start_index_of_target = sid.find(systemName)
    end_index_of_target = sid.find("_unrelaxed")
    def lookup(nameToFind):
        sub = df.loc[df["system_id"].astype(str).str.strip().eq(str(nameToFind).strip()), "fmax_eV_per_A"]
        if sub.empty:   # nothing matched → decide a fallback or raise a helpful error
            raise KeyError(f"system_id not found in fmax CSV: {sid}")
        return _compute_target_tol(float(sub.iloc[0]), mode, floor, lower, upper, factor)      
    if start_index_of_target != -1 and end_index_of_target != -1:
        chemicalHeading = chemicalDict[systemName]
        desiredStart = start_index_of_target + len(systemName) + 1
        nameToFind = f'{chemicalHeading}_{sid[desiredStart:end_index_of_target]}'
        return lookup(nameToFind)
    else:
        if sid == systemName:
            nameToFind = chemicalDict[systemName]
            return lookup(nameToFind)
        raise ValueError(f"Not correct format for: {sid}")

    
def benchmarking(parentDir:str, outputParent:str, calculateSystem: SystemDefinition, 
                 modelFile:str, logFile:str, templateFile:str, ScriptsDir:str, fmaxCsvBasic:str,
                 mode:str, floor:float, lower:float, upper:float, factor:float, using_dynamic:bool):
    refs = dict()
    systemsInfo = list()
    if not os.path.exists(logFile):
        logging.info(f"Log file {logFile} does not exist. Creating a new one.")
        with open(logFile, 'w') as f:
            f.write(f"Log file created for system: {calculateSystem.systemName}\n")
            f.write(str(calculateSystem) + "\n")
            logging.info(f"Log file {logFile} created.")
    else:
        logging.info(f"Log file {logFile} already exists. Appending to it.")
        
    # preprocess if the val csv contents overlapped with the train csv
    csvFileTrain = os.path.join(parentDir, "low_density_defects" if calculateSystem.lowHighDensity else "high_density_defects",
                            calculateSystem.systemName, "train.csv")
    csvFileVal = os.path.join(parentDir, "low_density_defects" if calculateSystem.lowHighDensity else "high_density_defects",
                            calculateSystem.systemName, "val.csv")
    csvTrain = pd.read_csv(csvFileTrain)
    csvVal = pd.read_csv(csvFileVal)
    if set(csvTrain['atoms_id']).intersection(set(csvVal['atoms_id'])):
        logging.info(f"Preprocessing {csvFileTrain} and {csvFileVal} due to overlapping atoms_id.")
        preprocess(csvFile1=csvFileTrain, csvFile2=csvFileVal, outputFile=os.path.join(parentDir, "low_density_defects" if calculateSystem.lowHighDensity else "high_density_defects",
                            calculateSystem.systemName))
    
    # for dataType in [DataType.TRAIN, DataType.TEST, DataType.VALIDATION]:
    for dataType in [DataType.TEST]:
        elements = transfer_data_files(
                parentDir=parentDir,
                lowHigh=calculateSystem.lowHighDensity,
                system=calculateSystem.systemName,
                data_type=dataType,
                outputParent=outputParent
            )
        logging.info(f"Transferred data files for {dataType.value} with elements: {elements}")
        
        fmax_csv = f"{fmaxCsvBasic}_{str(dataType)}.csv"
        dfFMax = pd.read_csv(fmax_csv)
        if "system_id" not in dfFMax.columns or "fmax_eV_per_A" not in dfFMax.columns:
            raise ValueError("CSV must contain columns: system_id, fmax_eV_per_A")
        
        systemsInfo.extend(
            SystemInfo(
                structure_file=os.path.join(outputParent, 
                                            "low_density_defects" if calculateSystem.lowHighDensity else "high_density_defects",
                                            calculateSystem.systemName, dataType.value, "features", f"{systemID}.data"),
                model_file=modelFile,
                elements=element,
                system_name=calculateSystem.systemName,
                output_file=os.path.join(outputParent, 
                                            "low_density_defects" if calculateSystem.lowHighDensity else "high_density_defects",
                                            calculateSystem.systemName, 
                                            dataType.value,
                                            "results", f"{systemID}.data"),
                dump_name=f"{systemID}", 
                ftol=extract_fmax(dfFMax, systemID, mode, floor, lower, upper, factor, calculateSystem.systemName) if using_dynamic else floor
            ) for systemID, element in elements.items() if "unrelaxed" in systemID 
        )
        logging.info(f"Prepared system info for {dataType.value} with {len(systemsInfo)} systems.")
        
    for refSystem in calculateSystem.referenceSystem:
        refs.update(
            transfer_data_files(
                parentDir=parentDir,
                lowHigh=calculateSystem.lowHighDensity,
                system=calculateSystem.systemName,
                data_type=DataType.REFERENCE,
                outputParent=outputParent,
                refSystem=refSystem
            )
        )
    systemsInfo.extend(
        SystemInfo(
            structure_file=os.path.join(outputParent,
                                            "low_density_defects" if calculateSystem.lowHighDensity else "high_density_defects",
                                            calculateSystem.systemName, "reference", f"{systemID}.data"),
            model_file=modelFile,
            elements=element,
            system_name=systemID.split('_')[0],
            output_file=os.path.join(outputParent,
                                            "low_density_defects" if calculateSystem.lowHighDensity else "high_density_defects",
                                            calculateSystem.systemName, "reference", "results", f"{systemID}.data"),
            dump_name=f"{systemID}",
            ftol=extract_fmax(dfFMax, calculateSystem.systemName, mode, floor, lower, upper, factor, calculateSystem.systemName)  if using_dynamic else floor
        ) for systemID, element in refs.items()
    )

    scriptsFile = prepare_scripts(
        tmplFile=templateFile,
        scriptsDir=ScriptsDir,
        systemsInfo=systemsInfo,
        totalName=calculateSystem.systemName
    )
    
    calculate(
        input_script=scriptsFile,
        log_file=logFile
    )
    
def main():
    
    argparser = argparse.ArgumentParser(description="Benchmarking workflow for DefiNet systems.")
    argparser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = argparser.parse_args()
    config_file = args.config
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file {config_file} does not exist.")
    with open(config_file, 'r') as file:
        configs = json.load(file)
        
    outerLog = configs["outer_log"]
    
    logging.basicConfig(
        filename=outerLog,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
        
    for config in configs["calculations"]:
        logging.info(f"Loaded configuration: {config}")
        
        ftol_setting = config["ftol_setting"]
        
        referenceSystems = config["reference_systems"]
        lowHighDensity = config["low_high_density"]
        systemName = config["system_name"]
        parentDir = config["parent_directory"]
        outputParent = config["output_directory"]
        modelFile = config["model_file"]
        logFile = get_create_log_filename_now(config["log_file"].replace(".log", f"_{systemName}") if config["log_file"].endswith(".log") else config["log_file"])
        templateFile = config["template_file"]
        ScriptsDir = config["scripts_directory"]
        if not os.path.exists(parentDir):
            raise FileNotFoundError(f"Parent directory {parentDir} does not exist.")
        if not os.path.exists(modelFile):
            raise FileNotFoundError(f"Model file {modelFile} does not exist.")
        if not os.path.exists(templateFile):
            raise FileNotFoundError(f"Template file {templateFile} does not exist.")
        
        mkdir_if_not_exists(outputParent)
        mkdir_if_not_exists(ScriptsDir)
        logFile = os.path.abspath(logFile)
        logging.info(f"*******************************\nLog file will be written to: {logFile}\n*******************************\n")
        logging.info(f"Output directory: {outputParent}")
        logging.info(f"Scripts directory: {ScriptsDir}")
        logging.info(f"Parent directory: {parentDir}")
        logging.info(f"Model file: {modelFile}")
        logging.info(f"Template file: {templateFile}")
        logging.info(f"Reference system: {referenceSystems}")
        logging.info(f"Low/High density: {lowHighDensity}")
        logging.info(f"System name: {systemName}")
        # Define the system to be calculated
        logging.info(f"Starting benchmarking for system: {systemName}")
        logging.info(f"Low/High density: {lowHighDensity}")
        
        calculateSystem = SystemDefinition(
            referenceSystem=referenceSystems,
            lowHighDensity=lowHighDensity,
            systemName= systemName
        )
        
        benchmarking(parentDir, outputParent, calculateSystem, modelFile, logFile, templateFile, ScriptsDir,
                     ftol_setting["basic"], ftol_setting["mode"], ftol_setting["floor"], 
                     ftol_setting["lower"], ftol_setting["upper"], ftol_setting["factor"], ftol_setting["using_dynamic"])

if __name__ == "__main__":
    main()
    logging.info("Benchmarking workflow completed successfully.")
# This script is designed to be run as a standalone module.