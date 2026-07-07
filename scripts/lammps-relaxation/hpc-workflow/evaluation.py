import numpy as np
from ase.io import read
from sklearn.metrics import mean_absolute_error
from scipy.spatial.transform import Rotation
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.io.ase import AseAtomsAdaptor
from dscribe.descriptors import SOAP as DScribeSOAP
import matplotlib.pyplot as plt
import os

def eval_pos_mae(labelDataFile, predDataFile, turnOffAlignment=False):
    # ---- 1. Read the two LAMMPS data files --------------------
    label  = read(labelDataFile, format='lammps-data')   # ground-truth
    pred = read(predDataFile, format='lammps-data')   # inference
    # ---- 2. Optional: rigid-body alignment (Kabsch) ----------
    # Coordinates as (N, 3) NumPy arrays
    R_ref  = label.get_positions()
    R_pred = pred.get_positions()
    # Superpose pred onto ref
    if not turnOffAlignment:
        # ---- 2.1. Kabsch algorithm (SciPy ≥1.9) ----------------
        rot, rmsd = Rotation.align_vectors(R_ref, R_pred)   # SciPy ≥1.9
        R_pred_aligned = R_pred @ rot.as_matrix().T
    # ---- 3. Mean-absolute-error in Å --------------------------
    mae = mean_absolute_error(R_ref.flatten(), R_pred_aligned.flatten())
    print(f"Cartesian MAE = {mae:.4f} Å   (RMSD after fit = {rmsd:.4f} Å)")
    return mae, rmsd

def eval_rmsd_xdap_xrd(labelDataFile, predDataFile):
    # read LAMMPS data
    ref = read(labelDataFile, format='lammps-data')
    pred = read(predDataFile, format='lammps-data')
    
    adaptor = AseAtomsAdaptor()
    ref_struct  = adaptor.get_structure(ref)     # pymatgen Structure
    pred_struct = adaptor.get_structure(pred)

    # --- lattice & RMS match (pymatgen) ---
    sm  = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
    rms = sm.get_rms_dist(ref_struct, pred_struct)

    # --- SOAP kernel distance ---
    species = list({*ref.get_chemical_symbols(),
                    *pred.get_chemical_symbols()})

    soap = DScribeSOAP(species=species, r_cut=5, n_max=8, l_max=6,
                    sigma=0.5, periodic=True)

    vec_ref  = soap.create(ref).mean(axis=0)     # average over atoms
    vec_pred = soap.create(pred).mean(axis=0)

    d_soap = np.linalg.norm(vec_ref - vec_pred)

    # --- Rwp from powder XRD ---
    xcalc = XRDCalculator(wavelength="CuKa")
    # scaled=False builds a uniform 2θ grid common to any structure
    p_ref  = xcalc.get_pattern(ref_struct,  scaled=False)
    p_pred = xcalc.get_pattern(pred_struct, scaled=False)

    i_pred_on_ref = np.interp(p_ref.x, p_pred.x, p_pred.y)  # 361-point vector
    Rwp = np.sqrt(((p_ref.y - i_pred_on_ref)**2).sum() /(p_ref.y**2).sum())

    return rms, d_soap, Rwp

def compare_in_batch(preds:list[str], labels:list[str], turnOffAlignment=False):
    mae_list = []
    rmsd1List = []
    rmsd2List1 = []
    rmsd2List2 = []
    dsoapList = []
    rwpList = []
    if len(preds) != len(labels):
        raise ValueError("The number of predictions and labels must match.")

    for pred, label in zip(preds, labels):
        mae, rmsd1 = eval_pos_mae(label, pred, turnOffAlignment)
        rmsd2, dsoap, rwp = eval_rmsd_xdap_xrd(label, pred)
        mae_list.append(mae)
        rmsd1List.append(rmsd1)
        if rmsd2 is not None:
            rmsd2List1.append(rmsd2[0])
            rmsd2List2.append(rmsd2[1])
        dsoapList.append(dsoap)
        rwpList.append(rwp)
    return mae_list, rmsd1List, rmsd2List1, rmsd2List2, dsoapList, rwpList

def draw_compare_graphgraph(systemName, resultsDir:str, outputDir:str, datatypes:list[str]=["train", "test", "val"]):
    resultsRecords = {}
    for dataType in datatypes:
        labelsTrain = []
        predsTrain = []
        for resultFile in os.listdir(os.path.join(resultsDir, systemName, dataType, "results")):
            if resultFile.endswith('.data'):
                predsTrain.append(os.path.join(resultsDir, systemName, dataType, "results", resultFile))
                systemID = resultFile.split('.')[0].split('_')[-2] if len(resultFile.split('.')[0].split('_')) <= 3 else f'{resultFile.split(".")[0].split("_")[-3]}_{resultFile.split(".")[0].split("_")[-2]}'
                labelsTrain.append(os.path.join(resultsDir, systemName, dataType, "labels", f"{systemName}_{systemID}_relaxed.data"))
        mae_train, rmsd1_train, rmsd2_train1, rmsd2_train2, dsoap_train, rwp_train = compare_in_batch(predsTrain, labelsTrain)
        resultsRecords[f"{dataType}_mae"] = mae_train
        resultsRecords[f"{dataType}_rmsd1"] = rmsd1_train
        resultsRecords[f"{dataType}_rmsd2_mean"] = rmsd2_train1
        resultsRecords[f"{dataType}_rmsd2_max"] = rmsd2_train2
        resultsRecords[f"{dataType}_dsoap"] = dsoap_train
        resultsRecords[f"{dataType}_rwp"] = rwp_train
    refSystemLabels = []
    refSystemPreds = []
    for refFile in os.listdir(os.path.join(resultsDir, systemName, "reference")):
        if refFile.endswith('.data') and refFile.startswith(systemName):
            refSystemLabels.append(os.path.join(resultsDir, systemName, "reference", refFile))
            refSystemPreds.append(os.path.join(resultsDir, systemName, "reference", "results", refFile))
    mae_ref, rmsd1_ref, rmsd2_ref1, rmsd2_ref2, dsoap_ref, rwp_ref = compare_in_batch(refSystemPreds, refSystemLabels)
    resultsRecords["ref_mae"] = mae_ref
    resultsRecords["ref_rmsd1"] = rmsd1_ref
    resultsRecords["ref_rmsd2_mean"] = rmsd2_ref1
    resultsRecords["ref_rmsd2_max"] = rmsd2_ref2
    resultsRecords["ref_dsoap"] = dsoap_ref
    resultsRecords["ref_rwp"] = rwp_ref
    
    print("Results Records:")
    for key, value in resultsRecords.items():
        print(f"{key}: {value}")
    
    avgs = {}
    for key in resultsRecords:
        avgs[key] = np.mean(resultsRecords[key])
    print("Average Results:")
    for key, value in avgs.items():
        print(f"{key}: {value:.4f}")
        
    with open(os.path.join(outputDir, f"{systemName}_results.txt"), 'w') as f:
        f.write("Average Results:\n")
        for key, value in avgs.items():
            f.write(f"{key}: {value:.4f}\n")
    
    # Draw graphs, with different colors for each data type and columns for each averages
    # Not draw the original data, only the averages
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.flatten()
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    labels = ['train', 'test', 'val', 'reference']
    for i, (key, color) in enumerate(zip(['mae', 'rmsd1', 'rmsd2_mean', 'rmsd2_max', 'dsoap', 'rwp'], colors)):
        for j, dataType in enumerate(datatypes + ['ref']):
            if f"{dataType}_{key}" in resultsRecords:
                axs[i].bar(j, avgs[f"{dataType}_{key}"], color=color, label=dataType if j == 0 else "")
        axs[i].set_title(f"{key.upper()} Comparison")
        axs[i].set_xticks(range(len(datatypes) + 1))
        axs[i].set_xticklabels(datatypes + ['ref'])
        axs[i].set_ylabel(key.upper())
        axs[i].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outputDir, f"{systemName}_comparison.png"))

def calculate(config:dict):
    systemName = config["name"]
    resultsDir = config["resultsDir"]
    outputDir = config["outputDir"]
    dataTypes = config["dataTypes"]
    
    draw_compare_graphgraph(systemName, resultsDir, outputDir, dataTypes)
    print(f"Evaluation completed for {systemName}.")

import concurrent.futures as cf

def calculate_batch(configs:list[dict]):
    with cf.ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(calculate, config) for config in configs]
        for future in cf.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error occurred: {e}")

import json

def load_configs(config_file:str):
    with open(config_file, 'r') as f:
        jsonDict = json.load(f)
    configs = jsonDict["systems"]
    return configs

import argparse

def main():
    parser = argparse.ArgumentParser(description="Evaluate LAMMPS data files.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()

    configs = load_configs(args.config)
    calculate_batch(configs)
    
if __name__ == "__main__":
    main()
    # Example usage:
    # calculate({"name": "MoS2", "resultsDir": "/path/to/results", "outputDir": "/path/to/output", "dataTypes": ["train", "test", "val"]})
    # calculate_batch([{"name": "MoS2", "resultsDir": "/path/to/results", "outputDir": "/path/to/output", "dataTypes": ["train", "test", "val"]}])
    
            