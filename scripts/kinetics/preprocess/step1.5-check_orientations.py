import pandas as pd
import numpy as np

df = pd.read_csv("../data_center/nearest_nerighbors_test.csv", low_memory=False)
p = df[(df["is_pristine"].astype(str).str.lower().isin(["true","1","t","yes","y"])) &
       (df["structure"].astype(str).str.strip() == "dft") &
       (df["natoms"].astype(int) == 144)].copy()

# use existing projections (only valid for the AC/ZZ you used when generating CSV!)
a = np.abs(p["proj_ac_A"].to_numpy())
z = np.abs(p["proj_zz_A"].to_numpy())
eps = 1e-12

def counts(thr):
    ratio = a / np.maximum(z, eps)
    inv   = z / np.maximum(a, eps)
    lab = np.where(ratio >= thr, "AC", np.where(inv >= thr, "ZZ", "MIXED"))
    vals, cnts = np.unique(lab, return_counts=True)
    return dict(zip(vals, cnts))

for thr in [1.1, 1.2, 1.3, 1.5, 1.8]:
    print(thr, counts(thr))