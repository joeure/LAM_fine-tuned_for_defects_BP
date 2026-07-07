import pandas as pd
a=pd.read_csv("results_foundation.csv")
b=pd.read_csv("results_finetuned.csv")
m=a.merge(b, on="system", suffixes=("_base","_ft"))
m["dE(meV/atom)"]=1000*(m.energy_rmse_per_atom_base-m.energy_rmse_per_atom_ft)
m["dF(eV/A)"]   =(m.force_rmse_base-m.force_rmse_ft)
print(m[["system","energy_rmse_per_atom_base","energy_rmse_per_atom_ft","dE(meV/atom)","dF(eV/A)"]].to_string(index=False))