import glob, os
UNIFIED = ["P","N"]  # add any other dopants here in a stable order
for root in ["data/train", "data/valid"]:
    for sysdir in sorted(glob.glob(f"{root}/system.*")):
        with open(os.path.join(sysdir, "type_map.raw"), "w") as f:
            f.write("\n".join(UNIFIED) + "\n")
