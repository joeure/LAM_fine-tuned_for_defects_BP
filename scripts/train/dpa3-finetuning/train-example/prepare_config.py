import glob, json, math, random

BASE = "../data-transfer/data/train"        # or "data/train" if you already put all systems there
systems = sorted(glob.glob(f"{BASE}/system.*"))

k = max(1, math.ceil(len(systems)*0.05))   # 5% with at least 1 system
rng = random.Random(42)                    # fixed seed for reproducibility, later change it!!!!
valid = set(rng.sample(systems, k))
train = [p for p in systems if p not in valid]

cfg = {
  "training": {
    "training_data":   {"systems": train,  "batch_size": 4},
    "validation_data": {"systems": sorted(valid), "batch_size": 4},
    "numb_steps": 1500, ## steps_per_pass = ceil(N_train_frames / batch_size) ~ 95-100  num_steps / steps_per_pass ~ 30 (Main run 25~ 50) -> 15 (quick test)
    "disp_file": "lcurve.out",
    "save_ckpt": "ckpt/ckpt", "save_freq": 300, "disp_freq": 50 # save 5 times, disp 30 times
  },
  "learning_rate": {"type":"exp","start_lr":1e-4,"stop_lr":1e-6,"decay_steps":1500},
  "loss": {"type":"ener","start_pref_e":0.2,"limit_pref_e":1.0,
           "start_pref_f":150.0,"limit_pref_f":1.0,
           "start_pref_v":0.0,"limit_pref_v":0.0}
}
with open("input.json","w") as f: json.dump(cfg, f, indent=2)
print(f"train={len(train)} systems, valid={len(valid)} systems")

