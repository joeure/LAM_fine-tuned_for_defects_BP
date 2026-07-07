#!/usr/bin/env python3
import argparse, os, sys, traceback, time
from glob import glob

from ase.io import read, write
from ase.optimize import LBFGS, FIRE
from ase.io.trajectory import Trajectory

# MACE imports: adjust if your install uses a different path
from mace.calculators import MACECalculator

def relax_once(
    cif_path: str,
    model_path: str,
    out_relaxed: str,
    out_traj: str,
    out_log: str,
    device: str = "cuda",
    fmax: float = 0.01,
    maxstep: float = 0.04,
    dtype: str = "float32",
    start_with_fire: bool = False,
):
    atoms = read(cif_path)
    atoms.pbc = True  # mirror paper's fixed-cell ionic relax

    calc = MACECalculator(
        model=model_path,
        device=device,
        default_dtype=dtype,
        # stress not needed for fixed-cell; leave default
    )
    atoms.calc = calc

    # --- Optimizer: (optional) FIRE warmup then LBFGS, else LBFGS only ---
    logfile = open(out_log, "w", buffering=1)
    def _run_opt(opt):
        # write trajectory every few steps
        traj = Trajectory(out_traj, "w", atoms)
        opt.attach(traj.write, interval=5)
        opt.run(fmax=fmax)
        traj.close()

    try:
        if start_with_fire:
            fire = FIRE(atoms, logfile=logfile, maxstep=maxstep)
            fire.run(fmax=min(fmax * 5.0, 0.05))  # loosened target to settle quickly
        lbfgs = LBFGS(atoms, logfile=logfile, maxstep=maxstep)
        _run_opt(lbfgs)

        write(out_relaxed, atoms)
        return True, None

    except RuntimeError as e:
        # GPU OOM or numerics? Fallbacks:
        msg = str(e)
        traceback.print_exc(file=logfile)
        try:
            # 1) try CPU float64 fallback
            logfile.write("\n[Fallback] Retrying on CPU in float64...\n")
            atoms.calc = MACECalculator(model=model_path, device="cpu", default_dtype="float64")
            lbfgs = LBFGS(atoms, logfile=logfile, maxstep=maxstep*0.75)
            _run_opt(lbfgs)
            write(out_relaxed, atoms)
            return True, f"GPU fallback used: {msg}"
        except Exception as e2:
            traceback.print_exc(file=logfile)
            return False, f"Failed on GPU and CPU: {msg} | {e2}"
    finally:
        logfile.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="Folder with CIF files")
    ap.add_argument("--model", required=True, help=".model path")
    ap.add_argument("--outputs", default="outputs", help="Output folder")
    ap.add_argument("--device", default="cuda", choices=["cuda","cpu"])
    ap.add_argument("--fmax", type=float, default=0.01)
    ap.add_argument("--maxstep", type=float, default=0.04)
    ap.add_argument("--dtype", default="float32", choices=["float32","float64"])
    ap.add_argument("--fire_warmup", action="store_true",
                    help="Run a short FIRE phase before LBFGS for extra robustness")
    args = ap.parse_args()

    os.makedirs(args.outputs, exist_ok=True)
    out_rel = os.path.join(args.outputs, "relaxed")
    out_trj = os.path.join(args.outputs, "traj")
    out_log = os.path.join(args.outputs, "logs")
    for d in (out_rel, out_trj, out_log):
        os.makedirs(d, exist_ok=True)

    cif_files = sorted(glob(os.path.join(args.inputs, "*.cif")))
    if not cif_files:
        print("No CIFs found in", args.inputs, file=sys.stderr)
        sys.exit(1)

    summary = []
    t0 = time.time()
    for i, cif in enumerate(cif_files, 1):
        stem = os.path.splitext(os.path.basename(cif))[0]
        out_relaxed = os.path.join(out_rel, f"{stem}_relaxed.cif")
        out_traj = os.path.join(out_trj, f"{stem}.traj")
        out_log = os.path.join(out_log, f"{stem}.log")
        print(f"[{i}/{len(cif_files)}] {stem} -> relaxing...", flush=True)

        ok, note = relax_once(
            cif_path=cif,
            model_path=args.model,
            out_relaxed=out_relaxed,
            out_traj=out_traj,
            out_log=out_log,
            device=args.device,
            fmax=args.fmax,
            maxstep=args.maxstep,
            dtype=args.dtype,
            start_with_fire=args.fire_warmup,
        )
        status = "OK" if ok else "FAIL"
        summary.append((stem, status, note or ""))

        # Friendly GPU memory hygiene between jobs
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    dt = time.time() - t0
    print("\n=== Batch summary ===")
    for s, status, note in summary:
        print(f"{s:30s} {status:4s} {note}")
    print(f"Total wall time: {dt/60:.1f} min for {len(cif_files)} systems")

if __name__ == "__main__":
    main()
