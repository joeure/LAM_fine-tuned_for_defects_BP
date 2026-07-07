#!/usr/bin/env python3
"""
Export last-K MACE checkpoints (that contain only a state_dict) to real .model files
by loading a template .model (same architecture) and then loading each epoch's state_dict.

Usage example:
  python export_from_state_dict.py \
    --ckpt_glob 'checkpoints/omat_finetune_BP_smoketest_run-*_epoch-*.pt' \
    --keep_last 4 \
    --template_model results/omat_finetune_BP_smoketest_run-42.model \
    --head omat_definet_ft_head \
    --compile --format libtorch
"""

import argparse, glob, os, re, subprocess, sys, copy
from pathlib import Path
from collections import OrderedDict
from collections.abc import Mapping

import torch

# ----------------- helpers -----------------
def epoch_key(p):
    m = re.search(r"_epoch-(\d+)\.pt$", p)
    return int(m.group(1)) if m else -1

def set_threads(omp=None, blas=None, torch_threads=None, cuda=False):
    if omp is not None:
        os.environ["OMP_NUM_THREADS"] = str(omp)
        os.environ.setdefault("OMP_PROC_BIND", "spread")
        os.environ.setdefault("OMP_PLACES", "cores")
    if blas is not None:
        os.environ["MKL_NUM_THREADS"] = str(blas)
        os.environ["OPENBLAS_NUM_THREADS"] = str(blas)
        os.environ["NUMEXPR_NUM_THREADS"] = str(blas)
    if torch_threads is not None:
        try:
            torch.set_num_threads(int(torch_threads))
            try:
                torch.set_num_interop_threads(max(1, int(torch_threads)//4))
            except Exception:
                pass
        except Exception:
            pass
    if cuda:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES","0"))

def find_heads_in_ckpt(ckpt_path):
    heads = set()
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # prefer state_dict if present
    sd = None
    if isinstance(ckpt, Mapping) and isinstance(ckpt.get("state_dict"), Mapping):
        sd = ckpt["state_dict"]
    # many runs store the state_dict under 'model' (your case)
    if sd is None and isinstance(ckpt, Mapping) and isinstance(ckpt.get("model"), Mapping):
        sd = ckpt["model"]
    # fallback: whole object might be an OrderedDict
    if sd is None and isinstance(ckpt, (Mapping, OrderedDict)):
        sd = ckpt

    if isinstance(sd, Mapping):
        for k in sd.keys():
            m = re.match(r"heads\.([^\.]+)\.", k)
            if m:
                heads.add(m.group(1))
    return sorted(heads)

def unpack_ckpt(ckpt):
    """
    Return (module_or_None, state_dict_or_None).
    Your files have: ckpt['model'] = OrderedDict(...) and no 'config'.
    """
    module = None
    state_dict = None

    if isinstance(ckpt, Mapping):
        mod = ckpt.get("model", None)
        # some forks put a live Module here; yours does not (it's an OrderedDict)
        if hasattr(mod, "state_dict"):
            module = mod
            try:
                state_dict = mod.state_dict()
            except Exception:
                state_dict = None
        # common case for your run: 'model' is the raw state_dict
        elif isinstance(mod, (Mapping, OrderedDict)):
            state_dict = mod
        elif isinstance(ckpt.get("state_dict"), Mapping):
            state_dict = ckpt["state_dict"]
        else:
            # maybe checkpoint itself is a raw state_dict
            if isinstance(ckpt, OrderedDict):
                state_dict = ckpt
    elif isinstance(ckpt, OrderedDict):
        state_dict = ckpt

    return module, state_dict

def compile_via_cli(model_path, head, python_exe=None, fmt="libtorch"):
    """Run the official converter on a serialized .model (Module)."""
    py = python_exe or sys.executable
    cmd = [
        py, "-m", "mace.cli.create_lammps_model", str(model_path),
        "--head", head or "", "--format", fmt
    ]
    print("[compile:cli]", " ".join(cmd))
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if res.returncode != 0:
        print(res.stdout)
        raise RuntimeError(f"CLI compile failed for {model_path} (head={head})")
    else:
        tail = "\n".join(res.stdout.splitlines()[-20:])
        print(tail)

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser(description="Export last-K state_dict checkpoints to real .model using a template model.")
    ap.add_argument("--ckpt_glob", required=True, help="e.g. checkpoints/run-42_epoch-*.pt")
    ap.add_argument("--keep_last", type=int, default=4, help="how many latest to export")
    ap.add_argument("--outdir", type=Path, default=Path("checkpoints"))

    # required for reconstruction in your case
    ap.add_argument("--template_model", type=Path, required=True,
                    help="Path to a real .model (same architecture) to load as a template")

    # conversion
    ap.add_argument("--compile", action="store_true", help="convert .model to LAMMPS after export")
    ap.add_argument("--format", choices=["libtorch","mliap"], default="libtorch",
                    help="LAMMPS format for CLI conversion")
    ap.add_argument("--python_exe", type=str, default=None,
                    help="Python executable for -m mace.cli.create_lammps_model")

    # heads
    ap.add_argument("--head", action="append", default=None,
                    help="Head name to export/compile. Repeat, or 'all'. If omitted and multiple heads exist, the script lists heads and aborts.")
    ap.add_argument("--list_heads", action="store_true", help="List heads in the first matched checkpoint and exit.")

    # threads
    ap.add_argument("--omp_threads", type=int, default=None)
    ap.add_argument("--blas_threads", type=int, default=None)
    ap.add_argument("--torch_threads", type=int, default=None)
    ap.add_argument("--cuda", action="store_true")

    args = ap.parse_args()
    set_threads(args.omp_threads, args.blas_threads, args.torch_threads, args.cuda)

    # load template Module once
    if not args.template_model.exists():
        raise SystemExit(f"--template_model not found: {args.template_model}")
    try:
        template = torch.load(args.template_model, map_location="cpu")
        template.eval()
    except Exception as e:
        raise SystemExit(f"Failed to load template model {args.template_model}: {e}")

    # collect checkpoints
    paths = sorted(glob.glob(args.ckpt_glob), key=epoch_key)
    if not paths:
        raise SystemExit("No checkpoints matched.")
    paths = paths[-args.keep_last:]
    args.outdir.mkdir(parents=True, exist_ok=True)

    # optional: head detection
    if args.list_heads:
        heads = find_heads_in_ckpt(paths[-1])
        print("Heads in", paths[-1], ":", ", ".join(heads) if heads else "(none detected)")
        return

    # decide requested heads
    requested_heads = args.head
    if requested_heads is None:
        detected = find_heads_in_ckpt(paths[-1])
        if len(detected) == 0:
            requested_heads = [None]
        elif len(detected) == 1:
            requested_heads = detected
        else:
            print("Multiple heads detected:", detected)
            raise SystemExit("Please specify --head <name> (repeatable) or --head all")
    elif len(requested_heads) == 1 and requested_heads[0].lower() == "all":
        detected = find_heads_in_ckpt(paths[-1])
        requested_heads = detected if detected else [None]

    # per-ckpt export
    for ckpt_path in paths:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        module_in_ckpt, state_dict = unpack_ckpt(ckpt)
        if state_dict is None and module_in_ckpt is None:
            print(f"[skip] {ckpt_path}: no state_dict found")
            continue

        heads_here = find_heads_in_ckpt(ckpt_path)

        for head in requested_heads:
            if head is not None and heads_here and head not in heads_here:
                print(f"[warn] head '{head}' not found in {ckpt_path}; available: {heads_here}. Skipping.")
                continue

            base = Path(ckpt_path).with_suffix("").name
            tag  = f".{head}" if head else ""
            out_model = args.outdir / f"{base}{tag}.model"
            print(f"[export] {ckpt_path} -> {out_model}")

            # clone the template, load this epoch's weights
            built = copy.deepcopy(template)
            # load state dict (strict=False to tolerate e.g., extra heads)
            missing, unexpected = built.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                print(f"  [warn] load_state_dict mismatches → missing: {len(missing)}, unexpected: {len(unexpected)}")

            try:
                built.eval()
            except Exception:
                pass

            # save as a real .model
            try:
                from mace.tools import save_model
                save_model(built, str(out_model))
                print(f"  [ok] wrote real .model → {out_model}")
            except Exception as e:
                # fallback to torch.save(module) (usually works too)
                torch.save(built, str(out_model))
                print(f"  [ok] wrote torch-saved Module → {out_model} (save_model failed: {e})")

            # optional: convert to LAMMPS
            if args.compile:
                try:
                    compile_via_cli(out_model, head or (heads_here[0] if heads_here else ""), args.python_exe, fmt=args.format)
                except Exception as e:
                    print(f"  [warn] CLI compile failed: {e}")

if __name__ == "__main__":
    main()
