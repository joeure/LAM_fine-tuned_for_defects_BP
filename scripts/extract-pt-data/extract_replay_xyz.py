#!/usr/bin/env python3
import argparse, random, math, sys
from pathlib import Path
from typing import List, Tuple, Optional, Iterable

import numpy as np
from ase import Atoms
from ase.io import read as ase_read, write as ase_write
from ase.io import iread

# ----------------------------- helpers -----------------------------

from typing import List, Iterable, Sequence, Set, Union, Optional
import random, numpy as np
from ase import Atoms
from ase.data import chemical_symbols, atomic_numbers

Elem = Union[str, int]

def _normalize_required(required: Optional[Iterable[Elem]]) -> Set[str]:
    if not required:
        return set()
    out = set()
    for e in required:
        if isinstance(e, int):
            if e <= 0 or e >= len(chemical_symbols):
                raise ValueError(f"Unknown Z={e}")
            out.add(chemical_symbols[e])
        else:
            s = str(e).strip()
            # accept case-insensitive symbols, e.g. 'si' → 'Si'
            if s.isdigit():
                z = int(s); out.add(chemical_symbols[z])
            else:
                # map to proper case if possible
                Z = atomic_numbers.get(s.capitalize(), None)
                if Z is None:
                    raise ValueError(f"Unknown element '{e}'")
                out.add(chemical_symbols[Z])
    return out

def _fps_within(indices: Sequence[int], k: int, frames: List[Atoms], rng: random.Random) -> List[int]:
    if k <= 0 or not indices:
        return []
    if k >= len(indices):
        return list(sorted(indices))
    # centroids for candidates only
    X = np.asarray([frames[i].get_positions(wrap=True).mean(axis=0) for i in indices], float)
    # start from a random candidate
    j0 = rng.randrange(len(indices))
    sel_local = [j0]
    d2 = np.sum((X - X[j0])**2, axis=1)
    for _ in range(1, k):
        j = int(np.argmax(d2))
        sel_local.append(j)
        d2 = np.minimum(d2, np.sum((X - X[j])**2, axis=1))
    # map local (0..len(indices)-1) back to original frame indices
    return sorted([indices[j] for j in sel_local])

def _pick_within(indices: Sequence[int], k: int, frames: List[Atoms], rng: random.Random, method: str) -> List[int]:
    if k <= 0 or not indices:
        return []
    if k >= len(indices):
        return list(sorted(indices))
    method = method.lower()
    if method == "random":
        return sorted(rng.sample(list(indices), k))
    elif method == "fps":
        return _fps_within(indices, k, frames, rng)
    else:
        raise ValueError(f"Unknown sampling method '{method}' (use 'random' or 'fps').")

def choose_subset(
    n: int,
    frames: List[Atoms],
    seed: int = 42,
    method: str = "random",
    required_elements: Optional[Iterable[Elem]] = None,
    require_all: bool = True,
) -> List[int]:
    """
    Return indices of a size-n subset.
    - required_elements: elements to prioritize (symbols or Z). Examples: {'N','P'} or {7,15}.
    - require_all=True: a frame is a 'hit' only if it contains *all* required elements.
      If False: it's a 'hit' if it contains *any* of them.
    Selection runs in two stages: first from hits, then (if needed) from the rest,
    using the same `method` within each pool.
    """
    N = len(frames)
    if n >= N:
        return list(range(N))
    rng = random.Random(seed)

    req = _normalize_required(required_elements)
    if not req:
        # no prioritization → original behavior
        return _pick_within(list(range(N)), n, frames, rng, method)

    # Build pools
    hits, others = [], []
    for i, a in enumerate(frames):
        syms = set(a.get_chemical_symbols())
        ok = (req.issubset(syms)) if require_all else (len(req & syms) > 0)
        (hits if ok else others).append(i)

    k_hits = min(n, len(hits))
    sel_hits = _pick_within(hits, k_hits, frames, rng, method)
    k_left = n - k_hits
    sel_rest = _pick_within(others, k_left, frames, rng, method) if k_left > 0 else []
    return sorted(sel_hits + sel_rest)

import os, glob, io
from typing import Iterable, List
import numpy as np
from ase import Atoms
from ase.io import iread, read

# ---------- helpers ----------
def expand_inputs(inputs: Iterable[str] | str) -> List[str]:
    if isinstance(inputs, str):
        parts = [p.strip() for p in inputs.split(",") if p.strip()]
    else:
        parts = list(inputs)
    out = []
    for p in parts:
        out.extend(sorted(glob.glob(os.path.expanduser(p))))
    # de-dup preserve order
    seen = set(); uniq = []
    for p in out:
        if p not in seen:
            uniq.append(p); seen.add(p)
    return uniq

def _parse_properties(header: str):
    # header contains ... Properties=species:S:1:pos:R:3:forces:R:3 ...
    hdr = header.strip()
    props = None
    for token in hdr.split():
        if token.startswith("Properties="):
            props = token[len("Properties="):]
            break
    if props is None:
        return []
    # split like "species:S:1:pos:R:3:forces:R:3" -> [('species','S',1), ('pos','R',3), ('forces','R',3)]
    parts = props.split(":")
    triplets = []
    i = 0
    while i < len(parts) - 2:
        name, typ, n = parts[i], parts[i+1], parts[i+2]
        # next field starts at i+3
        try:
            n = int(n)
        except Exception:
            break
        triplets.append((name, typ, n))
        i += 3
    return triplets

def _manual_read_extxyz(path: str):
    """Very small robust EXTXYZ reader for species/pos(/forces) + info energy-like fields."""
    frames = []
    with open(path, "r", errors="ignore") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                nat = int(line)
            except ValueError:
                # skip garbage until next valid block
                continue
            header = f.readline().strip()
            props = _parse_properties(header)
            # figure column layout
            cols = []
            for (name, typ, n) in props:
                for k in range(n):
                    cols.append((name, k))  # e.g. ('pos',0..2), ('forces',0..2)
            # read nat atom lines
            raw = [f.readline().split() for _ in range(nat)]
            # species first?
            # find 'species' field (string), 'pos' (3), and optionally 'forces' (3)
            # If Properties lists 'species' first (common), OK; if not, still OK via lookup.
            idx_species = None
            idx_pos = None
            idx_forces = None
            # Build column index map
            offset = 0
            name_offsets = {}
            for (name, typ, n) in props:
                name_offsets[name] = (offset, n)
                offset += n
            # Extract
            if "species" not in name_offsets or "pos" not in name_offsets:
                # not an atom block we recognize
                continue
            off_s, n_s = name_offsets["species"]
            off_p, n_p = name_offsets["pos"]
            sym = [row[off_s] for row in raw]
            pos = np.array([[float(row[off_p+i]) for i in range(3)] for row in raw], float)
            forces = None
            if "forces" in name_offsets:
                off_f, n_f = name_offsets["forces"]
                try:
                    forces = np.array([[float(row[off_f+i]) for i in range(3)] for row in raw], float)
                except Exception:
                    forces = None

            at = Atoms(symbols=sym, positions=pos, pbc=True)
            # very light parse of lattice if present
            if 'Lattice="' in header:
                try:
                    lat_str = header.split('Lattice="',1)[1].split('"',1)[0].strip()
                    L = np.fromstring(lat_str, sep=" ")
                    if L.size == 9:
                        at.set_cell(L.reshape(3,3), scale_atoms=False)
                        at.set_pbc([True, True, True])
                except Exception:
                    pass
            # parse energy and other info scalars from header
            info = dict(at.info)
            for kv in header.split():
                if "=" in kv and not kv.startswith("Properties=") and not kv.startswith("Lattice=") and not kv.startswith("pbc="):
                    k, v = kv.split("=",1)
                    v = v.strip('"')
                    try:
                        info[k] = float(v)
                    except ValueError:
                        info[k] = v
            at.info = info
            if forces is not None:
                at.new_array("forces", forces)
            frames.append(at)
    return frames

# ---------- main loader with fallback ----------
def load_all_xyz_strict(paths) -> list:
    frames = []
    paths = expand_inputs(paths)
    if not paths:
        return frames

    for p in paths:
        # First try ASE iread strictly as extxyz
        local = []
        try:
            for a in iread(p, index=slice(None), format="extxyz"):
                local.append(a)
        except Exception:
            local = []

        # If ASE produced frames but NONE have forces while header claims forces, fallback
        fallback_needed = False
        if local:
            any_forces = any(("forces" in a.arrays or "REF_forces" in a.arrays) for a in local)
            if not any_forces:
                # peek header to see if Properties advertises forces
                with open(p, "r", errors="ignore") as fh:
                    try:
                        _ = int(fh.readline().strip())
                        hdr = fh.readline()
                        fallback_needed = (":forces:R:3" in hdr) or ("Properties=" in hdr and "forces" in hdr)
                    except Exception:
                        fallback_needed = False

        if not local or fallback_needed:
            # manual robust read
            manual = _manual_read_extxyz(p)
            if manual:
                frames.extend(manual)
            else:
                frames.extend(local)
        else:
            frames.extend(local)

    return frames

# ---------- quick probe to verify ----------
def probe(paths):
    fr = load_all_xyz_strict(paths)
    n = len(fr)
    nf = sum(1 for a in fr if "forces" in a.arrays or "REF_forces" in a.arrays)
    ne = sum(1 for a in fr if any(k in a.info for k in ("energy","Energy","corrected_total_energy","REF_energy")))
    ak = list(fr[0].arrays.keys()) if n else []
    ik = list(fr[0].info.keys()) if n else []
    print(f"[dataset] frames: {n}, with forces: {nf}, with energy: {ne}")
    if n:
        print(f"[dataset] sample arrays: {ak}")
        print(f"[dataset] sample info keys: {ik[:10]}{' ...' if len(ik)>10 else ''}")


def load_all_xyz(paths):
    frames = []
    for p in paths:
        print(f"reading: {p}")
        for a in iread(p, index=":", format="extxyz"):  # <- force extxyz
            frames.append(a)
    return frames

def _as_float_array(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return np.array(x, dtype=float)
    arr = np.asarray(x)
    try:
        return arr.astype(float, copy=False)
    except Exception:
        return None

def stress_to_voigt6(S33: np.ndarray) -> np.ndarray:
    S = np.asarray(S33, float).reshape(3,3)
    # ASE’s Voigt order for stress is [xx, yy, zz, yz, xz, xy]
    return np.array([S[0,0], S[1,1], S[2,2], S[1,2], S[0,2], S[0,1]], float)

def _extract_stress_3x3_from_info(info: dict) -> np.ndarray | None:
    """
    Accepts info['stress'] or ['stresses'] as:
      - string with 9 floats (row-major 3x3)
      - list/ndarray with 9 or (3,3)
    Returns (3,3) or None.
    """
    for key in ("stress","stresses","Stress","Stresses"):
        if key not in info:
            continue
        raw = info[key]
        # reject scalars immediately (these break the writer)
        if isinstance(raw, (int, float)):
            return None

        if isinstance(raw, str):
            toks = raw.replace(",", " ").split()
            if len(toks) != 9:
                return None
            v = np.array([float(t) for t in toks], float)
        else:
            v = np.asarray(raw)
            if v.size not in (9,) and v.shape != (3,3):
                return None

        v = np.asarray(v, float)
        if v.shape == (3,3):
            return v
        return v.reshape(3,3)
    return None

def _extract_stress(info: dict) -> Optional[np.ndarray]:
    """
    Try to extract a 3x3 stress tensor from common keys.
    Accepts:
      - 3x3 array-like
      - 9-vector in row-major order (xx,xy,xz,yx,yy,yz,zx,zy,zz)
      - string with 9 floats (EXTXYZ)
    Returns (3,3) float array or None.
    """
    for key in ("stress", "stresses", "Stress", "Stresses"):
        if key in info:
            raw = info[key]
            if isinstance(raw, str):
                toks = raw.replace(",", " ").split()
                if len(toks) == 9:
                    v = np.array([float(t) for t in toks], dtype=float)
                else:
                    continue
            else:
                v = _as_float_array(raw)
                if v is None:
                    continue

            v = np.array(v, dtype=float)
            if v.shape == (3, 3):
                return v
            if v.size == 9:
                return v.reshape(3, 3)
    return None

def compute_virial_from_stress(a: Atoms, stress_3x3: np.ndarray) -> np.ndarray:
    """
    virial tensor (eV) from Cauchy stress σ (eV/Å³):  W = -σ * V
    Output is 3x3.
    """
    V = float(a.get_cell().volume)
    return -stress_3x3 * V

def remap_one(a: Atoms) -> Atoms:
    b = a.copy()

    info = dict(b.info)
    arrays = dict(b.arrays)

    # --- energy ---
    for k in ("corrected_total_energy", "REF_energy", "energy", "Energy"):
        if k in info:
            try:
                info["REF_energy"] = float(info[k])
                break
            except Exception:
                pass

    # --- forces ---
    F = None
    for k in ("REF_forces", "forces", "force", "Forces", "FORCES"):
        if k in arrays:
            F = np.asarray(arrays[k], dtype=float)
            if F.shape == (len(b), 3):
                arrays["REF_forces"] = F
            else:
                print(f"[warn] forces shape {F.shape} != (N,3) for frame with keys {list(info.keys())}")
            break
    else:
        # no forces in this frame
        # print once or set a flag if you want to count how many lack forces
        pass

    # ---------- stress / virials ----------
    S33 = _extract_stress_3x3_from_info(info)
    if S33 is not None:
        info["REF_stress"] = S33
        # keep a writer-safe Voigt-6 for ASE’s extxyz writer
        info["stress"] = stress_to_voigt6(S33)
        # (optional) if you compute virials from stress:
        # info["REF_virials"] = compute_virial_from_stress(b, S33)
    else:
        # if a bogus scalar stress exists, drop it so writer won’t choke
        st = info.get("stress", None)
        if isinstance(st, (int,float)) or np.asarray(st).ndim == 0:
            info.pop("stress", None)

    b.info = info
    b.arrays = arrays
    return b

def cast_to_fp32(a: Atoms) -> Atoms:
    """Down-cast numeric info/arrays to float32 (positions/forces/stress/virials/energy…)."""
    c = a.copy()
    # positions & cell are handled by ASE writer; info/arrays we cast here
    info2 = {}
    for k, v in c.info.items():
        if isinstance(v, (int, float, np.floating)):
            info2[k] = np.float32(v).item()
        elif isinstance(v, (list, tuple)):
            info2[k] = np.asarray(v, dtype=np.float32).tolist()
        elif isinstance(v, np.ndarray):
            # keep shape; cast
            info2[k] = v.astype(np.float32, copy=False)
        else:
            info2[k] = v
    c.info = info2

    arrays2 = {}
    for k, v in c.arrays.items():
        if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.floating):
            arrays2[k] = v.astype(np.float32, copy=False)
        else:
            arrays2[k] = v
    c.arrays = arrays2
    return c

def write_extxyz(path: str, frames: Iterable[Atoms], float_format: str = "%.10g"):
    """Write an iterable of Atoms to EXTXYZ. (ASE chooses format by extension.)"""
    # ASE’s xyz writer does not accept float_format kw; it applies a reasonable default.
    # To keep things simple/robust, we just call write without that kw.
    ase_write(path, list(frames), format="extxyz")

# ----------------------------- runner -----------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Extract 30k systems, remap fields to REF_* keys, compute virials, and write fp64/fp32 replay xyz."
    )
    ap.add_argument("--inputs", nargs="+", required=True, help="Input EXTXYZ files (one or many).")
    ap.add_argument("--out64", default="replay_np_30k_remap.xyz", help="Output (float64) EXTXYZ.")
    ap.add_argument("--out32", default="replay_np_30k_remap_fp32.xyz", help="Output (float32) EXTXYZ.")
    ap.add_argument("--n", type=int, default=30000, help="How many frames to sample.")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for random sampling.")
    ap.add_argument("--elements", type=str, default="", help="RNG seed for random sampling.")
    ap.add_argument("--method", choices=["random","fps"], default="random", help="Sampling method.")
    args = ap.parse_args()

    frames_all = load_all_xyz_strict(args.inputs)
    if not frames_all:
        raise SystemExit("No frames loaded from inputs.")
    
    probe(args.inputs)
    
    required_elements = [el.strip() for el in args.elements.split(",")]
    if len(required_elements) == 0:
        required_elements = None
    
    idx = choose_subset(args.n, frames_all, seed=args.seed, method=args.method,
                        required_elements=required_elements, require_all=False)

    chosen = [frames_all[i] for i in idx]

    # remap + compute virials
    remapped = [remap_one(a) for a in chosen]

    # write fp64 (native)
    write_extxyz(args.out64, remapped)

    # write fp32
    remapped_fp32 = [cast_to_fp32(a) for a in remapped]
    write_extxyz(args.out32, remapped_fp32)

    print(f"[OK] wrote {args.out64}  (n={len(remapped)})")
    print(f"[OK] wrote {args.out32}  (n={len(remapped_fp32)})")

if __name__ == "__main__":
    main()
