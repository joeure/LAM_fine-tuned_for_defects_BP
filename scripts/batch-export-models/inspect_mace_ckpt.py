#!/usr/bin/env python3
import argparse, json, sys, re, os
from pathlib import Path
from collections.abc import Mapping
from collections import OrderedDict

def _try_import_torch():
    try:
        import torch
        return torch
    except Exception as e:
        print("[error] PyTorch not importable:", e, file=sys.stderr)
        sys.exit(2)

def _type_name(o):
    return type(o).__name__

def _is_scalar_like(x):
    return isinstance(x, (int, float, bool, str)) or x is None

def _jsonify(obj):
    """Make nested config JSON-safe."""
    import numpy as np
    try:
        import torch
        torch_types = (torch.Tensor,)
    except Exception:
        torch_types = ()
    if isinstance(obj, Mapping):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        # keep small arrays inline; otherwise summarize
        if obj.size <= 32:
            return obj.tolist()
        return {"__ndarray__": list(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, torch_types):
        # do not dump full tensors
        try:
            shape = tuple(obj.shape)
        except Exception:
            shape = None
        return {"__tensor__": shape, "dtype": str(obj.dtype) if hasattr(obj, "dtype") else None}
    if isinstance(obj, (bytes, bytearray)):
        return {"__bytes__": len(obj)}
    if _is_scalar_like(obj):
        return obj
    # fallback: string repr (truncated)
    s = repr(obj)
    return s if len(s) <= 200 else (s[:200] + "...")

def _print_kv(k, v, indent=0, max_value_len=200):
    pref = "  " * indent
    if _is_scalar_like(v):
        sv = repr(v)
        if len(sv) > max_value_len:
            sv = sv[:max_value_len] + "..."
        print(f"{pref}- {k}: {sv}  ({_type_name(v)})")
    elif isinstance(v, (list, tuple)):
        print(f"{pref}- {k}: <{_type_name(v)}> len={len(v)}")
        if len(v) and indent < 3:
            for i, item in enumerate(v[:5]):
                _print_kv(f"[{i}]", item, indent+1)
            if len(v) > 5:
                print(f"{pref}  ... ({len(v)-5} more)")
    elif isinstance(v, Mapping):
        print(f"{pref}- {k}: <{_type_name(v)}> keys={len(v)}")
        if indent < 4:
            for kk in list(v.keys())[:20]:
                _print_kv(kk, v[kk], indent+1)
            if len(v) > 20:
                print(f"{pref}  ... ({len(v)-20} more)")
    else:
        # tensors/arrays/others summarized
        try:
            import numpy as np
            if "torch" in sys.modules and hasattr(v, "shape"):
                shape = tuple(getattr(v, "shape", ()))
                print(f"{pref}- {k}: <{_type_name(v)}> shape={shape}")
                return
            if isinstance(v, np.ndarray):
                print(f"{pref}- {k}: <ndarray> shape={v.shape} dtype={v.dtype}")
                return
        except Exception:
            pass
        s = repr(v)
        if len(s) > max_value_len:
            s = s[:max_value_len] + "..."
        print(f"{pref}- {k}: {s} <{_type_name(v)}>")

def _find_heads_from_state_dict(sd):
    """Return sorted head names via keys like 'heads.<name>.'"""
    heads = set()
    for k in sd.keys():
        m = re.match(r"heads\.([^\.]+)\.", k)
        if m: heads.add(m.group(1))
    return sorted(heads)

def _summarize_param_counts(sd):
    """Count parameters by prefix (heads.<name>. vs trunk)."""
    try:
        import torch
        numel = lambda t: (t.numel() if isinstance(t, torch.Tensor) else 0)
    except Exception:
        numel = lambda t: 0

    total = 0
    per_head = {}
    trunk = 0
    for k, v in sd.items():
        n = numel(v)
        total += n
        m = re.match(r"heads\.([^\.]+)\.", k)
        if m:
            per_head[m.group(1)] = per_head.get(m.group(1), 0) + n
        else:
            trunk += n
    return total, trunk, per_head

def _safe_len(x):
    try:
        return len(x)
    except Exception:
        return None

def inspect_checkpoint(path, dump_config=None, sample_keys=30):
    torch = _try_import_torch()

    print(f"=== Inspect: {path} ===")
    if not os.path.exists(path):
        print("  [error] file does not exist")
        return

    ckpt = torch.load(path, map_location="cpu")
    print(f"Top-level type: {_type_name(ckpt)}")

    if isinstance(ckpt, Mapping):
        print("Top-level keys:", ", ".join(list(ckpt.keys())[:50]) + (" ..." if len(ckpt) > 50 else ""))
        # common top-level fields
        for key in ["epoch", "step", "global_step", "trainer_state", "optimizer", "scheduler",
                    "model", "state_dict", "config"]:
            if key in ckpt:
                print(f"  has '{key}': type={_type_name(ckpt[key])}, len={_safe_len(ckpt[key])}")
    else:
        print("  (not a Mapping; may be a raw OrderedDict state_dict)")

    # state_dict
    if isinstance(ckpt, Mapping) and isinstance(ckpt.get("state_dict"), Mapping):
        sd = ckpt["state_dict"]
        print(f"\nstate_dict: {len(sd)} tensors")
        # sample keys
        keys = list(sd.keys())
        for k in keys[:sample_keys]:
            print("  ·", k)
        if len(keys) > sample_keys:
            print(f"  ... ({len(keys)-sample_keys} more)")

        # heads
        heads = _find_heads_from_state_dict(sd)
        print("Detected heads (from keys):", ", ".join(heads) if heads else "(none)")

        # parameter counts
        total, trunk, per_head = _summarize_param_counts(sd)
        def _fmt(n): 
            return f"{n:,}" if isinstance(n, int) else str(n)
        print(f"Parameter counts (numel): total={_fmt(total)}, trunk={_fmt(trunk)}")
        if per_head:
            for h, n in per_head.items():
                print(f"  head[{h}] = {_fmt(n)}")

    elif isinstance(ckpt, Mapping) and "model" in ckpt:
        # some training scripts store a live Module and no explicit sd
        model = ckpt["model"]
        print("\n[info] 'model' present (live Module):", _type_name(model))
        # try to show its own config if present
        cfg = getattr(model, "config", None)
        if cfg is not None:
            print("Model.config:")
            _print_kv("config", cfg, indent=1)
    else:
        # maybe the checkpoint itself is the state_dict
        if isinstance(ckpt, (Mapping, OrderedDict)):
            keys = list(ckpt.keys())
            looks_like_sd = all(isinstance(k, str) for k in keys[:5]) and any("weight" in k or "bias" in k for k in keys[:20])
            if looks_like_sd:
                print(f"\n[info] This looks like a raw state_dict with {len(keys)} keys.")
                for k in keys[:sample_keys]:
                    print("  ·", k)
                if len(keys) > sample_keys:
                    print(f"  ... ({len(keys)-sample_keys} more)")
                heads = _find_heads_from_state_dict(ckpt)
                print("Detected heads (from keys):", ", ".join(heads) if heads else "(none)")
            else:
                print("\n[warn] Mapping-like object but not clearly a state_dict or trainer dump.")
        else:
            print("\n[warn] Unknown checkpoint structure.")

    # config
    cfg = None
    if isinstance(ckpt, Mapping) and "config" in ckpt:
        cfg = ckpt["config"]

    if cfg is not None:
        print("\n=== CONFIG (structure) ===")
        if isinstance(cfg, Mapping):
            print(f"Config keys: {len(cfg)}")
            for k in list(cfg.keys())[:40]:
                _print_kv(k, cfg[k], indent=1)
            if len(cfg) > 40:
                print("  ... (more keys omitted)")
        else:
            # sometimes config is a Namespace or object; try dir
            print(f"config type: {_type_name(cfg)}")
            if hasattr(cfg, "__dict__"):
                d = {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_")}
                for k in list(d.keys())[:40]:
                    _print_kv(k, d[k], indent=1)
            else:
                print(repr(cfg)[:500])

        if dump_config:
            outp = Path(dump_config)
            try:
                outp.parent.mkdir(parents=True, exist_ok=True)
                with outp.open("w", encoding="utf-8") as f:
                    json.dump(_jsonify(cfg), f, indent=2, sort_keys=True)
                print(f"\n[ok] Wrote JSON config to: {outp}")
            except Exception as e:
                print(f"[warn] Could not dump JSON config: {e}")
    else:
        print("\n[info] No 'config' found in this checkpoint.")

def main():
    ap = argparse.ArgumentParser(description="Inspect a MACE checkpoint and print its structure (including config).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ckpt", type=Path, help="Path to a single checkpoint (.pt / .model / .pth).")
    g.add_argument("--ckpt_glob", type=str, help="Glob for multiple checkpoints (inspects the last by sort).")
    ap.add_argument("--dump_config", type=Path, help="Optional path to write the config as JSON.")
    ap.add_argument("--sample_keys", type=int, default=30, help="How many state_dict keys to sample in the printout.")
    args = ap.parse_args()

    if args.ckpt_glob:
        matches = sorted(Path(".").glob(args.ckpt_glob))
        if not matches:
            print("[error] No files matched:", args.ckpt_glob, file=sys.stderr)
            sys.exit(1)
        ckpt = matches[-1]
    else:
        ckpt = args.ckpt

    inspect_checkpoint(str(ckpt), dump_config=args.dump_config, sample_keys=args.sample_keys)

if __name__ == "__main__":
    main()
