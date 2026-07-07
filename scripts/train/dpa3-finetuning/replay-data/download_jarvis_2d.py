from jarvis.db.figshare import data

import os, requests, zipfile, dpdata, io
import numpy as np

d2d = data("dft_2d")     # ~1.1k curated 2D entries (metadata + structures)
print(len(d2d), d2d[0].keys())

from jarvis.core.atoms import Atoms  # to read element list from the stored structure

wanted = {"P","N", "Mo", "S", "Se", "W", "B", "C", "In", "Ga"}
keep = []
for rec in d2d:
    elems = set(Atoms.from_dict(rec["atoms"]).elements)
    if wanted.intersection(elems):
        keep.append(rec["jid"])      # JARVIS id, e.g. "JVASP-xxxx"
print("Chosen JIDs:", len(keep))

rf = data("raw_files")  # dict: category -> list of file dicts
def urls_for_jid(jid):
    urls = []
    for cat, items in rf.items():
        for it in items:
            if isinstance(it, dict) and str(it.get("name","")).startswith(jid):
                url = it.get("download_url") or it.get("url") or it.get("link")
                if url: urls.append(url)
    return urls

def labeled_from_zip_bytes(zbytes, jid):
    """Try vasprun.xml first, then OUTCAR; return a dpdata.LabeledSystem or None."""
    z = zipfile.ZipFile(io.BytesIO(zbytes))
    # Prefer vasprun.xml; fall back to OUTCAR
    xmls = [n for n in z.namelist() if n.endswith("vasprun.xml")]
    outs = [n for n in z.namelist() if n.endswith("OUTCAR")]
    for name, fmt in [(xmls[0] if xmls else None, "vasp/xml"),
                      (outs[0] if outs else None, "vasp/outcar")]:
        if not name: 
            continue
        try:
            os.makedirs(f"jarvis_raw/{jid}", exist_ok=True)
            path = f"jarvis_raw/{jid}/{os.path.basename(name)}"
            with open(path, "wb") as f: f.write(z.read(name))
            ls = dpdata.LabeledSystem(path, fmt=fmt)
            # sanity: at least one frame, and some labels present
            nf = ls.get_nframes()
            has_E = isinstance(ls.data.get("energies"), np.ndarray) and ls.data["energies"].size>0
            has_F = isinstance(ls.data.get("forces"),   np.ndarray) and ls.data["forces"].size>0
            if nf >= 1 and (has_E or has_F):
                return ls
        except Exception as e:
            print(f"[warn] {jid}: {fmt} parse failed ({e})")
    return None

sess = requests.Session()
sess.headers.update({"User-Agent":"jarvis-tools-script"})
os.makedirs("data/replay", exist_ok=True)

ok = 0
for jid in sorted(keep):
    try:
        urls = urls_for_jid(jid)
        if not urls:
            print(f"[skip] {jid}: no raw zip listed"); 
            continue
        got = None
        for url in urls:
            try:
                r = sess.get(url, timeout=120); r.raise_for_status()
                got = labeled_from_zip_bytes(r.content, jid)
                if got: break
            except Exception as e:
                print(f"[warn] {jid}: download/read failed: {e}")
        if not got:
            print(f"[skip] {jid}: no usable vasprun.xml/OUTCAR (0 frames / no labels)")
            continue
        # optional: keep only the final frame (typical for replay/eval)
        for k in ["energies","forces","virials","coords","cells"]:
            arr = got.data.get(k); 
            if isinstance(arr, np.ndarray) and arr.ndim>=1 and arr.shape[0]>1:
                got.data[k] = arr[-1:,...]
        got.to("deepmd/npy", f"data/replay/system.{jid}", set_size=got.get_nframes())
        print(f"[ok] {jid}")
        ok += 1
    except Exception as e:
        print(f"[warn] {jid}: {e}")
print(f"Converted {ok} systems into data/replay/")