"""Merged JAR paper, expanded molecular docking (analysis N5): the union of the key aroma-active compounds of the two systems
(fermentation stage, classes I-III of paper 2 table 1; distillation stage, classes I and III of paper 3 table 2) plus five polar
late-eluting contrast compounds, so that TPSA covers 0-52.6 A^2 -- against the same six human olfactory receptor models, the same
boxes, exhaustiveness 32, nine modes and the same three seeds as in paper 2.  The seven compounds docked for paper 2 keep their
original ligand files and their rows are copied unchanged from the old dock_results.csv, so the published numbers do not move.
Only PubChem CIDs are sent.  Run with the docking venv:  C:\\md\\mdenv\\python.exe jar_dock.py [--prep-only]"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request

import numpy as np
import pandas as pd
from meeko import MoleculePreparation, PDBQTWriterLegacy
from rdkit import Chem
from rdkit.Chem import AllChem

SCR = os.path.dirname(os.path.abspath(__file__))
OLD = os.path.join(os.path.dirname(SCR), "p2v12", "md_work")
W = os.path.join(SCR, "dock_work"); O = os.path.join(W, "dock")
VINA = os.environ.get("VINA_EXE", "vina")
SEEDS = [11, 22, 33]; CPU = "8"                                                # the MD run keeps the GPU and one core
os.makedirs(O, exist_ok=True)

OLDINP = json.load(open(os.path.join(OLD, "inputs.json"), encoding="utf-8"))
RECS = list(OLDINP["receptors"]); KEEP = list(OLDINP["ligands"])               # L1..L7, order fixed by paper 2
SEL = json.load(open(os.path.join(SCR, "dock_set.json"), encoding="utf-8"))
SEL.pop("苯乙醇乙酸酯", None)                                                   # same compound as 乙酸苯乙酯 (CAS 103-45-7, CID 7654)
LIGS = KEEP + sorted([n for n in SEL if n not in KEEP], key=lambda n: SEL[n]["cid"])
for f in ["boxes.json"] + [f"{r}.pdbqt" for r in RECS]:
    shutil.copy(os.path.join(OLD, f), os.path.join(W, f))
BOX = json.load(open(os.path.join(W, "boxes.json")))


def fetch(cid, tries=6):
    """PubChem over this connection drops TLS handshakes now and then, so retry with a growing pause"""
    url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/"
           "IsomericSMILES,MolecularFormula,MolecularWeight,IUPACName/JSON")
    for t in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "structure-download"}), timeout=120) as r:
                p = json.loads(r.read().decode("utf-8"))["PropertyTable"]["Properties"][0]
            break
        except Exception as e:
            if t == tries - 1:
                raise
            print(f"  retry {t + 1} (CID {cid}): {type(e).__name__}", flush=True); time.sleep(3 * (t + 1))
    return dict(cid=cid, smiles=p.get("IsomericSMILES") or p.get("SMILES"), formula=p["MolecularFormula"],
                mw=p["MolecularWeight"], iupac=p.get("IUPACName"))


FI = os.path.join(W, "inputs.json")                                            # prepared ligands are cached: PubChem drops TLS often
CACHE = json.load(open(FI, encoding="utf-8"))["ligands"] if os.path.exists(FI) else {}
inp = {"receptors": OLDINP["receptors"], "ligands": {}}
for k, name in enumerate(LIGS):
    code = f"L{k + 1}"
    if name in KEEP:                                                           # paper 2 ligand: keep its files and metadata
        inp["ligands"][name] = dict(OLDINP["ligands"][name], code=code, source="paper2")
        for ext in ("mol", "pdbqt"):
            shutil.copy(os.path.join(OLD, f"L{KEEP.index(name) + 1}.{ext}"), os.path.join(W, f"{code}.{ext}"))
        continue
    if CACHE.get(name, {}).get("code") == code and os.path.exists(os.path.join(W, f"{code}.pdbqt")):
        inp["ligands"][name] = CACHE[name]                                     # already fetched and prepared
        continue
    v = fetch(SEL[name]["cid"]); v.update(code=code, source=SEL[name].get("src"), tpsa=SEL[name].get("tpsa"),
                                          elution=SEL[name].get("type"), grade=SEL[name].get("grade"))
    inp["ligands"][name] = v
    m = Chem.AddHs(Chem.MolFromSmiles(v["smiles"]))
    assert AllChem.EmbedMolecule(m, randomSeed=2026 + k) == 0, name
    AllChem.MMFFOptimizeMolecule(m, maxIters=2000)
    Chem.MolToMolFile(m, os.path.join(W, f"{code}.mol"))
    s, ok, err = PDBQTWriterLegacy.write_string(MoleculePreparation().prepare(m)[0]); assert ok, (name, err)
    open(os.path.join(W, f"{code}.pdbqt"), "w").write(s)
    print(f"prep {code} {name:<22} CID {v['cid']:<8} heavy {m.GetNumHeavyAtoms():>2} torsions {s.count('BRANCH') // 2}", flush=True)
json.dump(inp, open(FI, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"\n{len(LIGS)} ligands, {len(RECS)} receptors; new pairs to dock: {(len(LIGS) - len(KEEP)) * len(RECS)}", flush=True)
if "--prep-only" in sys.argv:
    sys.exit()


def first_pose(path):
    xyz = []
    for line in open(path):
        if line.startswith("ENDMDL"):
            break
        if line.startswith(("ATOM", "HETATM")):
            xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.array(xyz)


rows = []
old = pd.read_csv(os.path.join(OLD, "dock_results.csv"))
for name in KEEP:                                                              # copy paper 2's rows unchanged
    r = old[old.ligand == name].copy(); r["code"] = inp["ligands"][name]["code"]; rows += r.to_dict("records")
done = {(d["receptor"], d["ligand"], d["seed"]) for d in rows}
todo = [(rec, n) for rec in RECS for n in LIGS if n not in KEEP]
t0 = time.time()
for i, (rec, name) in enumerate(todo, 1):
    code = inp["ligands"][name]["code"]; b = BOX[rec]
    for seed in SEEDS:
        if (rec, name, seed) in done:
            continue
        out = os.path.join(O, f"{rec}_{code}_s{seed}.pdbqt")
        cmd = [VINA, "--receptor", os.path.join(W, f"{rec}.pdbqt"), "--ligand", os.path.join(W, f"{code}.pdbqt"),
               "--center_x", str(b["center"][0]), "--center_y", str(b["center"][1]), "--center_z", str(b["center"][2]),
               "--size_x", "20", "--size_y", "20", "--size_z", "20", "--exhaustiveness", "32", "--num_modes", "9",
               "--energy_range", "3", "--seed", str(seed), "--cpu", CPU, "--out", out]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, res.stderr[-500:]
        modes = [float(m.group(1)) for m in re.finditer(r"^\s+\d+\s+(-?\d+\.\d+)\s+", res.stdout, flags=re.M)]
        dist = float(np.linalg.norm(first_pose(out).mean(0) - np.array(b["center"])))
        rows.append(dict(receptor=rec, ligand=name, code=code, seed=seed, best=modes[0], n_modes=len(modes),
                         dist_to_site=round(dist, 2)))
    pd.DataFrame(rows).to_csv(os.path.join(W, "dock_results.csv"), index=False, encoding="utf-8-sig")
    el = time.time() - t0
    print(f"[{i}/{len(todo)}] {rec} {name} {[r['best'] for r in rows[-3:]]} {el / 60:.1f} min "
          f"(eta {el / i * (len(todo) - i) / 60:.0f} min)", flush=True)
df = pd.DataFrame(rows)
print(df.groupby(["ligand", "receptor"])["best"].mean().unstack().round(2).to_string())
print("\nsaved", os.path.join(W, "dock_results.csv"))
