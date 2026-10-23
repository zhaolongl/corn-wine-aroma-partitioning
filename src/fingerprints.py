"""Structural features of the 115 distillate compounds (requires RDKit).
(1) optional: refresh SMILES and heavy-atom counts from PubChem by CID (only CIDs are sent);
(2) Morgan fingerprints (radius 2, 2048 bits) and the Tanimoto similarity matrix;
(3) Butina clusters at Tanimoto distance 0.5 / 0.6 / 0.7 for the main analysis set (96) and for all 115 compounds;
(4) heavy atoms, fraction sp3 and ring count.
Run:  python src/fingerprints.py [--fetch]"""
import io
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
from rdkit.ML.Cluster import Butina

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as CFG                                                                # noqa: E402

b = pd.read_csv(CFG.RAW / "distillate_115_compounds.csv")
smi_path = CFG.DERIVED / "smiles115.csv"
if "--fetch" in sys.argv or not smi_path.exists():
    url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/" + ",".join(map(str, b["CID"].astype(int)))
           + "/property/SMILES,ConnectivitySMILES,HeavyAtomCount,Complexity/CSV")
    for k in range(5):
        try:
            txt = urllib.request.urlopen(url, timeout=60).read().decode(); break
        except Exception as e:                                                       # noqa: BLE001
            print("retry", e); time.sleep(5)
    pd.read_csv(io.StringIO(txt)).to_csv(smi_path, index=False)
s = pd.read_csv(smi_path)
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fps = [gen.GetFingerprint(Chem.MolFromSmiles(x)) for x in s["SMILES"]]
n = len(fps); S = np.array([DataStructs.BulkTanimotoSimilarity(fps[i], fps) for i in range(n)])
np.save(CFG.DERIVED / "tanimoto115.npy", S)
s[["CID"]].to_csv(CFG.DERIVED / "tanimoto115_cids.csv", index=False)
rows = []
for cid, x in zip(s["CID"], s["SMILES"]):
    m = Chem.MolFromSmiles(x)
    rows.append(dict(CID=cid, HA=m.GetNumHeavyAtoms(), FSP3=rdMolDescriptors.CalcFractionCSP3(m), Rings=rdMolDescriptors.CalcNumRings(m)))
pd.DataFrame(rows).to_csv(CFG.DERIVED / "rdkit_extra115.csv", index=False)
cids = list(s["CID"]); out = pd.DataFrame({"CID": cids})
for name, mask in [("main", b["主分析集"].values), ("all", np.ones(len(b), bool))]:
    sub = b[mask]; idx = [cids.index(c) for c in sub["CID"]]; Sm = S[np.ix_(idx, idx)]; k = len(idx)
    dists = [1 - Sm[i, j] for i in range(1, k) for j in range(i)]
    for cut in [0.5, 0.6, 0.7]:
        lab = np.full(k, -1)
        for q, cl in enumerate(Butina.ClusterData(dists, k, cut, isDistData=True)):
            for i in cl:
                lab[i] = q
        col = f"butina_{name}_{cut}"; out[col] = np.nan
        for i, ii in enumerate(idx):
            out.loc[ii, col] = lab[i]
out.to_csv(CFG.DERIVED / "clusters115.csv", index=False)
print("fingerprints, similarity matrix and clusters written to", CFG.DERIVED)
