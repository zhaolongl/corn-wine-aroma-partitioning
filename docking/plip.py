"""Merged JAR paper, expanded docking: best poses and PLIP interaction profiles of all 31 ligands x 6 receptors, with exactly the
pipeline of paper 2 (md_poses.py: lowest-score pose over the three seeds, Meeko restores bond orders and hydrogens; md_render.py:
complex PDB = receptor with PDBFixer hydrogens + ligand HETATM LIG, PLIP 3.0 with NOHYDRO, hydrophobic contacts and hydrogen
bonds).  The OR1A2 rows of the seven paper 2 ligands must reproduce paper 2's plip_interactions.csv.  Pocket residues = any
heavy atom within 4.0 A of a ligand heavy atom.  Run with the docking env:  C:\\md\\mdenv\\python.exe jar_plip.py"""
import os

import numpy as np
import pandas as pd
from meeko import PDBQTMolecule, RDKitMolCreate
from openbabel import pybel
from rdkit import Chem

SCR = os.path.dirname(os.path.abspath(__file__)); W = os.path.join(SCR, "dock_work")
OLD = os.path.join(os.path.dirname(SCR), "p2v12", "md_work")
PO = os.path.join(W, "poses"); CO = os.path.join(W, "complex"); os.makedirs(PO, exist_ok=True); os.makedirs(CO, exist_ok=True)
_write = pybel.Molecule.write


def _patched(self, format="smi", filename=None, overwrite=False, opt=None):   # the Windows Open Babel wheel lacks the InChIKey plugin;
    return "NA" if format == "inchikey" else _write(self, format, filename, overwrite, opt)   # PLIP only uses it as a label


pybel.Molecule.write = _patched
from plip.basic import config                                                    # noqa: E402
from plip.structure.preparation import PDBComplex                                # noqa: E402
config.NOHYDRO = True

df = pd.read_csv(os.path.join(W, "dock_results.csv"))
best = df.loc[df.groupby(["receptor", "code"])["best"].idxmin()].reset_index(drop=True)
REC = {}
for rec in best.receptor.unique():
    lines = [l for l in open(os.path.join(OLD, f"{rec}_H.pdb")) if l.startswith(("ATOM", "HETATM"))]
    xyz = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in lines])
    heavy = np.array([l[76:78].strip() != "H" for l in lines])
    resid = [f"{l[17:20].strip()}{int(l[22:26])}" for l in lines]
    REC[rec] = (lines, xyz, heavy, resid)
rows, pocket = [], []
for _, r in best.iterrows():
    sdf = os.path.join(PO, f"{r.receptor}_{r.code}.sdf")
    old = os.path.join(OLD, "poses", f"{r.receptor}_{r.code}.sdf")
    if int(r.code[1:]) <= 7 and os.path.exists(old):                             # paper 2 ligand: its pose file, unchanged
        m1 = Chem.MolFromMolFile(old, removeHs=False)
    else:
        pm = PDBQTMolecule.from_file(os.path.join(W, "dock", f"{r.receptor}_{r.code}_s{int(r.seed)}.pdbqt"), skip_typing=True)
        mol = RDKitMolCreate.from_pdbqt_mol(pm)[0]
        m1 = Chem.Mol(mol); confs = [c.GetId() for c in m1.GetConformers()]
        m1.RemoveAllConformers(); m1.AddConformer(mol.GetConformer(confs[0]), assignId=True)
        m1 = Chem.AddHs(m1, addCoords=True)
    w = Chem.SDWriter(sdf); w.write(m1); w.close()
    for a in m1.GetAtoms():
        a.SetMonomerInfo(Chem.AtomPDBResidueInfo(f"{a.GetSymbol()}{a.GetIdx() + 1:<3d}"[:4], residueName="LIG", residueNumber=900,
                                                 chainId="L", isHeteroAtom=True))
    lig = [l for l in Chem.MolToPDBBlock(m1, flavor=4).splitlines() if l.startswith(("HETATM", "CONECT"))]
    lines, xyz, heavy, resid = REC[r.receptor]
    path = os.path.join(CO, f"{r.receptor}_{r.code}_complex.pdb")
    open(path, "w").write("".join(lines) + "TER\n" + "\n".join(lig) + "\nEND\n")
    pc = PDBComplex(); pc.load_pdb(path); pc.analyze()
    site = next(v for k, v in pc.interaction_sets.items() if k.startswith("LIG"))
    for h in site.hydrophobic_contacts:
        rows.append(dict(receptor=r.receptor, code=r.code, ligand=r.ligand, type="疏水", residue=f"{h.restype}{h.resnr}", dist=round(h.distance, 2)))
    for h in site.hbonds_pdon + site.hbonds_ldon:
        rows.append(dict(receptor=r.receptor, code=r.code, ligand=r.ligand, type="氢键", residue=f"{h.restype}{h.resnr}", dist=round(h.distance_ad, 2)))
    lx = m1.GetConformer().GetPositions()[[a.GetAtomicNum() > 1 for a in m1.GetAtoms()]]
    dmin = np.linalg.norm(xyz[heavy][:, None, :] - lx[None], axis=2).min(1)
    res = sorted({resid[i] for i in np.where(heavy)[0][dmin <= 4.0]}, key=lambda s: int("".join(ch for ch in s if ch.isdigit())))
    pocket.append(dict(receptor=r.receptor, code=r.code, ligand=r.ligand, seed=int(r.seed), score=r.best, residues_4A=" ".join(res), n_res=len(res)))
P = pd.DataFrame(rows); K = pd.DataFrame(pocket)
P.to_csv(os.path.join(W, "plip_interactions.csv"), index=False, encoding="utf-8-sig")
K.to_csv(os.path.join(W, "pocket_residues.csv"), index=False, encoding="utf-8-sig")
# reproduce paper 2 (OR1A2, L1-L7)
ref = pd.read_csv(os.path.join(OLD, "plip_interactions.csv"))
key = lambda d: sorted(map(tuple, d[["receptor", "code", "type", "residue", "dist"]].astype(str).values))
mine = P[P.code.isin([f"L{i}" for i in range(1, 8)])]
print("paper 2 PLIP rows reproduced:", key(mine[mine.receptor.isin(ref.receptor.unique())]) == key(ref), len(ref))
o = P[P.receptor == "OR1A2"]
print("OR1A2 interaction types:", o.type.value_counts().to_dict(), "| ligands with H-bonds:", sorted(o[o.type == "氢键"].ligand.unique()))
k = K[K.receptor == "OR1A2"]
freq = pd.Series([x for s in k.residues_4A for x in s.split()]).value_counts()
print(f"OR1A2 residues within 4 A (n ligands = {len(k)}):"); print((freq / len(k) * 100).round(0).head(20).to_string())
