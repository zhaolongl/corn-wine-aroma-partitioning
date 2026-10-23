"""JAR merged paper, sixth version: shared data layer for the continuous-partitioning analyses (C0).
Distillate side: 115 compounds of the second distillation (paper 3, thesis appendix 3, mg/L, one pooled sample per fraction).
Endpoints are computed from concentration x fraction volume, so the denominator is the five collected fractions, not the feed.
  p_ij  relative mass share of compound i in fraction j
  R_i   heart-fraction share (middle + second fraction)
  D_i   segment-midpoint approximation of the distillation centroid, sum_j p_ij * t_j, t_j = midpoint of fraction j / 700 mL
S0 takes the tabulated concentrations as fraction concentrations; S1 is the conditional scenario in which the first four fractions
were read after dilution to 10% vol and were not back-corrected (factors = mean fraction strength / 10; tail measured as is).
S1 is a sensitivity scenario only and is never treated as the true concentration basis."""
import os

import numpy as np
import pandas as pd

SCR = os.path.dirname(os.path.abspath(__file__))
import config as CFG
P3CSV = str(CFG.RAW / "distillate_115_compounds.csv")
DER = str(CFG.DERIVED)
FRAME = str(CFG.DERIVED / "framework_derived_endpoints.csv")
OUT = str(CFG.INTER)
FR = ["头馏分_mg/L", "中间馏分_mg/L", "次馏分_mg/L", "次尾馏分_mg/L", "尾馏分_mg/L"]
FRS = ["头", "中间", "次", "次尾", "尾"]
VOL = np.array([50., 300., 50., 100., 200.])
EDGE = np.array([0., 50., 350., 400., 500., 700.])
TMID = (EDGE[:-1] + EDGE[1:]) / 2 / 700.
ABV = np.array([78.8, 71.366667, 52.4, 29.85, 5.2])
S1F = np.array([7.88, 7.14, 5.24, 2.99, 1.0])
ETOH = np.array([39.40, 214.10, 26.20, 29.85, 10.40])           # mL ethanol collected in each fraction (14-cut strengths)
DESC = ["MW", "XLogP", "TPSA", "HBD", "HBA", "RotB", "芳香环", "O/C"]
DESC_LAB = {"MW": "MW", "XLogP": "XLogP3", "TPSA": "TPSA", "HBD": "HBD", "HBA": "HBA", "RotB": "RotB",
            "芳香环": "芳香环", "O/C": "O/C"}
REN = {"苯乙醇乙酸酯": "乙酸苯乙酯", "醋酸": "乙酸"}
CLS7 = ["酯类", "醇类", "醛酮类", "酸类", "缩醛类", "萜类及降异戊二烯类", "呋喃类"]


def shares(conc, factors=None):
    c = np.asarray(conc, float)
    if factors is not None:
        c = c * factors
    m = c * VOL
    return m / m.sum(1, keepdims=True)


def load():
    """115 compounds with descriptors, labels, SMILES, Butina clusters and S0/S1 endpoints"""
    b = pd.read_csv(P3CSV)
    b["采用名称"] = b["采用名称"].replace(REN)
    p0 = shares(b[FR].values); p1 = shares(b[FR].values, S1F)
    for k, s in enumerate(FRS):
        b[f"p_{s}"] = p0[:, k]; b[f"p1_{s}"] = p1[:, k]
    b["R"] = p0[:, 1] + p0[:, 2]; b["D"] = p0 @ TMID; b["H"] = p0[:, 0]; b["T"] = p0[:, 3] + p0[:, 4]
    b["R_S1"] = p1[:, 1] + p1[:, 2]; b["D_S1"] = p1 @ TMID; b["H_S1"] = p1[:, 0]
    b["late"] = (b["洗脱类型"] == "晚馏出型").astype(int)
    smi = pd.read_csv(os.path.join(DER, "smiles115.csv"))[["CID", "SMILES"]]
    ex = pd.read_csv(os.path.join(DER, "rdkit_extra115.csv"))
    cl = pd.read_csv(os.path.join(DER, "clusters115.csv"))
    b = b.merge(smi, on="CID", how="left").merge(ex, on="CID", how="left").merge(cl, on="CID", how="left")
    b["n_det"] = (b[FR] > 0).sum(1)
    return b


def tanimoto(b):
    S = np.load(os.path.join(DER, "tanimoto115.npy"))
    cids = list(pd.read_csv(os.path.join(DER, "tanimoto115_cids.csv")).iloc[:, 0].values)
    idx = [cids.index(c) for c in b["CID"]]
    return S[np.ix_(idx, idx)]


def check_frame(b):
    """C0: the derived endpoints reproduce the framework file row by row"""
    f = pd.read_csv(FRAME)
    m = b.merge(f, on="编号", suffixes=("", "_f"))
    d = dict(n=len(m),
             R=float(np.abs(m["R"] - m["S0_酒心占比"]).max()),
             D=float(np.abs(m["D"] - m["S0_分段中点近似馏出重心"]).max()),
             R_S1=float(np.abs(m["R_S1"] - m["S1条件性折算_酒心占比"]).max()),
             share_sum=float(np.abs(b[[f"p_{s}" for s in FRS]].sum(1) - 1).max()))
    return d


if __name__ == "__main__":
    b = load(); print(check_frame(b))
    print(b[["R", "D", "H", "R_S1", "D_S1"]].describe().round(3))
