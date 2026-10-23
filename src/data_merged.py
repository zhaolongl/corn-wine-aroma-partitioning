"""Merged JAR paper: shared data layer for figures, Table 1 and the figure-data workbook.
Wine side = paper 2 (thesis appendix 4, three representative co-fermented wines, ug/L, n = 3) with the n8 descriptors and fate
predictions; distillate side = paper 3 (thesis appendix 3, five fractions of the second distillation, mg/L) with its thresholds
(图8_OAV).  Compounds are matched by PubChem CID (70 shared after correcting the naphthalene CAS of the thesis).
Key aroma-active compounds (one criterion for both systems): OAV >= 1 in at least one sample of the system."""
import os

import numpy as np
import pandas as pd

SCR = os.path.dirname(os.path.abspath(__file__))
import config as CFG
P3DIR = str(CFG.RAW)
WINES = ["山楂", "梨", "石榴"]
WLAB = {"山楂": "山楂 8∶2", "梨": "梨 7∶3", "石榴": "石榴 8∶2"}
FR = ["头馏分_mg/L", "中间馏分_mg/L", "次馏分_mg/L", "次尾馏分_mg/L", "尾馏分_mg/L"]
FRS = ["头", "中间", "次", "次尾", "尾"]
VOL = np.array([50., 300., 50., 100., 200.])
CLASSES = ["酯类", "醇类", "醛酮类", "酸类", "萜类及降异戊二烯类", "缩醛类", "呋喃类", "芳香烃类", "烷烃及烯烃类", "其它类"]


def load():
    w = pd.read_csv(os.path.join(CFG.RAW, "cofermented_wine_compounds.csv"))
    b = pd.read_csv(os.path.join(CFG.RAW, "distillate_115_compounds.csv"))
    th = pd.read_excel(os.path.join(CFG.RAW, "distillation_figure_data_paper3.xlsx"), sheet_name="图8_OAV", header=3)
    th = th[th["编号"].notna()][["编号", "阈值（μg/L）", "分级", "头占比", "酒心占比", "酒尾占比"]]
    b = b.merge(th, on="编号", how="left")
    for f, s in zip(FR, FRS):
        b[f"{s}_OAV"] = b[f] * 1000 / b["阈值（μg/L）"]
    b["OAVmax"] = b[[f"{s}_OAV" for s in FRS]].max(1)
    b["共有"] = b["CID"].isin(w["CID"])
    return w, b


def key_table(w, b):
    """26 key aroma-active compounds (union), ordered: shared, wine-only, distillate-only; within groups by max OAV"""
    kw = set(w.loc[w["OAVmax"] >= 1, "CID"]); kd = set(b.loc[b["OAVmax"] >= 1, "CID"])
    rows = []
    for cid in kw | kd:
        rw = w[w.CID == cid]; rd = b[b.CID == cid]
        r = dict(CID=cid, 名称=(rd["采用名称"].iloc[0] if len(rd) else rw["化合物"].iloc[0]),
                 类别=(rd["新类别"].iloc[0] if len(rd) else rw["类别_统一"].iloc[0]),
                 组别=("两阶段" if cid in kw and cid in kd else "仅发酵" if cid in kw else "仅蒸馏"))
        for s in WINES:
            r[f"{s}_μg/L"] = rw[f"{s} 均值"].iloc[0] if len(rw) else np.nan
            r[f"{s}_OAV"] = rw[f"{s}_OAV"].iloc[0] if len(rw) else np.nan
        r["酒_OAVmax"] = rw["OAVmax"].iloc[0] if len(rw) else np.nan
        for s, f in zip(FRS, FR):
            r[f"{s}_mg/L"] = rd[f].iloc[0] if len(rd) else np.nan
            r[f"{s}_OAV"] = rd[f"{s}_OAV"].iloc[0] if len(rd) else np.nan
        r["馏_OAVmax"] = rd["OAVmax"].iloc[0] if len(rd) else np.nan
        for c in ["TPSA", "HBD", "XLogP", "MW"]:
            r[c] = rd[c].iloc[0] if len(rd) else rw[c].iloc[0]
        for c in ["洗脱类型", "馏出酒度", "头占比", "酒心占比", "酒尾占比", "分级"]:
            r[c] = rd[c].iloc[0] if len(rd) else np.nan
        r["阈值_μg/L"] = rd["阈值（μg/L）"].iloc[0] if len(rd) else rw["阈值（μg/L）"].iloc[0]
        if not len(rd):
            r["洗脱类型"] = "预测" + rw["预测类型"].iloc[0]
        rows.append(r)
    k = pd.DataFrame(rows)
    k["_g"] = k["组别"].map({"两阶段": 0, "仅发酵": 1, "仅蒸馏": 2})
    k["_o"] = np.where(k["组别"] == "仅蒸馏", -k["馏_OAVmax"], -k["酒_OAVmax"])
    return k.sort_values(["_g", "_o"]).drop(columns=["_g", "_o"]).reset_index(drop=True)


def shares(w, b):
    out = {}
    for s in WINES:
        out[WLAB[s]] = w.loc[w["共有"], f"{s} 均值"].sum() / w[f"{s} 均值"].sum() * 100
    M = b[FR].values * VOL; sh = b["共有"].values
    for i, s in enumerate(FRS):
        out[s + "馏分"] = M[sh, i].sum() / M[:, i].sum() * 100
    out["馏出液合计"] = M[sh].sum() / M.sum() * 100
    return pd.Series(out)


if __name__ == "__main__":
    w, b = load(); k = key_table(w, b)
    pd.set_option("display.width", 250)
    print(k[["名称", "组别", "类别", "酒_OAVmax", "馏_OAVmax", "洗脱类型", "TPSA", "酒心占比"]].round(2).to_string())
    print(k["组别"].value_counts().to_dict()); print(shares(w, b).round(2).to_string())
