"""Merged JAR paper: reader for the paper 3 figure-data workbook (the paper 3 plotting script was lost, so the merged figures are
rebuilt from the workbook that the audited draft was made from).  Every block below is the cell range of one labelled table in
'论文三_图数据_第二版.xlsx'; values are the audited ones, so the redrawn panels reproduce the published numbers exactly."""
import os

import numpy as np
import pandas as pd

import config as CFG
P3DIR = str(CFG.RAW)
BOOK = str(CFG.RAW / "distillation_figure_data_paper3.xlsx")
FRS = ["头", "中间", "次", "次尾", "尾"]
FRL = ["头馏分", "中间馏分", "次馏分", "次尾馏分", "尾馏分"]
FRCOL = ["#8C1C13", "#D9603F", "#E3A21A", "#5B8DB8", "#2F4B7C"]
CLS9 = ["酯类", "醇类", "酸类", "缩醛类", "呋喃类", "醛酮类", "萜类及降异戊二烯类", "芳香烃类", "烷烃及烯烃类"]
CLS9_COL = {"酯类": "#8C1C13", "醇类": "#E8734A", "酸类": "#E3A21A", "缩醛类": "#6A57A8", "呋喃类": "#2F6DB5",
            "醛酮类": "#2E9A6E", "萜类及降异戊二烯类": "#D96BA0", "芳香烃类": "#5D6D7E", "烷烃及烯烃类": "#A9CCE3"}
DESC = ["分子量", "XLogP3", "TPSA", "HBD", "HBA", "可旋转键数", "芳香环数", "O/C"]


def _sheet(name, header, nrows=None, usecols=None):
    df = pd.read_excel(BOOK, sheet_name=name, header=header, usecols=usecols, nrows=nrows)
    return df.dropna(how="all")


def fractions():
    """second distillation: 14 cuts (volume midpoint, strength, acid, solids with SD and Duncan letters) and the 5 cut fractions"""
    d = _sheet("图2_馏分理化", 17, nrows=14)
    d.columns = ["段号", "累计体积", "中点", "酒精度", "酒精度SD", "酒精度字母", "总酸", "总酸SD", "总酸字母",
                 "固形物", "固形物SD", "固形物字母", "段乙醇", "累计乙醇", "累计回收率", "所属馏分"][:len(d.columns)]
    cut = _sheet("图2_馏分理化", 40, nrows=5).iloc[:, :7]
    cut.columns = ["截取馏分", "段号范围", "体积", "平均酒精度", "乙醇", "占回收乙醇", "体积占比"]
    return d, cut


def classes():
    """class totals per fraction (mg/L) with SD and Duncan letters, 9 classes"""
    c = _sheet("图3_组成", 3, nrows=9)
    return c.set_index("类别").reindex(CLS9)


def compounds():
    """115 compounds: contents, row z-values, PCA loadings/correlation loadings, Ward cluster"""
    c = _sheet("图4_PCA与聚类", 3, nrows=115)
    return c


def pca():
    """PCA of the 5 fractions: eigenvalues, explained variance and fraction scores"""
    p = _sheet("图4_PCA与聚类", 122, nrows=4).iloc[:, :8]
    p.columns = ["主成分", "特征值", "方差贡献率"] + [f + "_得分" for f in FRS]
    return p.set_index("主成分")


def clusters():
    """mean z-curve of the four elution patterns"""
    k = _sheet("图4_PCA与聚类", 130, nrows=4).iloc[:, :7]
    k.columns = ["聚类", "化合物数"] + FRS
    return k.set_index("聚类")


def plsda():
    """PLS-DA model statistics and per-descriptor VIP / coefficients"""
    m = _sheet("图5_PLSDA", 3, nrows=2).iloc[:, :9].set_index("化合物集")
    v = _sheet("图5_PLSDA", 7, nrows=8).iloc[:, :5].set_index("描述符")
    return m, v


def xgb():
    """XGBoost: per-descriptor mean |SHAP|, top-3 frequency, univariate AUC; and the per-compound SHAP matrix"""
    g = pd.read_excel(BOOK, sheet_name="图6_XGBoost", header=3, nrows=8)
    g = g[["描述符", "mean |SHAP|（公式）", "前 3 位频率", "单变量 AUC（公式）", "P（Python）"]]
    g.columns = ["描述符", "mean|SHAP|", "前3频率", "单变量AUC", "P"]
    g = g.set_index("描述符").reindex(DESC)
    s = pd.read_excel(BOOK, sheet_name="图6_XGBoost", header=21, usecols="D:V").dropna(how="all")
    s = s.iloc[:96]
    return g, s


def pairs():
    """four structurally related compound pairs with different elution behaviour"""
    p = _sheet("图7_化合物对", 3, nrows=8)
    return p


def process():
    """single-factor trials, orthogonal main effects and the cut simulation"""
    sf = _sheet("图9_工艺", 3, nrows=15)
    me = _sheet("图9_工艺", 49, nrows=9).iloc[:, :4]
    me.columns = ["因素", "水平", "L9_k偏差", "单因素偏差"][:len(me.columns)]
    sim = _sheet("图9_工艺", 62, nrows=6)
    return sf, me, sim


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    d, cut = fractions(); print(d.head(3).to_string()); print(cut.to_string())
    print(classes().iloc[:, :6].to_string())
    print(pca().to_string()); print(clusters().to_string())
    m, v = plsda(); print(m.to_string()); print(v.to_string())
    g, s = xgb(); print(g.to_string()); print(s.shape, list(s.columns)[:8])
    print(pairs().iloc[:, :6].to_string())
    sf, me, sim = process(); print(sf.head(3).to_string()); print(me.to_string()); print(sim.to_string())
