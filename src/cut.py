"""JAR merged paper, sixth version: C5 cut scenarios with uncertainty propagation and C6 OAV / candidate-list robustness.
C5  Whole-fraction combinations are identifiable from the measured fraction shares.  Cuts inside the 0-50 mL head fraction are
    not identifiable (one pooled head sample): without a distribution assumption the retained share of the head portion can be
    anywhere in [0, 1].  Three labelled stress scenarios for the within-head distribution of each compound are used:
    front-loaded (density ~ exp(-3x)), uniform, back-loaded (density ~ exp(+3x)), x = position within the head volume.
    Each is combined with the two concentration bases (S0 tabulated, S1 conditional dilution scenario) -> 6 scenarios.
    Ethanol inside the head is taken as uniform (the first 50-mL cut is a single measured segment of 78.8% vol).
C6  Thresholds are multiplied by 0.25-4 as an arbitrary stress test (not a confidence interval of a measured threshold).
Run: python cut.py"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_merged as J                                                                # noqa: E402
import core as C                                                                 # noqa: E402

K = {"前置富集": 3.0, "均匀": 0.0, "后置富集": -3.0}
HEART_END = 400.0
FACT = [0.25, 0.5, 1.0, 2.0, 4.0]


def head_left(u, k):
    """share of the head-fraction mass left after removing the first u (0-1) of the head volume"""
    u = np.asarray(u, float)
    if abs(k) < 1e-12:
        return 1 - u
    return (np.exp(-k * u) - np.exp(-k)) / (1 - np.exp(-k))


def main():
    b = C.load(); w, bb = J.load(); kt = J.key_table(w, bb)
    b = b.merge(bb[["编号", "阈值（μg/L）"]], on="编号", how="left")
    od = kt[kt["馏_OAVmax"] >= 1]["CID"].tolist()                         # 25 distillate odorants
    acids = b["新类别"] == "酸类"
    out = {}
    hgrid = np.arange(0, 50.5, 1.0)
    # ---------------------------------------------------------------- ethanol (uniform within head)
    et = C.ETOH; etot = et.sum()
    eth = []
    for h in hgrid:
        u = h / 50
        e_heart = et[0] * (1 - u) + et[1] + et[2]
        vol = (50 - h) + 300 + 50
        eth.append(dict(h=h, 掐头百分比=h / 700 * 100, 乙醇回收率=e_heart / etot * 100, 酒心体积=vol, 酒心酒度=e_heart / vol * 100))
    out["ethanol"] = pd.DataFrame(eth)
    # ---------------------------------------------------------------- per compound curves, 6 scenarios
    rows = []
    for basis, pre in [("S0", "p_"), ("S1", "p1_")]:
        P = b[[pre + s for s in C.FRS]].values
        for kn, k in K.items():
            for h in hgrid:
                left = head_left(h / 50, k)
                Rh = P[:, 0] * left + P[:, 1] + P[:, 2]
                for i in range(len(b)):
                    rows.append((b["CID"].iloc[i], basis, kn, h, Rh[i]))
    cur = pd.DataFrame(rows, columns=["CID", "浓度口径", "头内分布", "h", "酒心占比"])
    out["curves"] = cur
    # ---------------------------------------------------------------- robust classification at 2% (14 mL) and 7.1% (50 mL)
    odd = b[b["CID"].isin(od)][["CID", "采用名称", "新类别", "p_头", "R", "R_S1", "洗脱类型"]].copy()
    for h in [14.0, 50.0, 0.0]:
        sub = cur[(cur["h"] == h) & cur["CID"].isin(od)]
        g = sub.groupby("CID")["酒心占比"]
        odd[f"min_{int(h)}"] = odd["CID"].map(g.min()); odd[f"max_{int(h)}"] = odd["CID"].map(g.max())
        odd[f"uni_{int(h)}"] = odd["CID"].map(sub[(sub["浓度口径"] == "S0") & (sub["头内分布"] == "均匀")].set_index("CID")["酒心占比"])
    def cls(r):
        if r["min_14"] >= 0.5:
            return "稳健保留"
        if r["max_14"] < 0.5:
            return "稳健损失"
        return "情景依赖"
    odd["2%掐头判断"] = odd.apply(cls, axis=1)
    def cls50(r):
        if r["min_50"] >= 0.5:
            return "稳健保留"
        if r["max_50"] < 0.5:
            return "稳健损失"
        return "口径依赖"
    odd["整段掐头判断"] = odd.apply(cls50, axis=1)
    out["odor"] = odd.sort_values("uni_14", ascending=False)
    # ---------------------------------------------------------------- whole-fraction combinations (identifiable)
    combos = {"中间": [1], "中间+次（基准酒心）": [1, 2], "头+中间+次（不掐头）": [0, 1, 2],
              "中间+次+次尾（延后去尾）": [1, 2, 3], "头+中间+次+次尾": [0, 1, 2, 3]}
    crow = []
    for nm, idx in combos.items():
        vol = C.VOL[idx].sum(); e = et[idx].sum()
        for basis, pre in [("S0", "p_"), ("S1", "p1_")]:
            P = b[[pre + s for s in C.FRS]].values
            ret = P[:, idx].sum(1)
            # acid mass share in the cut: mass of all acids / total volatile mass in the cut (S0 or S1 concentrations)
            conc = b[C.FR].values * (C.S1F if basis == "S1" else 1.0)
            mass = conc * C.VOL
            acid_in = mass[acids.values][:, idx].sum(); tot_in = mass[:, idx].sum()
            crow.append(dict(组合=nm, 浓度口径=basis, 体积=vol, 酒度=e / vol * 100, 乙醇回收率=e / etot * 100,
                             候选物保留中位数=np.median(ret[b["CID"].isin(od).values]) * 100,
                             候选物保留50以上=int((ret[b["CID"].isin(od).values] >= 0.5).sum()),
                             酸类回收率=mass[acids.values][:, idx].sum() / mass[acids.values].sum() * 100,
                             酸类占切段挥发物=acid_in / tot_in * 100,
                             **{f"保留_{n}": ret[b["采用名称"].values == n][0] * 100 for n in
                                ["β-大马士酮", "己酸乙酯", "乙酸异戊酯", "乙酸苯乙酯", "丁酸异戊酯", "辛酸", "苯乙醇", "异戊醇"]}))
    out["combos"] = pd.DataFrame(crow)
    # ---------------------------------------------------------------- counts vs h per scenario
    cnt = cur[cur["CID"].isin(od)].groupby(["浓度口径", "头内分布", "h"])["酒心占比"].agg(
        n50=lambda v: (v >= 0.5).sum(), med="median").reset_index()
    out["counts"] = cnt
    # ---------------------------------------------------------------- C6 OAV stress
    rows = []
    # distillate (S0 and S1 concentrations)
    for basis in ["S0", "S1"]:
        conc = b[C.FR].values * (C.S1F if basis == "S1" else 1.0)
        oav = conc * 1000 / b["阈值（μg/L）"].values[:, None]
        omax = np.nanmax(oav, 1)
        for i in range(len(b)):
            if not np.isfinite(omax[i]):
                continue
            rows.append(dict(体系="馏出液", 浓度口径=basis, CID=b["CID"].iloc[i], 名称=b["采用名称"].iloc[i], OAVmax=omax[i]))
    for i, r in w.iterrows():
        om = r["OAVmax"]
        if np.isfinite(om):
            rows.append(dict(体系="共发酵酒", 浓度口径="—", CID=r["CID"], 名称=r["化合物"], OAVmax=om))
    oa = pd.DataFrame(rows)
    def grade(v):
        if v >= 2:
            return "稳定"
        if v >= 1:
            return "阈值敏感"
        if v >= 0.5:
            return "临界"
        return "非候选"
    oa["等级"] = oa["OAVmax"].apply(grade)
    out["oav"] = oa
    st = []
    for (sys_, basis), g in oa.groupby(["体系", "浓度口径"]):
        for f in FACT:
            st.append(dict(体系=sys_, 浓度口径=basis, 阈值倍数=f, 候选数=int((g["OAVmax"] / f >= 1).sum())))
    out["stress"] = pd.DataFrame(st)
    # union candidate grades for Table 1: distillate S0 and wine
    with open(os.path.join(C.OUT, "cut.pkl"), "wb") as fh:
        pickle.dump(out, fh)
    pd.set_option("display.width", 250)
    print(out["ethanol"].iloc[[0, 14, 50]].round(2).to_string())
    print(odd.round(3).to_string())
    print(out["combos"].round(2).to_string())
    print(out["stress"].to_string())
    print(oa[oa["OAVmax"] >= 0.5].sort_values(["体系", "浓度口径", "OAVmax"]).round(2).to_string())
    print(cnt[cnt["h"].isin([0, 14, 50])].to_string())


if __name__ == "__main__":
    main()
