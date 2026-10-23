"""JAR merged paper, sixth version: C4 structural associations shared by distillation partitioning and multi-receptor docking.
Independent unit = ligand (31).  The three Vina seeds of each receptor-ligand pair are averaged first; the six receptors are
correlated end points (BH within each block).  Size correction: residual of the seed-mean Vina score on heavy-atom count
within each receptor (heavy-atom-corrected score; positive = weaker than expected for its size).
Partitioning associations use only the 30 ligands with measured fraction data (beta-ionone, predicted only, is excluded).
Run: python dock.py"""
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core as C                                                                 # noqa: E402

REC = ["OR1A1", "OR1A2", "OR1D2", "OR1G1", "OR2W1", "OR51E2"]
SEED = 2026


def load():
    lig = pd.read_csv(os.path.join(C.CFG.RAW, "docking", "ligands.csv"))
    dk = pd.read_csv(os.path.join(C.CFG.RAW, "docking", "dock_results.csv"))
    sc = dk.groupby(["receptor", "code"])["best"].agg(["mean", "std"]).reset_index()
    b = C.load()
    lig = lig.merge(b[["CID", "R", "D", "H", "芳香环", "XLogP", "RotB", "HBA", "O/C", "MW", "late"]].rename(
        columns={"芳香环": "arom", "XLogP": "XLogP_b", "MW": "MW_b"}), on="CID", how="left")
    # aromatic ring count for beta-ionone (not in the distillate table) from its structure: 0
    lig["arom"] = lig["arom"].fillna(0)
    for r in REC:
        s = sc[sc.receptor == r].set_index("code")
        lig[f"v_{r}"] = lig["code"].map(s["mean"]); lig[f"sd_{r}"] = lig["code"].map(s["std"])
        slope, icpt = np.polyfit(lig["HA"], lig[f"v_{r}"], 1)
        lig[f"c_{r}"] = lig[f"v_{r}"] - (icpt + slope * lig["HA"])
        lig[f"slope_{r}"] = slope
    lig["c_mean"] = lig[[f"c_{r}" for r in REC]].mean(1)
    return lig, sc


def zs(v):
    v = np.asarray(v, float); return (v - v.mean()) / v.std(ddof=0)


def ols(X, y):
    X1 = np.column_stack([np.ones(len(y)), X]); beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta; dof = len(y) - X1.shape[1]
    s2 = resid @ resid / dof; cov = s2 * np.linalg.inv(X1.T @ X1)
    se = np.sqrt(np.diag(cov)); t = beta / se; p = 2 * stats.t.sf(np.abs(t), dof)
    r2 = 1 - resid @ resid / ((y - y.mean()) ** 2).sum()
    return beta[1:], se[1:], p[1:], r2


def partial_spearman(x, y, Z):
    rx = stats.rankdata(x); ry = stats.rankdata(y); RZ = np.column_stack([stats.rankdata(z) for z in Z.T])
    Z1 = np.column_stack([np.ones(len(x)), RZ])
    ex = rx - Z1 @ np.linalg.lstsq(Z1, rx, rcond=None)[0]; ey = ry - Z1 @ np.linalg.lstsq(Z1, ry, rcond=None)[0]
    r = np.corrcoef(ex, ey)[0, 1]; dof = len(x) - 2 - Z.shape[1]
    t = r * np.sqrt(dof / (1 - r ** 2)); return r, 2 * stats.t.sf(abs(t), dof)


def main():
    lig, sc = load(); out = {"lig": lig, "sc": sc}
    rng = np.random.default_rng(SEED)
    # (1) structural model of the corrected score: z(TPSA) + z(aromatic rings), all 31 ligands, bootstrap over ligands
    rows = []
    X = np.column_stack([zs(lig["TPSA"]), zs(lig["arom"])])
    for r in REC:
        y = lig[f"c_{r}"].values
        beta, se, p, r2 = ols(X, y)
        bs = []
        for i in range(2000):
            ii = rng.integers(0, len(y), len(y))
            if np.std(X[ii, 1]) == 0:
                continue
            bs.append(ols(np.column_stack([zs(lig["TPSA"].values[ii]), zs(lig["arom"].values[ii])]), y[ii])[0])
        bs = np.array(bs)
        for k, nm in enumerate(["TPSA", "芳香环"]):
            rows.append(dict(受体=r, 项=nm, beta=beta[k], lo=np.percentile(bs[:, k], 2.5), hi=np.percentile(bs[:, k], 97.5),
                             P=p[k], R2=r2))
        rho = stats.spearmanr(lig["TPSA"], y)
        rows.append(dict(受体=r, 项="TPSA（Spearman）", beta=rho.correlation, lo=np.nan, hi=np.nan, P=rho.pvalue, R2=np.nan))
    st = pd.DataFrame(rows)
    st["q"] = np.nan
    for nm, g in st.groupby("项"):
        st.loc[g.index, "q"] = multipletests(g["P"], method="fdr_bh")[1]
    out["struct"] = st
    # leave-one-ligand-out influence on rho(TPSA, corrected score)
    loo = []
    for r in REC:
        y = lig[f"c_{r}"].values; t = lig["TPSA"].values
        for i in range(len(y)):
            k = np.arange(len(y)) != i
            loo.append(dict(受体=r, 去除=lig["ligand"].iloc[i], rho=stats.spearmanr(t[k], y[k]).correlation))
    out["loo_struct"] = pd.DataFrame(loo)
    # (2) partitioning end points vs corrected scores, 30 measured ligands
    mm = lig[lig["R"].notna()].reset_index(drop=True); out["n_meas"] = len(mm)
    rows = []
    for ep in ["R", "D", "H"]:
        for r in REC + ["mean"]:
            col = f"c_{r}" if r != "mean" else "c_mean"
            rho = stats.spearmanr(mm[ep], mm[col])
            pr, pp = partial_spearman(mm[ep].values, mm[col].values, mm[["TPSA", "arom", "HA"]].values)
            rows.append(dict(终点=ep, 受体=r, rho=rho.correlation, P=rho.pvalue, 偏rho=pr, 偏P=pp))
            raw = stats.spearmanr(mm[ep], mm[f"v_{r}" if r != "mean" else "v_OR1A2"])
            rows[-1]["rho_原始评分"] = raw.correlation
    ass = pd.DataFrame(rows)
    ass["q"] = np.nan; ass["偏q"] = np.nan
    for ep, g in ass[ass["受体"] != "mean"].groupby("终点"):
        ass.loc[g.index, "q"] = multipletests(g["P"], method="fdr_bh")[1]
        ass.loc[g.index, "偏q"] = multipletests(g["偏P"], method="fdr_bh")[1]
    out["assoc"] = ass
    # bootstrap CI of rho(D, c_mean) and rho(R, c_mean)
    ci = {}
    for ep in ["R", "D"]:
        bs = []
        for i in range(2000):
            ii = rng.integers(0, len(mm), len(mm))
            bs.append(stats.spearmanr(mm[ep].values[ii], mm["c_mean"].values[ii]).correlation)
        ci[ep] = (np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5))
    out["assoc_ci"] = ci
    loo = []
    for ep in ["R", "D"]:
        for i in range(len(mm)):
            k = np.arange(len(mm)) != i
            loo.append(dict(终点=ep, 去除=mm["ligand"].iloc[i],
                            rho=stats.spearmanr(mm[ep].values[k], mm["c_mean"].values[k]).correlation))
    out["loo_assoc"] = pd.DataFrame(loo)
    # (3) early vs late groups (label), Mann-Whitney per receptor
    rows = []
    for r in REC + ["mean"]:
        col = f"c_{r}" if r != "mean" else "c_mean"
        a = mm.loc[mm["late"] == 1, col]; e = mm.loc[mm["late"] == 0, col]
        u = stats.mannwhitneyu(a, e, alternative="two-sided")
        rows.append(dict(受体=r, n_late=len(a), n_early=len(e), AUC=u.statistic / (len(a) * len(e)), P=u.pvalue,
                         med_late=a.median(), med_early=e.median()))
    out["group"] = pd.DataFrame(rows)
    # (4) score vs heavy atoms
    out["ha_r"] = {r: stats.pearsonr(lig["HA"], lig[f"v_{r}"])[0] for r in REC}
    allv = np.concatenate([lig[f"v_{r}"].values for r in REC]); allha = np.tile(lig["HA"].values, len(REC))
    out["ha_r_all"] = stats.pearsonr(allha, allv)[0]
    out["seed_sd_max"] = float(sc["std"].max())
    with open(os.path.join(C.OUT, "dock.pkl"), "wb") as fh:
        pickle.dump(out, fh)
    pd.set_option("display.width", 250)
    print(st.round(3).to_string()); print(ass.round(3).to_string()); print(out["group"].round(3).to_string())
    print("CI", ci, "n", len(mm), out["ha_r"], out["ha_r_all"], out["seed_sd_max"])
    print(out["loo_struct"].groupby("受体")["rho"].agg(["min", "max"]).round(3))
    print(out["loo_assoc"].groupby("终点")["rho"].agg(["min", "max"]).round(3))


if __name__ == "__main__":
    main()
