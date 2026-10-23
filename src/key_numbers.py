"""JAR merged paper, sixth version: every number printed in the text, computed from the analysis outputs (v6out/*.pkl)
and the audited workbooks of the fifth version.  The text modules import F and format from it, so the manuscript and the
analyses cannot drift apart.  Run: python key_numbers.py (prints all facts)."""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_merged as J                                                                # noqa: E402
import core as C                                                                 # noqa: E402

MOD = pickle.load(open(os.path.join(C.OUT, "models.pkl"), "rb"))
EXT = pickle.load(open(os.path.join(C.OUT, "extra.pkl"), "rb"))
DK = pickle.load(open(os.path.join(C.OUT, "dock.pkl"), "rb"))
CUT = pickle.load(open(os.path.join(C.OUT, "cut.pkl"), "rb"))
MDRUN = str(C.CFG.RAW / "md")


def r2(x, nd=2):
    from decimal import ROUND_HALF_UP, Decimal
    q = Decimal(1).scaleb(-nd)
    out = str(Decimal(str(float(x))).quantize(q, rounding=ROUND_HALF_UP)).replace("-", "−")
    return out[1:] if out.startswith("−") and set(out[1:]) <= set("0.") else out


def build():
    F = {}
    b = C.load(); m = b[b["主分析集"]].reset_index(drop=True)
    w, bb = J.load(); kt = J.key_table(w, bb); sh = J.shares(w, bb)
    F["n_wine"] = len(w); F["n_dist"] = len(bb); F["n_shared"] = int(bb["共有"].sum())
    F["share_wine_lo"], F["share_wine_hi"] = sh.iloc[:3].min(), sh.iloc[:3].max()
    F["share_dist"] = sh["馏出液合计"]; F["share_fr_lo"], F["share_fr_hi"] = sh.iloc[3:8].min(), sh.iloc[3:8].max()
    F["n_key_wine"] = int((kt["组别"] != "仅蒸馏").sum()); F["n_key_dist"] = int((kt["组别"] != "仅发酵").sum())
    F["n_key_both"] = int((kt["组别"] == "两阶段").sum()); F["n_key_union"] = len(kt)
    # C0 endpoints
    F["n_main"] = len(m); F["n_late"] = int(m["late"].sum()); F["n_early"] = int((m["late"] == 0).sum())
    F["R0"] = int((m["R"] == 0).sum()); F["R1"] = int((m["R"] == 1).sum())
    F["R_med"] = m["R"].median(); F["D_med"] = m["D"].median()
    F["D_lo"], F["D_hi"] = m["D"].min(), m["D"].max()
    F["rho_DR"] = stats.spearmanr(m["D"], m["R"]).correlation
    u = stats.mannwhitneyu(m.loc[m.late == 1, "R"], m.loc[m.late == 0, "R"], alternative="two-sided")
    F["auc_R_label"] = u.statistic / (F["n_late"] * F["n_early"]); F["p_R_label"] = u.pvalue
    F["R_med_early"] = m.loc[m.late == 0, "R"].median(); F["R_med_late"] = m.loc[m.late == 1, "R"].median()
    F["kw_R_class"] = stats.kruskal(*[m.loc[m["新类别"] == c, "R"] for c in C.CLS7]).pvalue
    det = EXT["det"]; F.update({f"det_{k}": v for k, v in det.items()})
    d_ = (m["R_S1"] - m["R"]).abs() * 100
    F["S1_gt5_main"] = int((d_ > 5).sum()); F["S1_med_main"] = d_.median()
    da = (b["R_S1"] - b["R"]).abs() * 100
    F["S1_gt5_all"] = int((da > 5).sum()); F["S1_med_all"] = da.median(); F["S1_max_all"] = da.max()
    # C1-C2
    p = MOD["perf"]
    def pv(lbl, mo, col):
        r = p[(p["方案"] == lbl) & (p["模型"] == mo)].iloc[0]; return r[f"{col}_mean"], r[f"{col}_sd"]
    for ep in ["R", "D"]:
        for sch in ["random", "group", "group0.5", "group0.7", "all115_random", "all115_group"]:
            for mo in ["baseline", "ridge", "xgb"]:
                for col in ["MAE", "R2", "rho"]:
                    F[f"{ep}_{sch}_{mo}_{col}"], F[f"{ep}_{sch}_{mo}_{col}_sd"] = pv(f"{ep}|{sch}", mo, col)
    for ep in ["R_S1", "D_S1"]:
        for mo in ["baseline", "ridge", "xgb"]:
            F[f"{ep}_{mo}_R2"], _ = pv(f"{ep}|random", mo, "R2")
    pe = EXT["perf_extra"]
    for ep in ["R", "D"]:
        for mo in ["baseline", "ridge", "xgb"]:
            r = pe[(pe["方案"].str.startswith(ep + "|")) & (pe["模型"] == mo)].iloc[0]
            F[f"multi_{ep}_{mo}_R2"] = r["R2_mean"]
    F["n_multi"] = int((m["n_det"] >= 2).sum())
    F["perm_R"] = MOD["perm"]["R"]["p"]; F["perm_D"] = MOD["perm"]["D"]["p"]
    F["nn_rand"] = np.median(MOD["nnsim"]["random"]); F["nn_group"] = np.median(MOD["nnsim"]["group"])
    F["n_clusters"] = int(m["butina_main_0.6"].nunique()); F["max_cluster"] = int(m["butina_main_0.6"].value_counts().max())
    F["n_singleton"] = int((m["butina_main_0.6"].value_counts() == 1).sum())
    co = MOD["classout"]
    for c in C.CLS7:
        for mo in ["baseline", "ridge", "xgb"]:
            F[f"co_D_{c}_{mo}"] = co[(co["终点"] == "D") & (co["留出类别"] == c) & (co["模型"] == mo)]["MAE"].iloc[0]
    F["late_auc"] = MOD["late_auc"].mean(); F["late_auc_sd"] = MOD["late_auc"].std(ddof=1)
    # error vs similarity (group, D, XGB): mean |error| above / below 0.5
    ns = MOD["nnsim"]["group"].ravel(); yD = np.tile(m["D"].values, MOD["nnsim"]["group"].shape[0])
    e = np.abs(MOD[("D", "group")]["xgb"]["oof"].ravel() - yD); e0 = np.abs(MOD[("D", "group")]["baseline"]["oof"].ravel() - yD)
    F["err_lo_sim"] = e[ns < 0.4].mean(); F["err_hi_sim"] = e[ns >= 0.5].mean()
    F["err0_lo_sim"] = e0[ns < 0.4].mean(); F["err0_hi_sim"] = e0[ns >= 0.5].mean()
    F["rho_err_sim"] = stats.spearmanr(ns, e).correlation
    wp = EXT["wine_pred"]; F["n_wine_dom"] = len(wp)
    F["ionone_D"] = float(wp.loc[wp["化合物"] == "β-紫罗酮", "D_pred"].iloc[0])
    F["wine_pred_early"] = int((wp["D_pred"] < 0.5).sum())
    # C3
    abl = MOD["abl"]
    def ab(sub, key, mo, col="相对基线降低", ep="D"):
        return abl[(abl["子集"] == sub) & (abl["特征组"] == key) & (abl["模型"] == mo) & (abl["终点"] == ep)][col].iloc[0]
    for sub in ["主分析集", "去酸类", "有实验沸点"]:
        keys = ["类别", "极性", "大小/疏水", "芳香环", "8 项描述符", "8 项 + 类别", "极性 + 类别"]
        if sub == "有实验沸点":
            keys += ["沸点", "8 项 + 沸点", "极性 + 沸点"]
        for key in keys:
            for mo in ["ridge", "xgb"]:
                F[f"abl_{sub}_{key}_{mo}"] = ab(sub, key, mo); F[f"ablR2_{sub}_{key}_{mo}"] = ab(sub, key, mo, "R2")
                F[f"ablRR_{sub}_{key}_{mo}"] = ab(sub, key, mo, ep="R")
    F["abl_noacid_max"] = abl[(abl["子集"] == "去酸类") & (abl["终点"] == "D")]["相对基线降低"].max()
    F["ablR_max"] = abl[(abl["终点"] == "R")]["相对基线降低"].max()
    nb = EXT["nonacid_bp"]
    for key in ["沸点", "极性", "8 项描述符", "8 项 + 沸点", "极性 + 沸点"]:
        r = nb[(nb["特征组"] == key) & (nb["模型"] == "ridge")].iloc[0]
        F[f"nb_{key}"] = r["相对基线降低"]; F[f"nbR2_{key}"] = r["R2"]
    cor = MOD["cor"]
    def cr(sub, d, col="rho"):
        return cor[(cor["子集"] == sub) & (cor["终点"] == "D") & (cor["描述符"] == d)][col].iloc[0]
    for sub in ["主分析集", "去酸类", "酯类", "醇类"]:
        for d in ["TPSA", "bp_exp", "HBA", "XLogP", "MW", "O/C"]:
            F[f"cor_{sub}_{d}"] = cr(sub, d); F[f"corq_{sub}_{d}"] = cr(sub, d, "q")
    for sname in ["主分析集", "去酸类"]:
        bt = MOD["boot"][(sname, "D")]
        for j, d in enumerate(C.DESC):
            F[f"coef_{sname}_{d}"] = bt["coef"][j]; F[f"coeflo_{sname}_{d}"] = bt["lo"][j]; F[f"coefhi_{sname}_{d}"] = bt["hi"][j]
            F[f"coefss_{sname}_{d}"] = bt["same_sign"][j]
    shp = EXT["shap_D"].set_index("描述符")
    F["shap_TPSA"] = shp.loc["TPSA", "mean|SHAP|"]; F["shap_TPSA_sd"] = shp.loc["TPSA", "SD"]
    F["shap_TPSA_top3"] = shp.loc["TPSA", "前3频率"]; F["shap_arom_top3"] = shp.loc["芳香环", "前3频率"]
    F["shap_HBD_top3"] = shp.loc["HBD", "前3频率"]
    fx = MOD["thr_fixed"]; tf = MOD["thr_fold"]
    F.update({f"thr_{k}": v for k, v in fx.items()})
    F["thr_fold_med"] = tf["thr"].median(); F["thr_fold_min"] = tf["thr"].min(); F["thr_fold_max"] = tf["thr"].max()
    F["thr_fold_sens"] = tf["sens"].mean(); F["thr_fold_sens_sd"] = tf["sens"].std(ddof=1)
    F["thr_fold_spec"] = tf["spec"].mean(); F["thr_fold_spec_sd"] = tf["spec"].std(ddof=1)
    # C4
    lig = DK["lig"]; st = DK["struct"]; ass = DK["assoc"]; grp = DK["group"]
    allv = np.concatenate([lig[f"v_{r}"].values for r in ["OR1A1", "OR1A2", "OR1D2", "OR1G1", "OR2W1", "OR51E2"]])
    F["score_lo"], F["score_hi"] = allv.min(), allv.max(); F["seed_sd_max"] = DK["seed_sd_max"]
    F["ha_r_all"] = DK["ha_r_all"]; F["ha_r_lo"], F["ha_r_hi"] = min(DK["ha_r"].values()), max(DK["ha_r"].values())
    sT = st[st["项"] == "TPSA"]; sA = st[st["项"] == "芳香环"]; sS = st[st["项"] == "TPSA（Spearman）"]
    F["bT_lo"], F["bT_hi"] = sT["beta"].min(), sT["beta"].max(); F["bT_qmax"] = sT["q"].max()
    F["bA_lo"], F["bA_hi"] = sA["beta"].min(), sA["beta"].max(); F["bA_qmax"] = sA["q"].max()
    F["sT_lo"], F["sT_hi"] = sS["beta"].min(), sS["beta"].max(); F["sT_qmax"] = sS["q"].max()
    F["struct_R2_lo"], F["struct_R2_hi"] = sT["R2"].min(), sT["R2"].max()
    F["bT_OR2W1_lo"] = sT.set_index("受体").loc["OR2W1", "lo"]
    lo = DK["loo_struct"]; F["loo_lo"], F["loo_hi"] = lo["rho"].min(), lo["rho"].max()
    aD = ass[(ass["终点"] == "D") & (ass["受体"] != "mean")]; aR = ass[(ass["终点"] == "R") & (ass["受体"] != "mean")]
    F["aD_lo"], F["aD_hi"], F["aD_qmin"] = aD["rho"].min(), aD["rho"].max(), aD["q"].min()
    F["aR_lo"], F["aR_hi"], F["aR_qmin"] = aR["rho"].min(), aR["rho"].max(), aR["q"].min()
    F["paD_lo"], F["paD_hi"] = aD["偏rho"].min(), aD["偏rho"].max()
    F["paD_nsig"] = int((aD["偏q"] < 0.05).sum()); F["paD_qmin"] = aD["偏q"].min()
    F["paD_sig"] = "、".join(aD[aD["偏q"] < 0.05]["受体"])
    F["paD_qother"] = aD.loc[aD["偏q"] >= 0.05, "偏q"].min()
    F["paR_lo"], F["paR_hi"], F["paR_qmin"] = aR["偏rho"].min(), aR["偏rho"].max(), aR["偏q"].min()
    mD = ass[(ass["终点"] == "D") & (ass["受体"] == "mean")].iloc[0]; F["aDmean"] = mD["rho"]
    F["aDmean_lo"], F["aDmean_hi"] = DK["assoc_ci"]["D"]
    mR = ass[(ass["终点"] == "R") & (ass["受体"] == "mean")].iloc[0]; F["aRmean"] = mR["rho"]
    F["aRmean_lo"], F["aRmean_hi"] = DK["assoc_ci"]["R"]
    g6 = grp[grp["受体"] != "mean"]; F["grp_auc_lo"], F["grp_auc_hi"], F["grp_pmin"] = g6["AUC"].min(), g6["AUC"].max(), g6["P"].min()
    la = DK["loo_assoc"]; F["looD_lo"], F["looD_hi"] = la[la["终点"] == "D"]["rho"].min(), la[la["终点"] == "D"]["rho"].max()
    F["n_meas_lig"] = DK["n_meas"]
    dk = pd.read_csv(os.path.join(C.CFG.RAW, "docking", "dock_results.csv"))
    s2 = dk[dk.receptor == "OR1A2"].groupby("code")["best"].agg(["mean", "std"])
    for code in ["L1", "L2", "L3", "L5"]:
        F[f"or1a2_{code}"] = s2.loc[code, "mean"]
    pk = pd.read_csv(os.path.join(C.CFG.RAW, "docking", "pocket_residues.csv"))
    best = pk.loc[pk.groupby(["receptor", "code"])["score"].idxmin()]; b2 = best[best.receptor == "OR1A2"]
    for res in ["PHE206", "TYR258", "THR254", "VAL203"]:
        F[f"contact_{res}"] = b2["residues_4A"].str.contains(res).mean() * 100
    sm = pd.read_csv(os.path.join(MDRUN, "md_summary.csv")).set_index("code")
    F["md_ns"] = float(sm["ns"].min())
    F["md_bb_lo"], F["md_bb_hi"] = sm["rmsd_bb"].min(), sm["rmsd_bb"].max()
    for code in ["L1", "L2", "L3", "L5"]:
        F[f"md_lig_{code}"] = sm.loc[code, "rmsd_lig"]; F[f"md_hb_{code}"] = sm.loc[code, "hb_frac"] * 100
        F[f"md_etot_{code}"] = sm.loc[code, "etot"]
    F["md_rmsf_lo"], F["md_rmsf_hi"] = sm["rmsf_mean"].min(), sm["rmsf_mean"].max()
    # C5
    et = CUT["ethanol"].set_index("h"); odd = CUT["odor"]; cb = CUT["combos"]
    F["eth_0"], F["eth_14"], F["eth_50"] = et.loc[0.0, "乙醇回收率"], et.loc[14.0, "乙醇回收率"], et.loc[50.0, "乙醇回收率"]
    F["vol_14"], F["abv_14"], F["abv_50"] = et.loc[14.0, "酒心体积"], et.loc[14.0, "酒心酒度"], et.loc[50.0, "酒心酒度"]
    ho = odd[odd["p_头"] >= 0.999]; F["n_headonly"] = len(ho); F["headonly_names"] = "、".join(ho["采用名称"])
    F["ho_front"], F["ho_uni"], F["ho_back"] = ho["min_14"].mean() * 100, ho["uni_14"].mean() * 100, ho["max_14"].mean() * 100
    for h in [0, 14, 50]:
        lo_, hi_ = odd[f"min_{h}"], odd[f"max_{h}"]
        F[f"cat_{h}_ret"] = int((lo_ >= 0.5).sum()); F[f"cat_{h}_dep"] = int(((lo_ < 0.5) & (hi_ >= 0.5)).sum())
        F[f"cat_{h}_lost"] = int((hi_ < 0.5).sum())
    sp = odd[(odd["p_头"] < 0.999) & (odd["p_头"] > 0)]; F["n_spread"] = len(sp)
    F["sp_uni14_med"] = sp["uni_14"].median() * 100; F["sp_50_med"] = sp["min_50"].median() * 100
    def cbv(name, basis, col):
        return cb[(cb["组合"] == name) & (cb["浓度口径"] == basis)][col].iloc[0]
    base, nohead, ext = "中间+次（基准酒心）", "头+中间+次（不掐头）", "中间+次+次尾（延后去尾）"
    for nm, key in [(base, "base"), (nohead, "nohead"), (ext, "ext"), ("中间", "mid"), ("头+中间+次+次尾", "all4")]:
        for basis in ["S0", "S1"]:
            for col, k2 in [("乙醇回收率", "eth"), ("酒度", "abv"), ("候选物保留50以上", "n50"), ("酸类回收率", "acid"),
                            ("酸类占切段挥发物", "acidsh"), ("保留_乙酸苯乙酯", "pea"), ("保留_β-大马士酮", "bdam"),
                            ("保留_己酸乙酯", "eh"), ("保留_乙酸异戊酯", "ia"), ("保留_辛酸", "c8"), ("保留_苯乙醇", "pe")]:
                F[f"cb_{key}_{basis}_{k2}"] = cbv(nm, basis, col)
    for nm in ["乙酸异戊酯", "己酸乙酯", "β-大马士酮", "乙酸苯乙酯", "异丁酸异丙酯", "辛酸异戊酯"]:
        r = odd[odd["采用名称"] == nm].iloc[0]
        F[f"od_{nm}_uni14"] = r["uni_14"] * 100; F[f"od_{nm}_min14"] = r["min_14"] * 100; F[f"od_{nm}_max14"] = r["max_14"] * 100
        F[f"od_{nm}_R"] = r["R"] * 100; F[f"od_{nm}_RS1"] = r["R_S1"] * 100
    # C6
    st_ = CUT["stress"]
    for (sys_, basis), g in st_.groupby(["体系", "浓度口径"]):
        for _, r in g.iterrows():
            F[f"stress_{sys_}_{basis}_{r['阈值倍数']}"] = int(r["候选数"])
    oa = CUT["oav"]
    for (sys_, basis), g in oa.groupby(["体系", "浓度口径"]):
        for gr in ["稳定", "阈值敏感", "临界"]:
            F[f"grade_{sys_}_{basis}_{gr}"] = int((g["等级"] == gr).sum())
        F[f"sens_names_{sys_}_{basis}"] = "、".join(g[g["等级"] == "阈值敏感"].sort_values("OAVmax")["名称"])
    return F


F = build()

if __name__ == "__main__":
    for k, v in F.items():
        print(f"{k}: {v if not isinstance(v, float) else round(v, 4)}")
