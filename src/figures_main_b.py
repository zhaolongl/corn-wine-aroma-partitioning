"""JAR merged paper, sixth version: main figures 4, 5 and 6 (new models and docking associations).
Fig 4  out-of-fold prediction of R and D, random vs structure-grouped validation, applicability (C1, C2);
Fig 5  information carried by feature groups, chemical-class confounding and the 37 A2 rule (C3);
Fig 6  heavy-atom correction, structural terms of the docking scores, partitioning-score association and two OR1A2 poses (C4).
Run: python figures_main_b.py [4 5 6]"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

SCR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCR)
import matplotlib.colors as mcolors                                                 # noqa: E402
import matplotlib.image as mpimg                                                    # noqa: E402
import matplotlib.patches as mpatches                                               # noqa: E402
from matplotlib.lines import Line2D                                                 # noqa: E402
from scipy import stats                                                             # noqa: E402

import core as C                                                                 # noqa: E402
from style import (CLS_COL, CLS_SHORT, EARLY, FRC, GREY, INK, LATE, LGREY, MCOL, MLAB, P2DATA, PAPER, REC,  # noqa: E402
                      REC_COL, SEQ, FS, G, Canvas, fmt, hline, leg, mk, patch, plt, ptitle, save, soft, vline)

sys.path.insert(0, P2DATA)
import dockstyle                                                                    # noqa: E402
import ligdiagram                                                                   # noqa: E402
import poselabels                                                                   # noqa: E402

MOD = pickle.load(open(os.path.join(C.OUT, "models.pkl"), "rb"))
EXT = pickle.load(open(os.path.join(C.OUT, "extra.pkl"), "rb"))
DK = pickle.load(open(os.path.join(C.OUT, "dock.pkl"), "rb"))
R4 = os.path.join(C.CFG.RAW, "docking_renders")
SHORT = {k: CLS_SHORT.get(k, k).replace("萜类/降异戊二烯", "萜类") for k in C.CLS7}


def perf(label, model, col="MAE"):
    p = MOD["perf"]; r = p[(p["方案"] == label) & (p["模型"] == model)].iloc[0]
    return r[f"{col}_mean"], r[f"{col}_sd"]


# ============================================================================================== Fig 4
def fig4():
    m = MOD["meta"]
    cv = Canvas(160, 172)
    # (A)/(B) MAE of three models under the two validation schemes
    for k_, ep in enumerate(["R", "D"]):
        a = cv.panel(G.Cell(53.3 * k_, 0, 53.3, 56), "AB"[k_], L=13, B=10, T=8, R=2)
        for j, sch in enumerate(["random", "group"]):
            for i, mo in enumerate(["baseline", "ridge", "xgb"]):
                mu, sd = perf(f"{ep}|{sch}", mo)
                r2, _ = perf(f"{ep}|{sch}", mo, "R2")
                x = j + (i - 1) * 0.24
                a.errorbar(x, mu, yerr=sd, fmt="o", color=MCOL[mo], ms=4.6 if mo != "baseline" else 4.0,
                           mec="white", mew=0.5, elinewidth=0.9, capsize=1.6, zorder=3)
                a.text(x, mu - sd - (0.006 if ep == "R" else 0.004), fmt(r2), ha="center", va="top",
                       fontsize=FS["small"] - 1.3, color=MCOL[mo] if mo != "baseline" else GREY, rotation=90)
        a.set_xticks([0, 1]); a.set_xticklabels(["随机 5 折", "结构分组 5 折"], fontsize=FS["small"])
        a.set_xlim(-0.55, 1.55)
        if ep == "R":
            a.set_ylim(0.27, 0.36)
        else:
            a.set_ylim(0.10, 0.20)
        a.set_ylabel(f"折外 MAE（{ep}）"); soft(a)
        ptitle(a, "酒心占比 R" if ep == "R" else "馏出重心 D")
        if k_ == 0:
            leg(a, handles=[mk("o", MCOL[mo], MLAB[mo], mec="white") for mo in ["baseline", "ridge", "xgb"]],
                loc="upper left", fontsize=FS["small"] - 0.8, frameon=False, handlelength=0.8, ncol=3, columnspacing=0.6)
            a.text(0.98, 0.03, "竖排数字为折外 R²", transform=a.transAxes, ha="right", va="bottom",
                   fontsize=FS["small"] - 1.2, color=GREY)
    # (C) structural map: classical MDS of Tanimoto distances, Butina clusters
    S = MOD["S"]; Dm = 1 - S
    n = len(Dm); J_ = np.eye(n) - np.ones((n, n)) / n
    Bm = -0.5 * J_ @ (Dm ** 2) @ J_
    ev, evec = np.linalg.eigh(Bm); idx = ev.argsort()[::-1][:2]
    Y = evec[:, idx] * np.sqrt(np.maximum(ev[idx], 0))
    cl = m["butina_main_0.6"].values.astype(int)
    sizes = pd.Series(cl).value_counts()
    top = list(sizes.index[:6])
    CCOL = ["#B35C44", "#3F6E8C", "#D9A441", "#6F9E7A", "#8E7AA8", "#5E9EA0"]
    a = cv.panel(G.Cell(106.6, 0, 53.4, 56), "C", L=12, B=10, T=8, R=2)
    for j, c_ in enumerate(top):
        s_ = cl == c_
        a.scatter(Y[s_, 0], Y[s_, 1], s=14, c=CCOL[j], edgecolors="white", linewidths=0.4, zorder=3,
                  marker="s" if m.loc[s_, "late"].mean() > 0.5 else "o")
    s_ = ~np.isin(cl, top)
    a.scatter(Y[s_, 0], Y[s_, 1], s=10, c=LGREY, edgecolors=GREY, linewidths=0.3, zorder=2)
    a.set_xlabel("MDS 1（Tanimoto 距离）"); a.set_ylabel("MDS 2"); soft(a, grid=None)
    y0_, y1_ = Y[:, 1].min(), Y[:, 1].max(); a.set_ylim(y0_ - 0.05 * (y1_ - y0_), y1_ + 0.75 * (y1_ - y0_))
    a.set_xticks([]); a.set_yticks([])
    ptitle(a, f"Butina 聚类（{len(sizes)} 簇）")
    lab = []
    for j, c_ in enumerate(top):
        mem = m[cl == c_]
        lab.append(patch(CCOL[j], f"{len(mem)} 种：{mem['新类别'].mode().iloc[0].replace('萜类及降异戊二烯类', '萜类')}为主"))
    lab.append(patch(LGREY, f"其余 {int(s_.sum())} 种（{len(sizes) - 6} 簇）"))
    leg(a, handles=lab, loc="upper left", fontsize=FS["small"] - 1.5, frameon=False, handlelength=0.7, labelspacing=0.1,
        ncol=2, columnspacing=0.6)
    # (D)/(E) observed vs out-of-fold prediction (XGBoost, random scheme, mean of 10 repeats)
    for k_, ep in enumerate(["D", "R"]):
        a = cv.panel(G.Cell(53.3 * k_, 56, 53.3, 56), "DE"[k_], L=13, B=10, T=7, R=2)
        oof = MOD[(ep, "random")]["xgb"]["oof"]; pm = oof.mean(0); psd = oof.std(0)
        y = m[ep].values
        a.plot([0, 1], [0, 1], color=GREY, lw=0.6, ls=(0, (3, 2)), zorder=1)
        for c in C.CLS7:
            s_ = (m["新类别"] == c).values
            a.errorbar(y[s_], pm[s_], yerr=psd[s_], fmt="o", ms=3.0, color=CLS_COL[c], mec="white", mew=0.3,
                       elinewidth=0.4, alpha=0.95, zorder=3)
        r2, _ = perf(f"{ep}|random", "xgb", "R2"); mae, _ = perf(f"{ep}|random", "xgb")
        a.text(0.04, 0.96, f"XGBoost，随机 5 折 × 10\nR² = {fmt(r2, 2)}，MAE = {mae:.3f}", transform=a.transAxes,
               ha="left", va="top", fontsize=FS["small"] - 0.9)
        lim = (0, 0.9) if ep == "D" else (-0.03, 1.03)
        a.set_xlim(*lim); a.set_ylim(*lim)
        a.set_xlabel(f"实测 {ep}"); a.set_ylabel(f"折外预测 {ep}"); soft(a, grid=None)
    # (F) absolute error vs nearest-neighbour similarity to the training fold (structure-grouped scheme, D)
    a = cv.panel(G.Cell(106.6, 56, 53.4, 56), "F", L=12, B=10, T=7, R=2)
    ns = MOD["nnsim"]["group"].ravel()
    yD = np.tile(m["D"].values, MOD["nnsim"]["group"].shape[0])
    for mo, mkr in [("baseline", "o"), ("xgb", "o")]:
        e = np.abs(MOD[("D", "group")][mo]["oof"].ravel() - yD)
        bins = np.array([0, 0.3, 0.4, 0.5, 0.6, 1.01])
        idx = np.digitize(ns, bins) - 1
        mu = [e[idx == i].mean() for i in range(len(bins) - 1)]
        se = [e[idx == i].std(ddof=1) / np.sqrt((idx == i).sum()) for i in range(len(bins) - 1)]
        xc = (bins[:-1] + np.minimum(bins[1:], 1.0)) / 2
        if mo == "xgb":
            a.scatter(ns, e, s=3, c=MCOL["xgb"], alpha=0.18, lw=0, zorder=1, rasterized=True)
        a.errorbar(xc, mu, yerr=se, fmt="-o", color=MCOL[mo], ms=3.6, mec="white", mew=0.4, lw=1.1, capsize=1.5, zorder=3,
                   label=MLAB[mo] + "（分箱均值 ± SE）")
    a.set_xlabel("与训练折的最大 Tanimoto 相似度"); a.set_ylabel("|误差|（D）"); a.set_xlim(0.1, 1.0); a.set_ylim(0, 0.6)
    soft(a)
    leg(a, loc="upper right", fontsize=FS["small"] - 1.0, frameon=False, handlelength=1.4)
    med_r = np.median(MOD["nnsim"]["random"]); med_g = np.median(MOD["nnsim"]["group"])
    a.text(0.03, 0.04, f"中位相似度：随机 {med_r:.2f}，分组 {med_g:.2f}", transform=a.transAxes, fontsize=FS["small"] - 1.1,
           color=INK)
    # (G) class leave-out stress test for D
    co = MOD["classout"]; co = co[co["终点"] == "D"]
    a = cv.panel(G.Cell(0, 118, 106.6, 54), "G", L=13, B=10, T=7, R=2)
    xs = np.arange(len(C.CLS7))
    for i, mo in enumerate(["baseline", "ridge", "xgb"]):
        v = [co[(co["留出类别"] == c) & (co["模型"] == mo)]["MAE"].iloc[0] for c in C.CLS7]
        a.bar(xs + (i - 1) * 0.26, v, width=0.24, color=MCOL[mo], edgecolor="white", lw=0.4, zorder=2, label=MLAB[mo])
    nlab = [f"{SHORT[c]}\n（{int(co[co['留出类别'] == c]['n'].iloc[0])}）" for c in C.CLS7]
    a.set_xticks(xs); a.set_xticklabels(nlab, fontsize=FS["small"] - 0.5)
    a.set_ylabel("留出类别的 MAE（D）"); a.set_ylim(0, 0.5); soft(a)
    ptitle(a, "官能团类别留出压力测试：训练集不含该类别")
    leg(a, loc="upper right", fontsize=FS["small"] - 0.8, frameon=False, ncol=3, handlelength=1.0)
    # (H) permutation null of the ridge model (one 5-fold split, 300 permutations)
    a = cv.panel(G.Cell(106.6, 118, 53.4, 54), "H", L=12, B=10, T=7, R=2)
    for ep, col in [("D", MCOL["ridge"]), ("R", MCOL["baseline"])]:
        pr = MOD["perm"][ep]
        rel = pr["null"] / pr["null"].mean()
        a.hist(rel, bins=24, color=col, alpha=0.55, lw=0, zorder=2)
        a.axvline(pr["obs"] / pr["null"].mean(), color=col if ep == "D" else INK, lw=1.2, zorder=3)
        a.text(pr["obs"] / pr["null"].mean() + 0.006, 55 if ep == "D" else 45,
               f"{ep}：P = {pr['p']:.3f}", ha="left", va="center", fontsize=FS["small"] - 0.9, color=col if ep == "D" else INK)
    a.set_xlabel("MAE / 置换零分布均值"); a.set_ylabel("频数"); a.set_ylim(0, 62); a.set_xlim(0.82, 1.12); soft(a)
    ptitle(a, "Ridge 响应置换（300 次）")
    cv.legend([patch(CLS_COL[c], SHORT[c]) for c in C.CLS7], 80, 113.0, ncol=7, anchor="upper center",
              fontsize=FS["small"] - 1.0, handlelength=0.8, columnspacing=0.6, handletextpad=0.3)
    save(cv, "Fig4")


# ============================================================================================== Fig 5
def fig5():
    abl = MOD["abl"]; nb = EXT["nonacid_bp"]
    m = MOD["meta"]
    cv = Canvas(160, 150)
    KEYS = ["类别", "极性", "大小/疏水", "芳香环", "8 项描述符", "8 项 + 类别", "极性 + 类别"]

    def ablpanel(cell, letter, sub, keys, title, df=None):
        a = cv.panel(cell, letter, L=19, B=10, T=8, R=2)
        d = (df if df is not None else abl[(abl["子集"] == sub) & (abl["终点"] == "D")])
        y = np.arange(len(keys))[::-1]
        for yi, kk in zip(y, keys):
            for mo, off, fc in [("ridge", 0.13, MCOL["ridge"]), ("xgb", -0.13, MCOL["xgb"])]:
                r = d[(d["特征组"] == kk) & (d["模型"] == mo)].iloc[0]
                a.plot([0, r["相对基线降低"]], [yi + off] * 2, color=fc, lw=1.0, alpha=0.6, zorder=2)
                a.scatter(r["相对基线降低"], yi + off, s=18, c=fc, edgecolors="white", linewidths=0.4, zorder=3)
        vline(a, 0, color=INK, ls="-", lw=0.6)
        a.set_yticks(y); a.set_yticklabels(keys, fontsize=FS["small"] - 0.6)
        a.set_xlim(-14, 36); a.set_xlabel("MAE 降低（%，相对均值基线）"); soft(a, grid="x")
        ptitle(a, title)
        return a
    a = ablpanel(G.Cell(0, 0, 53.3, 64), "A", "主分析集", KEYS, "主分析集（n = 96）")
    leg(a, handles=[mk("o", MCOL["ridge"], "Ridge", mec="white"), mk("o", MCOL["xgb"], "XGBoost", mec="white")],
        loc="center right", fontsize=FS["small"] - 0.8, frameon=False, handlelength=0.8)
    ablpanel(G.Cell(53.3, 0, 53.3, 64), "B", "去酸类", KEYS, "去除 8 种酸（n = 88）")
    kb = ["沸点", "极性", "8 项描述符", "8 项 + 沸点", "极性 + 沸点"]
    d64 = abl[(abl["子集"] == "有实验沸点") & (abl["终点"] == "D")]
    a = cv.panel(G.Cell(106.6, 0, 53.4, 64), "C", L=19, B=10, T=8, R=2)
    y = np.arange(len(kb))[::-1]
    for (dd, off, ttl, mkr) in [(d64, 0.16, "有沸点（n = 64）", "o"), (nb, -0.16, "去酸且有沸点（n = 57）", "D")]:
        for yi, kk in zip(y, kb):
            for mo, fc in [("ridge", MCOL["ridge"])]:
                r = dd[(dd["特征组"] == kk) & (dd["模型"] == mo)].iloc[0]
                a.plot([0, r["相对基线降低"]], [yi + off] * 2, color=fc, lw=1.0, alpha=0.6, zorder=2)
                a.scatter(r["相对基线降低"], yi + off, s=18, marker=mkr, c=fc if mkr == "o" else "white",
                          edgecolors=fc if mkr == "D" else "white", linewidths=0.8, zorder=3)
    vline(a, 0, color=INK, ls="-", lw=0.6)
    a.set_yticks(y); a.set_yticklabels(kb, fontsize=FS["small"] - 0.6); a.set_xlim(-5, 40)
    a.set_xlabel("MAE 降低（%）"); soft(a, grid="x"); ptitle(a, "沸点的增量信息（Ridge）")
    a.set_ylim(-0.6, len(kb) + 0.4)
    leg(a, handles=[mk("o", MCOL["ridge"], "有沸点（n = 64）", mec="white"),
                    mk("D", "white", "去酸且有沸点（n = 57）", mec=MCOL["ridge"])],
        loc="upper right", fontsize=FS["small"] - 1.1, frameon=False, handlelength=0.8)
    # (D) standardized ridge coefficients for D with bootstrap 95% CI, main vs without acids
    a = cv.panel(G.Cell(0, 64, 53.3, 86), "D", L=14, B=10, T=8, R=2)
    lab = ["MW", "XLogP3", "TPSA", "HBD", "HBA", "RotB", "芳香环", "O/C"]
    y = np.arange(len(lab))[::-1]
    for sname, off, col, mkr in [("主分析集", 0.16, MCOL["ridge"], "o"), ("去酸类", -0.16, "#8E7AA8", "D")]:
        bt = MOD["boot"][(sname, "D")]
        a.errorbar(bt["coef"], y + off, xerr=[bt["coef"] - bt["lo"], bt["hi"] - bt["coef"]], fmt=mkr, color=col, ms=3.4,
                   mec="white", mew=0.4, elinewidth=0.9, capsize=1.2, zorder=3,
                   label=f"{sname}（n = {96 if sname == '主分析集' else 88}，α = {bt['alpha']:.2g}）")
    vline(a, 0, color=INK, ls="-", lw=0.6)
    a.set_yticks(y); a.set_yticklabels(lab, fontsize=FS["small"] - 0.3)
    a.set_xlabel("标准化 Ridge 系数（D）"); soft(a, grid="x"); ptitle(a, "系数与 95% 自助区间（2000 次）")
    a.set_ylim(-1.9, len(lab) - 0.4)
    leg(a, loc="lower right", fontsize=FS["small"] - 1.2, frameon=False, handlelength=0.8)
    # (E) Spearman rho heat map: descriptors x subsets for D
    cor = MOD["cor"]; cor = cor[cor["终点"] == "D"]
    subs = ["主分析集", "去酸类", "酯类", "醇类"]; dsc = ["MW", "XLogP", "TPSA", "HBA", "RotB", "O/C", "bp_exp"]
    Mx = np.full((len(dsc), len(subs)), np.nan); Q = np.full_like(Mx, np.nan)
    for j, s_ in enumerate(subs):
        for i, d_ in enumerate(dsc):
            r = cor[(cor["子集"] == s_) & (cor["描述符"] == d_)]
            if len(r):
                Mx[i, j] = r["rho"].iloc[0]; Q[i, j] = r["q"].iloc[0]
    a = cv.ax(70, 76, 34, 58); cv.letter(53.3, 64, "E")
    from style import DIV
    a.set_facecolor("#EFECE6")
    a.imshow(np.ma.masked_invalid(Mx), cmap=DIV, norm=mcolors.Normalize(-0.6, 0.6), aspect="auto")
    for i in range(len(dsc)):
        for j in range(len(subs)):
            v = Mx[i, j]
            a.text(j, i, "—" if v != v else fmt(v, 2), ha="center", va="center", fontsize=5.0,
                   fontweight="bold" if (Q[i, j] == Q[i, j] and Q[i, j] < 0.05) else "normal",
                   color="white" if v == v and abs(v) > 0.45 else INK)
    a.set_xticks(range(len(subs))); a.set_xticklabels(["主集\n96", "去酸\n88", "酯类\n34", "醇类\n15"], fontsize=FS["small"] - 1.0)
    a.xaxis.tick_top()
    a.set_yticks(range(len(dsc))); a.set_yticklabels(["MW", "XLogP3", "TPSA", "HBA", "RotB", "O/C", "沸点"],
                                                      fontsize=FS["small"] - 0.5); a.tick_params(length=0, pad=1)
    for kk in range(len(dsc) + 1):
        a.axhline(kk - 0.5, color="white", lw=0.8)
    for kk in range(len(subs) + 1):
        a.axvline(kk - 0.5, color="white", lw=0.8)
    for s_ in a.spines.values():
        s_.set_visible(False)
    cv.hcbar(70, 138, 34, plt.cm.ScalarMappable(norm=mcolors.Normalize(-0.6, 0.6), cmap=DIV),
             label="Spearman ρ（与 D；粗体 q < 0.05）", ticks=[-0.6, 0, 0.6])
    # (F) TPSA vs D with the 37 A2 rule
    a = cv.panel(G.Cell(106.6, 64, 53.4, 86), "F", L=12, B=10, T=8, R=2)
    rng = np.random.default_rng(2)
    jit = rng.uniform(-0.9, 0.9, len(m))
    for lab_, col, mkr in [(0, EARLY, "o"), (1, LATE, "s")]:
        s_ = (m["late"] == lab_).values
        acid = (m["新类别"] == "酸类").values
        a.scatter(m["TPSA"][s_ & ~acid] + jit[s_ & ~acid], m["D"][s_ & ~acid], s=12, marker=mkr, facecolors="white",
                  edgecolors=col, linewidths=0.7, zorder=3)
        a.scatter(m["TPSA"][s_ & acid] + jit[s_ & acid], m["D"][s_ & acid], s=14, marker=mkr, c=CLS_COL["酸类"],
                  edgecolors="white", linewidths=0.4, zorder=4)
    a.axvspan(37, 60, color="#EFECE6", lw=0, zorder=0); vline(a, 37, color=INK)
    fx = MOD["thr_fixed"]; tf = MOD["thr_fold"]
    a.text(38.2, 0.06, f"≥ 37 Å²：\n{fx['TP']} 种均\n晚馏出\n（含酸 8 种）", fontsize=FS["small"] - 1.3, va="bottom")
    a.text(-1.5, 1.15, f"37 Å² 规则：\n灵敏度 {fmt(fx['sens'])}，特异度 {fmt(fx['spec'])}\n折内阈值中位 {fmt(tf['thr'].median(), 1)} Å²\n"
                       f"折外灵敏度 {fmt(tf['sens'].mean())} ± {fmt(tf['sens'].std())}",
           fontsize=FS["small"] - 1.3, va="top", linespacing=1.25)
    a.set_xlim(-3, 60); a.set_ylim(0, 1.18)
    a.set_xlabel("TPSA（Å²，横向微移）"); a.set_ylabel("馏出重心 D"); soft(a, grid=None)
    leg(a, handles=[mk("o", EARLY, "早馏出型", mec=EARLY, mfc="white"), mk("s", LATE, "晚馏出型", mec=LATE, mfc="white"),
                    mk("s", CLS_COL["酸类"], "酸类", mec="white")], loc="upper right", fontsize=FS["small"] - 1.1,
        frameon=False, handlelength=0.8)
    save(cv, "Fig5")


# ============================================================================================== Fig 6
def fig6():
    lig = DK["lig"]; st = DK["struct"]; ass = DK["assoc"]
    cv = Canvas(160, 172)
    # (A) Vina score vs heavy atoms
    a = cv.panel(G.Cell(0, 0, 53.3, 58), "A", L=13, B=10, T=8, R=2)
    for r in REC:
        a.scatter(lig["HA"] + (REC.index(r) - 2.5) * 0.07, lig[f"v_{r}"], s=7, c=REC_COL[r], edgecolors="white",
                  linewidths=0.2, alpha=0.9, zorder=3)
        xx = np.array([4.5, 15.5]); sl = lig[f"slope_{r}"].iloc[0]
        ic = np.mean(lig[f"v_{r}"]) - sl * lig["HA"].mean()
        a.plot(xx, ic + sl * xx, color=REC_COL[r], lw=0.7, zorder=2)
    a.set_xlabel("配体重原子数"); a.set_ylabel("Vina 对接评分（kcal/mol）"); soft(a)
    a.text(0.97, 0.96, f"r = {DK['ha_r_all']:.2f}（186 组合）\n受体内 r = {min(DK['ha_r'].values()):.2f} 至 {max(DK['ha_r'].values()):.2f}",
           transform=a.transAxes, ha="right", va="top", fontsize=FS["small"] - 1.0)
    leg(a, handles=[mk("o", REC_COL[r], r, mec="white") for r in REC], loc="lower left", fontsize=FS["small"] - 1.3,
        frameon=False, ncol=2, handlelength=0.6, columnspacing=0.5, labelspacing=0.15)
    # (B) forest: standardized terms of the heavy-atom-corrected score
    a = cv.panel(G.Cell(53.3, 0, 53.3, 58), "B", L=14, B=10, T=8, R=2)
    y = np.arange(len(REC))[::-1]
    for nm, off, col, mkr in [("TPSA", 0.15, "#3F6E8C", "o"), ("芳香环", -0.15, "#B35C44", "D")]:
        s_ = st[st["项"] == nm].set_index("受体").loc[REC]
        a.errorbar(s_["beta"], y + off, xerr=[s_["beta"] - s_["lo"], s_["hi"] - s_["beta"]], fmt=mkr, color=col, ms=3.4,
                   mec="white", mew=0.4, elinewidth=0.9, capsize=1.2, zorder=3, label=f"z({nm})")
    vline(a, 0, color=INK, ls="-", lw=0.6)
    a.set_yticks(y); a.set_yticklabels(REC, fontsize=FS["small"] - 0.3)
    a.set_xlabel("标准化系数（kcal/mol 每 SD）"); soft(a, grid="x"); a.set_ylim(-1.5, len(REC) - 0.4)
    ptitle(a, "校正评分 ~ z(TPSA) + z(芳香环)")
    leg(a, loc="lower right", fontsize=FS["small"] - 0.9, frameon=False, handlelength=0.8)
    # (C) mean corrected score vs TPSA
    a = cv.panel(G.Cell(106.6, 0, 53.4, 58), "C", L=13, B=10, T=8, R=2)
    AC = {0: "#C9C3B8", 1: "#C8664B", 2: "#7A2E3A"}
    for ar in [0, 1, 2]:
        for late, mkr in [(0, "o"), (1, "s")]:
            s_ = (lig["arom"] == ar) & (lig["late"].fillna(0) == late)
            a.scatter(lig.loc[s_, "TPSA"], lig.loc[s_, "c_mean"], s=18, marker=mkr, c=AC[ar], edgecolors="white",
                      linewidths=0.4, zorder=3)
    for nm, off in {"β-大马士酮": (5, -3), "2-甲基萘": (5, -2), "丁二酸二乙酯": (-3, -8), "辛酸异戊酯": (-5, 1),
                    "乙缩醛": (5, 3)}.items():
        r = lig[lig["ligand"] == nm].iloc[0]
        a.annotate(nm, (r["TPSA"], r["c_mean"]), xytext=off, textcoords="offset points", fontsize=FS["small"] - 1.3,
                   ha="left" if off[0] > 0 else "right", color=INK)
    hline(a, 0, color=LGREY, ls="-")
    rho = st[(st["项"] == "TPSA（Spearman）")]["beta"]
    a.text(0.42, 0.04, f"6 受体 ρ = {rho.min():.2f}–{rho.max():.2f}", transform=a.transAxes, ha="left", va="bottom",
           fontsize=FS["small"] - 1.0)
    a.set_xlabel("TPSA（Å²）"); a.set_ylabel("重原子数校正评分（6 受体均值）"); soft(a, grid=None)
    leg(a, handles=[patch(AC[k], f"芳香环 {k}") for k in AC] + [mk("o", GREY, "早馏出型", mec="white"),
                                                              mk("s", GREY, "晚馏出型", mec="white")],
        loc="lower right", fontsize=FS["small"] - 1.3, frameon=False, handlelength=0.7, labelspacing=0.15)
    # (D) corrected score vs D (30 measured ligands)
    mm_ = lig[lig["R"].notna()]
    a = cv.panel(G.Cell(0, 58, 53.3, 58), "D", L=13, B=10, T=8, R=2)
    a.scatter(mm_["D"], mm_["c_mean"], s=18, c=[EARLY if v == 0 else LATE for v in mm_["late"]], edgecolors="white",
              linewidths=0.4, zorder=3)
    r_ = ass[(ass["终点"] == "D") & (ass["受体"] == "mean")].iloc[0]; ci = DK["assoc_ci"]["D"]
    a.text(0.97, 0.04, f"ρ = {fmt(r_['rho'], 2)}\n95% CI {fmt(ci[0], 2)} 至 {fmt(ci[1], 2)}\nn = {len(mm_)}",
           transform=a.transAxes, ha="right", va="bottom", fontsize=FS["small"] - 1.0)
    hline(a, 0, color=LGREY, ls="-")
    a.set_xlabel("馏出重心 D（实测）"); a.set_ylabel("校正评分（6 受体均值）"); soft(a, grid=None)
    leg(a, handles=[mk("o", EARLY, "早馏出型", mec="white"), mk("o", LATE, "晚馏出型", mec="white")], loc="lower left",
        fontsize=FS["small"] - 1.1, frameon=False, handlelength=0.6)
    # (E) per receptor rho with R and D, marginal vs partial (TPSA, aromatic rings, heavy atoms)
    a = cv.panel(G.Cell(53.3, 58, 53.3, 58), "E", L=14, B=10, T=8, R=2)
    y = np.arange(len(REC))[::-1]
    for ep, off, col in [("D", 0.15, "#3F6E8C"), ("R", -0.15, "#D9A441")]:
        s_ = ass[(ass["终点"] == ep) & (ass["受体"] != "mean")].set_index("受体").loc[REC]
        for yi, (_, rr) in zip(y, s_.iterrows()):
            a.plot([rr["rho"], rr["偏rho"]], [yi + off] * 2, color=col, lw=0.7, alpha=0.6, zorder=2)
        a.scatter(s_["rho"], y + off, s=16, c=col, edgecolors="white", linewidths=0.4, zorder=3, label=f"{ep}：边际 ρ")
        a.scatter(s_["偏rho"], y + off, s=16, facecolors="white", edgecolors=col, linewidths=0.8, zorder=3,
                  label=f"{ep}：偏 ρ")
    vline(a, 0, color=INK, ls="-", lw=0.6)
    a.set_yticks(y); a.set_yticklabels(REC, fontsize=FS["small"] - 0.3); a.set_xlim(-0.75, 0.6); a.set_ylim(-1.7, len(REC) - 0.4)
    a.set_xlabel("Spearman ρ（与校正评分）"); soft(a, grid="x"); ptitle(a, "偏相关控制 TPSA、芳香环与重原子数")
    leg(a, loc="lower right", fontsize=FS["small"] - 1.3, frameon=False, handlelength=0.6, ncol=2, columnspacing=0.5)
    # (F) leave-one-ligand-out influence on rho(TPSA, corrected score)
    lo = DK["loo_struct"]
    a = cv.panel(G.Cell(106.6, 58, 53.4, 58), "F", L=14, B=10, T=8, R=2)
    for yi, r in zip(y, REC):
        s_ = lo[lo["受体"] == r]["rho"]
        full = st[(st["项"] == "TPSA（Spearman）") & (st["受体"] == r)]["beta"].iloc[0]
        a.plot([s_.min(), s_.max()], [yi, yi], color=REC_COL[r], lw=3.0, solid_capstyle="round", alpha=0.55, zorder=2)
        a.scatter(full, yi, s=20, c=REC_COL[r], edgecolors="white", linewidths=0.5, zorder=3)
    a.set_yticks(y); a.set_yticklabels(REC, fontsize=FS["small"] - 0.3); a.set_xlim(0.4, 0.85)
    a.set_xlabel("ρ（TPSA，校正评分）"); soft(a, grid="x"); ptitle(a, "逐一剔除配体（31 次）的影响范围")
    # (G)-(J) two representative OR1A2 poses: pocket close-up + 2-D interaction diagram
    dk = pd.read_csv(os.path.join(C.CFG.RAW, "docking", "dock_results.csv"))
    sc = dk[dk.receptor == "OR1A2"].groupby("code")["best"].mean()
    FRm = dockstyle.FRAME
    types = set()
    for k, (code, nm) in enumerate([("L1", "乙酸异戊酯"), ("L5", "β-大马士酮")]):
        x0 = 80.0 * k; y0 = 118.0
        cv.letter(x0, y0, "GH"[k])
        geo = json.load(open(os.path.join(R4, f"OR1A2_{code}_geom.json"), encoding="utf-8"))
        axz = cv.ax(x0 + 1.0, y0 + 6.0, 40, 40, overlap_ok=True)
        axz.imshow(mpimg.imread(os.path.join(R4, f"OR1A2_{code}_zoom.png")), interpolation="lanczos", aspect="auto")
        axz.set_xticks([]); axz.set_yticks([])
        poselabels.place(axz, geo, fontsize=FS["small"] - 1.0)
        ax2 = cv.ax(x0 + 42.5, y0 + 12.0, 36, 28, overlap_ok=True)
        js = json.load(open(os.path.join(R4, f"OR1A2_{code}_2d.json"), encoding="utf-8"))
        types |= set(ligdiagram.draw(ax2, js, fs_res=FS["small"] - 1.0))
        for a_, ls_ in ((axz, (0, (4, 2.5))), (ax2, "-")):
            for sp in a_.spines.values():
                sp.set_color(FRm); sp.set_linewidth(0.7); sp.set_linestyle(ls_)
        s = fmt(sc[code])
        cv.text(x0 + 40, y0 + 2.6, f"OR1A2–{nm}（{s} kcal/mol）", ha="center", va="center", fontsize=FS["small"],
                fontweight="bold")
    order = sorted(types, key=dockstyle.TYPES.index)
    cv.legend(ligdiagram.legend_handles(order), 80, 165.5, ncol=len(order), anchor="upper center",
              fontsize=FS["small"] - 0.8, handlelength=2.4, columnspacing=1.0, handletextpad=0.4)
    save(cv, "Fig6")


if __name__ == "__main__":
    which = sys.argv[1:] or ["4", "5", "6"]
    for k in which:
        print("Fig", k); globals()[f"fig{k}"]()
