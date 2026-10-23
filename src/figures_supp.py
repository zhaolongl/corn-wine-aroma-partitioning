"""JAR merged paper, sixth version: supplementary figures.
New:   S3 differential volatiles of the representative wines (OPLS-DA heat map + three key odorants, redrawn from the
          audited paper-2 data), S7 validation sensitivity, S10 TreeSHAP of the D model, S11 full docking matrix and
          pocket contacts, S12 the two remaining OR1A2 poses, S13 molecular dynamics (current trajectories),
          S14 per-odorant cut curves under the six scenarios.
Reused unchanged from the fifth version (content identical, renumbered): S1 formulation effects, S2 physicochemical
          indices, S4 OPLS-DA scores/permutation, S5 literature formation pathways, S6 distillation profile,
          S8 binary early/late classifiers, S9 elution-type prediction of wine-only compounds, S15 single-factor trials.
Run: python figures_supp.py"""
import json
import os
import pickle
import shutil
import sys

import numpy as np
import pandas as pd

SCR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCR)
import matplotlib.colors as mcolors                                                 # noqa: E402
import matplotlib.image as mpimg                                                    # noqa: E402
import matplotlib.patches as mpatches                                               # noqa: E402
import matplotlib.transforms as mtrans                                              # noqa: E402
from matplotlib.lines import Line2D                                                 # noqa: E402

import core as C                                                                 # noqa: E402
from style import (CLS_COL, DIV, EARLY, FRC, FRUIT, FRUIT_COL, GREY, INK, LATE, LGREY, MCOL, MLAB, OUT, P2DATA,  # noqa: E402
                      REC, REC_COL, SCN_COL, SEQ, SEQB, FS, G, Canvas, fmt, hline, leg, mk, patch, plt, ptitle, save,
                      soft, vline)

sys.path.insert(0, P2DATA)
import dockstyle                                                                    # noqa: E402
import ligdiagram                                                                   # noqa: E402
import poselabels                                                                   # noqa: E402

V5FIG = str(C.CFG.LEGACY)
MDRUN = str(C.CFG.RAW / "md")
R4 = os.path.join(C.CFG.RAW, "docking_renders")
D2 = pickle.load(open(os.path.join(C.CFG.RAW, "cofermentation_paper2_data.pkl"), "rb"))
X2 = pickle.load(open(os.path.join(C.CFG.RAW, "cofermentation_paper2_extra.pkl"), "rb"))
MOD = pickle.load(open(os.path.join(C.OUT, "models.pkl"), "rb"))
EXT = pickle.load(open(os.path.join(C.OUT, "extra.pkl"), "rb"))
DK = pickle.load(open(os.path.join(C.OUT, "dock.pkl"), "rb"))
CUT = pickle.load(open(os.path.join(C.OUT, "cut.pkl"), "rb"))
REN = {"醋酸": "乙酸", "苯乙醇乙酸酯": "乙酸苯乙酯", "琥珀酸单乙酯": "丁二酸单乙酯", "活性2-甲基丁酸乙酯": "2-甲基丁酸乙酯",
       "顺式-乙基肉桂酸酯": "顺式-肉桂酸乙酯", "(S,S)-(+)-2,3-丁二醇": "2,3-丁二醇"}
OLDCLS = {"酯类": CLS_COL["酯类"], "醇类": CLS_COL["醇类"], "酸类": CLS_COL["酸类"], "醛酮类": CLS_COL["醛酮类"],
          "萜烯类": CLS_COL["萜类及降异戊二烯类"], "烷烃类": CLS_COL["烷烃及烯烃类"], "其它类": CLS_COL["其他类"]}
REUSE = {"FigS1": "Fig2", "FigS2": "FigS2", "FigS4": "FigS3", "FigS5": "Fig4", "FigS6": "Fig5", "FigS8": "Fig6",
         "FigS9": "FigS6", "FigS15": "FigS5"}


def reuse():
    os.makedirs(OUT, exist_ok=True)
    for new, old in REUSE.items():
        for ext in ("png", "pdf"):
            shutil.copyfile(os.path.join(V5FIG, f"{old}.{ext}"), os.path.join(OUT, f"{new}.{ext}"))


# ---------------------------------------------------------------------------------------------- S3
def figS3():
    vip = D2["vip"].copy(); vip["化合物"] = vip["化合物"].replace(REN)
    BEST = D2["best"]
    rank = {f: i for i, f in enumerate(FRUIT)}
    vip["_g"] = vip["最高样品"].map(rank); vip = vip.sort_values(["_g", "VIP"], ascending=[True, False]).reset_index(drop=True)
    M0 = np.nan_to_num(vip[FRUIT].values.astype(float))
    Z = (M0 - M0.mean(1, keepdims=True)) / (M0.std(1, ddof=0, keepdims=True) + 1e-9)
    n = len(vip); cv = Canvas(160, 112)
    rowh = 4.4; xh, yh = 34.0, 9.0; H = rowh * n
    cv.letter(0, 1, "A")
    axh = cv.ax(xh, yh, 24, H)
    im = axh.imshow(Z, aspect="auto", cmap=DIV, norm=mcolors.Normalize(-1.5, 1.5), interpolation="nearest")
    axh.set_xticks(range(3)); axh.set_xticklabels(FRUIT, fontsize=FS["small"]); axh.xaxis.tick_top()
    axh.tick_params(length=0, pad=1.5); axh.set_yticks([])
    for i in range(n + 1):
        axh.axhline(i - 0.5, color="white", lw=0.6)
    for j in range(4):
        axh.axvline(j - 0.5, color="white", lw=0.6)
    axs = cv.ax(xh - 3.0, yh, 2.2, H, overlap_ok=True)
    axs.imshow(np.arange(n)[:, None], aspect="auto", interpolation="nearest",
               cmap=mcolors.ListedColormap([OLDCLS.get(c, GREY) for c in vip["类别"]]))
    axs.set_xticks([]); axs.set_yticks([])
    for i, nm in enumerate(vip["化合物"]):
        cv.text(xh - 3.8, yh + rowh * (i + 0.5), nm, ha="right", va="center", fontsize=FS["small"] - 0.6)
    axb = cv.ax(xh + 26.5, yh, 16, H)
    axb.barh(np.arange(n), vip["VIP"].values, height=0.62, color="#9FB3C4", edgecolor="none", zorder=2)
    axb.set_ylim(n - 0.5, -0.5); axb.set_yticks([]); axb.set_xlim(0, 5); axb.set_xticks([0, 2, 4])
    vline(axb, 1, color=INK)
    axb.set_xlabel("VIP", fontsize=FS["small"]); axb.tick_params(axis="x", labelsize=FS["small"])
    for s_ in ("top", "right"):
        axb.spines[s_].set_visible(False)
    k1 = int((vip["_g"] == 0).sum())
    for a_ in (axh, axs, axb):
        a_.axhline(k1 - 0.5, color=INK, lw=0.9, zorder=5)
    cv.hcbar(xh, yh + H + 4, 24, im, label="行标准化 Z 分数", ticks=[-1.5, 0, 1.5])
    present = [c for c in OLDCLS if c in set(vip["类别"])]
    cv.legend([patch(OLDCLS[c], c.replace("其它类", "其他类")) for c in present], 20, yh + H + 3.5, ncol=2,
              anchor="upper center", fontsize=FS["small"] - 1.0, handlelength=0.8, columnspacing=0.6)
    v = X2["vol"].copy()
    ITEMS = [("A5", "乙酸异戊酯（香蕉）"), ("E6", "2,3-丁二酮（奶油）"), ("E15", "β-大马士酮（玫瑰）")]
    for k, (code, ttl) in enumerate(ITEMS):
        ax = cv.panel(G.Cell(80 + (k % 2) * 40, (k // 2) * 56, 40, 56), "BCD"[k], L=12, B=13, R=2, T=8)
        r = v[v["编号"] == code].iloc[0]
        m = [r[s + "_均值"] if r[s + "_均值"] == r[s + "_均值"] else 0.0 for s in FRUIT]
        sd = [r[s + "_SD"] if r[s + "_SD"] == r[s + "_SD"] else 0.0 for s in FRUIT]
        lt = [r[s + "_字母"] if isinstance(r[s + "_字母"], str) else "" for s in FRUIT]
        x = np.arange(3)
        ax.bar(x, m, width=0.6, color=[FRUIT_COL[s] for s in FRUIT], edgecolor="white", lw=0.4, zorder=2)
        ax.errorbar(x, m, yerr=sd, fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.6, capthick=0.6, zorder=3)
        top = max(np.array(m) + np.array(sd)); ax.set_ylim(0, top * 1.34)
        for xi, mi, si, li in zip(x, m, sd, lt):
            ax.text(xi, (mi + si + top * 0.03) if mi else top * 0.03, li if mi else "nd", ha="center", va="bottom",
                    fontsize=FS["small"] - 0.6, fontweight="bold" if mi else "normal", color=INK if mi else GREY)
        below = mtrans.offset_copy(ax.get_xaxis_transform(), fig=cv.fig, y=-10.5, units="points")
        for xi, s, mi in zip(x, FRUIT, m):
            o = r[s + "_OAV"]
            ax.text(xi, 0, "—" if (o != o or mi == 0) else fmt(o), transform=below, ha="center", va="top",
                    fontsize=FS["small"] - 0.6, color="#8A3340", fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(FRUIT, fontsize=FS["small"]); ax.set_xlim(-0.6, 2.6)
        ptitle(ax, ttl, fontsize=FS["small"])
        ax.set_ylabel("含量（μg/L）"); soft(ax)
        vv = r["VIP"]
        ax.text(0.96, 0.96, (f"VIP {vv:.2f}" if vv == vv else "非差异化合物"), transform=ax.transAxes, ha="right", va="top",
                fontsize=FS["small"] - 0.8, color=GREY)
    # (E) concentration ratio of isoamyl acetate to isoamyl alcohol and acetate-ester sums (descriptive)
    ax = cv.panel(G.Cell(120, 56, 40, 56), "E", L=12, B=13, R=2, T=8)
    vol = D2["vol"].copy(); vol["化合物"] = vol["化合物"].replace(REN)
    ace = vol[vol["化合物"].str.startswith("乙酸") & (vol["类别"] == "酯类")]
    s_ = [ace[f + "_均值"].fillna(0).sum() for f in FRUIT]
    ax.bar(np.arange(3), s_, width=0.6, color=[FRUIT_COL[f] for f in FRUIT], edgecolor="white", lw=0.4, zorder=2)
    for xi, vv in enumerate(s_):
        ax.text(xi, vv + max(s_) * 0.03, f"{vv:.0f}", ha="center", va="bottom", fontsize=FS["small"] - 0.8)
    ax.set_xticks(range(3)); ax.set_xticklabels(FRUIT, fontsize=FS["small"]); ax.set_ylim(0, max(s_) * 1.25)
    ax.set_ylabel("乙酸酯总量（μg/L）"); soft(ax); ptitle(ax, f"乙酸酯（{len(ace)} 种）之和", fontsize=FS["small"])
    save(cv, "FigS3")
    return float(s_[0]), [float(x) for x in s_]


# ---------------------------------------------------------------------------------------------- S7
def figS7():
    p = MOD["perf"]; pe = EXT["perf_extra"]
    allp = pd.concat([p, pe], ignore_index=True)
    rows = [("random", "随机 5 折（主）"), ("group", "结构分组 0.6（主）"), ("group0.5", "结构分组 0.5"), ("group0.7", "结构分组 0.7"),
            ("all115_random", "全部 115 种，随机"), ("all115_group", "全部 115 种，分组"), ("S1", "S1 浓度情景，随机"),
            ("multi", "多馏分检出 56 种，随机")]
    cv = Canvas(160, 80)
    for k_, ep in enumerate(["R", "D"]):
        a = cv.panel(G.Cell(80 * k_, 0, 80, 80), "AB"[k_], L=30, B=10, T=8, R=3)
        y = np.arange(len(rows))[::-1]
        for yi, (key, lab) in zip(y, rows):
            if key == "S1":
                lbl = f"{ep}_S1|random"
            elif key == "multi":
                lbl = [s for s in allp["方案"] if s.startswith(f"{ep}|多馏分")][0]
            else:
                lbl = f"{ep}|{key}"
            for i, mo in enumerate(["baseline", "ridge", "xgb"]):
                r = allp[(allp["方案"] == lbl) & (allp["模型"] == mo)].iloc[0]
                a.errorbar(r["R2_mean"], yi + (1 - i) * 0.22, xerr=r["R2_sd"], fmt="o", color=MCOL[mo], ms=3.2, mec="white",
                           mew=0.4, elinewidth=0.8, capsize=1.0, zorder=3)
        vline(a, 0, color=INK, ls="-")
        a.set_yticks(y); a.set_yticklabels([r[1] for r in rows], fontsize=FS["small"] - 0.5)
        a.set_xlabel(f"折外 R²（{ep}，10 次重复均值 ± SD）"); a.set_xlim(-0.35, 0.75); soft(a, grid="x")
        ptitle(a, "酒心占比 R" if ep == "R" else "馏出重心 D")
        if k_ == 0:
            leg(a, handles=[mk("o", MCOL[m_], MLAB[m_], mec="white") for m_ in ["baseline", "ridge", "xgb"]],
                loc="lower right", fontsize=FS["small"] - 0.8, frameon=False, handlelength=0.8)
    save(cv, "FigS7")


# ---------------------------------------------------------------------------------------------- S10
def figS10():
    sh = EXT["shap_D"]; sp = EXT["shap_D_point"]; m = MOD["meta"]
    lab = {"MW": "MW", "XLogP": "XLogP3", "TPSA": "TPSA", "HBD": "HBD", "HBA": "HBA", "RotB": "RotB", "芳香环": "芳香环", "O/C": "O/C"}
    cv = Canvas(160, 60)
    a = cv.panel(G.Cell(0, 0, 40, 60), "A", L=14, B=10, T=8, R=3)
    o = sh.sort_values("mean|SHAP|")
    y = np.arange(len(o))
    a.barh(y, o["mean|SHAP|"], xerr=o["SD"], color="#B86A4B", edgecolor="none", height=0.62, error_kw=dict(lw=0.6, capsize=1.2),
           zorder=2)
    for yi, f in zip(y, o["前3频率"]):
        a.text(0.103, yi, f"{f * 100:.0f}%", va="center", ha="right", fontsize=FS["small"] - 1.2, color=GREY)
    a.set_yticks(y); a.set_yticklabels([lab[d] for d in o["描述符"]], fontsize=FS["small"] - 0.3)
    a.set_xlim(0, 0.105); a.set_xlabel("mean |SHAP|（50 个外层模型）"); soft(a, grid="x"); ptitle(a, "右侧为前 3 位频率")
    for k_, d_ in enumerate(["TPSA", "芳香环", "HBD"]):
        a = cv.panel(G.Cell(40 + 40 * k_, 0, 40, 60), "BCD"[k_], L=12, B=10, T=8, R=2)
        j = C.DESC.index(d_)
        rng = np.random.default_rng(k_)
        xv = m[d_].values + (rng.uniform(-0.6, 0.6, len(m)) if d_ == "TPSA" else rng.uniform(-0.12, 0.12, len(m)))
        a.scatter(xv, sp[:, j], s=10, c=[LATE if v else EARLY for v in m["late"]], edgecolors="white", linewidths=0.3, zorder=3)
        hline(a, 0, color=LGREY, ls="-")
        if d_ == "TPSA":
            vline(a, 37, color=INK)
        a.set_xlabel(lab[d_] + ("（Å²）" if d_ == "TPSA" else "")); a.set_ylabel("SHAP 值（D）"); soft(a, grid=None)
        if k_ == 0:
            leg(a, handles=[mk("o", EARLY, "早馏出型", mec="white"), mk("o", LATE, "晚馏出型", mec="white")], loc="upper left",
                fontsize=FS["small"] - 1.0, frameon=False, handlelength=0.6)
    save(cv, "FigS10")


# ---------------------------------------------------------------------------------------------- S11
def figS11():
    lig = DK["lig"].copy()
    lig = lig.sort_values(["late", "TPSA"]).reset_index(drop=True)
    M = lig[[f"v_{r}" for r in REC]].values
    n = len(lig); rh = 3.6
    cv = Canvas(160, rh * n + 24)
    cv.letter(0, 1, "A")
    x0, y0 = 36.0, 10.0
    a = cv.ax(x0, y0, 6 * 7.2, rh * n)
    norm = mcolors.Normalize(-9.5, -2.5)
    cmap = SEQB.reversed()
    a.imshow(M, cmap=cmap, norm=norm, aspect="auto")
    for i in range(n):
        for j in range(6):
            a.text(j, i, fmt(M[i, j]), ha="center", va="center", fontsize=5.0,
                   color="white" if M[i, j] < -6.6 else INK)
    a.set_xticks(range(6)); a.set_xticklabels(REC, fontsize=FS["small"] - 0.8); a.xaxis.tick_top(); a.tick_params(length=0)
    a.set_yticks(range(n)); a.set_yticklabels(lig["ligand"], fontsize=FS["small"] - 0.9)
    for k in range(n + 1):
        a.axhline(k - 0.5, color="white", lw=0.4)
    for k in range(7):
        a.axvline(k - 0.5, color="white", lw=0.4)
    for s_ in a.spines.values():
        s_.set_visible(False)
    a2 = cv.ax(x0 + 6 * 7.2 + 1.5, y0, 7, rh * n, overlap_ok=True)
    lab = lig["late"].map({0: 0, 1: 1}).fillna(2).values
    a2.imshow(lab[:, None], cmap=mcolors.ListedColormap([EARLY, LATE, "#C9C3B8"]), aspect="auto",
              norm=mcolors.Normalize(-0.5, 2.5))
    for i, t in enumerate(lig["TPSA"]):
        a2.text(0, i, f"{t:.1f}", ha="center", va="center", fontsize=5.0, color="white")
    a2.set_xticks([0]); a2.set_xticklabels(["TPSA"], fontsize=FS["small"] - 1.0); a2.xaxis.tick_top(); a2.set_yticks([])
    a2.tick_params(length=0)
    cv.hcbar(x0, y0 + rh * n + 4, 6 * 7.2, plt.cm.ScalarMappable(norm=norm, cmap=cmap), label="Vina 对接评分（kcal/mol，3 种子均值）",
             ticks=[-9, -7, -5, -3])
    # (B) OR1A2 pocket residues within 4 A of the best pose: share of ligands in contact
    pk = pd.read_csv(os.path.join(C.CFG.RAW, "docking", "pocket_residues.csv"))
    best = pk.loc[pk.groupby(["receptor", "code"])["score"].idxmin()]
    cv.letter(98, 1, "B")
    a = cv.ax(116, y0, 40, rh * n * 0.62)
    cnt = {}
    for rr in best[best.receptor == "OR1A2"]["residues_4A"]:
        for res in str(rr).split():
            cnt[res] = cnt.get(res, 0) + 1
    s = pd.Series(cnt) / best[best.receptor == "OR1A2"].shape[0] * 100
    s = s[s >= 10].sort_values()
    a.barh(np.arange(len(s)), s.values, color="#B35C44", edgecolor="none", height=0.66, zorder=2)
    a.set_yticks(np.arange(len(s))); a.set_yticklabels([x[0] + x[1:3].lower() + x[3:] for x in s.index], fontsize=FS["small"] - 0.9)
    a.set_xlim(0, 105); a.set_xlabel("接触配体占比（%，OR1A2）"); soft(a, grid="x")
    # (C) reproducibility of the three random seeds: SD of the three Vina scores per pair
    sc = DK["sc"]
    cv.letter(98, y0 + rh * n * 0.62 + 10, "C")
    a = cv.ax(116, y0 + rh * n * 0.62 + 16, 40, rh * n * 0.38 - 20)
    for j, r in enumerate(REC):
        v = sc[sc["receptor"] == r]["std"].values
        rng = np.random.default_rng(j)
        a.scatter(j + rng.uniform(-0.18, 0.18, len(v)), v, s=6, c=REC_COL[r], edgecolors="white", linewidths=0.2, zorder=3)
        a.hlines(np.median(v), j - 0.3, j + 0.3, color=INK, lw=0.9, zorder=4)
    a.set_xticks(range(6)); a.set_xticklabels(REC, rotation=45, ha="right", fontsize=FS["small"] - 1.0)
    a.set_ylabel("3 种子评分 SD", fontsize=FS["small"]); a.set_ylim(0, 0.32); soft(a)
    save(cv, "FigS11")


# ---------------------------------------------------------------------------------------------- S12
def figS12():
    dk = pd.read_csv(os.path.join(C.CFG.RAW, "docking", "dock_results.csv"))
    sc = dk[dk.receptor == "OR1A2"].groupby("code")["best"].mean()
    FRm = dockstyle.FRAME; SH = 44.0
    POSE = [("L2", "己酸乙酯"), ("L3", "2,3-丁二酮")]
    cv = Canvas(160, 2 * SH + 8.0)
    types = set()
    for k, (code, nm) in enumerate(POSE):
        y0 = SH * k; yp = y0 + 5.5
        cv.letter(0, y0, "AB"[k])
        s = fmt(sc[code])
        cv.text(80, y0 + 2.4, f"OR1A2–{nm}（{s} kcal/mol）", ha="center", va="center", fontsize=FS["small"] + 0.3,
                fontweight="bold")
        geo = json.load(open(os.path.join(R4, f"OR1A2_{code}_geom.json"), encoding="utf-8")); ov = geo["overview"]
        XO, WO_, XZ, WZ_, X2_, W2_, HP = 0.5, 34.0, 39.5, 60.0, 101.5, 58.0, 36.0
        axo = cv.ax(XO, yp, WO_, HP, overlap_ok=True)
        axo.imshow(mpimg.imread(os.path.join(R4, f"OR1A2_{code}_overview.png")), interpolation="lanczos", aspect="auto")
        axo.set_xticks([]); axo.set_yticks([])
        win = np.array(ov["window"]); x_a, y_a = win[:, 0].min(), win[:, 1].min(); x_b, y_b = win[:, 0].max(), win[:, 1].max()
        axo.add_patch(mpatches.Rectangle((x_a, y_a), x_b - x_a, y_b - y_a, fill=False, ec=FRm, lw=0.75, ls=(0, (3, 2)),
                                         zorder=5))
        axo.text(0.04, 0.965, "胞外侧", transform=axo.transAxes, ha="left", va="top", fontsize=FS["small"] - 0.8, color=GREY)
        axo.text(0.04, 0.035, "胞内侧", transform=axo.transAxes, ha="left", va="bottom", fontsize=FS["small"] - 0.8, color=GREY)
        axz = cv.ax(XZ, yp, WZ_, HP, overlap_ok=True)
        axz.imshow(mpimg.imread(os.path.join(R4, f"OR1A2_{code}_zoom.png")), interpolation="lanczos", aspect="auto")
        axz.set_xticks([]); axz.set_yticks([])
        poselabels.place(axz, geo, fontsize=FS["small"] - 0.6)
        ax2 = cv.ax(X2_, yp, W2_, HP, overlap_ok=True)
        js = json.load(open(os.path.join(R4, f"OR1A2_{code}_2d.json"), encoding="utf-8"))
        types |= set(ligdiagram.draw(ax2, js, fs_res=FS["small"] - 0.6))
        for a_, ls_ in ((axo, "-"), (axz, (0, (4, 2.5))), (ax2, "-")):
            for sp in a_.spines.values():
                sp.set_color(FRm); sp.set_linewidth(0.8); sp.set_linestyle(ls_)

        def mm(px, py):
            return XO + (px + 0.5) / ov["W"] * WO_, yp + (py + 0.5) / ov["H"] * HP
        for (px, py), (qx, qy) in (((x_b, y_a), (XZ, yp)), ((x_b, y_b), (XZ, yp + HP))):
            ax_, ay_ = mm(px, py)
            cv.fig.add_artist(Line2D([ax_ / cv.W, qx / cv.W], [1 - ay_ / cv.H, 1 - qy / cv.H], transform=cv.fig.transFigure,
                                     color=FRm, lw=0.7, ls=(0, (3, 2))))
    order = sorted(types, key=dockstyle.TYPES.index)
    cv.legend(ligdiagram.legend_handles(order), 80, 2 * SH + 2.0, ncol=len(order), anchor="upper center",
              fontsize=FS["small"] - 0.5, handlelength=2.6, columnspacing=1.2, handletextpad=0.4)
    save(cv, "FigS12")


# ---------------------------------------------------------------------------------------------- S13
def figS13():
    z = np.load(os.path.join(MDRUN, "md_series.npz"), allow_pickle=True)
    sm = pd.read_csv(os.path.join(MDRUN, "md_summary.csv")); ct = pd.read_csv(os.path.join(MDRUN, "md_contacts.csv"))
    off = {t: json.load(open(os.path.join(MDRUN, t, "build.json")))["first"] - 1 for t in sm["complex"]}
    ns = float(sm["ns"].min())
    POSE = [("L1", "乙酸异戊酯"), ("L2", "己酸乙酯"), ("L3", "2,3-丁二酮"), ("L5", "β-大马士酮")]
    MDCOL = {"L1": "#B35C44", "L2": "#D9A441", "L3": "#6F9E7A", "L5": "#3F6E8C"}
    smooth = lambda v, w=25: pd.Series(v).rolling(w, center=True, min_periods=1).mean().values
    cv = Canvas(160, 118)
    CC = G.cells(0, 1, [53.3, 53.3, 53.4], [55, 57])
    PAN = [("rmsd_bb", "受体主链 RMSD（Å）", (0, 4.4), "time"), ("rmsd_lig", "配体 RMSD（Å）", (0, 8.5), "time"),
           ("nhb", "受体–配体氢键数", (0, 2.2), "time"), ("sasa", "受体 SASA（Å²）", None, "time"),
           ("rmsf", "残基 Cα RMSF（Å）", None, "res")]
    for k, (key, ylab, lim, kind) in enumerate(PAN):
        r, c = divmod(k, 3)
        a = cv.panel(CC[r][c], "ABCDE"[k], L=15, B=11, R=3, T=(10 if k == 0 else 6))
        for code, nm in POSE:
            t = f"OR1A2_{code}"
            if kind == "time":
                a.plot(z[f"{t}__time"], smooth(z[f"{t}__{key}"]), color=MDCOL[code], lw=0.8, zorder=3)
            else:
                res = np.asarray(z[f"{t}__resnum"], float) + off[t]
                a.plot(res, z[f"{t}__rmsf"], color=MDCOL[code], lw=0.7, zorder=3)
        if kind == "time":
            a.set_xlim(0, ns); a.set_xlabel("模拟时间（ns）"); a.set_xticks(np.arange(0, ns + 1, 20))
        else:
            a.set_xlabel("残基编号（UniProt）"); a.set_xlim(9, 308)
        if lim:
            a.set_ylim(*lim)
        a.set_ylabel(ylab, fontsize=FS["small"] + 0.3); soft(a)
    cv.legend([Line2D([], [], color=MDCOL[c], lw=1.4, label=n) for c, n in POSE], 80, 1.0, ncol=4, anchor="upper center",
              fontsize=FS["small"], handlelength=1.6, columnspacing=1.4)
    a = cv.panel(CC[1][2], "F", L=15, B=14, R=3, T=6)
    ct["UniProt"] = [f"{r.residue[:3]}{r.resSeq + off[r.complex]}" for r in ct.itertuples()]
    piv = ct.pivot_table(index="UniProt", columns="complex", values="occupancy", fill_value=0.0)
    piv = piv.reindex(columns=[f"OR1A2_{c}" for c, _ in POSE])
    keep = sorted(piv.index[piv.max(1) >= 0.5], key=lambda s: int("".join(ch for ch in s if ch.isdigit())))
    piv = piv.loc[keep]; xs = np.arange(len(keep)); w = 0.2
    for i, (code, nm) in enumerate(POSE):
        a.bar(xs + (i - 1.5) * w, piv[f"OR1A2_{code}"].values * 100, width=w, color=MDCOL[code], edgecolor="none", zorder=2)
    a.set_xticks(xs); a.set_xticklabels([s[0] + s[1:3].lower() + s[3:] for s in keep], rotation=90, fontsize=5.0)
    a.set_ylim(0, 105); a.set_ylabel("接触帧占比（%）"); a.set_xlim(-0.6, len(keep) - 0.4); soft(a)
    cv.text(80, 116.5, f"各复合物生产模拟已完成 {ns:.0f} ns（计划 100 ns，模拟仍在进行）", ha="center", va="bottom",
            fontsize=FS["small"] - 0.6, color=GREY)
    save(cv, "FigS13")
    return sm, ns


# ---------------------------------------------------------------------------------------------- S14
def figS14():
    cur = CUT["curves"]; odd = CUT["odor"].sort_values("uni_14", ascending=False).reset_index(drop=True)
    et = CUT["ethanol"]
    n = len(odd); ncol = 5; nrow = int(np.ceil(n / ncol))
    cv = Canvas(160, 26 * nrow + 10)
    for i, r in odd.iterrows():
        rr, cc = divmod(i, ncol)
        a = cv.panel(G.Cell(32 * cc, 26 * rr, 32, 26), None, L=8, B=(6 if rr == nrow - 1 else 2.5), T=4.5, R=1.5)
        for kn, col in SCN_COL.items():
            for basis, ls in [("S0", "-"), ("S1", (0, (3, 1.5)))]:
                s_ = cur[(cur["CID"] == r["CID"]) & (cur["浓度口径"] == basis) & (cur["头内分布"] == kn)]
                a.plot(s_["h"], s_["酒心占比"] * 100, color=col, lw=0.8, ls=ls, zorder=3)
        a.plot(et["h"], et["乙醇回收率"], color=LGREY, lw=0.8, zorder=2)
        a.set_xlim(0, 50); a.set_ylim(0, 105); a.set_xticks([0, 14, 50]); a.set_yticks([0, 50, 100])
        a.tick_params(labelsize=FS["small"] - 1.5, length=1.5, pad=1)
        if cc:
            a.set_yticklabels([])
        if rr < nrow - 1:
            a.set_xticklabels([])
        a.set_title(r["采用名称"], fontsize=FS["small"] - 1.0, pad=1.5, fontweight="bold")
        soft(a, grid=None)
    cv.legend([Line2D([], [], color=c, lw=1.2, label=f"头内{k}") for k, c in SCN_COL.items()] +
              [Line2D([], [], color=GREY, lw=1.0, label="S0"), Line2D([], [], color=GREY, lw=1.0, ls=(0, (3, 1.5)), label="S1"),
               Line2D([], [], color=LGREY, lw=1.0, label="乙醇")], 80, 26 * nrow + 2, ncol=6, anchor="upper center",
              fontsize=FS["small"] - 0.8, handlelength=1.6)
    cv.text(1, 26 * nrow + 8.5, "横轴：掐头体积 h（mL）；纵轴：进入酒心的比例（%）", fontsize=FS["small"] - 1.0, color=GREY)
    save(cv, "FigS14")


if __name__ == "__main__":
    reuse()
    print("S3", figS3())
    figS7(); figS10(); figS11(); figS12()
    print("S13", figS13()[1])
    figS14()
