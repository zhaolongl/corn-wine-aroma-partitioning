"""JAR merged paper, sixth version: main figures 1, 2, 3 and 7 (data that do not depend on the new models).
Fig 1  study design, evidence units and the compound basis shared by the two processing branches of the same raw corn;
Fig 2  representative co-fermented wines and the robustness of the candidate odorant list (C6);
Fig 3  five-fraction distributions and the two continuous endpoints R (heart share) and D (distillation centroid) (C0);
Fig 7  cut responses under identifiable whole-fraction combinations and labelled within-head / concentration scenarios (C5).
Run: python figures_main_a.py [1 2 3 7]"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

SCR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCR)
import matplotlib.colors as mcolors                                                 # noqa: E402
import matplotlib.patches as mpatches                                               # noqa: E402
import matplotlib.transforms as mtrans                                              # noqa: E402
from scipy import stats                                                             # noqa: E402

import data_merged as J                                                                # noqa: E402
import core as C                                                                 # noqa: E402
from style import (CLS_COL, CLS_SHORT, DIV, EARLY, FRC, FRS, FRUIT, FRUIT_COL, FRUIT_MK, GREY, HEART, INK, LATE,  # noqa: E402
                      LGREY, PAPER, SCN_COL, SEQ, SEQB, FS, G, Canvas, fmt, frame, hline, leg, mk, patch, plt, ptitle,
                      save, soft, vline)

P2DATA = str(C.CFG.HELPERS)
sys.path.insert(0, P2DATA)
from classbars import class_totals                                                  # noqa: E402

D2 = pickle.load(open(os.path.join(C.CFG.RAW, "cofermentation_paper2_data.pkl"), "rb"))
X2 = pickle.load(open(os.path.join(C.CFG.RAW, "cofermentation_paper2_extra.pkl"), "rb"))
REN = {"醋酸": "乙酸", "苯乙醇乙酸酯": "乙酸苯乙酯", "1,2,3,4-四氢-1,1,6-三甲基萘": "1,1,6-三甲基-1,2,3,4-四氢萘"}
BEST = D2["best"]
RR = lambda r: r.replace(":", "∶")


def box(ax, x, y, w, h, fc, ec, lines, title=None, tcol=None, fs=FS["small"], lw=0.8, r=1.2, tfs=None, gap=3.1):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                         zorder=2))
    tot = (gap + 0.9 if title else 0) + gap * len(lines)
    yy = y + (h - tot) / 2 + 0.6
    if title:
        ax.text(x + w / 2, yy, title, ha="center", va="top", fontsize=tfs or fs + 0.6, fontweight="bold",
                color=tcol or ec, zorder=3)
        yy += gap + 0.9
    for ln in lines:
        ax.text(x + w / 2, yy, ln, ha="center", va="top", fontsize=fs, color=INK, zorder=3)
        yy += gap


def arrow(ax, x1, y1, x2, y2, col=GREY, lw=0.9, rad=0.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=1,
                arrowprops=dict(arrowstyle="-|>", color=col, lw=lw, mutation_scale=7,
                                connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


# ============================================================================================== Fig 1
def fig1():
    w, b = J.load(); sh = J.shares(w, b)
    cv = Canvas(160, 156)
    # (A) design schematic, drawn in mm on an axes covering the top band
    ax = cv.ax(0, 0, 160, 80); ax.set_xlim(0, 160); ax.set_ylim(80, -3); ax.axis("off")
    cv.letter(0, 1, "A")
    box(ax, 3, 3, 34, 38, "#F4EEE4", "#8A6D3B", ["津黑糯 529（黑糯玉米）", "2024-10-10 采收", "剥粒后 −18 ℃ 冻藏",
                                                 "浸泡 → 蒸制糊化", "→ 打浆 → 糖化（玉米醪）"],
        title="同一批鲜食玉米原料", fs=FS["small"] - 0.2, gap=3.6)
    box(ax, 3, 45, 34, 32, "#FAFAF8", GREY, ["15 酒样 × 3 平行（理化、感官）", "3 代表酒样 × 3 平行（挥发物）",
                                             "1 批二次蒸馏（各馏分合并样）", "化合物 115 种（主分析集 96 种）",
                                             "31 配体 × 6 受体 × 3 种子"],
        title="独立单位与样本量", tcol=INK, fs=FS["small"] - 0.8, gap=3.4)
    box(ax, 43, 3, 56, 36, "#F7ECE8", "#A23B3B",
        ["玉米醪 + 山楂/梨/石榴汁（体积比 9∶1–5∶5）", "初始糖度 190 g/L，F15 酵母，20 ℃ 发酵",
         "15 种共发酵酒：理化 9 项、QDA 10 项", "3 种代表酒样：HS-SPME-GC-MS，150 种挥发物",
         "OAV ≥ 1 的候选物 7 种"],
        title="分支一  共发酵（原料调控）", fs=FS["small"] - 0.3, gap=3.5)
    box(ax, 43, 43, 56, 34, "#E9EFF4", "#3F5A78",
        ["玉米醪单独发酵 → 发酵酒 11.2% vol", "一次蒸馏 30.2% vol → 二次蒸馏 14 段 × 50 mL",
         "按酒精度截取头/中间/次/次尾/尾 5 个馏分", "HS-SPME-GC-MS，115 种挥发物",
         "OAV ≥ 1 的候选物 25 种"],
        title="分支二  发酵—两次釜式蒸馏（物理分离）", fs=FS["small"] - 0.3, gap=3.5)
    arrow(ax, 37, 18, 43, 18); arrow(ax, 37, 33, 43, 58, rad=-0.15)
    box(ax, 105, 22, 23, 36, "#F1F0EC", INK, ["PubChem CID", "对齐", "共有 70 种", "以化合物为", "分析单位"],
        title="化合物对齐", tcol=INK, fs=FS["small"] - 0.3, gap=3.6)
    arrow(ax, 99, 21, 105, 33); arrow(ax, 99, 60, 105, 48)
    mods = [("3.1", "候选物与阈值稳健性", "#A23B3B"), ("3.2", "连续端点 R 与 D", "#7A2E3A"),
            ("3.3", "预测与结构泛化", "#3F6E8C"), ("3.4", "特征消融与类别混杂", "#3F6E8C"),
            ("3.5", "对接评分的结构关联", "#6F9E7A"), ("3.6", "截取情景与不确定性", "#B86A4B")]
    for i, (sec, txt, col) in enumerate(mods):
        y0 = 3 + i * 12.4
        ax.add_patch(mpatches.FancyBboxPatch((134, y0), 24, 10.4, boxstyle="round,pad=0,rounding_size=1.0",
                                             fc="white", ec=col, lw=0.8, zorder=2))
        ax.add_patch(mpatches.Rectangle((134, y0), 2.2, 10.4, fc=col, ec="none", zorder=3))
        ax.text(137.5, y0 + 2.6, sec, fontsize=FS["small"], fontweight="bold", color=col, va="top")
        ax.text(137.5, y0 + 6.4, txt, fontsize=FS["small"] - 0.6, color=INK, va="top")
        arrow(ax, 128, 40, 134, y0 + 5.2, col=LGREY, lw=0.6)
    # (B) class composition of the shared and unique sets
    cls = ["酯类", "醇类", "醛酮类", "酸类", "萜类及降异戊二烯类", "缩醛类", "呋喃类", "芳香烃类", "烷烃及烯烃类", "其它类"]
    cnt = pd.DataFrame({"仅共发酵酒": w[~w["共有"]].groupby("类别_统一").size(), "共有": w[w["共有"]].groupby("类别_统一").size(),
                        "仅馏出液": b[~b["共有"]].groupby("新类别").size()}).reindex(cls).fillna(0).astype(int)
    a = cv.panel(G.Cell(0, 84, 56, 72), "B", L=22, B=10, T=7, R=2)
    y = np.arange(len(cls))[::-1]
    SC = {"共有": "#5F7F73", "仅共发酵酒": "#D7A99A", "仅馏出液": "#A9BCCD"}
    left = np.zeros(len(cls))
    for s in ["共有", "仅共发酵酒", "仅馏出液"]:
        a.barh(y, cnt[s].values, left=left, height=0.7, color=SC[s], edgecolor="white", lw=0.5, label=s, zorder=2)
        for yi, l0, v in zip(y, left, cnt[s].values):
            if v >= 3:
                a.text(l0 + v / 2, yi, str(v), ha="center", va="center", fontsize=FS["small"] - 0.8,
                       color="white" if s == "共有" else INK)
        left += cnt[s].values
    a.set_yticks(y); a.set_yticklabels([CLS_SHORT.get(c, c).replace("其它类", "其他类") for c in cls], fontsize=FS["small"])
    a.set_xlim(0, 72); a.set_xlabel("化合物数"); soft(a, grid="x")
    leg(a, loc="lower right", handlelength=1.0, fontsize=FS["small"] - 0.3, frameon=False)
    # (C) share of volatile content / mass carried by the 70 shared compounds (single derived values -> points)
    a = cv.panel(G.Cell(56, 84, 50, 72), "C", L=12, B=14, T=7, R=2)
    labs = ["山楂", "梨", "石榴"] + ["头", "中间", "次", "次尾", "尾"]
    vals = sh.values[:8]; col = [FRUIT_COL[f] for f in FRUIT] + FRC
    xs = np.r_[0, 1, 2, 3.4 + np.arange(5)]
    a.vlines(xs, 80, vals, color=col, lw=1.3, zorder=2)
    a.scatter(xs, vals, s=30, c=col, edgecolors="white", linewidths=0.6, zorder=3)
    for x, v in zip(xs, vals):
        a.text(x, v + 0.8, fmt(v, 1), ha="center", va="bottom", fontsize=FS["small"] - 1.0, color=INK, rotation=90)
    a.hlines(sh["馏出液合计"], 2.95, xs[-1] + 0.6, color=FRC[4], lw=0.7, ls=(0, (3, 2)), zorder=1)

    vline(a, 2.7, color=LGREY, ls="-")
    a.set_xticks(xs); a.set_xticklabels(labs, fontsize=FS["small"] - 0.3)
    a.set_xlim(-0.6, xs[-1] + 0.6); a.set_ylim(80, 102); a.set_ylabel("共有化合物占比（%）")
    tr = mtrans.blended_transform_factory(a.transData, a.transAxes)
    a.text(1.0, -0.13, "共发酵酒\n（含量）", transform=tr, ha="center", va="top", fontsize=FS["small"] - 0.5, fontweight="bold")
    a.text(5.4, -0.13, "二次蒸馏馏分\n（质量）", transform=tr, ha="center", va="top", fontsize=FS["small"] - 0.5, fontweight="bold")
    a.text(0.98, 0.02, f"虚线：馏出液合计 {fmt(sh['馏出液合计'], 1)}%", transform=a.transAxes, ha="right", va="bottom",
           fontsize=FS["small"] - 1.2, color=FRC[4], bbox=dict(fc="white", ec="none", pad=0.3))
    soft(a)
    # (D) analysis sets and their sizes (n) -- measured, derived or computed units
    sets = [("共发酵酒挥发物", 150, "实测"), ("馏出液挥发物", 115, "实测"), ("CID 共有化合物", 70, "实测"),
            ("主分析集（去烃类）", 96, "派生"), ("有实验沸点子集", 64, "派生"), ("去酸类子集", 88, "派生"),
            ("候选气味活性物质（并集）", 26, "派生"), ("对接配体", 31, "计算"), ("其中有实测分配", 30, "计算"),
            ("对接评分记录", 558, "计算")]
    a = cv.panel(G.Cell(106, 84, 54, 72), "D", L=27, B=10, T=7, R=3)
    TC = {"实测": "#5F7F73", "派生": "#C8A15A", "计算": "#7F93AD"}
    yy = np.arange(len(sets))[::-1]
    vals = [s[1] for s in sets]
    a.barh(yy, vals, height=0.66, color=[TC[s[2]] for s in sets], edgecolor="none", zorder=2)
    for yi, v in zip(yy, vals):
        a.text(v * 1.12, yi, str(v), va="center", fontsize=FS["small"] - 0.8, color=INK)
    a.set_xscale("log"); a.set_xlim(10, 3000); a.set_xticks([10, 100, 1000]); a.set_xticklabels(["10", "100", "1000"]); a.minorticks_off()
    a.set_yticks(yy); a.set_yticklabels([s[0] for s in sets], fontsize=FS["small"] - 0.5)
    a.set_xlabel("单位数 n（对数轴）"); soft(a, grid="x")
    leg(a, handles=[patch(TC[k], k) for k in TC], loc="center right", fontsize=FS["small"] - 0.5, frameon=False,
        handlelength=1.0)
    save(cv, "Fig1")


# ============================================================================================== Fig 2
def fig2():
    vol = D2["vol"].copy()
    vol["化合物"] = vol["化合物"].replace(REN)
    w, b = J.load(); k = J.key_table(w, b); k["名称"] = k["名称"].replace(REN)
    cut = pickle.load(open(os.path.join(C.OUT, "cut.pkl"), "rb"))
    cv = Canvas(160, 184)
    # (A) volatile class totals of the three representative wines (mean +- RSS SD, Duncan letters)
    keys = ["总量", "醇类", "酯类", "酸类", "醛酮类", "萜烯类", "其它类"]
    ct = class_totals(vol, FRUIT, [c for c in keys if c != "总量"])
    a = cv.panel(G.Cell(0, 0, 84, 60), "A", L=14, B=10, T=8, R=2)
    x = 0.0; centres = []; top = 0
    scale = {"总量": 1.0}
    for key in keys:
        m = np.asarray(ct[key]["均值"], float) / 1000; sd = np.asarray(ct[key]["SD"], float) / 1000
        xs = x + np.arange(3) * 0.9
        a.bar(xs, m, width=0.78, color=[FRUIT_COL[f] for f in FRUIT], edgecolor="white", lw=0.4, zorder=2)
        a.errorbar(xs, m, yerr=sd, fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.4, capthick=0.6, zorder=3)
        for xx, mm, ss, l in zip(xs, m, sd, ct[key]["字母"]):
            a.text(xx, mm + ss + 0.12, l, ha="center", va="bottom", fontsize=FS["small"] - 0.8, fontweight="bold")
        centres.append(x + 0.9); x += 3.4; top = max(top, (m + sd).max())
    a.set_yscale("symlog", linthresh=0.5, linscale=0.6)
    a.set_ylim(0, top * 2.2)
    a.set_yticks([0, 0.5, 1, 2, 5, 10]); a.set_yticklabels(["0", "0.5", "1", "2", "5", "10"])
    a.set_xticks(centres); a.set_xticklabels([kk.replace("其它类", "其他类") for kk in keys], fontsize=FS["small"], fontweight="bold")
    a.tick_params(axis="x", length=0); a.set_ylabel("含量（mg/L，对称对数轴）"); soft(a)
    for i in range(1, len(keys)):
        a.axvline(i * 3.4 - 0.8, color=LGREY, lw=0.6, ls=(0, (3, 2)))
    leg(a, handles=[patch(FRUIT_COL[f], f"{f} {RR(BEST[f])}") for f in FRUIT], loc="upper right", ncol=3,
        fontsize=FS["small"] - 0.5, frameon=False, handlelength=1.0, columnspacing=0.9)
    # (B) PCA scores of the volatile profiles
    a = cv.panel(G.Cell(84, 0, 38, 60), "B", L=12, B=10, T=8, R=2)
    from scipy.spatial import ConvexHull
    for fr in FRUIT:
        P = np.asarray(D2["pca"][fr], float)
        h = ConvexHull(P); pts = P[h.vertices]
        a.fill(pts[:, 0], pts[:, 1], color=FRUIT_COL[fr], alpha=0.16, lw=0, zorder=1)
        a.scatter(P[:, 0], P[:, 1], s=20, marker=FRUIT_MK[fr], c=FRUIT_COL[fr], edgecolors="white", linewidths=0.5, zorder=3)
    hline(a, 0, color=LGREY, ls="-"); vline(a, 0, color=LGREY, ls="-")
    a.set_xlim(-14, 14); a.set_ylim(-11, 16)
    a.set_xlabel(f"PC1（{D2['pca_var'][0]}%）"); a.set_ylabel(f"PC2（{D2['pca_var'][1]}%）"); soft(a, grid=None)
    # (C) aroma-series OAV sums (descriptive, from means)
    SERIES = ["果香", "花香", "甜香", "青草/蘑菇", "辛香/其他"]
    AS = X2["aroma_series"].loc[SERIES, FRUIT].astype(float).values
    a = cv.ax(133, 8, 25, 40); cv.letter(122, 0, "C")
    norm = mcolors.LogNorm(0.1, 100)
    a.imshow(np.ma.masked_less_equal(AS, 0), cmap=SEQ, norm=norm, aspect="auto")
    a.set_facecolor(PAPER)
    for i in range(AS.shape[0]):
        for j in range(3):
            v = AS[i, j]
            a.text(j, i, fmt(v, 1) if v > 0 else "—", ha="center", va="center", fontsize=FS["small"] - 0.9,
                   color="white" if v >= 20 else INK)
    a.set_xticks(range(3)); a.set_xticklabels(FRUIT, fontsize=FS["small"] - 0.3); a.xaxis.tick_top()
    a.set_yticks(range(5)); a.set_yticklabels(SERIES, fontsize=FS["small"] - 0.3); a.tick_params(length=0, pad=1.2)
    for kk in range(6):
        a.axhline(kk - 0.5, color="white", lw=1.0)
    for kk in range(4):
        a.axvline(kk - 0.5, color="white", lw=1.0)
    for s_ in a.spines.values():
        s_.set_visible(False)
    cv.hcbar(133, 52.5, 25, plt.cm.ScalarMappable(norm=norm, cmap=SEQ), label="香气系列 OAV 之和",
             ticks=[0.1, 1, 10, 100], ticklabels=["0.1", "1", "10", "100"])
    # (D) OAV map of the union candidates: 3 wines + 5 fractions, with stability grade columns
    oa = cut["oav"]
    g_w = oa[oa["体系"] == "共发酵酒"].set_index("CID")["等级"]; g_d = oa[(oa["体系"] == "馏出液") & (oa["浓度口径"] == "S0")].set_index("CID")["等级"]
    g_d1 = oa[(oa["体系"] == "馏出液") & (oa["浓度口径"] == "S1")].set_index("CID")["等级"]
    cols = [f"{s}_OAV" for s in J.WINES] + [f"{s}_OAV" for s in J.FRS]
    M = k[cols].astype(float).values
    M = np.where(M > 0, M, np.nan)
    n = len(k); rh = 3.55; x0, y0, cw = 30.0, 75.0, 6.2
    a = cv.ax(x0, y0, cw * 8, rh * n); cv.letter(0, 64, "D")
    a.set_facecolor("#EFECE6")
    norm = mcolors.LogNorm(0.03, 400)
    a.imshow(np.ma.masked_invalid(M), cmap=SEQ, norm=norm, aspect="auto", interpolation="nearest")
    for i in range(n):
        for j in range(8):
            v = M[i, j]
            if v == v:
                a.text(j, i, (f"{v:.0f}" if v >= 100 else f"{v:.1f}" if v >= 1 else f"{v:.2f}"), ha="center", va="center",
                       fontsize=5.0, color="white" if v >= 30 else INK, fontweight="bold" if v >= 1 else "normal")
    for kk in range(n + 1):
        a.axhline(kk - 0.5, color="white", lw=0.5)
    for kk in range(9):
        a.axvline(kk - 0.5, color="white", lw=0.5)
    a.axvline(2.5, color=INK, lw=0.8)
    grp = k["组别"].tolist(); cutr = [i for i in range(1, n) if grp[i] != grp[i - 1]]
    for c_ in cutr:
        a.axhline(c_ - 0.5, color=INK, lw=0.8, ls=(0, (3, 2)))
    a.set_xticks(range(8)); a.set_xticklabels(["山楂", "梨", "石榴"] + FRS, fontsize=FS["small"] - 0.5)
    a.xaxis.tick_top(); a.tick_params(length=0, pad=1.0)
    a.set_yticks(range(n)); a.set_yticklabels(k["名称"], fontsize=FS["small"] - 0.6); a.tick_params(axis="y", pad=1.0)
    for s_ in a.spines.values():
        s_.set_visible(False)
    cv.text(x0 + cw * 1.5, y0 - 5.6, "共发酵酒", ha="center", fontsize=FS["small"] - 0.3, fontweight="bold")
    cv.text(x0 + cw * 5.5, y0 - 5.6, "馏分（S0）", ha="center", fontsize=FS["small"] - 0.3, fontweight="bold")
    # grade strips (wine / distillate S0 / distillate S1)
    GC = {"稳定": "#5F7F73", "阈值敏感": "#D6A84F", "临界": "#E9DDC4", "非候选": "#F5F3EF", "未检出": "white"}
    ag = cv.ax(x0 + cw * 8 + 1.8, y0, 3 * 3.4, rh * n, overlap_ok=True)
    for i, cid in enumerate(k["CID"]):
        for j, src in enumerate([g_w, g_d, g_d1]):
            gr = src.get(cid, "未检出")
            ag.add_patch(mpatches.Rectangle((j, i - 0.5), 0.92, 0.92, fc=GC[gr], ec=LGREY if gr in ("未检出", "非候选") else "none",
                                            lw=0.3))
    ag.set_xlim(-0.05, 3); ag.set_ylim(n - 0.5, -0.5); ag.axis("off")
    for j, t in enumerate(["酒", "S0", "S1"]):
        cv.text(x0 + cw * 8 + 1.8 + 3.4 * (j + 0.46), y0 - 1.2, t, ha="center", fontsize=FS["small"] - 1.0, rotation=0)
    cv.text(x0 + cw * 8 + 1.8 + 5.1, y0 - 5.6, "等级", ha="center", fontsize=FS["small"] - 0.3, fontweight="bold")
    cv.hcbar(x0 + 4, y0 + rh * n + 3.0, cw * 6, plt.cm.ScalarMappable(norm=norm, cmap=SEQ), label="OAV（对数色标）",
             ticks=[0.1, 1, 10, 100], ticklabels=["0.1", "1", "10", "100"])
    cv.legend([patch(GC[g], g, ec=LGREY if g in ("非候选", "临界") else "none") for g in ["稳定", "阈值敏感", "临界", "非候选"]],
              x0 + cw * 8 + 7.0, y0 + rh * n + 1.5, ncol=1, anchor="upper center", fontsize=FS["small"] - 1.0,
              handlelength=0.9, labelspacing=0.2)
    # (E) threshold stress test: number of candidates vs threshold multiplier
    st = cut["stress"]
    a = cv.panel(G.Cell(100, 64, 60, 54), "E", L=12, B=10, T=7, R=3)
    SCOL = {("共发酵酒", "—"): FRUIT_COL["山楂"], ("馏出液", "S0"): FRC[4], ("馏出液", "S1"): FRC[3]}
    SLAB = {("共发酵酒", "—"): "共发酵酒", ("馏出液", "S0"): "馏出液（S0）", ("馏出液", "S1"): "馏出液（S1 条件情景）"}
    for key, col in SCOL.items():
        s = st[(st["体系"] == key[0]) & (st["浓度口径"] == key[1])]
        a.plot(s["阈值倍数"], s["候选数"], marker="o", color=col, ms=3.6, mec="white", mew=0.5, lw=1.1,
               ls="--" if key[1] == "S1" else "-")
        for xv, yv in zip(s["阈值倍数"], s["候选数"]):
            a.text(xv, yv + 1.6, str(yv), ha="center", va="bottom", fontsize=FS["small"] - 1.0, color=col)
        a.text(5.3, s["候选数"].iloc[-1], SLAB[key].replace("（S1 条件情景）", "\nS1 情景").replace("（S0）", "\nS0"),
               ha="left", va="center", fontsize=FS["small"] - 1.0, color=col, linespacing=1.1)
    a.set_xscale("log", base=2); a.set_xticks([0.25, 0.5, 1, 2, 4]); a.set_xticklabels(["×0.25", "×0.5", "×1", "×2", "×4"])
    a.set_ylim(0, 56); a.set_xlim(0.2, 11)
    a.set_xlabel("阈值倍数（压力测试）"); a.set_ylabel("OAV ≥ 1 的化合物数"); soft(a)
    # (F) composition of candidate grades in each system
    a = cv.panel(G.Cell(100, 118, 60, 54), "F", L=21, B=10, T=7, R=3)
    rows = [("共发酵酒", "—"), ("馏出液", "S0"), ("馏出液", "S1")]
    yy = np.arange(3)[::-1]
    for yi, key in zip(yy, rows):
        s = oa[(oa["体系"] == key[0]) & (oa["浓度口径"] == key[1])]
        left = 0
        for gname in ["稳定", "阈值敏感", "临界"]:
            v = int((s["等级"] == gname).sum())
            a.barh(yi, v, left=left, height=0.6, color=GC[gname], edgecolor="white", lw=0.5, zorder=2)
            if False:
                a.text(left + v / 2, yi, str(v), ha="center", va="center", fontsize=FS["small"] - 0.8,
                       color="white" if gname == "稳定" else INK)
            left += v
        tot = int(s["等级"].isin(["稳定", "阈值敏感", "临界"]).sum())
        parts = " / ".join(str(int((s["等级"] == g_).sum())) for g_ in ["稳定", "阈值敏感", "临界"])
        a.text(left + 1.0, yi, parts, ha="left", va="center", fontsize=FS["small"] - 0.6, color=INK)
    a.set_yticks(yy); a.set_yticklabels(["共发酵酒", "馏出液 S0", "馏出液 S1"], fontsize=FS["small"])
    a.set_xlabel("化合物数（OAVmax ≥ 0.5）"); a.set_xlim(0, 56); soft(a, grid="x")
    leg(a, handles=[patch(GC[g], g, ec=LGREY if g == "临界" else "none") for g in ["稳定", "阈值敏感", "临界"]],
        loc="upper right", fontsize=FS["small"] - 0.8, frameon=False, handlelength=1.0)
    save(cv, "Fig2")


# ============================================================================================== Fig 3
def fig3():
    b = C.load(); m = b[b["主分析集"]].reset_index(drop=True)
    w, bb = J.load(); kt = J.key_table(w, bb)
    od = set(kt[kt["馏_OAVmax"] >= 1]["CID"])
    cv = Canvas(160, 172)
    # (A) hero heat map: relative mass share in the five fractions, compounds ordered by D
    o = m.sort_values("D").reset_index(drop=True)
    P = o[[f"p_{s}" for s in FRS]].values
    n = len(o); y0, H = 9.0, 140.0; rh = H / n
    xh = 18.0; cw = 6.0
    cv.letter(0, 1, "A")
    a = cv.ax(xh, y0, cw * 5, H)
    a.imshow(P, cmap=SEQ, norm=mcolors.Normalize(0, 1), aspect="auto", interpolation="nearest")
    a.set_xticks(range(5)); a.set_xticklabels(FRS, fontsize=FS["small"] - 0.3); a.xaxis.tick_top(); a.tick_params(length=0, pad=1)
    a.set_yticks([])
    for s_ in a.spines.values():
        s_.set_linewidth(0.5)
    # class strip and early/late strip on the left
    acs = cv.ax(xh - 7.2, y0, 2.6, H, overlap_ok=True)
    acs.imshow(np.arange(n)[:, None], aspect="auto", interpolation="nearest",
               cmap=mcolors.ListedColormap([CLS_COL[c] for c in o["新类别"]]))
    acs.set_xticks([]); acs.set_yticks([])
    ael = cv.ax(xh - 3.9, y0, 2.6, H, overlap_ok=True)
    ael.imshow(np.arange(n)[:, None], aspect="auto", interpolation="nearest",
               cmap=mcolors.ListedColormap([LATE if v else EARLY for v in o["late"]]))
    ael.set_xticks([]); ael.set_yticks([])
    cv.text(xh - 5.9, y0 - 1.0, "类别", ha="center", va="bottom", fontsize=FS["small"] - 1.2, rotation=90)
    cv.text(xh - 2.6, y0 - 1.0, "标签", ha="center", va="bottom", fontsize=FS["small"] - 1.2, rotation=90)
    # R as horizontal bars to the right, odorant star
    ar = cv.ax(xh + cw * 5 + 1.5, y0, 14, H, overlap_ok=True)
    ar.barh(np.arange(n), o["R"].values, height=0.8, color=HEART, edgecolor="none")
    ar.set_ylim(n - 0.5, -0.5); ar.set_xlim(0, 1.05); ar.set_yticks([]); ar.set_xticks([0, 0.5, 1])
    ar.set_xticklabels(["0", "0.5", "1"], fontsize=FS["small"] - 1)
    ar.xaxis.tick_top(); ar.tick_params(axis="x", length=2, pad=1)
    for s_ in ("right", "bottom"):
        ar.spines[s_].set_visible(False)
    cv.text(xh + cw * 5 + 8.5, y0 - 5.4, "R", ha="center", fontsize=FS["small"], fontweight="bold")
    ist = [i for i, cid in enumerate(o["CID"]) if cid in od]
    ar.scatter([1.1] * len(ist), ist, marker="*", s=9, c="#7A2E3A", lw=0, clip_on=False, zorder=4)
    # D tick marks at quartiles on the right margin
    for q in [0.1, 0.2, 0.3, 0.5, 0.7]:
        idx = int(np.searchsorted(o["D"].values, q))
        if 0 < idx < n:
            cv.text(xh - 7.6, y0 + rh * idx, f"D={q:g}", ha="right", va="center", fontsize=5.0, color=GREY)
            cv.fig.add_artist(plt.Line2D([(xh - 7.4) / 160, (xh - 7.0) / 160],
                                         [1 - (y0 + rh * idx) / 172] * 2, color=GREY, lw=0.5))
    cv.hcbar(xh, y0 + H + 3.0, cw * 5, plt.cm.ScalarMappable(norm=mcolors.Normalize(0, 1), cmap=SEQ), label="质量份额 p")
    # (B) D versus R
    a = cv.panel(G.Cell(66, 0, 94, 60), "B", L=13, B=10, T=7, R=2)
    for lab, col, mkr in [(0, EARLY, "o"), (1, LATE, "s")]:
        s = m[m["late"] == lab]
        isod = s["CID"].isin(od).values
        a.scatter(s["D"][~isod], s["R"][~isod], s=16, marker=mkr, facecolors="white", edgecolors=col, linewidths=0.8, zorder=3)
        a.scatter(s["D"][isod], s["R"][isod], s=20, marker=mkr, c=col, edgecolors="white", linewidths=0.5, zorder=4)
    lab_ = {"β-大马士酮": (-8, 6), "丁酸异戊酯": (8, -2), "乙酸苯乙酯": (6, 2), "辛酸": (-10, 9), "己酸乙酯": (8, -6),
            "2,3-丁二酮": (8, 7), "1-辛烯-3-醇": (-8, -5), "苯乙醇": (10, -12), "乙酸": (-6, 6)}
    for nm, off in lab_.items():
        r = m[m["采用名称"] == nm]
        if len(r):
            r = r.iloc[0]
            a.annotate(nm, (r["D"], r["R"]), xytext=off, textcoords="offset points", fontsize=FS["small"] - 1.0, color=INK,
                       ha="left" if off[0] > 0 else "right", va="center",
                       arrowprops=dict(arrowstyle="-", lw=0.4, color=GREY, shrinkA=0, shrinkB=1.5))
    for x_, lbl in zip((C.TMID * 1.0), FRS):
        pass
    a.set_xlim(0, 0.9); a.set_ylim(-0.05, 1.08)
    a.set_xlabel("分段近似馏出重心 D（馏出体积分数）"); a.set_ylabel("酒心占比 R")
    rho = stats.spearmanr(m["D"], m["R"]).correlation
    a.text(0.015, 0.97, f"Spearman ρ(D, R) = {fmt(rho, 2)}\nn = {len(m)}", transform=a.transAxes, ha="left", va="top",
           fontsize=FS["small"] - 0.6, color=INK)
    # heart window on the D axis (fraction midpoints of middle and second fractions)
    a.axvspan(50 / 700, 400 / 700, color=HEART, alpha=0.13, lw=0, zorder=0)
    a.text(0.33, 0.04, "酒心收集区间（50–400 mL）", ha="center", va="center", fontsize=FS["small"] - 0.8, color="#8A6D3B")
    soft(a, grid=None)
    leg(a, handles=[mk("o", EARLY, "早馏出型", mec=EARLY, mfc="white"), mk("s", LATE, "晚馏出型", mec=LATE, mfc="white"),
                    mk("o", INK, "实心：馏出液候选物", mec="white")],
        loc="upper right", fontsize=FS["small"] - 0.8, frameon=False, handlelength=1.0)
    # (C) / (D) R and D by early-late label
    for k_, (ep, lbl) in enumerate([("R", "酒心占比 R"), ("D", "馏出重心 D")]):
        a = cv.panel(G.Cell(66 + 47 * k_, 60, 47, 54), "CD"[k_], L=12, B=10, T=7, R=2)
        data = [m.loc[m["late"] == 0, ep].values, m.loc[m["late"] == 1, ep].values]
        bp = a.boxplot(data, positions=[0, 1], widths=0.5, patch_artist=True, showfliers=False,
                       medianprops=dict(color=INK, lw=0.9), whiskerprops=dict(lw=0.6, color=INK),
                       capprops=dict(lw=0.6, color=INK), boxprops=dict(lw=0.6, color=INK))
        for p_, col in zip(bp["boxes"], [EARLY, LATE]):
            p_.set_facecolor(mcolors.to_rgba(col, 0.22))
        rng = np.random.default_rng(3)
        for j, (vals, col) in enumerate(zip(data, [EARLY, LATE])):
            a.scatter(j + rng.uniform(-0.17, 0.17, len(vals)), vals, s=7, c=col, edgecolors="white", linewidths=0.3, zorder=3)
        u = stats.mannwhitneyu(data[1], data[0], alternative="two-sided")
        auc = u.statistic / (len(data[0]) * len(data[1]))
        a.text(0.5, 1.02, f"AUC = {fmt(auc, 2)}，P {('< 0.001' if u.pvalue < 0.001 else '= ' + fmt(u.pvalue, 2))}",
               transform=a.transAxes, ha="center", va="bottom", fontsize=FS["small"] - 0.7)
        a.set_xticks([0, 1]); a.set_xticklabels([f"早馏出型\n（{len(data[0])}）", f"晚馏出型\n（{len(data[1])}）"], fontsize=FS["small"])
        a.set_ylabel(lbl); a.set_xlim(-0.55, 1.55); a.set_ylim(-0.04, 1.04 if ep == "R" else 0.92); soft(a)
    # (E) R by chemical class
    a = cv.panel(G.Cell(66, 114, 47, 58), "E", L=12, B=15, T=7, R=2)
    cls = [c for c in C.CLS7]
    for j, c in enumerate(cls):
        v = m.loc[m["新类别"] == c, "R"].values
        rng = np.random.default_rng(j)
        a.scatter(j + rng.uniform(-0.2, 0.2, len(v)), v, s=7, c=CLS_COL[c], edgecolors="white", linewidths=0.3, zorder=3)
        a.hlines(np.median(v), j - 0.3, j + 0.3, color=INK, lw=1.0, zorder=4)
    a.set_xticks(range(len(cls))); a.set_xticklabels([CLS_SHORT.get(c, c).replace("萜类/降异戊二烯", "萜类") for c in cls],
                                                      rotation=90, ha="center", fontsize=FS["small"] - 0.5)
    kw = stats.kruskal(*[m.loc[m["新类别"] == c, "R"].values for c in cls])
    a.text(0.5, 1.02, f"Kruskal–Wallis P = {fmt(kw.pvalue, 2)}", transform=a.transAxes, ha="center", va="bottom",
           fontsize=FS["small"] - 0.7)
    a.set_ylabel("酒心占比 R"); a.set_ylim(-0.04, 1.04); a.set_xlim(-0.6, len(cls) - 0.4); soft(a)
    # (F) concentration-basis sensitivity: R under S1 vs S0
    a = cv.panel(G.Cell(113, 114, 47, 58), "F", L=12, B=15, T=7, R=2)
    d_ = (m["R_S1"] - m["R"]) * 100
    big = d_.abs() > 5
    a.plot([0, 1], [0, 1], color=GREY, lw=0.6, ls=(0, (3, 2)), zorder=1)
    a.scatter(m.loc[~big, "R"], m.loc[~big, "R_S1"], s=9, c=LGREY, edgecolors=GREY, linewidths=0.3, zorder=2)
    a.scatter(m.loc[big, "R"], m.loc[big, "R_S1"], s=13, c=FRC[1], edgecolors="white", linewidths=0.4, zorder=3)
    a.set_xlabel("R（S0：表列浓度）"); a.set_ylabel("R（S1：稀释未回算情景）")
    a.set_xlim(-0.03, 1.03); a.set_ylim(-0.03, 1.03)
    a.text(0.04, 0.96, f"|ΔR| > 5 个百分点：{int(big.sum())}/{len(m)}\n中位 |ΔR| = {fmt(d_.abs().median(), 2)} 个百分点",
           transform=a.transAxes, ha="left", va="top", fontsize=FS["small"] - 0.8)
    soft(a, grid=None)
    # class legend for (A) and (E)
    cv.legend([patch(CLS_COL[c], CLS_SHORT.get(c, c)) for c in C.CLS7] + [patch(EARLY, "早馏出型"), patch(LATE, "晚馏出型")],
              32, 160.5, ncol=3, anchor="upper center", fontsize=FS["small"] - 1.0, handlelength=0.8, columnspacing=0.7,
              handletextpad=0.3)
    save(cv, "Fig3")


# ============================================================================================== Fig 7
def fig7():
    cut = pickle.load(open(os.path.join(C.OUT, "cut.pkl"), "rb"))
    cur = cut["curves"]; odd = cut["odor"]; et = cut["ethanol"]; combos = cut["combos"]
    head_only = odd[odd["p_头"] >= 0.999]["CID"].tolist()
    spread = odd[(odd["p_头"] < 0.999) & (odd["p_头"] > 0)]["CID"].tolist()
    cv = Canvas(160, 160)
    # (A) hero: retention in the heart vs head cut, labelled within-head scenarios
    a = cv.panel(G.Cell(0, 0, 88, 72), "A", L=13, B=10, T=11, R=3)
    vals14 = {}
    for kn, col in SCN_COL.items():
        s_ = cur[(cur["浓度口径"] == "S0") & (cur["头内分布"] == kn)]
        ho = s_[s_["CID"].isin(head_only)].groupby("h")["酒心占比"].mean() * 100
        sp = s_[s_["CID"].isin(spread)].groupby("h")["酒心占比"].median() * 100
        a.plot(ho.index, ho.values, color=col, lw=1.5, zorder=3)
        a.plot(sp.index, sp.values, color=col, lw=1.0, ls=(0, (4, 2)), zorder=3)
        vals14[kn] = ho.loc[14.0]
    a.plot(et["h"], et["乙醇回收率"], color=INK, lw=1.5, zorder=4)
    vline(a, 14, color="#8A6D3B"); vline(a, 50, color="#8A6D3B")
    lo, hi = vals14["前置富集"], vals14["后置富集"]
    a.annotate("", xy=(15.2, lo), xytext=(15.2, hi), arrowprops=dict(arrowstyle="<->", color="#8A6D3B", lw=0.7))
    a.text(16.0, (lo + hi) / 2 + 4, f"头馏分独占型：\n{lo:.0f}%–{hi:.0f}%\n（均匀 {vals14['均匀']:.0f}%）",
           fontsize=FS["small"] - 1.0, color="#8A6D3B", va="center",
           bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.6))
    a.set_xlim(0, 50); a.set_ylim(0, 102)
    a.set_xlabel("掐头体积 h（mL，二次蒸馏前 50 mL 内）"); a.set_ylabel("进入酒心的比例（%）")
    sec = a.secondary_xaxis("top", functions=(lambda h: h / 7.0, lambda p: p * 7.0))
    sec.set_xticks([0, 1, 2, 3, 4, 5, 6, 7]); sec.tick_params(labelsize=FS["small"] - 0.5, length=2, pad=1)
    sec.set_xlabel("掐头量（占 700 mL 收集体积的 %）", fontsize=FS["small"], labelpad=2)
    soft(a)
    leg(a, handles=[plt.Line2D([], [], color=INK, lw=1.5, label="乙醇（头内均匀）"),
                    plt.Line2D([], [], color=GREY, lw=1.5, label=f"头馏分独占型（{len(head_only)} 种，均值）"),
                    plt.Line2D([], [], color=GREY, lw=1.0, ls=(0, (4, 2)), label=f"跨馏分型（{len(spread)} 种，中位数）")] +
        [plt.Line2D([], [], color=c, lw=2.2, label=f"头内{k}") for k, c in SCN_COL.items()],
        loc="lower left", fontsize=FS["small"] - 1.0, frameon=False, handlelength=1.8, ncol=2, columnspacing=0.8)
    # (B) identifiable whole-fraction combinations (S0), values as text
    cb = combos[combos["浓度口径"] == "S0"].reset_index(drop=True)
    metr = [("乙醇回收率", "乙醇\n回收"), ("酒度", "酒度"), ("候选物保留50以上", "≥50%\n个数"),
            ("保留_β-大马士酮", "大马\n士酮"), ("保留_己酸乙酯", "己酸\n乙酯"), ("保留_丁酸异戊酯", "丁酸异\n戊酯"),
            ("保留_乙酸苯乙酯", "乙酸苯\n乙酯"), ("酸类回收率", "酸类\n回收")]
    V = cb[[m_ for m_, _ in metr]].values.astype(float)
    Vn = V.copy(); Vn[:, 2] = V[:, 2] / 25 * 100; Vn[:, 1] = (V[:, 1] - 55) / 20 * 100
    a = cv.ax(109, 15, 49, 46); cv.letter(89, 1, "B")
    a.imshow(Vn, cmap=SEQB, norm=mcolors.Normalize(0, 115), aspect="auto")
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            v = V[i, j]
            a.text(j, i, f"{v:.0f}" if j != 1 else f"{v:.1f}", ha="center", va="center", fontsize=5.2,
                   color="white" if Vn[i, j] > 70 else INK)
    a.set_xticks(range(len(metr))); a.set_xticklabels([t for _, t in metr], fontsize=5.0, linespacing=1.15)
    a.xaxis.tick_top()
    names = ["中间", "中间+次（基准）", "头+中间+次", "中间+次+次尾", "头+中间+次+次尾"]
    a.set_yticks(range(len(cb))); a.set_yticklabels(names, fontsize=FS["small"] - 0.9); a.tick_params(length=0, pad=1)
    for kk in range(len(cb) + 1):
        a.axhline(kk - 0.5, color="white", lw=0.8)
    for kk in range(len(metr) + 1):
        a.axvline(kk - 0.5, color="white", lw=0.8)
    for s_ in a.spines.values():
        s_.set_visible(False)
    cv.text(124, 64.5, "整馏分组合：由实测份额直接计算（S0）", ha="center", fontsize=FS["small"] - 1.0, color=GREY)
    cv.text(124, 68.0, "单位：%；酒度为 % vol；候选物为 25 种中的个数", ha="center", fontsize=FS["small"] - 1.0, color=GREY)
    # (C) per-odorant retention at a 2% head cut: range over the 6 scenarios
    o = odd.sort_values("uni_14").reset_index(drop=True)
    a = cv.panel(G.Cell(0, 72, 74, 88), "C", L=27, B=10, T=7, R=3)
    JC = {"稳健保留": "#5F7F73", "情景依赖": "#D6A84F", "稳健损失": "#A64B3C"}
    for i, r in o.iterrows():
        col = JC[r["2%掐头判断"]]
        a.plot([r["min_14"] * 100, r["max_14"] * 100], [i, i], color=col, lw=2.6, solid_capstyle="round", alpha=0.9,
               zorder=2)
        a.scatter(r["uni_14"] * 100, i, s=12, c="white", edgecolors=INK, linewidths=0.6, zorder=3)
        a.scatter(r["min_50"] * 100, i, s=10, marker="|", c=INK, linewidths=0.8, zorder=3)
    vline(a, 50)
    a.set_yticks(range(len(o))); a.set_yticklabels(o["采用名称"], fontsize=FS["small"] - 1.0)
    a.set_xlim(-2, 102); a.set_ylim(-0.7, len(o) - 0.3)
    a.set_xlabel("掐头 2% 时进入酒心的比例（%）"); soft(a, grid="x")
    leg(a, handles=[plt.Line2D([], [], color=JC[k], lw=2.6, label=k + "（6 情景）") for k in JC] +
        [mk("o", "white", "S0 + 头内均匀", mec=INK), mk("|", INK, "整段掐头 7.1%（S0）", mec=INK)],
        loc="upper left", fontsize=FS["small"] - 1.1, frameon=False, handlelength=1.2)
    # (D) ethanol vs acid recovery for whole-fraction combinations, S0 -> S1
    a = cv.panel(G.Cell(74, 72, 43, 88), "D", L=12, B=10, T=7, R=2)
    s0 = combos[combos["浓度口径"] == "S0"].reset_index(drop=True)
    s1 = combos[combos["浓度口径"] == "S1"].reset_index(drop=True)
    for i, nm in enumerate(names):
        a.annotate("", xy=(s1.loc[i, "酸类回收率"], s1.loc[i, "乙醇回收率"]),
                   xytext=(s0.loc[i, "酸类回收率"], s0.loc[i, "乙醇回收率"]),
                   arrowprops=dict(arrowstyle="-|>", color=LGREY, lw=0.7, mutation_scale=6, shrinkA=2.5, shrinkB=2.5),
                   zorder=1)
    a.scatter(s0["酸类回收率"], s0["乙醇回收率"], s=26, c=FRC[4], edgecolors="white", linewidths=0.5, zorder=3)
    a.scatter(s1["酸类回收率"], s1["乙醇回收率"], s=22, marker="D", c="white", edgecolors=FRC[3], linewidths=0.9, zorder=3)
    off = {0: (3, 6), 1: (5, -9), 2: (-4, 5), 3: (-4, -9), 4: (-4, 5)}
    for i, nm in enumerate(names):
        a.annotate(nm, (s0.loc[i, "酸类回收率"], s0.loc[i, "乙醇回收率"]), xytext=off[i], textcoords="offset points",
                   fontsize=FS["small"] - 1.3, color=INK, ha="left" if off[i][0] > 0 else "right")
    a.set_xlabel("酸类回收率（%）"); a.set_ylabel("乙醇回收率（%）")
    a.set_xlim(0, 100); a.set_ylim(60, 100); soft(a)
    leg(a, handles=[mk("o", FRC[4], "S0", mec="white"), mk("D", "white", "S1", mec=FRC[3])], loc="lower right",
        fontsize=FS["small"] - 0.8, frameon=False)
    # (E) decision categories across the 6 scenarios for three cut options
    a = cv.panel(G.Cell(117, 72, 43, 88), "E", L=12, B=14, T=7, R=2)
    xs = np.arange(3)
    cats = {}
    for k_, h in enumerate([0, 14, 50]):
        lo_, hi_ = odd[f"min_{h}"], odd[f"max_{h}"]
        cats[k_] = [int((lo_ >= 0.5).sum()), int(((lo_ < 0.5) & (hi_ >= 0.5)).sum()), int((hi_ < 0.5).sum())]
    bottom = np.zeros(3)
    for j, kname in enumerate(["稳健保留", "情景依赖", "稳健损失"]):
        v = np.array([cats[k_][j] for k_ in range(3)])
        a.bar(xs, v, bottom=bottom, width=0.6, color=JC[kname], edgecolor="white", lw=0.6, zorder=2)
        for x_, b_, vv in zip(xs, bottom, v):
            if vv:
                a.text(x_, b_ + vv / 2, str(vv), ha="center", va="center", fontsize=FS["small"] - 0.6,
                       color="white" if kname != "情景依赖" else INK)
        bottom += v
    a.set_xticks(xs); a.set_xticklabels(["不掐头", "掐头\n2%", "掐头\n7.1%"], fontsize=FS["small"])
    a.set_ylabel("候选物数（酒心占比 50% 判据）"); a.set_ylim(0, 25.5); soft(a)
    save(cv, "Fig7")


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "7"]
    for k in which:
        print("Fig", k); globals()[f"fig{k}"]()
