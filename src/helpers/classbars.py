"""Volatile-class totals for papers 1 and 2 (data-audit revision): class sums of the appendix means, SD by root-sum-of-squares
of the compound SDs (independence assumed), Duncan letters from the summary statistics (n = 3). Replaces the stacked
composition bars, which carried neither dispersion nor significance."""
import numpy as np
from duncan import duncan_letters
from style2 import grouped_bars


def class_totals(v, keys, classes):
    out = {}
    for c in ["总量"] + list(classes):
        sub = v if c == "总量" else v[v["类别"] == c]
        m = [float(sub[k + "_均值"].fillna(0).sum()) for k in keys]
        s = [float(np.sqrt((sub[k + "_SD"].fillna(0) ** 2).sum())) for k in keys]
        L = duncan_letters(m, s)[0] if max(m) > 0 else [""] * len(keys)
        n = [int(sub[k + "_均值"].notna().sum()) for k in keys]
        out[c] = {"均值": m, "SD": s, "字母": [l if mm > 0 else "nd" for l, mm in zip(L, m)], "数量": n}
    return out


def class_axes(fig, rects, v, keys, labels, colors, groups_of_classes, ylab="含量（μg/L）", tab_y=(-0.012, -0.17, -0.33), lev_fs=None, ylims=None):
    """one grouped-bar axis per entry of groups_of_classes (e.g. [['总量','醇类','酯类'], ['酸类','醛酮类','其它类']])"""
    ct = class_totals(v, keys, sorted({c for g in groups_of_classes for c in g if c != "总量"}))
    axes = []
    for i, (rect, cl) in enumerate(zip(rects, groups_of_classes)):
        ax = fig.add_axes(rect)
        g = [(c, ("总量" if c == "总量" else c), [labels[k] for k in keys], ct[c]["均值"], ct[c]["SD"], ct[c]["字母"]) for c in cl]
        top = max(max(np.array(ct[c]["均值"]) + np.array(ct[c]["SD"])) for c in cl)
        yl = ylims[i] if ylims else (0, top * 1.22)
        grouped_bars(ax, g, ylab, ylim=yl, colors={c: [colors[k] for k in keys] for c in cl}, tab_y=tab_y, lev_fs=lev_fs, width=0.74)
        axes.append(ax)
    return axes, ct
