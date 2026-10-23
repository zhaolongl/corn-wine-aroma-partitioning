"""Figure style modelled on Liu et al. (2026, Food Research International 235, 119204):
closed frames, bold serif labels, black-edged bars with per-group colour gradients,
dashed group separators, table-like two-level category axis, bold significance letters."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb

OUT = "."
MM = 1 / 25.4
plt.rcParams.update({
    "font.family": ["Times New Roman", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC", "PingFang SC", "DejaVu Sans"],
    "font.size": 7.5,
    "axes.labelsize": 8,
    "axes.labelweight": "bold",
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.8,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.edgecolor": "black", "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black",
    "axes.spines.top": True, "axes.spines.right": True,
    "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600,
    "axes.unicode_minus": False,
    "mathtext.fontset": "stix",
})

def ramp(c_dark, c_light, n):
    a = np.array(to_rgb(c_dark)); b = np.array(to_rgb(c_light))
    return [tuple(a + (b - a) * t) for t in np.linspace(0, 1, n)]

# group base colours (dark -> light gradients within a group, as in the reference paper)
HEAT_BASE = {"WB": ("#2F6DB5", "#CFE2F5"), "HK": ("#D9603F", "#FAD6C7"), "ZZ": ("#2E9A6E", "#CDEBDD"), "SZ": ("#6A57A8", "#DCD5EE")}
HEAT_SOLID = {"WB": "#2F6DB5", "HK": "#D9603F", "ZZ": "#2E9A6E", "SZ": "#6A57A8"}
HEATL = {"WB": "微波", "HK": "烘烤", "ZZ": "蒸制", "SZ": "水煮"}
HEATM = {"WB": "o", "HK": "s", "ZZ": "^", "SZ": "D"}
VAR_SOLID = {"HN": "#6A57A8", "BT": "#3E9B5F", "HT": "#E3A21A"}
VARM = {"HN": "o", "BT": "s", "HT": "^"}
VARL = {"HN": "黑糯玉米（HN）", "BT": "白甜玉米（BT）", "HT": "黄甜玉米（HT）"}
MODE_SOLID = {"全浆发酵": "#D9603F", "浸渍后发酵": "#2F6DB5", "清汁发酵": "#E3A21A"}
MODEM = {"全浆发酵": "o", "浸渍后发酵": "s", "清汁发酵": "^"}
CLASS_COL = {"酯类": "#8C1C13", "醇类": "#C0392B", "酸类": "#E67E22", "萜烯类": "#F5B041", "醛酮类": "#F9D5A7", "烷烃类": "#A9CCE3", "其它类": "#5D6D7E"}
RDBU = LinearSegmentedColormap.from_list("rdbu", ["#2166AC", "#67A9CF", "#D1E5F0", "#F7F7F7", "#FDDBC7", "#EF8A62", "#B2182B"])
REDS = LinearSegmentedColormap.from_list("reds", ["#FFF5EB", "#FDD0A2", "#FD8D3C", "#D94801", "#7F2704"])

def panel(ax, s, x=-0.14, y=1.03):
    ax.text(x, y, f"({s})", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")

def save(fig, name):
    import os
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

def grouped_bars(ax, groups, ylabel, letters=True, ylim=None, level_fmt=None, table_axis=True, colors=None, width=0.78, tab_y=(-0.012, -0.11, -0.21), lev_fs=None):
    """groups: list of (group_key, group_label, levels[list of str], means, sds, letters)
    Draw bars with black edges, error bars, bold letters, dashed separators and a table-like axis."""
    x = 0.0; pos = []; bounds = []; allhi = []
    for gi, (gk, glab, levels, means, sds, lets) in enumerate(groups):
        n = len(levels)
        cols = colors[gk] if colors and isinstance(colors[gk], list) else ramp(*HEAT_BASE[gk], n) if gk in HEAT_BASE else [colors[gk]] * n
        xs = x + np.arange(n)
        sds = np.nan_to_num(np.asarray(sds, float))
        ax.bar(xs, means, width=width, color=cols, edgecolor="black", linewidth=0.6, zorder=2)
        ax.errorbar(xs, means, yerr=sds, fmt="none", ecolor="black", elinewidth=0.7, capsize=2.2, capthick=0.7, zorder=3)
        allhi += list(np.asarray(means) + sds)
        pos.append(xs); bounds.append((x - 0.5, x + n - 0.5))
        x += n + 0.6
    top = max(allhi)
    if ylim is None:
        ylim = (0, top * 1.18)
    ax.set_ylim(*ylim)
    rng = ylim[1] - ylim[0]
    if letters:
        for (gk, glab, levels, means, sds, lets), xs in zip(groups, pos):
            for xx, m, s, l in zip(xs, means, np.nan_to_num(np.asarray(sds, float)), lets):
                if l:
                    ax.text(xx, m + s + rng * 0.015, l, ha="center", va="bottom", fontsize=7, fontweight="bold")
    # dashed separators between groups
    for (a, b), (a2, b2) in zip(bounds[:-1], bounds[1:]):
        ax.axvline((b + a2) / 2, color="black", lw=0.8, ls=(0, (5, 4)), zorder=1)
    ax.set_xlim(bounds[0][0] - 0.3, bounds[-1][1] + 0.3)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.concatenate(pos))
    if table_axis:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
        trans = ax.get_xaxis_transform()
        y1, y2, y3 = tab_y
        x0, x1 = bounds[0][0] - 0.3, bounds[-1][1] + 0.3
        kw = dict(transform=trans, clip_on=False, color="black", lw=0.6)
        ax.plot([x0, x1], [y2, y2], **kw); ax.plot([x0, x1], [y3, y3], **kw)
        ax.plot([x0, x0], [0, y3], **kw); ax.plot([x1, x1], [0, y3], **kw)
        for gi, ((gk, glab, levels, *_), xs, (a, b)) in enumerate(zip(groups, pos, bounds)):
            for xx, lev in zip(xs, levels):
                ax.text(xx, (y1 + y2) / 2 - 0.005, lev, transform=trans, ha="center", va="center", fontsize=6.5, fontweight="bold")
            for xx in xs[:-1]:
                ax.plot([xx + 0.5, xx + 0.5], [0, y2], **kw)
            ax.text((a + b) / 2, (y2 + y3) / 2, glab, transform=trans, ha="center", va="center", fontsize=7, fontweight="bold")
            if gi < len(groups) - 1:
                nxt = bounds[gi + 1][0]
                mid = (b + nxt) / 2
                ax.plot([mid, mid], [0, y3], **kw)
    return pos

def radar_poly(ax, df, cols, colors, markers, labels, rmin, rmax, ticks, lw=1.1):
    """Radar chart with polygon grid (reference-paper style)."""
    attrs = list(df.index); n = len(attrs)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    ax.set_ylim(rmin, rmax)
    ax.grid(False); ax.spines["polar"].set_visible(False)
    ax.set_yticks([]); ax.set_xticks([])
    for t in ticks:
        aa = np.concatenate([ang, ang[:1]])
        ax.plot(aa, [t] * (n + 1), color="#BDBDBD", lw=0.5, zorder=0)
        if t <= rmin:
            continue  # centre tick: no label (it would sit on the data)
        ax.text(0, t, f"{t:g}", fontsize=5.8, color="#555555", ha="right", va="center")
    for a in ang:
        ax.plot([a, a], [rmin, rmax], color="#BDBDBD", lw=0.5, zorder=0)
    aa = np.concatenate([ang, ang[:1]])
    ax.plot(aa, [rmax] * (n + 1), color="#555555", lw=0.8, zorder=0)
    for c in cols:
        v = df[c].values.astype(float); vv = np.concatenate([v, v[:1]])
        ax.plot(aa, vv, color=colors[c], lw=lw, marker=markers[c], ms=3, mec="white", mew=0.4, label=labels[c], zorder=3)
        ax.fill(aa, vv, color=colors[c], alpha=0.08, zorder=2)
    for a, lab in zip(ang, attrs):
        ax.text(a, rmax + (rmax - rmin) * 0.13, lab, ha="center", va="center", fontsize=6.8, fontweight="bold")
