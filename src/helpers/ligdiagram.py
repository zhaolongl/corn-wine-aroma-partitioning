"""Two-dimensional receptor-ligand interaction diagram (Discovery Studio style) drawn with matplotlib from the layout written by
md_layout2d.py (paper 2, twelfth version, Fig. 6C-F): ligand skeleton (heteroatoms labelled), one disc per residue coloured by its
highest-ranked interaction (dockstyle.TYPES), dashed lines from the residue to each ligand atom it interacts with."""
import numpy as np
import matplotlib.patches as mpatches
from dockstyle import TYPES, LINE, FILL, EDGE

HET = {"O": "#D62728", "N": "#1F4FB4", "S": "#B8860B"}


def _trim(p, q, a, b):
    """segment p->q shortened by a at p and b at q"""
    d = q - p; L = np.hypot(*d); u = d / max(L, 1e-9)
    return p + u * a, q - u * b


def draw(ax, js, fs_res=6.0, fs_atom=6.8, lw_bond=0.95, lw_line=0.85, bond_color="#3C3C3C"):
    x0, y0, x1, y1 = js["box"]; R = js["R"]
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    P = np.array([[a["x"], a["y"]] for a in js["atoms"]]); el = [a["el"] for a in js["atoms"]]
    pad = {i: (0.36 if el[i] in HET else 0.0) for i in range(len(el))}
    disc = {d["resi"]: np.array([d["x"], d["y"]]) for d in js["discs"]}
    # interaction lines (below the ligand and the discs)
    for ln in js["lines"]:
        p, q = _trim(P[ln["atom"]], disc[ln["resi"]], max(pad[ln["atom"]], 0.12), R)
        ax.plot([p[0], q[0]], [p[1], q[1]], color=LINE[ln["type"]], lw=lw_line, ls=(0, (2.6, 1.6)), dash_capstyle="round", zorder=2)
    # ligand bonds
    for b in js["bonds"]:
        i, j = b["a"], b["b"]; p, q = _trim(P[i], P[j], pad[i], pad[j])
        if b["order"] == 2 and b["side"] is not None:                      # inner line towards the ring centre / substituents
            ax.plot([p[0], q[0]], [p[1], q[1]], color=bond_color, lw=lw_bond, solid_capstyle="round", zorder=3)
            d = q - p; nrm = np.array([-d[1], d[0]]) / max(np.hypot(*d), 1e-9); s = np.asarray(b["side"]) - (p + q) / 2
            nrm = nrm if nrm @ s > 0 else -nrm; L = np.hypot(*d)
            pi, qi = _trim(P[i] + nrm * 0.24, P[j] + nrm * 0.24, max(pad[i], L * 0.14), max(pad[j], L * 0.14))
            ax.plot([pi[0], qi[0]], [pi[1], qi[1]], color=bond_color, lw=lw_bond, solid_capstyle="round", zorder=3)
        elif b["order"] == 2:                                               # terminal C=O: two lines centred on the bond axis
            d = q - p; nrm = np.array([-d[1], d[0]]) / max(np.hypot(*d), 1e-9)
            for sgn in (-1, 1):
                ax.plot([p[0] + sgn * 0.11 * nrm[0], q[0] + sgn * 0.11 * nrm[0]], [p[1] + sgn * 0.11 * nrm[1], q[1] + sgn * 0.11 * nrm[1]],
                        color=bond_color, lw=lw_bond, solid_capstyle="round", zorder=3)
        else:
            ax.plot([p[0], q[0]], [p[1], q[1]], color=bond_color, lw=lw_bond, solid_capstyle="round", zorder=3)
    for i, e in enumerate(el):
        if e in HET:
            ax.text(P[i, 0], P[i, 1], e, ha="center", va="center", fontsize=fs_atom, color=HET[e], fontweight="bold", zorder=4)
    # residue discs
    for d in js["discs"]:
        t = d["type"]
        ax.add_patch(mpatches.Circle((d["x"], d["y"]), R, fc=FILL[t], ec=EDGE[t], lw=0.6, zorder=5))
        ax.text(d["x"], d["y"] + 0.03, f"{d['name']}\n{d['num']}", ha="center", va="center", fontsize=fs_res, linespacing=0.95,
                fontweight="bold", color="#1A1A1A", zorder=6)
    return sorted({d["type"] for d in js["discs"]} | {ln["type"] for ln in js["lines"]}, key=TYPES.index)


def legend_handles(types, ms=6.5):
    """legend entries: disc swatch with the dashed line colour (van der Waals contacts have no line)"""
    from matplotlib.lines import Line2D
    from dockstyle import LABEL
    hs = []
    for t in types:
        if LINE[t] is None:
            hs.append(Line2D([], [], marker="o", ls="none", ms=ms, mfc=FILL[t], mec=EDGE[t], mew=0.6, label=LABEL[t]))
        else:
            hs.append(Line2D([], [], marker="o", ls=(0, (2.6, 1.6)), color=LINE[t], lw=0.85, ms=ms, mfc=FILL[t], mec=EDGE[t], mew=0.6,
                             label=LABEL[t]))
    return hs
