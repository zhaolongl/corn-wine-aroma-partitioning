"""Overlap-free residue labels for the docking close-ups of Fig. 6C-F.  The anchors (side-chain centroids of the contact
residues, projected onto the image by md_render3.py) are labelled with text boxes placed by a greedy search followed by
refinement passes over candidate positions around each anchor.  The cost penalises overlap with other labels, ligand atoms
and residue atoms, leader lines that cross labels or the ligand, the distance from the anchor and positions facing the
ligand; a leader line is drawn when a label sits away from its residue."""
import numpy as np
import matplotlib.patheffects as pe

GAPS = [4, 14, 28, 45, 65, 90, 120, 155]                     # px between the anchor and the nearest point of the label box
THETA = np.radians(np.arange(0, 360, 15))
LEADER = 12                                                  # px; labels farther than this get a leader line


def _closest(b, p):
    return np.array([min(max(p[0], b[0]), b[2]), min(max(p[1], b[1]), b[3])])


def _seg_dist(p, q, P):
    d = q - p; L2 = max(float(d @ d), 1e-9); t = np.clip(((P - p) @ d) / L2, 0, 1)
    return np.hypot(*(P - (p + t[:, None] * d)).T)


def _seg_hits_box(p, q, b):
    t = np.linspace(0, 1, 30)[:, None]; s = p + t * (q - p)
    return bool(np.any((s[:, 0] > b[0]) & (s[:, 0] < b[2]) & (s[:, 1] > b[1]) & (s[:, 1] < b[3])))


def _n_inside(P, b):
    return int(np.sum((P[:, 0] > b[0]) & (P[:, 0] < b[2]) & (P[:, 1] > b[1]) & (P[:, 1] < b[3]))) if len(P) else 0


def place(ax, geo, fontsize=6.0, color="black", line_color="#4D4D4D", iters=6):
    fig = ax.figure; fig.canvas.draw(); r = fig.canvas.get_renderer(); xl, yl = ax.get_xlim(), ax.get_ylim()
    W, H = geo["W"], geo["H"]; bb = ax.get_window_extent(r); sx, sy = W / bb.width, H / bb.height
    kw = dict(fontsize=fontsize, fontweight="bold", color=color, ha="center", va="center", zorder=5,
              path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])
    A = geo["anchors"]; n = len(A); size = []
    for a in A:
        t = ax.text(0, 0, a["label"], **kw); e = t.get_window_extent(r); t.remove(); size.append((e.width * sx, e.height * sy))
    lig = np.array(geo["lig"]); lc = lig.mean(axis=0); atoms = [np.array(a["atoms"]) for a in A]; anc = np.array([a["xy"] for a in A])

    def box(i, c, m=0.0):
        w, h = size[i]; return (c[0] - w / 2 - m, c[1] - h / 2 - m, c[0] + w / 2 + m, c[1] + h / 2 + m)

    def leader(i, g, c):
        return (anc[i], _closest(box(i, c), anc[i])) if g > LEADER else None

    cands = []
    for i in range(n):
        w, h = size[i]; ci = []
        for g in GAPS:
            for th in THETA:
                u = np.array([np.cos(th), np.sin(th)]); e = min(w / 2 / max(abs(u[0]), 1e-9), h / 2 / max(abs(u[1]), 1e-9))
                ci.append((g, u, anc[i] + (g + e) * u))
        cands.append(ci)

    def cost(i, g, u, c, placed):
        b = box(i, c)
        if b[0] < 8 or b[1] < 8 or b[2] > W - 8 or b[3] > H - 8:
            return np.inf
        k = 1.6 * g; bm = box(i, c, 5); seg = leader(i, g, c)
        for j, (gj, cj) in placed.items():
            if j == i:
                continue
            bj = box(j, cj, 5); ix = min(bm[2], bj[2]) - max(bm[0], bj[0]); iy = min(bm[3], bj[3]) - max(bm[1], bj[1])
            if ix > 0 and iy > 0:
                k += 5000 + 0.5 * ix * iy
            sj = leader(j, gj, cj)
            if sj is not None and _seg_hits_box(sj[0], sj[1], box(i, c, 3)):
                k += 800
            if seg is not None and _seg_hits_box(seg[0], seg[1], box(j, cj, 3)):
                k += 800
        k += 600 * _n_inside(lig, box(i, c, 18))
        for j in range(n):
            k += (20 if j == i else 60) * _n_inside(atoms[j], box(i, c, 6))
        if seg is not None:
            k += 250 * int(np.sum(_seg_dist(seg[0], seg[1], lig) < 12))
        v = anc[i] - lc; nv = np.linalg.norm(v)
        if nv > 1e-6:
            k += 80 * (1 - float(u @ v) / nv) / 2
        return k

    crowd = [sum(np.hypot(*(anc[j] - anc[i])) < 160 for j in range(n) if j != i) for i in range(n)]
    order = sorted(range(n), key=lambda i: -crowd[i]); placed = {}
    for it in range(iters):
        for i in order:
            placed.pop(i, None)
            g, u, c = min(cands[i], key=lambda t: cost(i, t[0], t[1], t[2], placed))
            placed[i] = (g, c)
    for i, (g, c) in placed.items():
        ax.text(c[0], c[1], A[i]["label"], **kw)
        seg = leader(i, g, c)
        if seg is not None:
            p0, p1 = seg; d = p1 - p0; L = np.hypot(*d); p1 = p0 + d * max(L - 3, 0) / max(L, 1e-9)
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=line_color, lw=0.45, zorder=4, solid_capstyle="round")
    ax.set_xlim(xl); ax.set_ylim(yl)
    # report residual conflicts (label-label overlaps and ligand atoms under labels) for the build log
    over = sum(1 for i in placed for j in placed if i < j and
               min(box(i, placed[i][1])[2], box(j, placed[j][1])[2]) > max(box(i, placed[i][1])[0], box(j, placed[j][1])[0]) and
               min(box(i, placed[i][1])[3], box(j, placed[j][1])[3]) > max(box(i, placed[i][1])[1], box(j, placed[j][1])[1]))
    on_lig = sum(_n_inside(lig, box(i, placed[i][1], 4)) for i in placed)
    return {"labels": n, "label_overlaps": over, "ligand_atoms_under_labels": on_lig,
            "leaders": sum(1 for i in placed if placed[i][0] > LEADER)}
