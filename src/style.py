"""JAR merged paper, sixth version: figure style.  Builds on the millimetre layout of figgrid (fixed print size 160 mm,
fixed font sizes, panel letters at cell corners) and replaces the saturated colours of earlier versions with one restrained
palette shared by all figures: warm earth tones for the distillation fractions, slate blue / terracotta for the
early-late contrast, muted qualitative colours for chemical classes and receptors."""
import os
import sys

import numpy as np

SCR = os.path.dirname(os.path.abspath(__file__))
import config as CFG
P2DATA = str(CFG.HELPERS)
sys.path.insert(0, P2DATA); sys.path.insert(0, SCR)
import matplotlib.colors as mcolors                                                 # noqa: E402

import figgrid as G                                                                 # noqa: E402
from figgrid import FS, Canvas, cells, leg, plt                                     # noqa: E402

OUT = str(CFG.FIGS)

# ---- fractions (head -> tail) and the heart band
FRS = ["头", "中间", "次", "次尾", "尾"]
FRL = ["头馏分", "中间馏分", "次馏分", "次尾馏分", "尾馏分"]
FRC = ["#7A2E3A", "#C8664B", "#E0A458", "#86A6BF", "#3F5A78"]
HEART = "#E9C46A"
# ---- early / late and key contrasts
EARLY, LATE = "#C8664B", "#3F5A78"
INK, GREY, LGREY, PAPER = "#2B2B2B", "#8C8C8C", "#D9D6D0", "#F7F4EF"
# ---- models and validation schemes
MCOL = {"baseline": "#BDB8B0", "ridge": "#3F6E8C", "xgb": "#B86A4B"}
MLAB = {"baseline": "均值基线", "ridge": "Ridge", "xgb": "XGBoost"}
SCH = {"random": "随机 5 折", "group": "结构分组 5 折"}
# ---- chemical classes
CLS_COL = {"酯类": "#B35C44", "醇类": "#D9A441", "醛酮类": "#6F9E7A", "酸类": "#4C6A92", "缩醛类": "#8E7AA8",
           "萜类及降异戊二烯类": "#C77E9F", "呋喃类": "#5E9EA0", "芳香烃类": "#7F7F7F", "烷烃及烯烃类": "#B7C4CF",
           "其他类": "#A89F91"}
CLS_SHORT = {"萜类及降异戊二烯类": "萜类/降异戊二烯", "烷烃及烯烃类": "烷烃/烯烃"}
# ---- receptors
REC = ["OR1A1", "OR1A2", "OR1D2", "OR1G1", "OR2W1", "OR51E2"]
REC_COL = {"OR1A1": "#3F5A78", "OR1A2": "#B35C44", "OR1D2": "#6F9E7A", "OR1G1": "#D9A441", "OR2W1": "#8E7AA8",
           "OR51E2": "#5E9EA0"}
# ---- fruits (co-fermented wines)
FRUIT = ["山楂", "梨", "石榴"]
FRUIT_COL = {"山楂": "#A23B3B", "梨": "#C9A227", "石榴": "#6B4C8A"}
FRUIT_MK = {"山楂": "o", "梨": "s", "石榴": "^"}
# ---- cut scenarios
SCN_COL = {"前置富集": "#A64B3C", "均匀": "#3F6E8C", "后置富集": "#6F9E7A"}

SEQ = mcolors.LinearSegmentedColormap.from_list("seq6", ["#FBF7F0", "#F3DDBF", "#E3A97C", "#C2664B", "#8A3340", "#4A1D2A"])
SEQB = mcolors.LinearSegmentedColormap.from_list("seqb6", ["#F7F9FB", "#D5E1EA", "#9FB8CC", "#5F86A6", "#34536F", "#1D3047"])
DIV = mcolors.LinearSegmentedColormap.from_list("div6", ["#34536F", "#7F9DB6", "#DCE4EA", "#F7F4EF", "#F0D2C1", "#C77A5C", "#8A3340"])


def fmt(v, nd=2):
    """round half up (as in the text) and print a typographic minus"""
    from decimal import ROUND_HALF_UP, Decimal
    if v != v:
        return "—"
    q = Decimal(1).scaleb(-nd)
    return str(Decimal(str(float(v))).quantize(q, rounding=ROUND_HALF_UP)).replace("-", "−")


def frame(ax, lw=0.7):
    for s in ax.spines.values():
        s.set_linewidth(lw); s.set_color(INK)
    ax.tick_params(width=lw, length=2.6, color=INK)


def soft(ax, grid="y"):
    """closed thin frame; grid lines are omitted (they ran through annotations in the collision audit)"""
    frame(ax)
    ax.set_axisbelow(True)


_leg0 = leg


def leg(ax, **kw):
    """axes legend; a frameless legend gets an opaque white, borderless patch so that data never show through the text"""
    if kw.get("frameon") is False:
        kw["frameon"] = True; kw["edgecolor"] = "none"; kw.setdefault("facecolor", "white"); kw["framealpha"] = 0.92
        kw.setdefault("borderpad", 0.25)
    return _leg0(ax, **kw)


_letter0 = G.Canvas.letter


def _letter(self, x, y, s):
    """panel letters 0.9 mm below / 0.4 mm right of the cell corner so that no glyph touches the page edge"""
    return _letter0(self, x + 0.4, y + 0.9, s)


G.Canvas.letter = _letter


def save(cv, name, out=OUT):
    os.makedirs(out, exist_ok=True)
    return cv.save(out, name)


def mk(marker, color, label, ms=4.0, mec="white", **kw):
    kw.setdefault("ls", "none")
    return plt.Line2D([], [], marker=marker, color=color, mec=mec, mew=0.4, ms=ms, label=label, **kw)


def patch(color, label, ec="none", **kw):
    import matplotlib.patches as mpatches
    return mpatches.Patch(fc=color, ec=ec, lw=0.4, label=label, **kw)


def hline(ax, y, **kw):
    kw.setdefault("color", GREY); kw.setdefault("lw", 0.6); kw.setdefault("ls", (0, (3, 2))); kw.setdefault("zorder", 1)
    ax.axhline(y, **kw)


def vline(ax, x, **kw):
    kw.setdefault("color", GREY); kw.setdefault("lw", 0.6); kw.setdefault("ls", (0, (3, 2))); kw.setdefault("zorder", 1)
    ax.axvline(x, **kw)


def ptitle(ax, s, **kw):
    kw.setdefault("fontsize", FS["title"]); kw.setdefault("fontweight", "bold"); kw.setdefault("pad", 3)
    ax.set_title(s, **kw)
