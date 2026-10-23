"""JAR merged paper, sixth version: figure-data workbook with live formulas.
图3: mass shares, R, D and the S1 scenario computed by formulas from the tabulated concentrations and fraction volumes;
图7: ethanol balance of the 14 cuts, whole-fraction combinations and head-cut scenarios by formulas;
图2: threshold stress counts by COUNTIF formulas;  图4-图6: model, ablation and docking values (from the Python outputs).
After writing, the workbook is recalculated in Excel (COM) and every formula cell is compared with the Python value.
Run: python figure_data.py <out.xlsx>"""
import os
import pickle
import sys

import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter as CL

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_distill as P3                                                                 # noqa: E402
import core as C                                                                 # noqa: E402

MOD = pickle.load(open(os.path.join(C.OUT, "models.pkl"), "rb"))
EXT = pickle.load(open(os.path.join(C.OUT, "extra.pkl"), "rb"))
DK = pickle.load(open(os.path.join(C.OUT, "dock.pkl"), "rb"))
CUT = pickle.load(open(os.path.join(C.OUT, "cut.pkl"), "rb"))
HF = PatternFill("solid", fgColor="E9E4DA")


def hdr(ws, r, cols, c0=1):
    for j, c in enumerate(cols):
        x = ws.cell(row=r, column=c0 + j, value=c); x.font = Font(bold=True); x.fill = HF


def frame(ws, r0, df):
    hdr(ws, r0, list(df.columns))
    for i, row in enumerate(df.itertuples(index=False), r0 + 1):
        for j, v in enumerate(row, 1):
            if isinstance(v, (np.floating, float)):
                v = None if v != v else float(v)
            elif isinstance(v, np.integer):
                v = int(v)
            elif isinstance(v, np.bool_):
                v = bool(v)
            ws.cell(row=i, column=j, value=v)


def build(path):
    wb = Workbook(); info = wb.active; info.title = "说明"
    checks = []                                     # (sheet, cell, python value)
    b = C.load()
    # ------------------------------------------------------------------ 图3 endpoints
    ws = wb.create_sheet("图3_连续端点")
    ws["A1"] = "图 3 与 Table S9：质量份额 p、酒心占比 R、馏出重心 D 及 S1 情景（公式）"; ws["A1"].font = Font(bold=True)
    ws["A2"] = "体积（mL）"; ws["A3"] = "区间中点/700"; ws["A4"] = "S1 稀释倍数"
    for j in range(5):
        col = 5 + j
        ws.cell(row=2, column=col, value=float(C.VOL[j])); ws.cell(row=3, column=col, value=float(C.TMID[j]))
        ws.cell(row=4, column=col, value=float(C.S1F[j]))
    cols = ["编号", "化合物", "类别", "主分析集"] + [f"{s}（mg/L）" for s in C.FRS] + [f"p_{s}" for s in C.FRS] + \
           ["R", "D", "R_S1", "D_S1"]
    hdr(ws, 6, cols)
    for i, r in enumerate(b.itertuples(index=False), 7):
        ws.cell(row=i, column=1, value=r.编号); ws.cell(row=i, column=2, value=r.采用名称); ws.cell(row=i, column=3, value=r.新类别)
        ws.cell(row=i, column=4, value=bool(r.主分析集))
        conc = b.loc[b["编号"] == r.编号, C.FR].values[0]
        for j in range(5):
            ws.cell(row=i, column=5 + j, value=float(conc[j]))
        rng_c = f"$E{i}:$I{i}"
        for j in range(5):
            cc = CL(5 + j)
            ws.cell(row=i, column=10 + j, value=f"={cc}{i}*{cc}$2/SUMPRODUCT({rng_c},$E$2:$I$2)")
        ws.cell(row=i, column=15, value=f"=K{i}+L{i}")
        ws.cell(row=i, column=16, value=f"=SUMPRODUCT(J{i}:N{i},$E$3:$I$3)")
        den = f"SUMPRODUCT({rng_c},$E$2:$I$2,$E$4:$I$4)"
        ws.cell(row=i, column=17, value=f"=(F{i}*F$2*F$4+G{i}*G$2*G$4)/{den}")
        ws.cell(row=i, column=18, value=f"=SUMPRODUCT({rng_c},$E$2:$I$2,$E$4:$I$4,$E$3:$I$3)/{den}")
        bi = b[b["编号"] == r.编号].iloc[0]
        checks += [("图3_连续端点", f"O{i}", bi["R"]), ("图3_连续端点", f"P{i}", bi["D"]), ("图3_连续端点", f"Q{i}", bi["R_S1"]),
                   ("图3_连续端点", f"R{i}", bi["D_S1"])]
    n_last = 6 + len(b)
    # ------------------------------------------------------------------ 图7 ethanol + combinations + head-cut scenarios
    ws = wb.create_sheet("图7_截取情景")
    ws["A1"] = "图 7、表 3 与 Table S17：14 段乙醇衡算、整馏分组合与头馏分内部情景（公式）"; ws["A1"].font = Font(bold=True)
    seg, cut = P3.fractions()
    hdr(ws, 3, ["段号", "体积（mL）", "酒精度（% vol）", "段乙醇（mL）", "所属馏分"])
    for i, r in enumerate(seg.itertuples(index=False), 4):
        ws.cell(row=i, column=1, value=int(r.段号)); ws.cell(row=i, column=2, value=50.0)
        ws.cell(row=i, column=3, value=float(r.酒精度)); ws.cell(row=i, column=4, value=f"=B{i}*C{i}/100")
        ws.cell(row=i, column=5, value=r.所属馏分)
    ws["G3"] = "乙醇总量（mL）"; ws["H3"] = "=SUM(D4:D17)"
    ws["G4"] = "头馏分乙醇"; ws["H4"] = '=SUMIF(E4:E17,"头馏分",D4:D17)'
    ws["G5"] = "中间馏分乙醇"; ws["H5"] = '=SUMIF(E4:E17,"中间馏分",D4:D17)'
    ws["G6"] = "次馏分乙醇"; ws["H6"] = '=SUMIF(E4:E17,"次馏分",D4:D17)'
    ws["G7"] = "次尾馏分乙醇"; ws["H7"] = '=SUMIF(E4:E17,"次尾馏分",D4:D17)'
    ws["G8"] = "尾馏分乙醇"; ws["H8"] = '=SUMIF(E4:E17,"尾馏分",D4:D17)'
    ws["G10"] = "掐头体积 h（mL）"; ws["H10"] = 14.0
    ws["G11"] = "乙醇回收率（头内均匀，%）"; ws["H11"] = "=(H4*(1-H10/50)+H5+H6)/H3*100"
    ws["G12"] = "情景参数 k（前置 3、后置 −3）"; ws["H12"] = 3.0; ws["I12"] = -3.0
    ws["G13"] = "头馏分部分保留（前置）"; ws["H13"] = "=(EXP(-H12*H10/50)-EXP(-H12))/(1-EXP(-H12))"
    ws["G14"] = "头馏分部分保留（均匀）"; ws["H14"] = "=1-H10/50"
    ws["G15"] = "头馏分部分保留（后置）"; ws["H15"] = "=(EXP(-I12*H10/50)-EXP(-I12))/(1-EXP(-I12))"
    et = CUT["ethanol"].set_index("h")
    checks += [("图7_截取情景", "H11", et.loc[14.0, "乙醇回收率"])]
    from cut import head_left
    checks += [("图7_截取情景", "H13", float(head_left(14 / 50, 3.0))), ("图7_截取情景", "H14", 1 - 14 / 50),
               ("图7_截取情景", "H15", float(head_left(14 / 50, -3.0)))]
    # combos for the 25 odorants: reference p_ cells in 图3
    od = CUT["odor"]
    hdr(ws, 19, ["化合物", "图3 行号", "中间", "中间+次", "头+中间+次", "中间+次+次尾", "头+中间+次+次尾", "掐头 2%（前置）",
                 "掐头 2%（均匀）", "掐头 2%（后置）"])
    rowmap = {cid: 7 + k for k, cid in enumerate(b["CID"])}
    for i, r in enumerate(od.itertuples(index=False), 20):
        rr = rowmap[r.CID]; ref = "'图3_连续端点'!"
        ws.cell(row=i, column=1, value=r.采用名称); ws.cell(row=i, column=2, value=rr)
        ws.cell(row=i, column=3, value=f"={ref}K{rr}*100")
        ws.cell(row=i, column=4, value=f"=({ref}K{rr}+{ref}L{rr})*100")
        ws.cell(row=i, column=5, value=f"=({ref}J{rr}+{ref}K{rr}+{ref}L{rr})*100")
        ws.cell(row=i, column=6, value=f"=({ref}K{rr}+{ref}L{rr}+{ref}M{rr})*100")
        ws.cell(row=i, column=7, value=f"=({ref}J{rr}+{ref}K{rr}+{ref}L{rr}+{ref}M{rr})*100")
        for j, cell in enumerate(["$H$13", "$H$14", "$H$15"]):
            ws.cell(row=i, column=8 + j, value=f"=({ref}J{rr}*{cell}+{ref}K{rr}+{ref}L{rr})*100")
        bi = b[b["CID"] == r.CID].iloc[0]
        checks += [("图7_截取情景", f"D{i}", bi["R"] * 100), ("图7_截取情景", f"I{i}", r.uni_14 * 100)]
    # ------------------------------------------------------------------ 图2 stress counts
    ws = wb.create_sheet("图2_阈值压力")
    ws["A1"] = "图 2E/F 与 Table S7：各化合物 OAVmax 与阈值倍数压力测试（COUNTIF 公式）"; ws["A1"].font = Font(bold=True)
    oa = CUT["oav"]
    blocks = [("共发酵酒", "—"), ("馏出液", "S0"), ("馏出液", "S1")]
    hdr(ws, 3, ["阈值倍数"] + [f"{a}{'' if c == '—' else '（' + c + '）'}" for a, c in blocks])
    facs = [0.25, 0.5, 1.0, 2.0, 4.0]
    col0 = 8
    for k, (sys_, basis) in enumerate(blocks):
        g = oa[(oa["体系"] == sys_) & (oa["浓度口径"] == basis)]
        c = col0 + 3 * k
        hdr(ws, 3, [f"{sys_}{basis if basis != '—' else ''}_化合物", "OAVmax"], c0=c)
        for i, r in enumerate(g.itertuples(index=False), 4):
            ws.cell(row=i, column=c, value=r.名称); ws.cell(row=i, column=c + 1, value=float(r.OAVmax))
        last = 3 + len(g); colL = CL(c + 1)
        for i, f in enumerate(facs, 4):
            ws.cell(row=i, column=1, value=f)
            ws.cell(row=i, column=2 + k, value=f'=COUNTIF({colL}4:{colL}{last},">="&A{i})')
            st = CUT["stress"]
            checks.append(("图2_阈值压力", f"{CL(2 + k)}{i}",
                           float(st[(st["体系"] == sys_) & (st["浓度口径"] == basis) & (st["阈值倍数"] == f)]["候选数"].iloc[0])))
    # ------------------------------------------------------------------ value sheets
    ws = wb.create_sheet("图4_预测性能"); ws["A1"] = "图 4、表 2 与 Table S10：模型性能（Python 输出）"; ws["A1"].font = Font(bold=True)
    frame(ws, 3, pd.concat([MOD["perf"], EXT["perf_extra"]], ignore_index=True))
    ws = wb.create_sheet("图5_消融"); ws["A1"] = "图 5 与 Table S14：特征组消融（Python 输出）"; ws["A1"].font = Font(bold=True)
    frame(ws, 3, pd.concat([MOD["abl"], EXT["nonacid_bp"]], ignore_index=True))
    ws = wb.create_sheet("图6_对接关联"); ws["A1"] = "图 6 与 Table S15：配体层面评分、校正评分与分配端点（Python 输出）"
    ws["A1"].font = Font(bold=True)
    frame(ws, 3, DK["lig"].drop(columns=[c for c in DK["lig"].columns if c.startswith("slope_") or c.startswith("sd_")]))
    info["A1"] = "论文合并稿第六版 图数据工作簿"; info["A1"].font = Font(bold=True, size=12)
    info["A3"] = "图3_连续端点、图7_截取情景与图2_阈值压力中的派生量均为 Excel 公式；其余工作表为 Python 输出值。"
    info["A4"] = "核对：本工作簿经 Excel 重算后，全部公式单元格与 Python 结果逐一比较，结果写入“核对”工作表。"
    wb.save(path)
    return checks


def recalc_and_check(path, checks):
    import win32com.client as w32
    xl = w32.DispatchEx("Excel.Application"); xl.Visible = False; xl.DisplayAlerts = False
    wbx = xl.Workbooks.Open(os.path.abspath(path)); xl.CalculateFull(); wbx.Save(); wbx.Close(); xl.Quit()
    wbv = load_workbook(path, data_only=True)
    bad = []
    for sh, cell, pv in checks:
        v = wbv[sh][cell].value
        if v is None or abs(float(v) - float(pv)) > 1e-9 * max(1, abs(pv)):
            bad.append((sh, cell, v, pv))
    nform = sum(1 for ws in load_workbook(path).worksheets for row in ws.iter_rows() for c in row
                if isinstance(c.value, str) and c.value.startswith("="))
    wb = load_workbook(path)
    ws = wb.create_sheet("核对")
    ws["A1"] = f"公式单元格 {nform} 个；与 Python 逐一比较 {len(checks)} 个关键单元格，不一致 {len(bad)} 个。"
    for i, (sh, cell, v, pv) in enumerate(bad, 3):
        ws.cell(row=i, column=1, value=sh); ws.cell(row=i, column=2, value=cell); ws.cell(row=i, column=3, value=v)
        ws.cell(row=i, column=4, value=pv)
    wb.save(path)
    # re-open in Excel once more so the cached values of the formula cells are stored after the openpyxl write
    xl = w32.DispatchEx("Excel.Application"); xl.Visible = False; xl.DisplayAlerts = False
    wbx = xl.Workbooks.Open(os.path.abspath(path)); xl.CalculateFull(); wbx.Save(); wbx.Close(); xl.Quit()
    return nform, len(checks), bad


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(C.CFG.TABLES, "Figure_source_data.xlsx")
    ch = build(out)
    try:
        nform, nchk, bad = recalc_and_check(out, ch)
        print(f"formulas {nform}; checked {nchk}; mismatches {len(bad)}", bad[:5])
    except ImportError:
        print("pywin32/Excel not available: workbook written without the Excel recalculation check")
