# Structure–partitioning relationships of aroma compounds in fresh corn wine

Code and data for the manuscript

> **Structure–partitioning relationships and cut-dependent retention of aroma compounds in fresh corn (*Zea mays* L.) wine** (submitted to *Journal of Advanced Research*).

The repository reproduces every computational result, main figure (Figs 1–7), supplementary figure (Figs S3, S7, S10–S14) and supplementary table (Tables S5, S7, S9–S17) of the manuscript from the deposited data. Experimental procedures are described in the manuscript; this repository contains no manuscript text.

中文说明见文末。

## Study design in brief

One batch of fresh waxy corn (cv. Jinheinuo 529) was processed along two branches: (i) co-fermentation of corn mash with hawthorn, pear or pomegranate juice (15 wines, 3 representative wines analysed for volatiles) and (ii) fermentation of the same corn mash alone followed by double pot-still distillation into five cut fractions (head, middle, second, sub-tail, tail). Volatiles were measured by HS-SPME-GC-MS and aligned by PubChem CID. The compound is the unit of analysis:

| Step | What is computed | Script |
|---|---|---|
| C0 | relative mass shares *p*, heart-fraction share *R* and distillation centroid *D* under two concentration bases (S0 tabulated, S1 dilution-not-corrected scenario) | `core.py` |
| C1–C2 | out-of-fold prediction of *R* and *D* (mean baseline, Ridge, shallow XGBoost; random and Butina-cluster-grouped nested CV; class leave-out; permutation test) | `models.py`, `extra.py` |
| C3 | feature-group ablation, acid-free and within-class analyses, coefficient bootstrap, TPSA rule | `models.py`, `extra.py` |
| C4 | heavy-atom-corrected Vina scores (31 ligands × 6 receptors × 3 seeds) versus structure and partitioning | `dock.py` |
| C5 | whole-fraction cut combinations and labelled within-head scenarios | `cut.py` |
| C6 | odour-activity-value threshold stress test | `cut.py` |

## Repository layout

```
data/raw/            measured or previously audited inputs (see "Data" below)
data/derived/        SMILES, Morgan-fingerprint Tanimoto matrix, Butina clusters (from fingerprints.py)
src/                 analysis and figure scripts; src/config.py holds all paths
src/helpers/         plotting layout helpers (millimetre grid, docking diagrams)
docking/             docking driver kept for provenance (needs AutoDock Vina 1.2.7, VINA_EXE)
results/intermediate model outputs (*.pkl, *.csv) written by the pipeline
results/figures/     Figs 1–7 and supplementary figures (PNG 600 dpi + PDF)
results/legacy_figures/  supplementary figures reused unchanged from earlier analyses (Figs S1, S2, S4–S6, S8, S9, S15)
results/tables/      Supplementary_Tables_S1-S18.xlsx and Figure_source_data.xlsx (live Excel formulas)
```

## Data

| File | Content |
|---|---|
| `data/raw/distillate_115_compounds.csv` | 115 volatiles of the five cut fractions (mg/L, pooled fraction samples), PubChem CID, class, descriptors, experimental boiling point, early/late label |
| `data/raw/distillation_figure_data_paper3.xlsx` | ABV of the 14 second-distillation cuts, fraction ethanol balance, class totals, odour thresholds (sheet 图8_OAV), process trials |
| `data/raw/cofermented_wine_compounds.csv` | 150 volatiles of the three representative co-fermented wines (μg/L, mean of n = 3), thresholds, OAV, descriptors, applicability-domain flags |
| `data/raw/cofermentation_paper2_*.pkl` | audited co-fermentation summaries (class totals, PCA scores, OPLS-DA VIP, aroma series, within-fruit correlations) used for Fig. 2 and Table S5 |
| `data/raw/docking/` | receptor/ligand PDBQT and MOL inputs, docking boxes, all Vina scores (`dock_results.csv`), PLIP interactions, 4-Å pocket residues, best poses (SDF) |
| `data/raw/docking_renders/` | pre-rendered OR1A2 pocket images and 2-D interaction layouts used in Fig. 6 and Fig. S12 |
| `data/raw/md/` | molecular-dynamics time series and summaries of four OR1A2 complexes (trajectories not deposited because of size) |
| `data/raw/legacy_supplementary_tables_v5.xlsx` | supplementary tables of earlier analyses reused in Tables S1–S4, S6, S8, S12, S15, S18 |

Units, column meanings and the exact denominators are given in the manuscript (Sections 2.7–2.13) and in the notes of each supplementary table. Concentrations of the fractions are tabulated values; whether they were back-corrected for the 10% vol dilution before GC-MS is unresolved, so every retention estimate is reported under both S0 and S1.

## Reproducing the results

1. Create the environment (Python 3.11):
   ```
   conda env create -f environment.yml
   conda activate corn-wine-aroma
   ```
   or `pip install -r requirements.txt`. RDKit is needed only for `fingerprints.py`; its outputs are already in `data/derived/`.
2. Run the pipeline from the repository root:
   ```
   run_all.bat          (Windows)
   bash run_all.sh      (Linux / macOS)
   ```
   `models.py` (nested cross-validation with XGBoost) takes about 10–20 min on 10 CPU cores; all other steps take seconds to a few minutes.
3. Optional: `python src/fingerprints.py --fetch` re-downloads SMILES from PubChem (only CIDs are sent) and rebuilds the similarity matrix and clusters.

Random seeds are fixed (2026). `figure_data.py` recalculates the Excel workbook with Microsoft Excel through COM when available (Windows) and compares every formula cell with the Python value; on other systems the workbook is written without that check.

Figures use Times New Roman for Latin text and a CJK font for Chinese labels (SimHei on Windows; Noto Sans CJK SC, Source Han Sans SC or PingFang SC are used as fallbacks).

## Mapping to the manuscript

| Manuscript item | Output |
|---|---|
| Fig. 1, 2, 3, 7 | `figures_main_a.py` → `results/figures/Fig1–Fig3, Fig7` |
| Fig. 4, 5, 6 | `figures_main_b.py` → `results/figures/Fig4–Fig6` |
| Figs S3, S7, S10–S14 (+ reused S1, S2, S4–S6, S8, S9, S15) | `figures_supp.py` |
| Table 2 and Tables S10, S11, S14 | `models.py`, `extra.py` → `results/intermediate/*.csv`, `supp_tables.py` |
| Table 3 and Table S17 | `cut.py`, `supp_tables.py` |
| Table S15 | `dock.py`, `supp_tables.py` |
| Numbers quoted in the text | `key_numbers.py` (prints every value used in the manuscript) |

## Licence

Code: MIT licence (see `LICENSE`). Data in `data/`: Creative Commons Attribution 4.0 International (CC BY 4.0). Please cite the article when using the data.

## Citation

See `CITATION.cff`. The article reference and DOI will be added after publication.

---

## 中文说明

本仓库提供论文《鲜食玉米酒香气物质的结构—分配关系与截取保留：分子结构可预测馏出位置，但不能预测酒心保留》全部计算分析的数据与代码，可复现正文图 1–7、附图与附表中的计算结果（不含论文正文）。运行方法：建立 `environment.yml` 环境后，在仓库根目录执行 `run_all.bat`（Windows）或 `bash run_all.sh`。馏分浓度的稀释回算口径尚待原始记录确认，因此保留率均同时给出 S0（表列浓度）与 S1（稀释未回算情景）两种结果。
