"""Repository paths.  Every script imports this module, so the analysis runs from any clone location."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DERIVED = ROOT / "data" / "derived"
RESULTS = ROOT / "results"
INTER = RESULTS / "intermediate"
FIGS = RESULTS / "figures"
TABLES = RESULTS / "tables"
LEGACY = RESULTS / "legacy_figures"
HELPERS = ROOT / "src" / "helpers"
for p in (INTER, FIGS, TABLES):
    p.mkdir(parents=True, exist_ok=True)
if str(HELPERS) not in sys.path:
    sys.path.insert(0, str(HELPERS))
os.environ.setdefault("MPLBACKEND", "Agg")
