#!/usr/bin/env bash
# Run the full analysis pipeline from the repository root.
set -euo pipefail
PY=${PY:-python}
$PY src/core.py
$PY src/models.py
$PY src/extra.py
$PY src/dock.py
$PY src/cut.py
$PY src/figures_main_a.py
$PY src/figures_main_b.py
$PY src/figures_supp.py
$PY src/supp_tables.py
$PY src/figure_data.py
$PY src/key_numbers.py > results/tables/key_numbers.txt
echo done
