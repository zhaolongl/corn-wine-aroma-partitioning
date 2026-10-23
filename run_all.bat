@echo off
rem Run the full analysis pipeline from the repository root.
set PY=python
%PY% src\core.py || exit /b 1
%PY% src\models.py || exit /b 1
%PY% srcxtra.py || exit /b 1
%PY% src\dock.py || exit /b 1
%PY% src\cut.py || exit /b 1
%PY% srcigures_main_a.py || exit /b 1
%PY% srcigures_main_b.py || exit /b 1
%PY% srcigures_supp.py || exit /b 1
%PY% src\supp_tables.py || exit /b 1
%PY% srcigure_data.py || exit /b 1
%PY% src\key_numbers.py > results	ables\key_numbers.txt || exit /b 1
echo done
