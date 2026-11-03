@echo off
rem Run all detconf stages in order, then rebuild results\summary.md.
rem Uses cached predictions/attack images where their metadata matches; add --force to a
rem stage below to recompute it. Expects the venv from the README (.venv).
setlocal
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe

%PY% run.py --stage baseline || exit /b 1
%PY% run.py --stage shift || exit /b 1
%PY% run.py --stage reliability || exit /b 1
%PY% run.py --stage attack || exit /b 1
%PY% summarize.py || exit /b 1
echo All stages completed.
