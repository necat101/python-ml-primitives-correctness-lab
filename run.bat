@echo off
setlocal
cd /d "%~dp0"
python -m py_compile run_lab.py test_lab.py
if errorlevel 1 exit /b %errorlevel%
python run_lab.py
if errorlevel 1 exit /b %errorlevel%
python -m unittest -v
