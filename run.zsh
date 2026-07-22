#!/usr/bin/env zsh
set -euo pipefail
cd "${0:A:h}"
python -m py_compile run_lab.py test_lab.py
python run_lab.py
python -m unittest -v
