#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python -m py_compile run_lab.py test_lab.py
python run_lab.py
python -m unittest -v
