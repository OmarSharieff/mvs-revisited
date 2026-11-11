#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
python -m pytest tests -q
python code/main.py --suite --size 48
python code/download.py
python code/main.py --data data/templeSparseRing --box -0.073568 0.021728 -0.012445 0.028855 0.181892 0.062736
python code/report.py
