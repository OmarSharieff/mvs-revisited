#!/usr/bin/env bash
set -euo pipefail

# Go to the project folder.
cd -- "$(dirname -- "$0")"

# Run the tests and all six cube experiments.
python -m pytest tests -q
python code/main.py --suite --size 48

# Get the real photos and rebuild the temple.
python code/download.py
python code/main.py \
    --data data/templeSparseRing \
    --box -0.073568 0.021728 -0.012445 0.028855 0.181892 0.062736

# Build the report from saved results.
python code/report.py
