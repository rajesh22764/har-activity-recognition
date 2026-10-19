#!/usr/bin/env bash
# Reproduce every result and figure:  conda activate har-zs && ./run_all.sh
set -euo pipefail
cd "$(dirname "$0")/src"
for s in s01_eda_tests s02_dimensionality s03_models s04_deep s05_explain_fairness; do
  echo "=== $s ==="; python -W ignore "$s.py"
done
echo "Done. Launch the app with:  streamlit run app/streamlit_app.py"
