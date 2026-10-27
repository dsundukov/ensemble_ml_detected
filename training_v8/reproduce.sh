#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
if [[ ! -x .venv/bin/python ]]; then
    "$PY" -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip -q
if [[ "${LOCK:-1}" == "1" ]]; then
    python -m pip install -r requirements-lock.txt -q || python -m pip install -r requirements.txt -q
else
    python -m pip install -r requirements.txt -q
fi
cd code
python check_env.py
python train.py --data-dir ../data \
    --threshold-data ../data/ru_promptsc_threshold.csv \
    --eval-data ../data/ru_promptsc_eval.csv \
    --output ../artifacts_repro 2>&1 | tee ../train_log_repro.txt
cd ..
python compare_runs.py --expected expected --actual artifacts_repro
