#!/usr/bin/env bash
# Regenerates every manuscript figure/table/source-data/manifest from frozen result files. Plotting only; no experiment is run.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PY="${PY:-$HOME/miniconda3/envs/cpgpt/bin/python}"
N="nice -n 19"
for s in make_fig1 make_fig2 make_fig3 make_fig4 make_fig5_scaffold make_tables make_supp; do $N "$PY" paper/scripts/$s.py; done
$N "$PY" paper/scripts/make_qa.py
