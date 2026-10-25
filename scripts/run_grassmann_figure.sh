#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

EXPERIMENT_PYTHON="${EXPERIMENT_PYTHON:-python}"
GRASSMANN_RESULTS_DIR="${GRASSMANN_RESULTS_DIR:-results/grassmann_dense_n1024}"
GRASSMANN_REPORT_DIR="${GRASSMANN_REPORT_DIR:-reports-grassmann/dense_action_periods_n1024}"
GRASSMANN_PLAN="experiments/transport_revision/plans/dense_action_periods_n1024.json"

if [[ -e "$GRASSMANN_RESULTS_DIR" || -e "$GRASSMANN_REPORT_DIR" ]]; then
  echo "Choose new GRASSMANN_RESULTS_DIR and GRASSMANN_REPORT_DIR paths; existing outputs are never overwritten." >&2
  exit 1
fi

"$EXPERIMENT_PYTHON" experiments/transport_revision/dense_action_cache/run_action_cache_experiments.py \
  --plan "$GRASSMANN_PLAN" --out "$GRASSMANN_RESULTS_DIR"

"$EXPERIMENT_PYTHON" experiments/transport_revision/tools/report_dense_action_periods.py \
  --results "$GRASSMANN_RESULTS_DIR" --plan "$GRASSMANN_PLAN" --out "$GRASSMANN_REPORT_DIR"

echo "Grassmann outputs are under $GRASSMANN_REPORT_DIR"
