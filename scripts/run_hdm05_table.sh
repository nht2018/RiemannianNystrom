#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

HDM05_RAW_DIR="${HDM05_RAW_DIR:-data/raw/hdm05_extracted/hdm05}"
RESULTS_DIR="${RESULTS_DIR:-results}"

if [[ ! -d "$HDM05_RAW_DIR" ]]; then
  echo "HDM05 data directory not found: $HDM05_RAW_DIR" >&2
  echo "Place files named id_length_label.npy there, or set HDM05_RAW_DIR=/path/to/hdm05." >&2
  exit 1
fi

mkdir -p "$RESULTS_DIR"

python PGA/src/run_hdm05_downstream.py \
  --raw-dir "$HDM05_RAW_DIR" \
  --output-dir "$RESULTS_DIR/hdm05_table_logreg_pairwise_storage" \
  --ells 20,40,80 \
  --ks 20 \
  --seed 0 \
  --log-eps 1e-6 \
  --n-perm 0 \
  --clf-eval holdout \
  --test-size 0.2 \
  --clf-task multiclass \
  --min-class-count 4 \
  --compare-exact \
  --pairwise-t2 \
  --report-storage \
  --materialize-covariance

python PGA/src/run_hdm05_downstream.py \
  --raw-dir "$HDM05_RAW_DIR" \
  --output-dir "$RESULTS_DIR/hdm05_table_svm_tuned" \
  --ells 20,40,80 \
  --ks 20 \
  --seed 0 \
  --log-eps 1e-6 \
  --n-perm 0 \
  --clf-eval holdout \
  --test-size 0.2 \
  --clf-task multiclass \
  --min-class-count 4 \
  --compare-exact \
  --clf-model svm \
  --svm-tune \
  --svm-c-grid 0.1,1,10,100 \
  --svm-gamma-grid 0.001,0.01,0.1,1 \
  --svm-cv 3

python PGA/src/run_hdm05_downstream.py \
  --raw-dir "$HDM05_RAW_DIR" \
  --output-dir "$RESULTS_DIR/hdm05_table_mlp" \
  --ells 20,40,80 \
  --ks 20 \
  --seed 0 \
  --log-eps 1e-6 \
  --n-perm 0 \
  --clf-eval holdout \
  --test-size 0.2 \
  --clf-task multiclass \
  --min-class-count 4 \
  --compare-exact \
  --clf-model mlp \
  --mlp-hidden-layers 100,50 \
  --mlp-max-iter 2000 \
  --mlp-learning-rate-init 0.001 \
  --mlp-early-stopping

python scripts/summarize_hdm05_table.py --results-dir "$RESULTS_DIR"
