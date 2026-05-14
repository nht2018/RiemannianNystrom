#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python code-grassmann/reports/report_compare_grassmann.py \
  -f code-grassmann/reports/compare_grassmann_transport_t2_n20000_l20.json

echo "Grassmann outputs are under reports-grassmann/grassmann_transport_t2_n20000_l20"
