#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python code-spd/reports/report_compare_spd.py \
  -f code-spd/reports/compare_spd_nystrom_cubic_arc_newton_phase_probe_nogradsketch_timing_d40_subset_l20_l80_l320.json

python code-spd/reports/report_compare_spd.py \
  -f code-spd/reports/compare_spd_nystrom_cubic_arc_newton_phase_probe_nogradsketch_hesserr_d40_subset_l20_l80_l320.json

echo "SPD outputs are under reports-spd/spd_nystrom_cubic_arc_newton_phase_probe_nogradsketch_d40_subset_l20_l80_l320"
