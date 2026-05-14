#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

scripts/run_hdm05_table.sh
scripts/run_spd_figure.sh
scripts/run_grassmann_figure.sh
