#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

python src/feature_engineering.py
python src/descriptive_statistics.py
python src/rq1_mediation_moderation.py
python src/hypothesis_tests_rq1_rq2_rq3.py

echo "Pipeline completed. Review outputs/ for results."
