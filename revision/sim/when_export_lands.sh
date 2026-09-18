#!/bin/bash
# when_export_lands.sh -- everything the simulation study does, in one command,
# the moment script 50's export is on disk.
#
#   bash revision/sim/when_export_lands.sh <output_50 dir> [outdir]
#
# 1 predictive validation of every design on REAL data (no MONA needed)
# 2 acceptance tests, now calibrated: they must pass before any ranking counts
# 3 the simulation study itself
# 4 the ranking, written to <outdir>/ranking.txt
set -euo pipefail
EXPORT="${1:?usage: when_export_lands.sh <output_50 dir> [outdir]}"
OUT="${2:-$(dirname "$0")/results}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT"

echo "=== 1. predictive validation on real data ==============================="
python3 "$HERE/../local/l13_validate_edu_designs.py" "$EXPORT" | tee "$OUT/validation.txt"

echo; echo "=== 2. acceptance tests, calibrated ====================================="
if ! python3 "$HERE/test_sim.py" --export "$EXPORT" | tee "$OUT/acceptance.txt"; then
    echo "ACCEPTANCE TESTS FAILED -- the ranking below is NOT to be used." | tee -a "$OUT/acceptance.txt"
fi

echo; echo "=== 3. simulation study ================================================="
python3 "$HERE/run_sim.py" --export "$EXPORT" --full --seeds 5 --out "$OUT" --resume \
    2>&1 | tee "$OUT/run.log"

echo; echo "=== 4. ranking =========================================================="
cat "$OUT/ranking.txt"
