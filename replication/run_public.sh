#!/usr/bin/env bash
# run_public.sh: every result of the paper that rests on public data, in order.
#
#   bash run_public.sh              # packs 1, 2 and 5, then the exhibits of pack 4
#   bash run_public.sh --no-download   # the archives are already in place
#   bash run_public.sh --cold          # recompute the HonestDiD bounds (hours)
#
# Settings, all optional (see config.py):
#   CANARIES_JOBADS_DIR  the Platsbanken archives (default data/raw/platsbanken)
#   CANARIES_FIRM_CUBE   the employer-by-month advertisement counts of Part V
#   CANARIES_SCB_BULK    Statistics Sweden's business-register bulk file
#   PYTHON, RSCRIPT      the interpreters (default python3, Rscript)
#
# Each step prints its wall-clock time; a failing step stops the run. The
# measured times on an Apple M2 laptop with 16 GB of memory are in the README.

set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
RSCRIPT="${RSCRIPT:-Rscript}"
DOWNLOAD=1
COLD=0
for a in "$@"; do
  case "$a" in
    --no-download) DOWNLOAD=0 ;;
    --cold) COLD=1 ;;
    *) echo "unknown option $a"; exit 2 ;;
  esac
done

step() {
  local t0=$SECONDS
  echo "=== $*"
  "$@"
  echo "    done in $((SECONDS - t0)) s"
}

# -- Pack 1: public data ------------------------------------------------------
if [ "$DOWNLOAD" = 1 ]; then
  step "$PYTHON" 1_data_public/01_download_platsbanken.py
fi
step "$PYTHON" 1_data_public/01_download_platsbanken.py --verify || \
  echo "    (an archive differs from the one the paper used; see the README)"
step "$PYTHON" 1_data_public/02_process_platsbanken.py
step "$PYTHON" 1_data_public/03_market_and_policy_series.py
step "$PYTHON" 1_data_public/04_merge_and_classify.py

# -- Pack 2: the posting margin -------------------------------------------------
step "$PYTHON" 2_postings/01_postings_accounting.py
step "$PYTHON" 2_postings/02_coverage_diagnostics.py
step "$PYTHON" 2_postings/03_extend_2026_and_did.py
step "$PYTHON" 2_postings/04_decile_gradient.py
step "$PYTHON" 2_postings/05_poisson_estimators.py
step "$PYTHON" 2_postings/06_seasonality.py
step "$PYTHON" 2_postings/07_event_study.py
if [ "$COLD" = 0 ] && [ ! -f output/results/posting_rr_honestdid_v3_cache.csv ]; then
  cp 2_postings/cache/posting_rr_honestdid_v3_cache.csv output/results/
fi
step "$RSCRIPT" 2_postings/08_honestdid.R
step "$PYTHON" 2_postings/09_honestdid_figure.py
step "$PYTHON" 2_postings/10_summary_statistics.py
step "$PYTHON" 2_postings/11_rate_sensitivity.py
step "$PYTHON" 2_postings/12_telework_split.py
step "$PYTHON" 2_postings/13_top_bottom_occupations.py
step "$PYTHON" 2_postings/14_within_employer.py
step "$PYTHON" 2_postings/14_within_employer.py --variants
step "$PYTHON" 2_postings/15_within_employer_heterogeneity.py
step "$PYTHON" 2_postings/16_appendix_tables.py
step "$PYTHON" 2_postings/17_accounting_to_june_2026.py
step "$PYTHON" 2_postings/18_figures.py
step "$PYTHON" 2_postings/19_telework_split_extended.py
step "$PYTHON" 2_postings/20_eloundou_postings.py
step "$PYTHON" 2_postings/21_posting_robustness.py
step "$PYTHON" 2_postings/22_tab_remote_measures.py
step "$PYTHON" 2_postings/23_fig_posting_coverage_monthly.py

# -- Pack 5: published occupational statistics ----------------------------------
step "$PYTHON" 5_occupation_register_public/01_published_age_gap.py
step "$PYTHON" 5_occupation_register_public/02_occupation_mix_by_sex.py

# -- Pack 4: exhibits built from the exported register aggregates ---------------
step "$PYTHON" 4_exhibits/run_all.py

# -- Pack 0: verification ---------------------------------------------------------
step "$PYTHON" 0_verification/check_mona_scripts.py
step "$PYTHON" 0_verification/check_manifest.py --quiet || \
  echo "    (see VERIFICATION.md for the rows known to differ from print)"
