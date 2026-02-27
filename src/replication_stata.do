/*
  replication_stata.do — Cross-language replication of core DiD results

  Purpose: Independently verify Python (linearmodels/statsmodels) results
           using Stata, following Cunningham (CC 24/26) protocol.

  Target results to verify:
    Spec 1: beta1 = -0.178***
    Spec 2: beta1 = -0.127***, beta2 = -0.062 (p=0.11)
    Spec 3: beta1 = -0.068,    beta2 = 0.018
    Spec 4: beta1 = -0.039,    beta2 = -0.032
    Telework split: beta2 = -0.005 (teleworkable), -0.233*** (non-teleworkable)

  Author: Claude Code (replication agent)
  Date: 2026-02-27
*/

clear all
set more off

// ── Paths ─────────────────────────────────────────────────────────────────
local projdir "/Users/mslk/Documents/Workspace/papers/canaries-sweden"
local datadir "`projdir'/data/processed"
local tabdir  "`projdir'/tables"

// ── Check reghdfe, install if needed ──────────────────────────────────────
capture which reghdfe
local has_reghdfe = (_rc == 0)
if !`has_reghdfe' {
    display "reghdfe not found. Attempting install..."
    capture ssc install ftools, replace
    capture ssc install reghdfe, replace
    capture which reghdfe
    local has_reghdfe = (_rc == 0)
}
if `has_reghdfe' {
    display "Using reghdfe for FE estimation"
}
else {
    display "reghdfe unavailable — falling back to areg"
}

// ══════════════════════════════════════════════════════════════════════════
// PART 1: Main DiD Replication (Specs 1-4)
// ══════════════════════════════════════════════════════════════════════════

// Load pre-computed panel (same data as Python)
import delimited "`datadir'/did_panel.csv", clear varnames(1)

// ssyk4 imports as numeric (e.g. 1112). We need a numeric ID for FE.
// It's already numeric, so just use it directly.
describe ssyk4
count
display "Expected: 26,672 observations"

// Create numeric month ID from the string year_month
encode year_month, gen(ym_id)

// Create group × time ID for Spec 4
// ssyk1 is already in the data as a numeric variable (first digit of ssyk4)
// group_time is already a string like "1_2020-01"
encode group_time, gen(gt_id)

// ── Specification 1: Monetary policy only ─────────────────────────────────
display _newline "========================================================================"
display "SPEC 1: Monetary policy interaction only"
display "Python target: beta1 = -0.178***"
display "========================================================================"

if `has_reghdfe' {
    reghdfe ln_ads post_rb_x_high, absorb(ssyk4 ym_id) vce(cluster ssyk4)
}
else {
    areg ln_ads post_rb_x_high i.ym_id, absorb(ssyk4) vce(cluster ssyk4)
}

// ── Specification 2: + ChatGPT ────────────────────────────────────────────
display _newline "========================================================================"
display "SPEC 2: + ChatGPT interaction"
display "Python target: beta1 = -0.127***, beta2 = -0.062 (p=0.11)"
display "========================================================================"

if `has_reghdfe' {
    reghdfe ln_ads post_rb_x_high post_gpt_x_high, absorb(ssyk4 ym_id) vce(cluster ssyk4)
}
else {
    areg ln_ads post_rb_x_high post_gpt_x_high i.ym_id, absorb(ssyk4) vce(cluster ssyk4)
}

// ── Specification 3: + occupation-specific trends ─────────────────────────
display _newline "========================================================================"
display "SPEC 3: + occupation-specific time trends"
display "Python target: beta1 = -0.068, beta2 = 0.018, trend = -0.003**"
display "========================================================================"

if `has_reghdfe' {
    reghdfe ln_ads post_rb_x_high post_gpt_x_high time_x_high, absorb(ssyk4 ym_id) vce(cluster ssyk4)
}
else {
    areg ln_ads post_rb_x_high post_gpt_x_high time_x_high i.ym_id, absorb(ssyk4) vce(cluster ssyk4)
}

// ── Specification 4: SSYK 1-digit × month FE ─────────────────────────────
display _newline "========================================================================"
display "SPEC 4: Occupation group x month FE"
display "Python target: beta1 = -0.039, beta2 = -0.032"
display "========================================================================"

if `has_reghdfe' {
    reghdfe ln_ads post_rb_x_high post_gpt_x_high, absorb(ssyk4 gt_id) vce(cluster ssyk4)
}
else {
    areg ln_ads post_rb_x_high post_gpt_x_high i.gt_id, absorb(ssyk4) vce(cluster ssyk4)
}


// ══════════════════════════════════════════════════════════════════════════
// PART 2: Teleworkability Split Replication
// ══════════════════════════════════════════════════════════════════════════

display _newline "========================================================================"
display "TELEWORKABILITY SPLIT (Dingel-Neiman 2020)"
display "========================================================================"

// Step 1: Load telework mapping and save as .dta
import delimited "`tabdir'/telework_ssyk_mapping.csv", clear varnames(1)
// ssyk4 here is numeric (e.g. 1112)
rename ssyk4 ssyk4_tw
tempfile tw
save `tw'

// Step 2: Load main merged data
import delimited "`datadir'/postings_daioe_merged.csv", clear varnames(1)

// ssyk4 is numeric in both files — merge directly
rename ssyk4 ssyk4_tw
merge m:1 ssyk4_tw using `tw', keep(match) nogen
rename ssyk4_tw ssyk4

// Create variables
gen date_num = date(year_month + "-01", "YMD")
format date_num %td

gen ln_ads = ln(n_ads + 1)
gen high = (exposure_quartile == "Q4 (highest)")

gen post_rb = (date_num >= date("2022-04-01", "YMD"))
gen post_gpt = (date_num >= date("2022-12-01", "YMD"))
gen rb_high = post_rb * high
gen gpt_high = post_gpt * high

// Create FE variables
encode year_month, gen(ym_id)

// Split at median teleworkability (computed across occupation-level means)
bysort ssyk4: egen tw_occ = mean(teleworkable)
qui summ tw_occ, detail
local med_tw = r(p50)
display "Median teleworkability: `med_tw'"
gen is_telework = (tw_occ >= `med_tw')

// ── Teleworkable occupations ──────────────────────────────────────────────
display _newline "========================================================================"
display "TELEWORKABLE occupations"
display "Python target: beta2 = -0.005 (p=0.917)"
display "========================================================================"

if `has_reghdfe' {
    reghdfe ln_ads rb_high gpt_high if is_telework == 1, absorb(ssyk4 ym_id) vce(cluster ssyk4)
}
else {
    areg ln_ads rb_high gpt_high i.ym_id if is_telework == 1, absorb(ssyk4) vce(cluster ssyk4)
}

// ── Non-teleworkable occupations ──────────────────────────────────────────
display _newline "========================================================================"
display "NON-TELEWORKABLE occupations"
display "Python target: beta2 = -0.233*** (p=0.001)"
display "========================================================================"

if `has_reghdfe' {
    reghdfe ln_ads rb_high gpt_high if is_telework == 0, absorb(ssyk4 ym_id) vce(cluster ssyk4)
}
else {
    areg ln_ads rb_high gpt_high i.ym_id if is_telework == 0, absorb(ssyk4) vce(cluster ssyk4)
}

// ── All occupations (pooled) ──────────────────────────────────────────────
display _newline "========================================================================"
display "ALL occupations (pooled, with telework merge)"
display "Python target: beta2 = -0.062 (p=0.082)"
display "========================================================================"

if `has_reghdfe' {
    reghdfe ln_ads rb_high gpt_high, absorb(ssyk4 ym_id) vce(cluster ssyk4)
}
else {
    areg ln_ads rb_high gpt_high i.ym_id, absorb(ssyk4) vce(cluster ssyk4)
}

// ══════════════════════════════════════════════════════════════════════════
// SUMMARY
// ══════════════════════════════════════════════════════════════════════════
display _newline "========================================================================"
display "REPLICATION COMPLETE"
display "Compare Stata output above with Python targets."
display "Differences of <0.005 in point estimates and <0.002 in SEs"
display "are acceptable (floating-point / solver differences)."
display "Any larger discrepancy = potential bug in one implementation."
display "========================================================================"
