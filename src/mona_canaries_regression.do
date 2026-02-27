/*==============================================================================
  CANARIES REGRESSION — Brynjolfsson-style Employment DiD

  Paper:  "Two Economies? Stock Markets, Job Postings, and AI Exposure"
  Author: Lydia Löthman (to run on MONA)
  Date:   February 2026

  PURPOSE:
  Test whether young workers in AI-exposed occupations experienced
  disproportionate employment declines after ChatGPT launch.

  Mirrors Eq. 4.1 in Brynjolfsson, Chandar & Chen (2025):
    log(E[y_{f,q,t}]) = γ_{q,t} + α_{f,q} + β_{f,t} + ε_{f,q,t}

  where f = employer, q = DAIOE quartile, t = month.
  Employer×month FE absorb ALL firm-level macro shocks (including
  interest rate sensitivity), so identification is within-firm,
  within-month variation across AI exposure levels.

  Run SEPARATELY for each age group. The "canaries" finding is that
  γ coefficients diverge for young workers (16-24) but not older ones.

  INPUT FILES (on MONA):
    1. AGI individual records: person_id, employer_id, ssyk4, age, ym
    2. daioe_quartiles.csv: ssyk4, exposure_quartile (1-4)

  OUTPUT FILES (export from MONA):
    1. canaries_did_results.csv    — DiD coefficient table
    2. canaries_eventstudy_*.csv   — event study coefficients for plotting
    3. canaries_eventstudy_*.png   — event study figures (if graph works)
    4. canaries_summary.txt        — sample sizes, diagnostics
==============================================================================*/

clear all
set more off
set matsize 11000
cap log close
log using "canaries_regression.log", replace text

* ============================================================================
* SECTION 0: PATHS AND GLOBALS — EDIT THESE
* ============================================================================

* Path to AGI individual-level data (adjust to your MONA project)
global agi_data "FILL_IN_PATH/agi_monthly.dta"

* Path to DAIOE quartile file (import this CSV to MONA first)
global daioe_file "FILL_IN_PATH/daioe_quartiles.csv"

* Output directory
global outdir "FILL_IN_PATH/output"
cap mkdir "$outdir"

* Reference period for event study (October 2022 = month before ChatGPT)
global ref_ym = ym(2022, 10)

* Treatment dates
global riksbank_ym = ym(2022, 4)   /* Riksbank first rate hike */
global chatgpt_ym  = ym(2022, 12)  /* First full month after ChatGPT */

di "Reference period: $ref_ym (October 2022)"
di "Riksbank hike:   $riksbank_ym (April 2022)"
di "ChatGPT:         $chatgpt_ym (December 2022)"


* ============================================================================
* SECTION 1: PREPARE DATA
* ============================================================================

* --- 1a. Load DAIOE quartiles ---
import delimited "$daioe_file", clear varnames(1)
keep ssyk4 exposure_quartile
rename exposure_quartile daioe_q
* Ensure ssyk4 is string with leading zeros
tostring ssyk4, replace format(%04.0f)
tempfile daioe
save `daioe'


* --- 1b. Load AGI data and aggregate ---
use "$agi_data", clear

* IMPORTANT: Adjust variable names below to match your AGI extract.
* Expected variables: person_id, employer_id, ssyk4, age (or birth_year), ym
* If you have birth_year instead of age:
*   gen age = year(dofm(ym)) - birth_year
* If ym is not a Stata monthly date:
*   gen ym = mofd(date_variable)
*   format ym %tm

* Ensure ssyk4 is 4-digit string
tostring ssyk4, replace format(%04.0f)

* Filter ages 16-69
keep if age >= 16 & age <= 69

* Create age group variable
gen age_group = .
replace age_group = 1 if age >= 16 & age <= 24
replace age_group = 2 if age >= 25 & age <= 30
replace age_group = 3 if age >= 31 & age <= 40
replace age_group = 4 if age >= 41 & age <= 49
replace age_group = 5 if age >= 50 & age <= 69

label define age_lbl 1 "16-24" 2 "25-30" 3 "31-40" 4 "41-49" 5 "50+"
label values age_group age_lbl


* --- 1c. Merge DAIOE quartiles ---
merge m:1 ssyk4 using `daioe', keep(match) nogen
* Report match rate
di "Matched observations: " _N


* --- 1d. Aggregate to employer × DAIOE quartile × age group × month ---
* This is the unit of analysis (mirrors Brynjolfsson's firm × quintile × month)
collapse (count) n_emp = person_id, by(employer_id daioe_q age_group ym)

* Create panel identifier: employer × quartile × age group
egen panelid = group(employer_id daioe_q age_group)

* Report sample sizes
di "=== SAMPLE SUMMARY ==="
di "Observations (cells): " _N
distinct employer_id
di "Unique employers: " r(ndistinct)
distinct panelid
di "Panel units (employer × quartile × age): " r(ndistinct)
tab age_group, m
tab daioe_q, m
sum n_emp, d


* --- 1e. Create treatment variables ---
gen post_rb  = (ym >= $riksbank_ym) if ym != .
gen post_gpt = (ym >= $chatgpt_ym) if ym != .
gen high_ai  = (daioe_q == 4) if daioe_q != .

* Interactions for simple DiD
gen post_rb_high  = post_rb * high_ai
gen post_gpt_high = post_gpt * high_ai

* For event study: half-year period dummies (Erik's suggestion)
gen half_year = .
replace half_year = 1  if ym >= ym(2019,1) & ym <= ym(2019,6)
replace half_year = 2  if ym >= ym(2019,7) & ym <= ym(2019,12)
replace half_year = 3  if ym >= ym(2020,1) & ym <= ym(2020,6)
replace half_year = 4  if ym >= ym(2020,7) & ym <= ym(2020,12)
replace half_year = 5  if ym >= ym(2021,1) & ym <= ym(2021,6)
replace half_year = 6  if ym >= ym(2021,7) & ym <= ym(2021,12)
replace half_year = 7  if ym >= ym(2022,1) & ym <= ym(2022,6)   /* contains RB hike */
replace half_year = 8  if ym >= ym(2022,7) & ym <= ym(2022,12)  /* contains ChatGPT */
replace half_year = 9  if ym >= ym(2023,1) & ym <= ym(2023,6)
replace half_year = 10 if ym >= ym(2023,7) & ym <= ym(2023,12)
replace half_year = 11 if ym >= ym(2024,1) & ym <= ym(2024,6)
replace half_year = 12 if ym >= ym(2024,7) & ym <= ym(2024,12)
replace half_year = 13 if ym >= ym(2025,1) & ym <= ym(2025,6)
replace half_year = 14 if ym >= ym(2025,7) & ym <= ym(2025,12)

label define hy_lbl 1 "2019H1" 2 "2019H2" 3 "2020H1" 4 "2020H2" ///
    5 "2021H1" 6 "2021H2" 7 "2022H1" 8 "2022H2" ///
    9 "2023H1" 10 "2023H2" 11 "2024H1" 12 "2024H2" ///
    13 "2025H1" 14 "2025H2"
label values half_year hy_lbl


* FE identifiers for reghdfe/ppmlhdfe
egen fe_emp_q   = group(employer_id daioe_q)   /* employer × quartile FE */
egen fe_emp_t   = group(employer_id ym)        /* employer × month FE */

* Log employment for OLS fallback
gen ln_emp = ln(n_emp)

* Save analysis dataset
compress
save "$outdir/canaries_analysis.dta", replace


* ============================================================================
* SECTION 2: MAIN DiD — SEPARATE BY AGE GROUP
* ============================================================================

* This mirrors Brynjolfsson Eq 4.1, but with a simple DiD instead of
* full event study. Two post-period interactions: PostRB×High and PostGPT×High.
*
* KEY: employer×month FE absorb all firm-level macro shocks.
* Identification: within same employer, in same month, do workers in
* high-AI-exposure occupations decline more than low-AI workers?

di ""
di "============================================================"
di "  MAIN DiD REGRESSIONS — BY AGE GROUP"
di "============================================================"

* Store results
tempname results
postfile `results' age_grp str20(variable) coef se pval n_obs ///
    using "$outdir/canaries_did_results.dta", replace

forvalues g = 1/5 {

    local glabel : label age_lbl `g'
    di ""
    di "--- Age group: `glabel' ---"

    * --- Specification A: Poisson with employer×quartile + employer×month FE ---
    * (Closest to Brynjolfsson)
    cap {
        ppmlhdfe n_emp post_rb_high post_gpt_high if age_group == `g', ///
            absorb(fe_emp_q fe_emp_t) cluster(employer_id)

        * Store PostRB × High
        local b1 = _b[post_rb_high]
        local s1 = _se[post_rb_high]
        local p1 = 2*normal(-abs(`b1'/`s1'))
        local n1 = e(N)
        post `results' (`g') ("PostRB_x_High") (`b1') (`s1') (`p1') (`n1')

        * Store PostGPT × High
        local b2 = _b[post_gpt_high]
        local s2 = _se[post_gpt_high]
        local p2 = 2*normal(-abs(`b2'/`s2'))
        post `results' (`g') ("PostGPT_x_High") (`b2') (`s2') (`p2') (`n1')

        di "  Poisson: PostRB×High  = " %7.4f `b1' " (p=" %5.3f `p1' ")"
        di "  Poisson: PostGPT×High = " %7.4f `b2' " (p=" %5.3f `p2' ")"
    }
    if _rc != 0 {
        di "  NOTE: ppmlhdfe failed (may not be installed). Using OLS fallback."

        * --- Specification B: OLS fallback with ln(employment) ---
        reghdfe ln_emp post_rb_high post_gpt_high if age_group == `g', ///
            absorb(fe_emp_q fe_emp_t) cluster(employer_id)

        local b1 = _b[post_rb_high]
        local s1 = _se[post_rb_high]
        local p1 = 2*ttail(e(df_r), abs(`b1'/`s1'))
        local n1 = e(N)
        post `results' (`g') ("PostRB_x_High") (`b1') (`s1') (`p1') (`n1')

        local b2 = _b[post_gpt_high]
        local s2 = _se[post_gpt_high]
        local p2 = 2*ttail(e(df_r), abs(`b2'/`s2'))
        post `results' (`g') ("PostGPT_x_High") (`b2') (`s2') (`p2') (`n1')

        di "  OLS: PostRB×High  = " %7.4f `b1' " (p=" %5.3f `p1' ")"
        di "  OLS: PostGPT×High = " %7.4f `b2' " (p=" %5.3f `p2' ")"
    }
}

postclose `results'

* Export results as CSV
use "$outdir/canaries_did_results.dta", clear
export delimited "$outdir/canaries_did_results.csv", replace

* Display summary table
list, clean noobs
use "$outdir/canaries_analysis.dta", clear


* ============================================================================
* SECTION 3: EVENT STUDY — HALF-YEAR PERIODS × QUARTILE
* ============================================================================

* Following Erik's suggestion: use half-year periods instead of monthly.
* Reference period: 2022H1 (half_year == 7), the last pre-hike period.
* This traces out the time path of the AI exposure differential.

di ""
di "============================================================"
di "  EVENT STUDY (HALF-YEAR) — BY AGE GROUP"
di "============================================================"

* Generate half-year × high-AI interactions (reference: 2022H1 = period 7)
forvalues h = 1/14 {
    if `h' != 7 {
        gen hy`h'_high = (half_year == `h') * high_ai
    }
}

forvalues g = 1/5 {

    local glabel : label age_lbl `g'
    di ""
    di "--- Event study: Age group `glabel' ---"

    * List of interaction terms (excluding reference period 7)
    local hvars ""
    forvalues h = 1/14 {
        if `h' != 7 {
            local hvars "`hvars' hy`h'_high"
        }
    }

    cap {
        ppmlhdfe n_emp `hvars' if age_group == `g', ///
            absorb(fe_emp_q fe_emp_t) cluster(employer_id)
    }
    if _rc != 0 {
        reghdfe ln_emp `hvars' if age_group == `g', ///
            absorb(fe_emp_q fe_emp_t) cluster(employer_id)
    }

    * Store coefficients for this age group
    preserve
    clear
    set obs 14
    gen half_year = _n
    gen coef = .
    gen se = .
    gen age_grp = `g'

    forvalues h = 1/14 {
        if `h' == 7 {
            replace coef = 0 if half_year == `h'
            replace se = 0 if half_year == `h'
        }
        else {
            replace coef = _b[hy`h'_high] if half_year == `h'
            replace se = _se[hy`h'_high] if half_year == `h'
        }
    }

    gen ci_lo = coef - 1.96*se
    gen ci_hi = coef + 1.96*se

    save "$outdir/canaries_es_age`g'.dta", replace
    export delimited "$outdir/canaries_es_age`g'.csv", replace

    * Print coefficients
    list half_year coef se ci_lo ci_hi, clean noobs

    restore
}

* --- Combine all age groups for plotting ---
clear
forvalues g = 1/5 {
    append using "$outdir/canaries_es_age`g'.dta"
}
save "$outdir/canaries_es_all.dta", replace
export delimited "$outdir/canaries_es_all.csv", replace


* ============================================================================
* SECTION 4: PLOT EVENT STUDIES (if graph capabilities available)
* ============================================================================

use "$outdir/canaries_es_all.dta", clear

* Labels for half-year periods
label define hy_lbl2 1 "19H1" 2 "19H2" 3 "20H1" 4 "20H2" ///
    5 "21H1" 6 "21H2" 7 "22H1" 8 "22H2" ///
    9 "23H1" 10 "23H2" 11 "24H1" 12 "24H2" ///
    13 "25H1" 14 "25H2"
label values half_year hy_lbl2

* --- Plot for ages 16-24 (the canaries) ---
twoway (rarea ci_lo ci_hi half_year if age_grp == 1, ///
            color(orange%20) lwidth(none)) ///
       (connected coef half_year if age_grp == 1, ///
            mcolor(orange) lcolor(orange) msymbol(circle)), ///
    xline(7.5, lcolor(cranberry) lpattern(dash) lwidth(thin)) ///
    xline(8.5, lcolor(navy) lpattern(dash) lwidth(thin)) ///
    yline(0, lcolor(gs10) lpattern(solid)) ///
    xlabel(1(1)14, valuelabel angle(45) labsize(small)) ///
    ylabel(, format(%5.2f)) ///
    title("Age 16-24: Employment by AI exposure (event study)") ///
    subtitle("Coefficient on High-AI × half-year (ref: 2022H1)") ///
    note("Dashed lines: Riksbank hike (Apr 2022) and ChatGPT (Dec 2022)." ///
         "Employer×quartile and employer×month FE. SE clustered by employer.") ///
    legend(off) scheme(s2color)
graph export "$outdir/canaries_es_young.png", width(2400) replace

* --- Plot for ages 25-30 (comparison) ---
twoway (rarea ci_lo ci_hi half_year if age_grp == 2, ///
            color(navy%20) lwidth(none)) ///
       (connected coef half_year if age_grp == 2, ///
            mcolor(navy) lcolor(navy) msymbol(circle)), ///
    xline(7.5, lcolor(cranberry) lpattern(dash) lwidth(thin)) ///
    xline(8.5, lcolor(navy) lpattern(dash) lwidth(thin)) ///
    yline(0, lcolor(gs10) lpattern(solid)) ///
    xlabel(1(1)14, valuelabel angle(45) labsize(small)) ///
    ylabel(, format(%5.2f)) ///
    title("Age 25-30: Employment by AI exposure (event study)") ///
    subtitle("Coefficient on High-AI × half-year (ref: 2022H1)") ///
    note("Dashed lines: Riksbank hike (Apr 2022) and ChatGPT (Dec 2022).") ///
    legend(off) scheme(s2color)
graph export "$outdir/canaries_es_25to30.png", width(2400) replace

* --- Plot for ages 41-49 (null control) ---
twoway (rarea ci_lo ci_hi half_year if age_grp == 4, ///
            color(gs8%20) lwidth(none)) ///
       (connected coef half_year if age_grp == 4, ///
            mcolor(gs8) lcolor(gs8) msymbol(circle)), ///
    xline(7.5, lcolor(cranberry) lpattern(dash) lwidth(thin)) ///
    xline(8.5, lcolor(navy) lpattern(dash) lwidth(thin)) ///
    yline(0, lcolor(gs10) lpattern(solid)) ///
    xlabel(1(1)14, valuelabel angle(45) labsize(small)) ///
    ylabel(, format(%5.2f)) ///
    title("Age 41-49: Employment by AI exposure (event study)") ///
    subtitle("Coefficient on High-AI × half-year (ref: 2022H1)") ///
    note("Dashed lines: Riksbank hike (Apr 2022) and ChatGPT (Dec 2022).") ///
    legend(off) scheme(s2color)
graph export "$outdir/canaries_es_41to49.png", width(2400) replace


* ============================================================================
* SECTION 5: SIMPLIFIED TRIPLE-DIFF (occupation-level, as backup)
* ============================================================================

* If employer×month FE is computationally infeasible (too many FE groups),
* fall back to occupation-level triple-diff. Less demanding but weaker ID.

di ""
di "============================================================"
di "  BACKUP: Occupation-level triple-diff"
di "============================================================"

use "$outdir/canaries_analysis.dta", clear

* Collapse to occupation × age group × month
collapse (sum) n_emp, by(ssyk4 daioe_q age_group ym high_ai ///
    post_rb post_gpt half_year)

gen ln_emp = ln(n_emp)
gen young = (age_group == 1)

* Triple-diff interactions
gen post_gpt_high       = post_gpt * high_ai
gen post_gpt_young      = post_gpt * young
gen post_gpt_young_high = post_gpt * young * high_ai
gen post_rb_high        = post_rb * high_ai
gen post_rb_young       = post_rb * young
gen post_rb_young_high  = post_rb * young * high_ai

* Entity = occupation × age group
egen entity = group(ssyk4 age_group)

* FE: entity + month + occupation×month (absorbs occupation-level macro)
egen occ_month = group(ssyk4 ym)

reghdfe ln_emp post_rb_high post_rb_young post_rb_young_high ///
    post_gpt_high post_gpt_young post_gpt_young_high, ///
    absorb(entity ym) cluster(ssyk4)

di ""
di "KEY COEFFICIENT: PostGPT × Young × High"
di "  Coef = " %7.4f _b[post_gpt_young_high]
di "  SE   = " %7.4f _se[post_gpt_young_high]
di "  p    = " %5.3f 2*ttail(e(df_r), abs(_b[post_gpt_young_high]/_se[post_gpt_young_high]))
di ""
di "PLACEBO: PostRB × Young × High"
di "  Coef = " %7.4f _b[post_rb_young_high]
di "  SE   = " %7.4f _se[post_rb_young_high]
di "  p    = " %5.3f 2*ttail(e(df_r), abs(_b[post_rb_young_high]/_se[post_rb_young_high]))


* ============================================================================
* SECTION 6: DIAGNOSTICS AND SUMMARY
* ============================================================================

use "$outdir/canaries_analysis.dta", clear

file open summ using "$outdir/canaries_summary.txt", write replace

file write summ "=== CANARIES REGRESSION SUMMARY ===" _n
file write summ "Date: $S_DATE $S_TIME" _n _n

* Sample sizes by age group
file write summ "--- Sample sizes ---" _n
forvalues g = 1/5 {
    local glabel : label age_lbl `g'
    qui count if age_group == `g'
    file write summ "Age `glabel': " (r(N)) " cells" _n
}

* Total observations
qui count
file write summ "Total cells: " (r(N)) _n

* Unique employers
qui distinct employer_id
file write summ "Unique employers: " (r(ndistinct)) _n

* Distribution of DAIOE quartiles
file write summ _n "--- DAIOE quartile distribution ---" _n
forvalues q = 1/4 {
    qui count if daioe_q == `q'
    file write summ "Q`q': " (r(N)) " cells" _n
}

* Employment distribution
file write summ _n "--- Employment count per cell ---" _n
qui sum n_emp, d
file write summ "Mean:   " (r(mean)) _n
file write summ "Median: " (r(p50)) _n
file write summ "P10:    " (r(p10)) _n
file write summ "P90:    " (r(p90)) _n
file write summ "Max:    " (r(max)) _n
file write summ "Zeros:  " _n
qui count if n_emp == 0
file write summ (r(N)) " cells with zero employment" _n

file write summ _n "--- Estimator used ---" _n
file write summ "Attempted: ppmlhdfe (Poisson PML)" _n
file write summ "Fallback:  reghdfe with ln(employment)" _n
file write summ "FE structure: employer×quartile + employer×month" _n
file write summ "SE: clustered by employer" _n

file close summ

di ""
di "=== ALL DONE ==="
di "Results saved to: $outdir"
di "Key files:"
di "  canaries_did_results.csv  — main DiD coefficients"
di "  canaries_es_all.csv       — event study coefficients"
di "  canaries_es_young.png     — event study figure (ages 16-24)"
di "  canaries_summary.txt      — diagnostics"

log close
