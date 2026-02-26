========================================================================
DST PACKAGE — Canaries Test Using Danish Monthly Register Data
========================================================================
Paper: "Two Economies? Stock Markets, Job Postings, and AI Exposure
        in Sweden"
Authors: Lodefalk, Engberg, Koch, Lothman
Date: February 2026

========================================================================
WHAT THIS IS
========================================================================

This folder contains everything needed to replicate one supplementary
analysis in DST: test whether young workers (16-24) in AI-exposed
occupations experienced disproportionate employment declines after
ChatGPT — the "canaries in the coal mine" hypothesis (Brynjolfsson
et al. 2025).

The specification is IDENTICAL to the Swedish MONA analysis. Only
variable names and data paths differ.

The analysis produces FIVE output files:
  1. figA8a_dst_canaries_softwaredevelopers.png (spotlight: software devs by age)
  2. figA8b_dst_canaries_customerservice.png    (spotlight: customer service by age)
  3. figA8c_dst_canaries_economy.png            (broad canaries: young×high vs others)
  4. dst_canaries_regression.csv                (main regression coefficients)
  5. dst_canaries_regression_robust.csv         (robustness: no rate hike controls)


========================================================================
FOLDER CONTENTS
========================================================================

dst_package/
  README.txt                  <-- You are reading this
  daioe_quartiles.csv         <-- AI exposure data (369 SSYK4 occupations)
  09_dst_canaries.R           <-- R script (main analysis)
  output/                     <-- Empty folder for results


========================================================================
BEFORE YOU START
========================================================================

1. CHECK IF DAIOE IS ALREADY ON DST
   Mark and/or Sarah should have an up-to-date version of the DAIOE
   index on DST, possibly already mapped to DISCO codes. Check with
   them first. If they have it, use that version instead of the
   daioe_quartiles.csv in this folder.

   If using their version, adjust DAIOE_PATH and DAIOE_OCC_COL in the
   configuration block of 09_dst_canaries.R.

2. OCCUPATION CODE CROSSWALK
   The daioe_quartiles.csv uses Swedish SSYK 4-digit codes. SSYK and
   DISCO are both national adaptations of ISCO-08, so 4-digit codes
   align closely. If the Danish data uses DISCO codes that differ at
   4-digit level, you may need a simple crosswalk. In most cases,
   SSYK4 == DISCO4 == ISCO-08 4-digit.

3. R PACKAGES REQUIRED
   - data.table
   - fixest    (fast fixed-effects estimation, CRAN)
   - ggplot2
   If fixest is not available on DST, see fallback note in the script.


========================================================================
HOW TO RUN
========================================================================

1. Copy this folder into your DST project.

2. Open 09_dst_canaries.R and edit the CONFIGURATION block (lines 38-58).
   There are 7 variables to set — the rest can stay as-is:

   # Data file path
   INPUT_PATH     <- "path/to/your/employment_extract.csv"

   # Column names in YOUR extract (defaults are guesses — check yours)
   COL_PERSON_ID  <- "PNR"           # Encrypted person ID
   COL_YEAR_MONTH <- "PERIOD"        # Year-month (YYYY-MM or similar)
   COL_DISCO4     <- "DISCO4"        # 4-digit DISCO/ISCO occupation code
   COL_BIRTH_YEAR <- "FOEDSELSAAR"   # Birth year

   # DAIOE file (use Mark/Sarah's version on DST if available)
   DAIOE_PATH     <- "daioe_quartiles.csv"
   DAIOE_OCC_COL  <- "ssyk4"         # Change to "disco4" if remapped

   The rate hike date defaults to July 2022 (ECB's first hike, +50 bps).
   This gives 5 months of separation from ChatGPT (Dec 2022).
   Nationalbanken formally followed in Sep 2022 (+75 bps), but that
   leaves only 3 months of separation, risking collinearity.
   Adjust RATE_HIKE in the config if you prefer a different date.

3. Run:
     Rscript 09_dst_canaries.R

4. Copy the three output files back.


========================================================================
THE SPECIFICATION (identical to Sweden)
========================================================================

Triple-difference regression on monthly panel:

  ln(emp_it) = alpha_i + gamma_t
      + beta1 * Post_RateHike * High
      + beta2 * Post_RateHike * Young
      + beta3 * Post_RateHike * Young * High
      + beta4 * Post_ChatGPT * High
      + beta5 * Post_ChatGPT * Young
      + beta6 * Post_ChatGPT * Young * High
      + epsilon_it

Where:
  i = entity = occ4 + "_" + young  (occupation x age group)
  t = year-month
  alpha_i = entity fixed effects
  gamma_t = year-month fixed effects
  Post_RateHike = 1 if year_month >= "2022-07" (ECB first hike)
  Post_ChatGPT  = 1 if year_month >= "2022-12"
  High = 1 if top quartile genAI exposure (from DAIOE)
  Young = 1 if age group 16-24

THE KEY COEFFICIENT: beta6 (Post_ChatGPT x Young x High)
  - If negative and significant: canaries effect
  - If zero/insignificant: no canaries effect

Clustered SE at entity level.

ROBUSTNESS SPECIFICATION: The script also runs a second regression
WITHOUT rate hike interactions (only Post_ChatGPT interactions),
saved as dst_canaries_regression_robust.csv. This addresses the
potential collinearity between the rate hike and ChatGPT dummies,
which are only 5 months apart.

SPOTLIGHT ANALYSIS (new): The script also produces two "spotlight"
figures for specific occupations (software developers and customer
service agents), plotting employment by fine age band (22-25, 26-30,
..., 50+) indexed to October 2022 = 100. This mirrors the approach
in Brynjolfsson et al. and allows visual inspection of whether the
youngest workers in these occupations show divergent trajectories.
The DISCO codes should align with SSYK/ISCO at 4-digit level.
If any codes are absent in your data, the script warns and skips.

NOTE ON RATE HIKE DATE: The default is July 2022 (ECB's first hike,
+50 bps). Danish markets priced this in immediately; Nationalbanken
formally followed in September 2022 (+75 bps). We use the ECB date
for better separation from ChatGPT (5 months vs 3). The Swedish
analysis uses April 2022 (Riksbanken). Adjust RATE_HIKE in the
config if you prefer a different date.


========================================================================
WHAT TO REPORT
========================================================================

1. figA8a_dst_canaries_softwaredevelopers.png — spotlight: software devs
2. figA8b_dst_canaries_customerservice.png    — spotlight: customer service
3. figA8c_dst_canaries_economy.png            — broad trajectory figure
4. dst_canaries_regression.csv                — main regression table
5. dst_canaries_regression_robust.csv         — robustness (no rate hike)
6. Also useful: N observations, N entities, N months,
   DAIOE match rate, and the beta6 coefficient + p-value


========================================================================
QUESTIONS? Contact Magnus (magnus.lodefalk@oru.se)
========================================================================
