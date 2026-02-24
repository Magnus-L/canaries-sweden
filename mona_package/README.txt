========================================================================
MONA PACKAGE — Canaries Test Using AGI Monthly Register Data
========================================================================
Paper: "Two Economies? Stock Markets, Job Postings, and AI Exposure
        in Sweden"
Authors: Lodefalk, Engberg, Koch, Lothman
Date: February 2026

========================================================================
WHAT THIS IS
========================================================================

This folder contains everything needed to run one supplementary analysis
in MONA: test whether young workers (16-24) in AI-exposed occupations
experienced disproportionate employment declines after ChatGPT — the
"canaries in the coal mine" hypothesis (Brynjolfsson et al. 2025).

The analysis produces TWO output files to bring back:
  1. mona_canaries_regression.csv  (regression coefficients)
  2. figA8_mona_canaries.png       (trajectory figure)


========================================================================
FOLDER CONTENTS
========================================================================

mona_package/
  README.txt                  <-- You are reading this
  daioe_quartiles.csv         <-- REQUIRED: AI exposure data (369 occupations)
  09_mona_agi_canaries.py     <-- Option A: full Python script
  MONA_INSTRUCTIONS.md        <-- Option B: step-by-step specification
  output/                     <-- Empty folder for results


========================================================================
OPTION A: If you CAN import .py files into MONA
========================================================================

This is the easy path. Three steps:

1. Copy the entire mona_package/ folder into your MONA project.

2. Open 09_mona_agi_canaries.py and edit lines 63-75 (the CONFIGURATION
   block at the top). You need to set:

   INPUT_PATH = Path("path/to/your/agi_extract.parquet")
     -- or .csv or .sas7bdat, all formats supported

   AGI_COLUMNS = {
       "person_id": "LopNr",       # adjust to your column name
       "year_month": "Period",      # adjust to your column name
       "ssyk4": "SSYK4",            # adjust to your column name
       "birth_year": "FodelseAr",   # adjust to your column name
   }

   DAIOE_PATH = Path("daioe_quartiles.csv")
     -- should work if you run from this folder

3. Run:
     python 09_mona_agi_canaries.py

   Output appears in output/ folder. Copy those files back.

Expected runtime: < 5 minutes.


========================================================================
OPTION B: If you can ONLY import .csv files into MONA
========================================================================

You can only bring in daioe_quartiles.csv. You write the code yourself
in MONA's Python environment, following the specification below.

1. Copy daioe_quartiles.csv into MONA.

2. Open MONA_INSTRUCTIONS.md — it contains:
   - Exact variable definitions
   - The complete regression equation
   - Copy-pasteable code snippets (both linearmodels and statsmodels)
   - What to aggregate, how to merge, what to export

3. The short version of what to do (details in MONA_INSTRUCTIONS.md):

   a) Load AGI individual records (monthly, 2019-2025)
   b) Compute age = year - birth_year
   c) Create: young = 1 if age 16-24, else 0
   d) Aggregate to: ssyk4 x year_month x young → count distinct persons
   e) Merge with daioe_quartiles.csv on ssyk4 (inner join)
   f) Create interactions:
        post_gpt      = 1 if year_month >= "2022-12"
        post_high      = post_gpt * high_exposure
        post_young     = post_gpt * young
        post_young_high = post_gpt * young * high_exposure
   g) Run panel regression with entity FE + time FE:
        entity = ssyk4 + "_" + str(young)
        ln(n_employed) ~ post_high + post_young + post_young_high
   h) Report the coefficient on post_young_high — that is the canaries test
   i) Save regression table as CSV, plot four employment trajectories

4. Export from MONA:
   - The regression CSV (one row per coefficient)
   - The figure (PNG, 300 dpi)


========================================================================
ABOUT daioe_quartiles.csv
========================================================================

This file maps SSYK 2012 4-digit occupation codes to AI exposure.
369 occupations. Columns:

  ssyk4              4-digit occupation code (zero-padded string)
  pctl_rank_genai    Generative AI exposure percentile (0-100)
  exposure_quartile  Q1 (lowest) to Q4 (highest)
  high_exposure      1 if Q4, 0 otherwise
  pctl_rank_allapps  All-apps AI exposure percentile (not needed here)

Source: DAIOE index (Lodefalk & Engberg 2024), 2023 cross-section.


========================================================================
THE KEY RESULT WE NEED
========================================================================

The coefficient on post_young_high (beta3 in the regression).

  - If beta3 < 0 and significant: evidence of "canaries" effect
    (young workers in AI-exposed occupations declining more)
  - If beta3 ~ 0 or insignificant: no canaries effect

Our preliminary test using public annual SCB data (2020-2024) found
beta3 = +0.038 (insignificant). The AGI monthly data should give a
much more precise estimate.

Please also report: N observations, N entities, N months, DAIOE match
rate, and the coefficients on post_high and post_young.


========================================================================
QUESTIONS? Contact Magnus (magnus.lodefalk@oru.se)
========================================================================
