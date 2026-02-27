# MONA Analysis: Canaries Regression

## Purpose

Formally test whether young workers (16-24) in high-AI-exposure occupations
experienced disproportionate employment declines after ChatGPT. This upgrades
the current descriptive Figure 2 in the paper to a causally identified result.

The design mirrors **Brynjolfsson, Chandar & Chen (2025), Eq. 4.1** —
an event study with employer×quartile and employer×month fixed effects,
run separately by age group. The employer×month FE absorb ALL firm-level
macro shocks (interest rates, energy crisis, etc.), so identification comes
from within-firm, within-month variation across AI exposure levels.

## Files

- **Python script**: `14_mona_employer_regression.py` — the complete analysis
- **DAIOE quartiles**: `daioe_quartiles.csv` — crosswalk from SSYK4 to exposure quartile (numeric 1-4)
- **Stata do-file**: `mona_canaries_regression.do` — Stata version (for replication/verification only)

## What You Need to Do

1. **Copy** `14_mona_employer_regression.py` and `daioe_quartiles.csv` to MONA
2. **Edit** the three `FILL_IN_PATH` variables at the top of the script (lines ~80-90):
   - `INPUT_PATH` — your AGI monthly extract (parquet, CSV, or SAS)
   - `DAIOE_PATH` — the DAIOE quartile file
   - `OUTPUT_DIR` — an output directory
3. **Check** the `AGI_COLUMNS` dict (line ~85) — map your column names:
   - `person_id` — encrypted individual identifier
   - `employer_id` — encrypted employer/workplace identifier
   - `ssyk4` — SSYK 2012 four-digit occupation code
   - `birth_year` — birth year (to compute age)
   - `year_month` — period as YYYY-MM string
4. **Run**: `python 14_mona_employer_regression.py`
5. **Export** all files from the output directory

## What the Code Does

### Step 2: Main DiD (the key result)

For each age group {16-24, 25-30, 31-40, 41-49, 50+}, estimates:

```
ln(n_emp_{f,q,t} + 1) = γ₁·PostRB·High + γ₂·PostGPT·High + α_{f,q} + β_{f,t}
```

where f = employer, q = DAIOE quartile, t = month.

- Uses `linearmodels.PanelOLS` if available
- Falls back to manual within-transformation (double-demeaning)
- Falls back to occupation-level regression if employer FE infeasible
- **Employer×quartile FE** absorb baseline differences within firms
- **Employer×month FE** absorb all firm-level time shocks
- SEs clustered by employer

**The canaries finding**: γ₂ (PostGPT × High) should be negative and significant
for ages 16-24, but not for older groups.

### Step 3: Half-year event study

Traces the time path using half-year period dummies × High-AI interactions.
Reference period: 2022H1 (pre-Riksbank hike).

### Backup

If employer×month FE is computationally infeasible, the code automatically
falls back to an occupation-level regression (weaker identification but
always feasible).

## Expected Output

| File | Description |
|------|-------------|
| `canaries_did_results.csv` | DiD coefficients by age group (main table) |
| `canaries_es_all.csv` | Event study coefficients for all age groups |
| `canaries_es_young.png` | Event study figure for ages 16-24 |
| `canaries_es_25to30.png` | Event study figure for ages 25-30 |
| `canaries_es_41to49.png` | Event study figure for ages 41-49 |
| `canaries_summary.txt` | Sample sizes and diagnostics |

## Computational Notes

- The employer×month FE can be very large (potentially millions of groups).
  The script filters to employers with ≥5 workers by default.
  If memory is still an issue, increase `MIN_EMPLOYER_SIZE` to 10 or 20.
- If `linearmodels` is not installed on MONA, the code falls back to a
  manual within-transformation using only pandas + statsmodels (always available).
- If even the within-transformation is too memory-intensive, the code
  falls back to occupation-level regressions automatically.

## For Koch (Danish Analysis)

The Python script can be adapted for DST data by:
1. Replacing SSYK4 with DISCO 4-digit codes in the column mapping
2. Crosswalking DAIOE to DISCO via ISCO-08
3. Adjusting age group definitions if needed
4. Adjusting the employer identifier variable name

## Reference

Brynjolfsson, E., Chandar, B., & Chen, R. (2025). "Canaries in the Coal
Mine? Six Facts about the Recent Employment Effects of Artificial Intelligence."
Stanford Digital Economy Lab Working Paper, November 2025.
