# MONA Analysis: Canaries Regression

## Purpose

Formally test whether young workers (16-24) in high-AI-exposure occupations
experienced disproportionate employment declines after ChatGPT. This upgrades
the current descriptive Figure 2 in the paper to a causally identified result.

The design mirrors **Brynjolfsson, Chandar & Chen (2025), Eq. 4.1** —
a Poisson event study with employer×quartile and employer×month fixed effects,
run separately by age group. The employer×month FE absorb ALL firm-level
macro shocks (interest rates, energy crisis, etc.), so identification comes
from within-firm, within-month variation across AI exposure levels.

## Files

- **Stata do-file**: `mona_canaries_regression.do` — the complete analysis
- **DAIOE quartiles**: `daioe_quartiles.csv` — crosswalk from SSYK4 to exposure quartile

## What You Need to Do

1. **Copy** `mona_canaries_regression.do` and `daioe_quartiles.csv` to MONA
2. **Edit** the three `FILL_IN_PATH` globals at the top of the do-file to point to:
   - Your AGI monthly data (individual-level)
   - The DAIOE quartile file
   - An output directory
3. **Check** that your AGI data has these variables (rename if needed):
   - `person_id` — encrypted individual identifier
   - `employer_id` — encrypted employer identifier
   - `ssyk4` — SSYK 2012 four-digit occupation code
   - `age` — worker age (or `birth_year`, in which case uncomment the gen line)
   - `ym` — Stata monthly date (e.g., from `mofd()`)
4. **Run** the full do-file: `do mona_canaries_regression.do`
5. **Export** all files from the output directory

## What the Code Does

### Section 2: Main DiD (the key result)

For each age group {16-24, 25-30, 31-40, 41-49, 50+}, estimates:

```
y_{f,q,t} = γ₁ × PostRB × HighQ4 + γ₂ × PostGPT × HighQ4 + α_{f,q} + β_{f,t}
```

where f = employer, q = DAIOE quartile, t = month.

- Uses `ppmlhdfe` (Poisson) if available, `reghdfe` (OLS on ln) as fallback
- **Employer×quartile FE** absorb baseline differences within firms
- **Employer×month FE** absorb all firm-level time shocks (monetary policy, etc.)
- SEs clustered by employer

**The canaries finding**: γ₂ (PostGPT × High) should be negative and significant
for ages 16-24, but not for older groups.

### Section 3: Half-year event study

Traces the time path using half-year period dummies × High-AI interactions.
Reference period: 2022H1 (pre-Riksbank hike). This shows whether divergence
appears (a) pre-ChatGPT, (b) post-ChatGPT, or (c) was already present earlier.

### Section 5: Backup triple-diff

If employer×month FE is computationally infeasible (too many FE groups on MONA),
Section 5 runs a simpler occupation-level triple-diff:

```
ln(emp) = ... + β × PostGPT × Young × HighAI + ... + entity FE + month FE
```

This is weaker identification but feasible with any dataset size.

## Expected Output

| File | Description |
|------|-------------|
| `canaries_did_results.csv` | DiD coefficients by age group (main table) |
| `canaries_es_all.csv` | Event study coefficients for all age groups |
| `canaries_es_young.png` | Event study figure for ages 16-24 |
| `canaries_es_25to30.png` | Event study figure for ages 25-30 |
| `canaries_es_41to49.png` | Event study figure for ages 41-49 |
| `canaries_summary.txt` | Sample sizes and diagnostics |
| `canaries_regression.log` | Full Stata log |

## Computational Notes

- The employer×month FE can be very large (potentially millions of groups).
  If Stata runs out of memory, try:
  1. Increase memory: `set maxvar 32767` and `set matsize 11000`
  2. Restrict to employers with ≥10 workers (add: `bysort employer_id: egen emp_size = count(person_id)` then `keep if emp_size >= 10`)
  3. Use the Section 5 backup (occupation-level) instead
- `ppmlhdfe` handles zero cells properly. If not installed on MONA, the
  `reghdfe` fallback drops zero-employment cells (logged DV). This is acceptable
  for an Economics Letters paper but note it in the output.
- If `ppmlhdfe` is available but slow, try: `ppmlhdfe ..., tolerance(1e-6)`

## For Koch (Danish Analysis)

The same do-file can be adapted for DST data by:
1. Replacing SSYK4 with DISCO 4-digit codes
2. Crosswalking DAIOE to DISCO via ISCO-08
3. Adjusting age group definitions if needed
4. Adjusting the employer identifier variable name

## Reference

Brynjolfsson, E., Chandar, B., & Chen, R. (2025). "Canaries in the Coal
Mine? Six Facts about the Recent Employment Effects of Artificial Intelligence."
Stanford Digital Economy Lab Working Paper, November 2025.
