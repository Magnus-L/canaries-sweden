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

- **`14_mona_canaries_descriptive.py`** — descriptive canaries analysis (triple-diff, spotlight figures)
- **`15_mona_employer_did.py`** — employer-level DiD regression (Brynjolfsson-style)
- **DAIOE quartiles**: `daioe_quartiles.csv` on the MONA network share at `\\micro.intra\Projekt\P1207$\P1207_Gem\Lydia P1207\`
- **Stata do-file**: `mona_canaries_regression.do` — Stata version (for replication/verification only)

## Data Access

Both Python scripts query AGI tables directly via pyodbc (SQL Server on `monasql.micro.intra`, database `P1207`). No file-based extract needed.

**SQL table structure:**
- AGI tables: `dbo.Arb_AGIIndivid{YYYYMM}{suffix}` where suffix is `_def` (years < 2025) or `_prel` (2025, months 1-6)
- Individ tables: `dbo.Individ_{year}` (e.g., `Individ_2023`)
- Key columns: `P1207_LOPNR_PERSONNR` (person ID), `P1207_LOPNR_PEORGNR` (employer ID), `PERIOD` (YYYYMM), `Ssyk4_2012_J16` (SSYK 4-digit), `FodelseAr` (birth year)

**Cascading SSYK lookup (years >= 2023):**
For 2023-2025 data, the scripts join to `Individ_2023`, `Individ_2022`, and `Individ_2021` using COALESCE to recover individuals who lack a 2023 SSYK code but had one in an earlier register year. For years < 2023, the scripts join to that year's own Individ table.

## What You Need to Do

1. **Copy** the Python script(s) to your MONA project folder
2. **Verify** the pyodbc connection string at the top of the script matches your MONA setup
3. **Verify** `DAIOE_PATH` points to the correct location of `daioe_quartiles.csv`
4. **Run**: `python 14_mona_canaries_descriptive.py` (or `14_mona_employer_regression.py`)
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

- Both scripts query SQL Server year-by-year, using UNION ALL across months
  within each year. This is the bottleneck; expect ~1-2 minutes per year.
- The employer regression script (`15_mona_employer_did.py`) has
  employer×month FE that can be very large (potentially millions of groups).
  It filters to employers with >=5 workers by default.
  If memory is still an issue, increase `MIN_EMPLOYER_SIZE` to 10 or 20.
- If `linearmodels` is not installed on MONA, both scripts fall back to
  manual within-transformation using only pandas + statsmodels (always available).
- If even the within-transformation is too memory-intensive, the code
  falls back to occupation-level regressions automatically.
- **pyodbc** must be installed in the MONA Python environment. If not
  available, contact SCB support (it is standard on MONA).

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
