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

## Outstanding MONA Tasks (pre-submission)

### Priority A — Must complete before submission

1. **Employment summary statistics** for appendix Table A2: N employers, N employer×quartile cells, mean/median cell sizes, **share of zero-employment cells by age group × quartile × half-year period**. The zero-cell share is critical: referees will ask whether results are driven by small cells going to zero (idiosyncratic turnover) vs systematic intensive-margin declines.
2. **Spotlight figures** for payroll administrators (SSYK 4112) and receptionists (SSYK 4225) — same format as software developers and customer service figures.
3. **Rambachan-Roth sensitivity analysis** for the employment event study (at minimum ages 22-25). Use the `honestdid` R package or implement in Python following Rambachan & Roth (2023, RES). Report the breakdown value $\bar{M}$ and honest confidence intervals under relative magnitudes restrictions. This is the single most important robustness check — three simulated referees all flagged this as the #1 concern.
4. **Alternative reference period (2021H2)**: Script 18 already computes this. Export the event study coefficients CSV and **include the figure in the appendix** (not just state "qualitatively unchanged"). Referees will want to see it given the significant γ₁.
5. **Poisson QMLE robustness**: re-estimate the main DiD (Equation A1) and event study using Poisson quasi-maximum likelihood instead of OLS on ln(count+1), following Chen & Roth (2024, QJE). Also report results **excluding zero-employment cells** to confirm the intensive margin alone shows the pattern. Script 15 has a pyfixest Poisson fallback — adapt for the balanced panel.

### Priority B — Should complete (raised by 2+ simulated referees)

6. **Teleworkability split for employment** (not just postings): crosswalk Dingel-Neiman teleworkability to SSYK and re-estimate the employer-level DiD + event study separately for teleworkable vs non-teleworkable occupations. If the employment age gradient also concentrates in non-teleworkable occupations, this strongly corroborates the main result.
7. **Alternative SE clustering**: report SEs clustered at (a) the employer level and (b) the occupation level, in addition to the current employer×quartile clustering. Present as a robustness table.
8. **SSYK attrition robustness**: re-estimate the employment DiD + event study restricting the sample to 2019-2023 only (where SSYK coverage is strongest, 90%+). The non-match rate rises to 20% in 2025, precisely when event study effects are largest. If results are qualitatively similar with the restricted sample, this is a strong defence.
9. **Employment age gradient with Eloundou measure**: re-estimate the employer-level DiD using the Eloundou et al. (2024) GPT exposure score instead of DAIOE. A referee noted that the posting timing test is measure-dependent (DAIOE loads effect onto Riksbank period; Eloundou loads onto ChatGPT period). Showing the employment result holds across measures would address this.
10. **Placebo treatment date test**: re-estimate the event study with a fake "PostGPT" cutoff at November 2021 (one year early). If no acceleration appears at the placebo date, this confirms the pattern is not a pre-existing trend. Simple and convincing.
11. **Composition entrant share over time**: report the share of young entrants (first observed in AGI) entering Q4 occupations, by year (2019-2025). The current "13% in 2023" statistic does not rule out a composition channel — a referee needs to see whether this share changes differentially after ChatGPT.

## Reference

Brynjolfsson, E., Chandar, B., & Chen, R. (2025). "Canaries in the Coal
Mine? Six Facts about the Recent Employment Effects of Artificial Intelligence."
Stanford Digital Economy Lab Working Paper, November 2025.
