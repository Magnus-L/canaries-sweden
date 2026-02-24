# MONA Analysis: Canaries Test Using AGI Data

## Purpose

Test whether young workers (16-24) in high-AI-exposure occupations
experienced disproportionate employment declines after ChatGPT
(the "canaries in the coal mine" hypothesis from Brynjolfsson et al. 2025).

This analysis supplements the paper's main finding (job postings show
no AI-driven differential decline) with register employment data.

## What You Need

1. **AGI data** — monthly employer declarations, individual level,
   2019-01 to latest available (~2025-06).
   Variables: person ID, year-month, SSYK 4-digit occupation code,
   birth year (or age).

2. **DAIOE quartiles** — the file `daioe_quartiles.csv` (import as CSV).
   Columns: ssyk4, pctl_rank_genai, exposure_quartile, high_exposure.
   `high_exposure = 1` means top quartile of genAI exposure.

## Step-by-Step Analysis

### Step 1: Prepare AGI data

```
- Load individual-level AGI records
- Compute age = year of record - birth year
- Create binary: young = 1 if age 16-24, else 0
- Filter to ages 16-69
- Clean SSYK code to 4-digit string (zero-padded, e.g. "0110")
- Aggregate to: ssyk4 x year_month x young
  Count: n_employed = number of DISTINCT persons per cell
- Result: a DataFrame with columns:
  ssyk4, year_month (format "YYYY-MM"), young (0/1), n_employed
```

### Step 2: Merge with DAIOE

```
- Load daioe_quartiles.csv
- Merge on ssyk4 (inner join)
- Keep columns: ssyk4, year_month, young, n_employed,
  pctl_rank_genai, high_exposure
- Report: how many of N occupations matched (expect ~92-98%)
```

### Step 3: Triple-difference regression

The specification:

```
ln(n_employed_it) = alpha_i + gamma_t
    + beta1 * PostChatGPT_t * High_i
    + beta2 * PostChatGPT_t * Young_i
    + beta3 * PostChatGPT_t * Young_i * High_i
    + epsilon_it
```

Where:
- i = entity = ssyk4 + "_" + str(young)  (occupation x age group)
- t = year-month
- alpha_i = entity fixed effects
- gamma_t = year-month fixed effects
- PostChatGPT_t = 1 if year_month >= "2022-12"
- High_i = 1 if top quartile genAI exposure (from DAIOE)
- Young_i = 1 if age group 16-24

THE KEY COEFFICIENT: beta3 (PostChatGPT x Young x High)
- If negative and significant: evidence of "canaries" — young workers
  in AI-exposed occupations declining disproportionately
- If zero/insignificant: no canaries effect

Cluster standard errors at the entity level (occupation x age group).

Using linearmodels (if available in MONA):
```python
from linearmodels.panel import PanelOLS
import pandas as pd, numpy as np

df["entity"] = df["ssyk4"] + "_" + df["young"].astype(str)
df["date"] = pd.to_datetime(df["year_month"] + "-01")
df["ln_emp"] = np.log(df["n_employed"])
df["post_gpt"] = (df["year_month"] >= "2022-12").astype(int)
df["post_high"] = df["post_gpt"] * df["high_exposure"]
df["post_young"] = df["post_gpt"] * df["young"]
df["post_young_high"] = df["post_gpt"] * df["young"] * df["high_exposure"]

panel = df.set_index(["entity", "date"])
exog = ["post_high", "post_young", "post_young_high"]
mod = PanelOLS(panel["ln_emp"], panel[exog],
               entity_effects=True, time_effects=True)
res = mod.fit(cov_type="clustered", cluster_entity=True)
print(res.summary)
```

Using statsmodels (fallback):
```python
import statsmodels.api as sm

entity_dum = pd.get_dummies(df["entity"], prefix="e", drop_first=True)
time_dum = pd.get_dummies(df["year_month"], prefix="t", drop_first=True)
exog = ["post_high", "post_young", "post_young_high"]
X = pd.concat([df[exog], entity_dum, time_dum], axis=1).astype(float)
X = sm.add_constant(X)
mod = sm.OLS(df["ln_emp"].values, X)
res = mod.fit(cov_type="cluster", cov_kwds={"groups": df["entity"].values})
print(res.params[exog])
print(res.pvalues[exog])
```

### Step 4: Save results

Save a CSV with one row per coefficient:

| variable | coefficient | std_error | p_value |
|----------|------------|-----------|---------|
| post_high | ... | ... | ... |
| post_young | ... | ... | ... |
| post_young_high | ... | ... | ... |

Filename: `mona_canaries_regression.csv`

### Step 5: Figure (optional but strongly preferred)

Plot employment indexed to the earliest month (= 100), four lines:
- Young (16-24), High AI exposure (orange, solid, thick)
- Young (16-24), Low AI exposure (orange, dashed, thin)
- Older (25+), High AI exposure (dark blue, solid, thick)
- Older (25+), Low AI exposure (dark blue, dashed, thin)

Add vertical dotted lines at:
- April 2022 (Riksbanken first rate hike)
- December 2022 (ChatGPT launch)

Save as: `figA8_mona_canaries.png` (300 dpi)

## What to Export from MONA

1. `mona_canaries_regression.csv` — the regression table
2. `figA8_mona_canaries.png` — the trajectory figure
3. Optionally: a text file with N observations, N entities, N months,
   match rate with DAIOE, and the beta3 coefficient + p-value

## Expected Runtime

< 5 minutes on MONA hardware for typical AGI extract sizes.

## Reference

This analysis replicates the age-specific test in:
- Brynjolfsson, E., Chandar, B., & Chen, J. (2025). "Generative AI
  at Work: Canaries in the Coal Mine." Working paper.

And parallels the Finnish employment analysis in:
- Kauhanen, A. & Rouvinen, P. (2025). "Canaries in the Finnish Coal
  Mine?" Applied Economics Letters.
