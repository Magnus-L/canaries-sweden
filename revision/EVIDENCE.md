# Canaries: what the evidence supports
*Assembled 2026-09-20 10:16 by `revision/assemble.py`. Submitted headline, ages 22-25: -0.174.*

> Pre-committed before the runs (47b's docstring, repeated in 47h, 47i and 47j): an artefact below 0.05 in absolute value at BOTH truncations means the design can carry register evidence; between that and half the occupation artefact (0.153 at T=2021, 0.081 at T=2022) it is usable only with the artefact stated beside every estimate; at or above half, the route is closed by lag.

## Where every number comes from

- **44** `decile_pooled.csv` from `round2_20260918-2008` (18 Sep 20:08)
- **45** `asof_estimates.csv` from `round2_20260918-1736-script45` (18 Sep 17:36)
- **46** `wfh_horserace.csv` from `round2_20260918-2008` (18 Sep 20:08)
- **47b** `edu_asof_estimates.csv` from `round2_20260918-2219-script47b` (18 Sep 22:19)
- **47h** `horserace_estimates.csv` from `round2_20260919-1645` (19 Sep 16:45)
- **48** `gender_poisson.csv` from `round2_20260919-1645` (19 Sep 16:45)
- **50** `m7_validation_t2023_k2.csv` from `round2_20260919-1759-output50slim` (19 Sep 17:59)

## 1. The paper's claim, with education-based exposure (47h)

**Not supported.** Every worker-level education design manufactures the result out of register lag, as 47b did. The paper's original claim cannot be re-made this way; go to 47j.

| design | artefact T2021 | artefact T2022 | true 22-25 | 50+ placebo | verdict |
|---|---|---|---|---|---|
| OL_daioe | -0.1624 | +nan | +0.0062 | nan | PENDING |


## 2. The age gradient within employers (47j)

**Pending** -- `triple_estimates.csv` not found.

## 3. Exposed firms employ fewer young workers (47i)

**Pending** -- `firmmix_estimates.csv` not found.

## 4. Register lag manufactures the result (45 and 47b)

Occupation design (45): artefact -0.3068 at T=2021 and -0.1627 at T=2022, against a true coefficient of +0.019 in the fully covered years, and a submitted headline of -0.174.

Worker-level education design (47b): -0.3596 and -0.2941, worse than the occupation design, with a 50+ placebo of -0.003 and mapped shares moving by 0.1 to 0.4 per cent between the arms.

**Supported, and independent of everything above.** A design in wide use manufactures this result out of register lag; the natural fix manufactures more of it; the damage is specific to the ages whose human capital is still moving; and no coverage diagnostic detects it.

## 5. Does education predict the job at all (50 + l13)

```
PREDICTIVE VALIDATION OF EDUCATION-EXPOSURE DESIGNS (ages 22-25)
quartile agreement / Q4 recall at lag 0 -> lag 2, pooled over t

  OL_daioe       agreement 0.405 -> 0.384   Q4 recall 0.599 -> 0.447
  fresh_stock    agreement 0.404 -> 0.384   Q4 recall 0.599 -> 0.447
  entrant        agreement 0.411 -> 0.400   Q4 recall 0.599 -> 0.444
  entrant_share  agreement 0.345 -> 0.340   Q4 recall 0.609 -> 0.460
  expband        agreement 0.461 -> 0.453   Q4 recall 0.303 -> 0.044
  enrol          agreement 0.411 -> 0.400   Q4 recall 0.600 -> 0.444
  full_nontier   agreement 0.318 -> 0.307   Q4 recall 0.599 -> 0.448
```

## 6. Gender split (48)

Whether the decline is concentrated among young women.

_8 rows in `gender_poisson.csv`; read it before writing the sentence._

## 7. Code vintage (41)

Whether the decline is steeper where the occupation code is older, which would corroborate the lag mechanism directly.

**Pending.**

## 8. Decile gradient (44)

Whether the decline rises monotonically with exposure.

_54 rows in `decile_pooled.csv`; read it before writing the sentence._

## 9. Telework horse race (46)

How much of the young-worker decline telework absorbs.

_24 rows in `wfh_horserace.csv`; read it before writing the sentence._

## 8b. The register-immune family

**Pending**: none of 47L, 54 or 57 found.

## 8c. The paper's own estimand on contemporaneous codes (53)

Young against young, inside the employer, monthly, restricted to worker-months whose occupation code was assigned in the observation year. Window ends 2023. Read the fresh arm against the stale arm: the contrast is internal.

**Pending.**

## 8d. The settled sample (47k)

Restricted to young workers whose education record is correct.

**Pending.**

## 8e. By age and over time (56)

Event studies on stock, hires and separations. The PRE-PERIOD is the test; quote it.

**Pending.**

## 8f. Seasonal control and the pre-committed rule (58)

Whether the 2025H1 reading survives honest seasonal handling. The rule can refuse, and on 20 Sep it did.

**Pending.**

## 8g. Quarterly and monthly path (59)

Whether any late movement is a drift or one odd month.

**Pending.**

## 8h. When did it start? (60)

Four pre-specified treatment dates, anchored on SCB's measured Swedish firm adoption (10.4 per cent in 2023, 25.2 in 2024, 35.0 in 2025), plus an exploratory profile. Earlier estimates all define post as Dec 2022 and therefore average an untreated year into the post window.

**Pending.**

## 10. Advertisements, no register involved

**Supported, and unaffected by any of the above.** Within-employer design on public advertisements: PostGPT x High -0.158 on 12,141 employers, -0.196 on entry-level advertisements, event study to -0.47 by 2026H1.
