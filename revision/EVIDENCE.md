# Canaries: what the evidence supports
*Assembled 2026-09-21 07:36 by `revision/assemble.py`. Submitted headline, ages 22-25: -0.174.*

> Pre-committed before the runs (47b's docstring, repeated in 47h, 47i and 47j): an artefact below 0.05 in absolute value at BOTH truncations means the design can carry register evidence; between that and half the occupation artefact (0.153 at T=2021, 0.081 at T=2022) it is usable only with the artefact stated beside every estimate; at or above half, the route is closed by lag.

## Where every number comes from

- **41** `output_41__vintage_es.csv` from `round2_20260920-exportpack` (20 Sep 17:23)
- **44** `output_44__decile_pooled.csv` from `round2_20260920-exportpack` (20 Sep 17:23)
- **45** `output_45__asof_estimates.csv` from `round2_20260920-exportpack` (20 Sep 17:35)
- **46** `output_46__wfh_horserace.csv` from `round2_20260920-exportpack` (20 Sep 17:26)
- **47L** `output_47L__agebase_estimates.csv` from `round2_20260920-exportpack` (20 Sep 17:32)
- **47Lg** `output_47L__agebase_gradient.csv` from `round2_20260920-exportpack` (20 Sep 17:32)
- **47b** `output_47b__edu_asof_estimates.csv` from `round2_20260920-exportpack` (20 Sep 17:32)
- **47h** `output_47h__horserace_estimates.csv` from `round2_20260920-exportpack` (20 Sep 17:23)
- **47i** `output_47i__firmmix_estimates.csv` from `round2_20260920-exportpack` (20 Sep 17:22)
- **47j** `output_47j__triple_estimates.csv` from `round2_20260920-exportpack` (20 Sep 17:32)
- **47k** `output_47k__settled_estimates.csv` from `round2_20260920-exportpack` (20 Sep 17:26)
- **48** `output_48__gender_poisson.csv` from `round2_20260920-exportpack` (20 Sep 17:32)
- **50** `output_50_slim__m7_validation_t2023_k2.csv` from `round2_20260920-exportpack` (20 Sep 17:18)
- **53** `output_53__fresh_pooled.csv` from `round2_20260920-exportpack` (20 Sep 17:23)
- **54** `output_54__flow_estimates.csv` from `round2_20260920-exportpack` (20 Sep 17:35)
- **54g** `output_54__flow_gradient.csv` from `round2_20260920-exportpack` (20 Sep 17:29)
- **56** `output_56__dynamics_young.csv` from `round2_20260920-exportpack` (20 Sep 17:23)
- **57** `output_57__reliability.csv` from `round2_20260920-exportpack` (20 Sep 17:29)
- **57v** `output_57__vintage_estimates.csv` from `round2_20260920-exportpack` (20 Sep 17:35)
- **58** `output_58__readrule.csv` from `round2_20260920-exportpack` (20 Sep 17:32)
- **59** `output_59__path_quarter.csv` from `round2_20260920-exportpack` (20 Sep 17:23)
- **60** `output_60__prespecified.csv` from `round2_20260920-exportpack` (20 Sep 17:29)
- **60p** `output_60__profile.csv` from `round2_20260920-exportpack` (20 Sep 17:32)
- **61** `output_61__redated_step.csv` from `round2_20260920-2148-jobs616263` (20 Sep 21:48)
- **61p** `output_61__redated_pooled.csv` from `round2_20260920-2148-jobs616263` (20 Sep 21:48)
- **62** `output_62__gradient_by_variant.csv` from `round2_20260920-2148-jobs616263` (20 Sep 21:48)
- **63** `output_63__robustness_gradient.csv` from `round2_20260920-2148-jobs616263` (20 Sep 21:48)
- **63h** `output_63__horserace.csv` from `round2_20260920-2148-jobs616263` (20 Sep 21:48)

## 1. The paper's claim, with education-based exposure (47h)

**Not supported.** Every worker-level education design manufactures the result out of register lag, as 47b did. The paper's original claim cannot be re-made this way; go to 47j.

| design | artefact T2021 | artefact T2022 | true 22-25 | 50+ placebo | verdict |
|---|---|---|---|---|---|
| OL_exact | -0.1145 | -0.1296 | +0.0206 | 0.0013 | CLOSED |
| OL_daioe | -0.1624 | -0.1446 | +0.0062 | 0.0022 | CLOSED |
| fresh_stock | -0.1628 | -0.1452 | +0.0064 | not estimated | CLOSED |
| entrant_share | -0.3360 | -0.2290 | -0.0206 | not estimated | CLOSED |
| entrant | -0.3840 | -0.2871 | -0.0127 | not estimated | CLOSED |
| enrol | -0.3849 | -0.2877 | -0.0127 | not estimated | CLOSED |
| expband | -0.4421 | -0.4068 | +0.0049 | not estimated | CLOSED |
| full | -0.4930 | -0.3554 | -0.0072 | not estimated | CLOSED |


## 2. The age gradient within employers (47j)

**Clean but imprecise.** The design passes the artefact test, which is what the rule tests, and the estimate is negative at every specification. It is NOT distinguishable from zero: -0.0132 with a standard error of 0.0111, t -1.19. What this licenses is a bound, not a finding.

| design | young band | gamma3 (true) | SE | t | artefact T2021 | artefact T2022 | verdict |
|---|---|---|---|---|---|---|---|
| OL_daioe | 22-25 | -0.0132 | 0.0111 | -1.19 | +0.0051 | +0.0044 | CLEAN |
| OL_daioe | 26-30 | +nan |  |  | +nan | +nan | PENDING |
| entrant | 22-25 | -0.0153 | 0.0114 | -1.34 | +0.0009 | +0.0013 | CLEAN |
| entrant | 26-30 | +nan |  |  | +nan | +nan | PENDING |


## 3. Exposed firms employ fewer young workers (47i)

**Not supported as stated.** See the table.

| design | age | gamma2 (true) | artefact T2021 | artefact T2022 | verdict |
|---|---|---|---|---|---|
| OL_daioe | 22-25 | +0.0243 | -0.0080 | -0.0084 | CLEAN |
| OL_daioe | 26-30 | +0.0375 | -0.0051 | -0.0036 | CLEAN |
| OL_daioe | 31-34 | +0.0297 | -0.0024 | -0.0009 | CLEAN |
| OL_daioe | 35-40 | +0.0163 | -0.0020 | -0.0002 | CLEAN |
| OL_daioe | 41-49 | +0.0104 | -0.0011 | -0.0003 | CLEAN |
| OL_daioe | 50+ | +0.0630 | -0.0007 | +0.0003 | CLEAN |
| entrant | 22-25 | +0.0149 | -0.0029 | -0.0044 | CLEAN |
| entrant | 26-30 | +0.0315 | -0.0032 | -0.0029 | CLEAN |
| entrant | 31-34 | +0.0241 | -0.0033 | -0.0030 | CLEAN |
| entrant | 35-40 | +0.0113 | -0.0019 | -0.0012 | CLEAN |
| entrant | 41-49 | +0.0106 | -0.0009 | -0.0001 | CLEAN |
| entrant | 50+ | +0.0562 | -0.0016 | -0.0021 | CLEAN |


## 4. Register lag manufactures the result (45 and 47b)

Occupation design (45): artefact -0.3068 at T=2021 and -0.1627 at T=2022, against a true coefficient of +0.019 in the fully covered years, and a submitted headline of -0.174.

Worker-level education design (47b): -0.3596 and -0.2941, worse than the occupation design, with a 50+ placebo of -0.003 and mapped shares moving by 0.1 to 0.4 per cent between the arms.

**Supported, and independent of everything above.** A design in wide use manufactures this result out of register lag; the natural fix manufactures more of it; the damage is specific to the ages whose human capital is still moving; and no coverage diagnostic detects it.

## 5. Does education predict the job at all (50 + l13)

```
l13 failed: ore/reshape/concat.py", line 407, in concat
    objs, keys, ndims = _clean_keys_and_objs(objs, keys)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/pandas/core/reshape/concat.py", line 808, in _clean_keys_and_objs
    raise ValueError("No objects to concatenate")
ValueError: No objects to concatenate
```

## 6. Gender split (48)

Whether the decline is concentrated among young women.

_8 rows in `output_48__gender_poisson.csv`; read it before writing the sentence._

## 7. Code vintage (41)

Whether the decline is steeper where the occupation code is older, which would corroborate the lag mechanism directly.

_39 rows in `output_41__vintage_es.csv`; read it before writing the sentence._

## 8. Decile gradient (44)

Whether the decline rises monotonically with exposure.

_54 rows in `output_44__decile_pooled.csv`; read it before writing the sentence._

## 9. Telework horse race (46)

How much of the young-worker decline telework absorbs.

_24 rows in `output_46__wfh_horserace.csv`; read it before writing the sentence._

## 8b. The register-immune family

- **47L, employment stock, exposure frozen 2019**: +0.0070 (SE 0.0089). Uses no occupation code after 2019 and no education register at all.
- **57, reliability of that frozen exposure**: lambda = 0.890 at the last measured year. Attenuation is mild, so a null is a null.

These share the DAIOE measure and differ in their register dependence and identifying variation, so agreement between them is corroboration only against measurement error, not against a mismeasured exposure concept.

## 8c. The paper's own estimand on contemporaneous codes (53)

Young against young, inside the employer, monthly, restricted to worker-months whose occupation code was assigned in the observation year. Window ends 2023. Read the fresh arm against the stale arm: the contrast is internal.

_36 rows in `output_53__fresh_pooled.csv`; read it before writing the sentence._

## 8d. The settled sample (47k)

Restricted to young workers whose education record is correct.

_36 rows in `output_47k__settled_estimates.csv`; read it before writing the sentence._

## 8e. By age and over time (56)

Event studies on stock, hires and separations. The PRE-PERIOD is the test; quote it.

_39 rows in `output_56__dynamics_young.csv`; read it before writing the sentence._

## 8f. Seasonal control and the pre-committed rule (58)

Whether the 2025H1 reading survives honest seasonal handling. The rule can refuse, and on 20 Sep it did.

_2 rows in `output_58__readrule.csv`; read it before writing the sentence._

## 8g. Quarterly and monthly path (59)

Whether any late movement is a drift or one odd month.

_50 rows in `output_59__path_quarter.csv`; read it before writing the sentence._

## 8h. When did it start? (60)

Four pre-specified treatment dates, anchored on SCB's measured Swedish firm adoption (10.4 per cent in 2023, 25.2 in 2024, 35.0 in 2025), plus an exploratory profile. Earlier estimates all define post as Dec 2022 and therefore average an untreated year into the post window.

_12 rows in `output_60__prespecified.csv`; read it before writing the sentence._

## 8i. The within-employer design, dated where adoption happened (61)

Young against older inside one employer, exposure from the 2019 education mix of incumbents aged 31 and over, and the treatment dated on SCB's measured firm adoption rather than on the launch.

- adoption window, from 2024-01: **-0.0584** (SE 0.0167, t -3.50)
- launch window, 2022-12 to 2023-11: -0.0270, which is the period every earlier estimate averaged in The artefact at the new dating is +0.0067.

Read rule: the windows were fixed by script 60 before any of this was estimated, and the adoption window is the shortest, so compare the windows on their t and not on their size.

## 8j. Why the two clean designs disagree about age (62)

47L and 47j name different age bands. Their exposures differ in source, unit and form at once, so this holds the outcome, the panel and the fixed effects at 47L's and varies one ingredient at a time. At 22-25:

- A0_47L_anchor: +0.0081 (as published; -0.0276 measured from 50+)
- A_occ_age_cont: +0.0085 (as published; -0.0273 measured from 50+)
- B_occ_firm_cont: -0.0278 (already measured from 50+)
- C_edu_age_cont: +0.0233 (as published; -0.0026 measured from 50+)
- D_edu_firm_quart: -0.0067 (already measured from 50+)

on a common base, since B_occ_firm_cont, D_edu_firm_quart lose the 50+ band to collinearity and are differences from it; UNIT alone moves 22-25 by -0.0004; SOURCE alone by +0.0248; unit and form together by -0.0041.

Read rule: whichever ingredient moves the answer is the one the two designs were disagreeing about, and the steps above are computed on a common base for the reason given. A step that survives rebasing is a measurement difference; one that does not was a normalisation difference and says nothing about the world.

## 8k. Does the answer depend on the exposure measure? (63)

Every other design in this round assigns exposure from DAIOE at four-digit occupation, so if that measure is wrong they all fail together. Age gradient at 22-25 and 41-49, by measure:

- hires, adoption-dated, 22-25: daioe -0.0754, eloundou -0.0400, telework -0.0233
- hires, adoption-dated, 41-49: daioe -0.0612, eloundou -0.0421, telework -0.0290
- seps, adoption-dated, 22-25: daioe -0.0124, eloundou -0.0117, telework +0.0139
- seps, adoption-dated, 41-49: daioe -0.0430, eloundou -0.0317, telework -0.0209
- stock, adoption-dated, 22-25: daioe -0.0132, eloundou -0.0132, telework +0.0069
- stock, adoption-dated, 41-49: daioe -0.0208, eloundou -0.0311, telework -0.0034
- hires, launch-dated, 22-25: daioe -0.0273, eloundou -0.0111, telework +0.0103
- hires, launch-dated, 41-49: daioe -0.0345, eloundou -0.0271, telework -0.0071
- seps, launch-dated, 22-25: daioe -0.0090, eloundou -0.0048, telework +0.0020
- seps, launch-dated, 41-49: daioe -0.0388, eloundou -0.0278, telework -0.0200
- stock, launch-dated, 22-25: daioe +0.0081, eloundou +0.0017, telework +0.0176
- stock, launch-dated, 41-49: daioe -0.0193, eloundou -0.0320, telework -0.0006

Horse race, both measures in one regression, which is the comparison that holds the sample fixed and the one worth quoting:

- hires, adoption-dated, 22-25: AI exposure **-0.1032** (SE 0.0500, t -2.07), teleworkability +0.0394
- hires, adoption-dated, 41-49: AI exposure **-0.0853** (SE 0.0184, t -4.65), teleworkability +0.0293
- seps, adoption-dated, 22-25: AI exposure **-0.0434** (SE 0.0213, t -2.03), teleworkability +0.0445
- seps, adoption-dated, 41-49: AI exposure **-0.0706** (SE 0.0167, t -4.23), teleworkability +0.0351
- stock, adoption-dated, 22-25: AI exposure **-0.0318** (SE 0.0143, t -2.22), teleworkability +0.0271
- stock, adoption-dated, 41-49: AI exposure **-0.0493** (SE 0.0134, t -3.68), teleworkability +0.0350
- hires, launch-dated, 22-25: AI exposure **-0.0606** (SE 0.0332, t -1.83), teleworkability +0.0457
- hires, launch-dated, 41-49: AI exposure **-0.0711** (SE 0.0178, t -3.99), teleworkability +0.0443
- seps, launch-dated, 22-25: AI exposure **-0.0227** (SE 0.0227, t -1.00), teleworkability +0.0201
- seps, launch-dated, 41-49: AI exposure **-0.0673** (SE 0.0166, t -4.06), teleworkability +0.0350
- stock, launch-dated, 22-25: AI exposure **-0.0062** (SE 0.0152, t -0.41), teleworkability +0.0206
- stock, launch-dated, 41-49: AI exposure **-0.0568** (SE 0.0141, t -4.04), teleworkability +0.0456

Read rule: the measures correlate 0.66 to 0.87 across occupations, so teleworkability being negative on its own is not evidence against the AI reading, and the horse race is what separates them. If teleworkability wins it, the interpretation fails however clean the identification is. The launch-dated rows are comparable with the rest of this round; the adoption-dated rows are where scripts 60 and 61 put the treatment.

## 10. Advertisements, no register involved

**Supported, and unaffected by any of the above.** Within-employer design on public advertisements: PostGPT x High -0.158 on 12,141 employers, -0.196 on entry-level advertisements, event study to -0.47 by 2026H1.
