# main_v2.tex: what changes, where, and what it costs in words

18 Sep 2026. `main_v1.tex`, `appendix_v1.tex` and `appendix_offline_v1.tex` are the submitted
files, frozen. Work happens in the `_v2` copies. **Overleaf must be repointed to `main_v2.tex`**
as the main document, or it will keep compiling a file nobody is editing.

**Word budget.** The submitted body is 1,537 words against the 2,000 limit (figures, equations and
footnotes excluded). Headroom is about 460 words, and the editor's demands are mostly appendix
work, so the main text should end near 1,900. Every addition below is costed.

## A. Abstract and frontmatter

| # | Change | Words |
|---|---|---|
| A1 | **"falls 5.5 per cent" leaves the abstract.** The editor's ln(n+1) objection is correct, so the headline becomes the Poisson incidence ratio already in submitted OA III.6: employment of 22-25 year olds in exposed occupations is **16.0 per cent lower** within employers by mid-2025. | 0 |
| A2 | Posting window becomes 2020-2026H1; the ad count follows the new accounting table (4.59 m for 2020-2025, plus 289,601 in 2026H1). | 0 |
| A3 | Version line and date. | 0 |

## B. Data and empirical strategy

| # | Change | Words | Status |
|---|---|---|---|
| B1 | **Posting sample accounting** (editor, point 1). Raw 4,631,741 ads 2020-2025; 4,951 dropped for a missing occupation field, 2 out of range, 40,615 duplicates, 0 parse errors; 4,586,173 kept. One sentence in text, the year-by-year table in the appendix. | +35 | ready |
| B2 | **Coverage over time.** Valid-SSYK share is 100.0 per cent in every month 2022-2025 and the two thin years are 2020-2021 (1.0 per cent and 0.7 per cent dropped), so the posting decline cannot be a coding artefact. | +30 | ready |
| B3 | **400 versus 369.** The 31 unmatched occupations are military and senior-manager codes that carry no DAIOE score, not a coverage loss. | +25 | ready |
| B4 | **Estimator.** Equation (2) becomes Poisson PML with the same two fixed effects; ln(n+1) OLS moves to robustness. State why: the log-of-one-plus transform is not a percentage change when zeros are common, and 70 per cent of 22-25 cells are zero. | +45 | ready |
| B5 | **Descriptive series cut at December 2025**, regressions extended to June 2026 on closed-quarter files. The submitted window ran to February 2026, whose last two months the collection under-counts. | +30 | ready |
| B6 | **Occupation-code recency**, one sentence pointing to the new appendix section on how workers entering after the last register year are treated. | +20 | **blocked on 40** |

## C. Results

| # | Change | Words | Status |
|---|---|---|---|
| C1 | Posting numbers to the extended window: OLS $\beta_1$ -0.127 (SE 0.039), $\beta_2$ -0.059 (SE 0.038, p = 0.12); Poisson -0.129 (p = 0.012) and -0.060 (p = 0.15). The conclusion is unchanged and now runs through June 2026. | 0 | ready |
| C2 | **Poisson headline.** Pooled post-ChatGPT $\gamma_2$ = -0.174 (SE 0.011) for 22-25, a 16.0 per cent lower level; -0.053 for 26-30; null at 31-49; +0.024 at 50+. | +20 | ready |
| C3 | **Dynamics, stated carefully.** The Poisson event study is flat through 2023 and breaks at 2024H1 (-0.283), reaching -0.588 at 2025H1. Two things must be said in the same breath: the pre-period carries small significant deviations (2021H2 -0.051), and the 2024H1 break coincides with the start of the code cascade, so part of the dynamic path may be mechanical. | +55 | **blocked on 45** |
| C4 | **The firm lane** (new, answers the coverage objection with data whose coverage is not ours). Within-employer posting DiD on public ads with organisationsnummer, 12,141 firms, 2021-2026H1: PostRB x High +0.004 (p = 0.72), PostGPT x High **-0.158** (p < 0.001); -0.180 excluding staffing agencies and public employers; **-0.196** on entry-level ads only, whose event study deepens to -0.47 by 2026H1. | +70 | ready |
| C5 | **Public register check.** In SCB's own published aggregates, which we do not construct, 16-24 is the only age band whose top-quartile employment gap falls by 2024; every older band rises. | +35 | ready |
| C6 | **Seasonality and ln(x+1)**, one clause each: occupation-group x calendar-month fixed effects move the coefficients by less than 0.004, and the estimator change answers the transform objection. | +25 | ready |
| C7 | **Monetary lags** (Reviewer 2). The differential decline deepens through 2024 and 2025, after the Riksbank began cutting in May 2024, which is the wrong sign for a monetary explanation. | +30 | ready |
| C8 | **Decile gradient** (Kallberg). Postings are non-monotone across deciles with a hump at d4-d7, so fine granularity is noisy on the posting margin; the employment deciles decide the figure. | +30 | **blocked on 44** |
| C9 | **Coverage-immune cohort.** Workers coded by December 2023, followed forward: onset in 2023H1 and the same sign, in a design where differential nonmatching of later entrants cannot enter. Magnitude held back pending the backtest. | +40 | **blocked on Michael's ruling** |
| C10 | **The as-of backtest**, the affirmative answer: impose 2024-25 staleness on the fully covered 2019-2023 panel and measure the artificial coefficient. | +50 | **blocked on 45** |

### Word budget as it stands

The body is now **1,987 words**, from 1,537. C3, C9 and C10 still have to fit, which is about 90
words, so roughly that much has to come out when the runs report. The reserve is identified: the
gender paragraph can go to the appendix in full (about 60 words), and the OMXSPI and placebo
asides another 30. Do not spend it on anything else.

### Two things for Michael to rule on

1. **The 16 per cent coincidence.** Brynjolfsson et al. report a 16 per cent decline for young US
   workers; our Poisson headline is now 16.0 per cent. These are different estimands, theirs a
   relative decline by mid-2025 and ours a pooled post-period average, and the paper currently
   states both without comment. Either we note the difference in a clause or a referee will ask.
2. **The gender and margin results are still ln(n+1).** Script 48, written today, re-runs the
   gender split in Poisson. The hires and separations decomposition is better replaced by 41's
   incumbents-versus-new-matches contrast than re-run in Poisson, because 41 also answers the
   coverage question and is already queued. Both paragraphs currently name their estimator.

## D. Conclusion

| # | Change | Words |
|---|---|---|
| D1 | One sentence that the finding survives three designs whose coverage properties differ: the register panel, the coverage-immune cohort, and public postings at the firm level. | +30 |
| D2 | Limitation sentence rewritten: we still do not observe adoption, and the register's occupation codes lag, which is why the paper now carries the backtest. | +15 |

## E. Citations to place

- **Azar, Gine and Sanz-Espin (2026)**, wage margin and the rank-versus-age reading (R1.10-R1.11).
- **Facius and Iacono (2026)**, Norway: their nulls are power, 45 and 82 identifying firms, not sign. Magnus is thanked in it.
- **Kallberg (2026)**, the independent replication on public aggregates; cite at the decile gradient.
- **Vossos et al. (2026)** and **Pizzinelli et al. (2023)**: drafted sentences exist in `notes/`, unplaced.
- **Baker et al. (2026)**, practitioner's guide, only if a covariate choice needs defending. Venue unverified: run `/crossref` before it enters the file.

## F. Appendix work

`appendix_v2.tex` gains: posting accounting (B1), monthly coverage by source and exposure (B2),
the 400/369 reconciliation (B3), the extended-window table (B5), the seasonality panel (C6), the
decile profiles (C8), the firm lane in full (C4, new section), the public register check (C5), and
the coverage battery from 40 and 41 when they land. `appendix_offline_v2.tex` keeps whatever the
word limit forces out of the online appendix.

Order of work: everything marked ready, then C3, C9 and C10 as the runs report, then a full read.
