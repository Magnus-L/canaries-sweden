# Lane 34, exported 23 September 2026 18:14, filed 18:20

Handed over as `~/Downloads/MyFiles1814`, eleven files, every one hash-checked before the
original was deleted. 89 ran 92.3 min and completed. **90 did not: `wfh_margin_split.csv` holds
hires only, `vcov_s90_seps` is absent and there is no `90_summary.txt`, so it died during the
separations fit.** No log came out with the export.

## THE TEST DOES NOT DISCRIMINATE, AND THE REASON IS MEASURED

**The two firm scores are too close.** Spearman $+0.862$, Pearson $+0.881$, and 74.6 per cent of
employers sit in the same quartile on both. The 2x2 is almost empty off the diagonal: 112,834
low-AI and low-teleworkability, 112,831 high on both, and only 18,211 and 18,213 in the two
off-diagonal cells. **Thirteen point nine per cent of employers carry the variation the test
needs.**

| fit | coefficient | t | firms | |
|---|---|---|---|---|
| AI, among LOW-teleworkability firms | $+0.0032$ (0.0826) | $+0.04$ | 59,325 | **discriminating** |
| AI, among HIGH-teleworkability firms | $-0.0687$ (0.0205) | $-3.35$ | 44,892 | |
| Teleworkability, among LOW-AI firms | $-0.0920$ (0.0616) | $-1.49$ | 62,860 | **discriminating** |
| Teleworkability, among HIGH-AI firms | $-0.0556$ (0.0209) | $-2.66$ | 41,357 | |

**Read rule 1 does not license the sentence.** It licensed calling AI the operative score only if
the young decline appeared among low-teleworkability firms and not among low-AI firms. It appears
in neither: the AI cell is $+0.003$ on a standard error of 0.083, five times the headline's, and
the teleworkability cell is negative but at $t=-1.49$. **Neither score can be called operative on
this evidence, and the paper does not get to say the remote-work rival is answered on the
employment margin.**

Note what the standard errors do. Where the two scores overlap, both terms are precise and both
bite. Where they are made to disagree, both go imprecise. That is collinearity behaving exactly
as collinearity does, and it is why the design was not led with a horse race; the off-diagonal
turns out to be too thin for the same reason a horse race would have been.

## PART B IS UNINFORMATIVE, AND PARTLY BY MY OWN SPECIFICATION

The two gradients are near mirror images, quarter by quarter: $+0.1247$ against $-0.1106$ in
2021Q3, $+0.1533$ against $-0.1208$ in 2022Q3, $+0.1496$ against $-0.1216$ in 2023Q3. Two
standardised regressors correlating at 0.88 split a common signal with opposite signs, which is
what this is.

**And the common signal is mostly seasonal, because I specified no calendar terms.** Q3 and Q4
are large in every year and Q1 and Q2 near zero, in both gradients and in every year including
the pre-period. The paper's own Equation (2) removes that cycle with three quarter-of-year terms.
A path meant to show whether the AI gradient opens in 2024 cannot be read while the seasonal
cycle is still in it. **That is a defect in the specification I wrote, not in the run.** Any
re-run puts the calendar terms back.

Two smaller things from the same part: the path starts at 2021Q1 and not 2019Q1, because the
cached counts start there, and the exported quarter label for the teleworkability rows carries a
trailing underscore from the term-name parsing, which is cosmetic.

## WHAT PART C SAYS, AS FAR AS IT RAN

Hires, both scores entered together on 90,997 employers: AI $-0.0047$ (0.0441) and
teleworkability $-0.0176$ (0.0412). Both are indistinguishable from zero and both intervals are
wide, which is the same collinearity again. **Separations, the margin the reconciliation turns
on, did not run.**

## WHAT THIS IS WORTH, AND IT IS NOT NOTHING

The honest finding is an asymmetry between levels, and it is reportable:

  - **At occupation level the split discriminates.** Online Appendix II.3 splits occupations at
    the median Dingel and Neiman score and finds the posting decline entirely in the
    NON-teleworkable half, $-0.233$ against $-0.005$.
  - **At firm level it cannot.** Averaging over a firm's incumbents strips the idiosyncratic
    occupation variation that made the occupation split work, and leaves two scores correlating
    at 0.88 with 14 per cent of employers off the diagonal.

That is a statement about what the Swedish firm data can and cannot separate, it explains why the
occupation-margin evidence is the evidence we have, and it is a limitation to state rather than a
result to bury.

## WHAT IS OWED

1. **Diagnose 90's separations failure.** No log was exported; the folder would need `90_log.txt`.
2. **Decide whether Part A is re-run at all.** With 13.9 per cent of employers off the diagonal, a
   re-run buys precision only if the cut moves to the extremes, for instance top and bottom
   terciles or deciles of each score, which trades sample for separation. That is a real option
   and it should be decided on the numbers above rather than attempted by reflex.
3. **Part B must be re-run with the calendar terms in** before its path means anything.
4. **The appendix should carry the asymmetry**, whichever way 2 and 3 are decided.
