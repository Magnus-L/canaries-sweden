# Lane 35, exported 24 September 2026 10:24, filed the same morning

Handed over as `~/Downloads/MyFiles1024`, sixteen files, every one hash-checked before the
original was deleted. Both stages completed. No log came out with the export.

**Gates.** 91: the 2024-01 arm reproduced the paper's estimates at both bands to four decimals
(stated in `91_summary.txt`). 92: the recomposition gate raises `SystemExit` on failure and the
script reached its fits, so 22-23 plus 24-25 recomposed to 22-25 exactly.

## 91: the headline under five adoption boundaries

Step from before (adoption minus interim), boundaries 2023-07 to 2024-07:

| band | range of the step | spread | reported (2024-01) |
|---|---|---|---|
| 22-25 | -0.0411 to -0.0392 | 0.0019 | -0.0399 (0.0102) |
| 26-30 | -0.0497 to -0.0399 | 0.0098 | -0.0403 (0.0067) |

At 22-25 the number the paper prints does not hinge on the boundary. At 26-30 the step grows as
the boundary moves later (-0.044 at 2024-04, -0.050 at 2024-07), which is what a gradual decline
does, and the reported boundary gives the smallest step of the five. Per the read rule: report the
range, claim nothing about timing or cause.

## 92: the reduced youth payroll contribution as a rival

| half | eligible | tightening | interim | adoption | step from 2023 | employers |
|---|---|---|---|---|---|---|
| 22-23 | in part | +0.0272 (0.0110) | +0.0263 (0.0115) | +0.0002 (0.0182) | -0.0261 (0.0124) | 86,373 |
| 24-25 | never | +0.0108 (0.0084) | -0.0601 (0.0107) | -0.1097 (0.0167) | -0.0496 (0.0112) | 88,460 |

**Rule 3 by the letter**: 24-25 declines distinguishably (t = -4.4) but is not within one SE of
22-23 (gap 0.0235 against SE 0.0112), so rule 1 does not fire; 22-23 also declines at five per
cent (t = -2.1), so rule 2 does not fire either.

**Substantively the pattern runs against the rival.** The rival predicts the decline in the
eligible half. It is instead about twice as large in the half the reduction never reached, and
the eligible half sits ABOVE its baseline through the interim (+0.026), the period straight after
the reduction ended on 31 March 2023, when the rival predicts a fall.

**Two cautions.** (1) The rule promised the difference between the halves with a standard error
from one joint fit; the script ran two separate fits and prints the difference (-0.0235) without
one, correctly refusing to test it. So "about twice as large" is a description, not a test.
(2) 22-23 and 24-25 differ in more than eligibility (student share, labour-market entry), so the
split is not a clean eligibility contrast either way.

## For the paper (not applied; ML is editing in Overleaf)

- OA payroll paragraph: it says the 22-25 split "needs a finer age banding than the panel carries
  and is not reported". Now false. Replace with the two steps and the plain reading.
- The dating section can cite the 91 range for both bands.
