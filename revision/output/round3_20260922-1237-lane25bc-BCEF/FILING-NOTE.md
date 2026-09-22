# Lane 25, parts B, C, E and F (filed 22 September 2026, 12:37)

Handed over as `~/Downloads/MyFiles1237`, eleven files, every one hash-checked against the
filed copy before the originals were deleted. Lane 25a (parts A, D and G: the pre-period path
and drift test, the skill-intensity placebo and the May 2022 boundary) was still running.

## B. The sex split on Equation (2) exactly

Every treatment term interacted with female, sex-specific employer-by-age and month-by-age
effects, 40,520,358 cells. On Table 1's base, the level of the tightening months:

| | coefficient | SE |
|---|---|---|
| men (post x high x young) | -0.0047 | 0.0177 |
| female differential | -0.0746 | 0.0142 |
| women (sum, SE from the covariance) | -0.0793 | 0.0156 |

Women's step is 5.1 standard errors from zero; men's is not distinguishable from zero. From the
2023 level the steps are -0.0015 for men and -0.0542 for women. The differential is larger than
script 68's -0.0659, which was measured on a different base; 68's number stays in the appendix
with its base stated. Women were already falling in the interim period (interim differential
-0.0219, SE 0.0087), which the paper should say.

## C. Industry clustering beside employer clustering

Coefficients reproduce script 68's to four decimals (the check column is True in every row), so
this is inference only. Three-digit industry gives 5,545 clusters at 22-25 and 9,368 at 26-30.

| | employer SE | industry SE | t on industry |
|---|---|---|---|
| 22-25, step from the tightening level (-0.0408) | 0.0150 | 0.0301 | -1.35 |
| 22-25, step from the 2023 level (-0.0265) | 0.0099 | 0.0189 | -1.40 |
| 26-30, step from the tightening level (-0.0394) | 0.0102 | 0.0199 | -1.98 |
| 26-30, step from the 2023 level (-0.0360) | 0.0066 | 0.0137 | -2.62 |

The youngest band's step is not distinguishable from zero once the standard errors allow common
industry disturbances; the older young band's step survives. This is the single most consequential
result of the lane and the paper must report it.

## E. The profile with 50 and over split at 65

Against 41-49, cycle removed, 111,659 employers: 22-25 -0.0111 (0.0120), 26-30 -0.0107 (0.0085),
31-34 +0.0150 (0.0065), 35-40 +0.0097 (0.0054), 50-64 +0.0343 (0.0038), 65-69 +0.1672 (0.0313).
The gain is not confined to 65-69, so the 2023 retirement-age change does not account for it; but
the advantage of the 50-64 band is 3.4 per cent, not the 5.9 per cent the pooled 50-and-over band
shows, and the paper's "about 6 per cent" should become the two figures.

## F. Industry by age by month effects with the calendar terms

On the 106,174 employers with a 2019 industry code, the same-sample baseline gives -0.0406
(0.0151) and the fit with industry by age by month effects -0.0333 (0.0134), so 82 per cent of
the adoption step survives. This replaces the appendix's comparison, which was on the
specification without the calendar terms (-0.0509 to -0.0448).
