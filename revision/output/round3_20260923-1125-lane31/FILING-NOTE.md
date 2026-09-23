# Lane 31, exported 23 September 2026 11:25, filed 12:30

Handed over as `~/Downloads/MyFiles1125`, seven files, every one hash-checked against the filed
copy before the original was deleted.

**THE SUMMARY IN THIS FOLDER IS THE EARLIER RUN'S AND IS OWED.** `85_summary.txt` here is the
09:21 job, which ran parts P only: its title block names the two arms, its runtime is 37.8 min,
and it carries no descriptive and no split section. The three-part job (parts PDS) started at
10:13 and had written `occ_route_descriptive_full.csv` and `occ_route_split65.csv` by the time
the folder was copied at 11:25, but had not yet written its own summary. So the exports are
complete and the script's own text for parts D and S is missing; the verdicts below were read
off the CSVs, which is not how the read rules are meant to be settled. **If `output_85` is still
on the share, its `85_summary.txt` is now the PDS one and is worth collecting.**

## What the exports say

**Part P, the profile arms** (153,845 employers, 37,826,298 cells). The gate passed in the 09:21
job: the arm with the calendar terms reproduces lane 28b's profile, so the panel is the one the
paper reports. At 22--25 the plain arm gives -0.0381 (0.0134) and the arm with the cycle removed
-0.0192 (0.0125); the terms move it by +0.0189. At 50 and over, +0.0709 (0.0065) plain against
+0.0675 (0.0063). This is Figure 2 of the paper, and it was already filed from the 09:21 job.

**Part D, the descriptive counterpart**: 48 cells, four quartiles by six bands by two windows,
with the total, the per-cell mean, the employer count and the cell count. THE TOTALS ARE WINDOW
SUMS AND NOT PER-MONTH FIGURES: the pre window holds 24 months and the post window 18, so a
change computed on them carries a mechanical -25 per cent. `l24_tab_descriptive_bands.py`
divides by 24 and 18 before it computes anything, and the console line in script 85 that calls
them "per month" is wrong and is being corrected.

**Part S, the oldest band split at 65** (104,333 employers, seven bands, reference 41--49):
50--64 +0.0392 (SE 0.0039), t 10.18, and 65--69 +0.1902 (0.0321), t 5.93. READ RULE 5 IS MET AND
THE PENSION-AGE RIVAL IS DISMISSED: the half the 2020 and 2023 reforms do not reach gains on its
own and is distinguishable from zero well inside five per cent. On the education route the same
split gave +0.0343 (0.0038) and +0.1672 (0.0313).

## What it replaces

Online Appendix Table III.1 (the descriptive) and Table III.3 (the split at 65) were on the
education route and said so in the text. Both are rebuilt on this export and the two disclosure
sentences are deleted.
