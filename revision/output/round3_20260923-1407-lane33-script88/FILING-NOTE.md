# Lane 33, script 88, exported 23 September 2026 14:07, filed 14:12

Handed over as `~/Downloads/MyFiles147`, eight files, every one hash-checked against the filed
copy before the original was deleted. Runtime 31.0 min, twelve fits, no SQL. This closes lane
33: with 87 (filed 13:58) **nothing in the paper measures exposure with the education register
any more**.

## No four-decimal gate, by design, and the base is fitted here

77's gate was the education route's three-band contrast, and this route has no three-band
all-worker figure to reproduce, so the all-worker fit run here IS the base and every track cell
is read against it. The summary says so and reports the two comparisons for reference only: the
education route's three-band contrast was $-0.0153$ (0.0126), and this route's six-band profile
gives $-0.0192$ (0.0125) on another panel.

Every check the table builder runs reproduces: each exported standard error is the square root
of its own `gpt_x_high_<band>` diagonal in the matching `vcov_s88_*` file, every printed t is the
estimate over that standard error, and all twelve fits report status ok.

## What it says

**The base moved and is no longer comfortably a null.** The all-worker contrast at 22--25 is
$-0.0241$ (0.0130), t $-1.85$, against the education route's $-0.0153$ (0.0126). It is still not
distinguishable from zero at five per cent, so the reading holds, but "a null" was the old
table's word and it is now too strong. The note says not distinguishable at five per cent.

| Track | 22--25 vs 41--49 | 26--30 vs 41--49 | Employers |
|---|---|---|---|
| All workers | $-0.0241$ (0.0130) | $-0.0090$ (0.0088) | 110,653 |
| ICT | $-0.1194$ (0.0500) | $-0.1407$ (0.0350) | 6,837 |
| Engineering | $-0.0253$ (0.0222) | $-0.0177$ (0.0158) | 48,374 |
| Business, law, social | $-0.0733$ (0.0215) | $-0.0171$ (0.0157) | 37,689 |
| Health, education, care | $-0.0251$ (0.0233) | $-0.0438$ (0.0168) | 25,379 |
| Other | $-0.0029$ (0.0183) | $-0.0421$ (0.0100) | 72,889 |

**Two cells reverse, and both were quoted in the appendix.**

  - **The residual fields no longer gain at 22--25.** The education route gave $+0.033$ (0.018),
    a gain the appendix reported in so many words; this route gives $-0.0029$ (0.0183), a null.
    The sentence that said the young gain there is deleted rather than renumbered.
  - **At 26--30 the decline is no longer general.** The appendix said every track but
    engineering declines by about three log points. Here business, law and administration is
    $-0.0171$ (0.0157) and not distinguishable from zero, and the decline is carried by health,
    education and care ($-0.0438$) and the residual fields ($-0.0421$), about four log points.

**Everything the paper leads on is stable.** ICT is steepest at both bands and deeper at 26--30
than the education route had it ($-0.141$ against $-0.132$); business, law and administration
is clear at 22--25 ($-0.073$ against $-0.069$); engineering and health remain absent at 22--25.
The employer counts fall by 6 to 9 per cent on every track, which is the scorable-set shift the
lane 28 plan predicted and reported before any fit ran.

## Applied the same hour

`l31_tab_contrast_by_track.py` repointed and `tableA_contrast_by_track.tex` rebuilt. Two gates
needed real work rather than a rename:

  - **The summary layout differs from 77's.** 88 names its base rows "all workers," and carries
    "vs 41-49" only there, while the track rows carry neither; both print a t ratio where 77
    printed a trailing star. One pattern now reads both shapes.
  - **The star is no longer taken from the summary.** It is derived from the estimate and its
    standard error, and checked against the t the run printed, which is a stronger check than
    reading a character the run happened to emit.
  - The two figures the note compares against were on the education route and are repointed:
    the six-band contrast now comes from this route's own profile, $-0.0192$ (0.0125), and the
    ICT shares from script 87's mix, 2.9 per cent of exposed firms' young women and 6.8 of
    their young men.

The appendix paragraph at `:672` is rewritten on these numbers, including both reversals, and
the sentence added at 15:05 disclosing that the contrast was still on the education score is
removed, because it no longer is. `main_v3.tex:137`'s contrast clause needed no change: ICT
steepest, engineering and health absent, all three still hold. **The cross-route inconsistency
flagged when 87 landed is closed**: that sentence now sits on one measure throughout.

`l37_v3_prose_numbers.py` prints the twelve contrast cells beside the split block. Both
manuscripts compile, main_v3 15 pp and appendix_v3 66 pp.

## The standing caveat the summary repeats

Note 82's cascade audit: 712,124 employer-band cells, 39 disagreeing with 47L's head count, a
total of 5,111,523 against 5,111,278. A disagreement means a vintage holds more than one row for
a person, so the totals are NOT head counts. The route scores 262,089 employers at a floor of 5
person-months on the `uniform3` arm, median coverage of the code 100.0 per cent, 7.4 per cent of
coded incumbents carried from a year before 2019.
