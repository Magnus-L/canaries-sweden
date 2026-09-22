#!/usr/bin/env python3
"""
run_lane28.py -- LANE 28. Submit this file to BatchClient.

  82  the paper's findings on a route with no education in it
                                                 about five hours for ABC

WHY. The paper's firm exposure is routed through education. An education
group is given the employment-weighted mean DAIOE generative-AI percentile
of the occupations its holders worked in during 2019, and an employer is
ranked by the mean of that score over its incumbents aged 31 to 69.
Education therefore carries the exposure from the occupation to the firm,
and a referee may reasonably ask how much of the result is the education
register rather than the work. Script 82 deletes the intermediate step: an
employer is ranked directly by the employment-weighted mean DAIOE
percentile of the 2019 four-digit occupations of its OWN incumbents aged
31 to 69. Same freeze year, same incumbent restriction, same person floor,
same employment-weighted quartile cuts, no education record anywhere.

WHAT WAS WRONG WITH THE OBVIOUS WAY OF BUILDING IT, AND IS FIXED HERE.
Script 65's occupation_exposure drops the uncoded incumbents and only then
applies the floor, so its floor counts CODED incumbents; and its weight is
a single November head count while script 47j's is person-months summed
over 2019. A floor of five therefore means five coded persons in one month
on one route and five person-months on the other, which is about an order
of magnitude stricter before any coverage question arises. That, and not
missing codes, is most of why script 70's route ladder falls from 172,396
employers to 60,704, and it is why that ladder is NOT the comparison this
lane makes. Two things are changed, both reported.

  The floor is on the firm's INCUMBENTS, not on its coded incumbents, and
  on the education route's own unit: incumbent person-months in 2019,
  summed from script 47L's monthly counts, which this lane pulls through
  47L's own query if they are not on the share. The mix is then formed
  over whatever share of those incumbents carries a code, and that share
  is reported per firm and in aggregate. If the monthly counts cannot be
  had the November head count is used, the summary says so in those words,
  and the sensitivity at floors of 1, 3 and 5 is reported either way.

  The code is completed by a cascade over years, as script 80 completes
  the industry code: each incumbent PERSON takes the occupation recorded
  for him in 2019, failing that 2018, then 2017, 2016 and 2015. BACKWARD
  ONLY for the reported score: a code recorded after 2019 would break the
  paper's claim that no occupation code recorded after the freeze year
  enters anything, which is the sentence the design rests on. The source
  year travels with each person, so every fit reports the share of its
  employers coded from a year other than 2019. A forward variant (2020 and
  2021, after the backward steps) is built and reported as a SEPARATE arm,
  never as the score, so the question of what it would add has an answer.

WHAT RUNS, AND FOR HOW LONG. Parts are chosen with CANARIES_82_PARTS
(default and the setting below: ABC) and the folder with CANARIES_82_OUT.

  A  the score and what it covers, no fit. Where the cascade resolves each
     incumbent; coverage of the code among incumbents by age band before
     and after the cascade, coded and scored; the decomposition of the
     employers the old rule lost into those the floor lost, those missing
     codes lost after the full backward cascade and those both lost; the
     floor sensitivity at 1, 3 and 5; the employers each route can place,
     alone and on the two young panels; the occupation quartile against
     the education quartile with the diagonal share and the rank
     correlation; and the size and longevity of the employers one route
     places and the other does not. Thirty to forty minutes, of which the
     cascade pull is a few and the 2019 counts about seventy seconds.
  B  the headline. Equation (2) on the stock at both young bands with the
     calendar terms, exactly as script 68 estimates it, then the six-band
     profile against 41-49 as script 74 builds it, then the 22-25 headline
     at floors of 1 and 3 and on the forward arm. Six fits, an hour and a
     half to two hours.
  C  the sex split, the margins and the vintage check. The sex
     specification of Equation (2) at 22-25, hires and separations at
     22-25, and three vintage arms (the backward cascade, the 2019 code
     alone, and the 2019 incumbents re-scored from the 2021 register) on
     one panel and on the employers all three can score. Six fits and one
     small pull, two to two and a half hours.

About five hours for ABC. A stage that dies costs its own part.

SQL. One read of Arb_AGIIndivid201911_def joined to the Individ tables of
2015 to 2021 for the cascade, preceded by an INFORMATION_SCHEMA probe so
that a vintage the delivery does not hold shortens the cascade rather than
failing the pull; one read of the 2019 monthly declarations through 47L's
own query if L_counts_2019 is not on the share; and, in Part C, one read
of the same November table joined to Individ_2019 for the birth year and
Individ_2021 for the code. Every one of them is cached. No other monthly
declarations are touched, so this is safe beside any job.

THREE SLOTS ARE BETTER. The three parts share nothing but the caches, so
run_lane28a, 28b and 28c run them in three jobs at once: about two and a
half hours of wall clock instead of five. Submit 28a first and give it a
few minutes to get past the cascade pull, then submit the other two, which
will read the cache rather than pull again; three jobs pulling at once
write the same frame and the write is atomic, so a race is harmless but
costs the pull three times. Submit those three rather than this file; this
one is kept for the record and for a single-slot night.

READ RULES, fixed before the runs and printed by the script at the start
and in the summary. There is NO coefficient gate: this is a different
measure of the same object, so the estimates will differ from the
education route and that is expected. What is judged is whether the
paper's findings survive the change of route, on three questions settled
in advance.
  1. REPRODUCES if the adoption step at 22-25 is negative and
     distinguishable from zero at the five per cent level on employer
     clustering.
  2. THE PROFILE REPRODUCES if the 50-and-over band gains against 41-49
     and the young band is the lowest or second lowest of the six.
  3. THE SEX RESULT REPRODUCES if the female differential is negative and
     distinguishable from zero at the one per cent level.
All three are read on the REPORTED score, the backward cascade at a floor
of five; the floor variants and the forward arm are reported beside it and
settle nothing. Each verdict is reported explicitly and whatever the
numbers are, and every point estimate is reported beside the
education-route one: -0.0408 (0.0150) at 22-25 and -0.0394 (0.0102) at
26-30; -0.0099 (0.0121) for 22-25 against 41-49 and +0.0589 (0.0062) for
50 and over; -0.0746 (0.0142) for the female differential; +0.0787
(0.0203) for separations and -0.0032 (0.0363) for hires.

Output goes to output_82/. Export: output_82/82_summary.txt,
occ_route_coverage.csv, occ_route_headline.csv, occ_route_profile.csv,
occ_route_gender.csv, occ_route_flows.csv, occ_route_vintage.csv and the
vcov_s82_*.csv files (twelve, small). Employer and person counts below
five are suppressed before anything leaves MONA.
"""

import os
import sys
from pathlib import Path

PARTS = "ABC"
os.environ.setdefault("CANARIES_82_PARTS", PARTS)
os.environ.setdefault("CANARIES_82_OUT", "output_82")
# One exchange directory per job, so lane 28 can run beside anything else.
os.environ["CANARIES_RWORK_TAG"] = "_28"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lane  # noqa: E402

STAGES = [
    ("82_occupation_route.py", "output_82/82_summary.txt", 300),
]

if __name__ == "__main__":
    _lane.run("28", STAGES)
