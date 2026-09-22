#!/usr/bin/env python3
"""
82_occupation_route.py -- the paper's findings re-estimated on a firm
                          score that uses no education record at all.

======================================================================
  RUNS IN MONA. Every part needs the occupation cascade, which is one
  read of the November 2019 declarations joined to the Individ tables
  of 2015 to 2021; it is cached as soon as it is built, so the part
  that runs after another reads it rather than pulling again. Part C
  pulls the 2021 vintage separately. Parts are chosen with the
  environment variable CANARIES_82_PARTS (default ABC) and the folder
  with CANARIES_82_OUT (default output_82); the lane runners set both.
======================================================================

QUESTION
The paper's firm exposure is routed through education. Each education
group is given the employment-weighted mean DAIOE generative-AI
percentile of the occupations its holders worked in during 2019, and an
employer is ranked by the mean of that score over its incumbents aged 31
to 69. Education therefore carries the exposure from the occupation to
the firm, and a reader may reasonably ask how much of the result is the
education register rather than the work. This script answers it by
deleting the intermediate step: an employer is scored directly by the
employment-weighted mean DAIOE percentile of the four-digit occupations
its own incumbents aged 31 to 69 held in 2019. The freeze year, the
incumbent restriction, the person floor and the quartile logic are the
paper's; no education record enters at any point.

WHY THE SCORE IS BUILT HERE AND NOT TAKEN FROM SCRIPT 65
Script 65's occupation_exposure drops the uncoded incumbents and only
then applies the floor, so its floor counts CODED incumbents; and its
weight is a single November head count while script 47j's is person-
months summed over 2019. A floor of five therefore means five coded
persons in one month on one route and five person-months on the other,
which is about an order of magnitude stricter before any coverage
question arises. That, and not missing codes, is most of why script
70's route ladder falls from 172,396 employers to 60,704, and it is why
that ladder is not the comparison this script makes. Two things are
changed and both are reported.

  The floor is on the firm's INCUMBENTS, not on its coded incumbents,
  and on the same unit as the education route: incumbent person-months
  in 2019, summed from script 47L's own monthly counts. The occupation
  mix is then formed over whatever share of those incumbents carries a
  code, and that share is reported per firm and in aggregate. When the
  2019 monthly counts cannot be obtained the November head count is
  used instead, the summary says so in those words, and the sensitivity
  at floors of 1, 3 and 5 is reported either way.

  The code is completed by a cascade over years, as script 80 completes
  the industry code: each incumbent PERSON takes the occupation recorded
  for him in 2019, failing that 2018, then 2017, 2016 and 2015. The
  cascade is BACKWARD ONLY for the reported score. A post-2019 code
  would break the paper's claim that no occupation code recorded after
  2019 enters anything, which is the sentence the design rests on. The
  source year travels with each person, so every fit can report the
  share of its employers coded from a year other than 2019. A forward
  variant (2020 and 2021, after the backward steps) is built and
  reported as a SEPARATE arm, never as the score, so that the question
  of what it would add has an answer.

  The score is taken at THREE digits for every incumbent, from a book
  built once as the 2019 national employment-weighted mean of the
  four-digit DAIOE scores within each three-digit group. The three-digit
  code is read from the register's own column and never by truncating
  the four-digit field, which would inherit that field's missingness.

THE SCORING ARM, AND WHY THE UNIFORM ONE IS PRIMARY
The workers who lack a four-digit code are not a random subset. Under a
mixed four-then-three rule a firm scored mostly at four digits gets a
sharp score and a firm scored mostly at three gets one smoothed toward
group means and so compressed away from the extremes; quartile
assignment then depends partly on how completely the firm's workers
happen to be coded, which is a bias channel into the treatment variable
itself and not merely noise in it. Under a uniform rule the smoothing is
common to every firm and the ranking survives it, and the register
clinches it: from 2019 every coded occupation carries at least a
three-digit code, so three-digit coverage is near complete and uniform
where four-digit coverage is partial and non-random. Three arms are
fitted and printed side by side and the primary is named in every
export: uniform3 (the reported score), mixed43 and four_only (both
robustness). The read rules are read on uniform3 and on nothing else.
Do not reverse this: the mixed arm looks more precise and is not. What
the coarsening costs is measured rather than assumed, by decomposing the
employment-weighted variance of the four-digit score into between and
within three-digit groups and by the mean absolute distance of a
four-digit score from its group's book value, both reported in the
summary against an unweighted benchmark computed on the released DAIOE
panel.

DESIGN
Exposure (occ_route_exposure): the head-count-weighted mean DAIOE
percentile of the cascade occupations of the employer's incumbents aged
31 to 69 in November 2019; an employer with fewer incumbent person-
months than the floor, or with no coded incumbent at all, is not scored;
the quartile cut points are weighted by the same incumbent employment
the floor is applied to, so the top quartile holds a quarter of
incumbent employment rather than a quarter of employers. That is script
47j's incumbent_exposure with the score taken from the occupation
register instead of the education register, and nothing else changed.

build_exposure() is the whole chain in one call, and it is the ONLY
place the score is built. Script 83, which puts the rest of the paper's
exhibits on this same score, imports it rather than rebuilding it: two
constructions of one treatment variable would drift apart the first time
either was corrected, and a published table would then mix two scores
under one name.

Part A (part_a). No fit. Where the cascade resolves each incumbent;
the coverage of the code among incumbents by age band, before and after
the cascade, as a coded share and as a scored share (a code outside the
DAIOE file carries no exposure); the decomposition of the employers the
old rule lost into those the floor lost, those missing codes lost after
the full backward cascade, and those both lost; the floor sensitivity at
1, 3 and 5; the occupation quartile against the education quartile among
the employers both routes score, with the share on the diagonal and the
Spearman rank correlation; and the size and longevity of the employers
one route scores and the other does not, read off the panels the fits
themselves build.

Part B (part_b). Equation (2) on the employment stock at 22-25 and at
26-30: the cumulative tightening switch from April 2022, the interim
window from the launch to December 2023, the adoption step from January
2024, and the three calendar-quarter terms with the fourth quarter
omitted, each interacted with High x Young, under employer-by-month,
employer-by-age and month-by-age effects, Poisson pseudo-maximum
likelihood, standard errors clustered by employer. The term set is
script 78's, which is script 68's. Then the six-band profile of script
74's seasonal arm on script 70's six-band skeleton, so every coefficient
is a difference from the prime-aged band; then the 22-25 headline at
floors of 1 and 3, and on the forward-cascade arm.

Part C (part_c). The sex specification of Equation (2) at 22-25 on
script 67's panel, every term entered as High x Young, High x Female and
High x Young x Female, with employer-by-age-and-sex and
month-by-age-and-sex effects; hires and separations at 22-25 on script
54's flows with the same terms as the stock; and a vintage check in
three arms on one panel and one set of employers, the reported backward
cascade, the 2019 code alone and the 2019 incumbents re-scored from the
2021 register, so that what the later register moves and what the
cascade adds are separated rather than summed. The three do not score
the same employers, so every fit is restricted to the ones all three
score and the difference between the arms is the score and not the
sample; how many each can score is Part A's question. The birth year and therefore the population come
from the 2019 register in every arm, so a person absent from a later
register loses his code rather than leaving the sample.

READ RULES
Fixed before the run and printed at the start and in the summary. There
is no coefficient gate: a different measure gives different estimates.
Three questions are settled in advance and answered explicitly whichever
way they fall, with every point estimate beside the education-route one.

INPUTS AND OUTPUTS
Reads the caches L_counts_2019 to 2025 (script 47L), flows_2021 to 2025
(script 54), L_counts_sex_2021 to 2025 (script 67), edu_hr_weights_2019
to 2021 and edu_hr_2019 (script 47h, for the education route Part A
compares against) and L_baseline_2019 (script 47L, for the head-count
audit), and the input file daioe_quartiles.dta. Pulls and caches
L_baseline_2019_cascade.parquet and L_baseline_2019_asof2021.parquet,
and pulls L_counts_2019 through 47L's own query if it is not on the
share. Writes to output_82/: occ_route_coverage.csv,
occ_route_headline.csv, occ_route_profile.csv, occ_route_gender.csv,
occ_route_flows.csv, occ_route_vintage.csv, the vcov_s82_*.csv files and
82_summary.txt.

IN THE PAPER
Online Appendix III.2, the robustness of the exposure route: whether the
findings of Section 3 and Table 1 survive a firm score built without the
education register. Nothing here replaces a number in Table 1.
"""

import gc
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mona_common as mc

HERE = Path(__file__).resolve().parent
OUT = HERE / os.environ.get("CANARIES_82_OUT", "output_82")
PARTS = os.environ.get("CANARIES_82_PARTS", "ABC").upper()
OUT.mkdir(exist_ok=True)
CACHE = mc.CACHE_DIR

BASE_YEAR = 2019                 # the year the score is frozen at
VINTAGE = 2021                   # the later register of the vintage check
POST_FROM = "2024-01"            # adoption, as in 68 and 78
BANDS = ["22-25", "26-30"]       # the two young bands the paper reports
SEX_BAND = "22-25"               # the band the paper's sex result is on
FLOW_BAND = "22-25"              # and the band the margins are read at
PROFILE_REF = "41-49"            # the omitted band of the profile
FLOOR = 5                        # the export floor, as in mona_common
SIG5 = 1.959963984540054         # two-sided five per cent
SIG1 = 2.5758293035489004        # two-sided one per cent

# The cascade. BACKWARD ONLY for the reported score: a code recorded
# after 2019 would break the paper's claim that no occupation code from
# after the freeze year enters anything. The forward years are a
# separate arm and are never the score.
CASCADE_BACK = [2019, 2018, 2017, 2016, 2015]
CASCADE_FWD = [2020, 2021]
MAIN_ARM = "backward"
ARM_YEARS = {"backward": CASCADE_BACK, "2019_only": [BASE_YEAR],
             "forward": CASCADE_BACK + CASCADE_FWD}
PERSON_COL = "P1207_LopNr_PersonNr"

# WHICH YEARS EXIST AND WHICH COLUMN EACH ONE NEEDS. From the register
# dictionary (lab-infrastructure/oru-micro-ai/documentation/
# mona-dictionary/lisa-individual-tables.md), which is the layer that
# answers this: the verified-schema catalogue covers neither LISA nor AGI
# nor RTB and says so.
#
#   Individ_YYYY runs 1990 to 2023, so every year the cascade wants
#   exists. The years are NOT the constraint.
#
#   The column is. Ssyk4_2012_J16, which script 47L uses and which
#   includes imputed occupations for the self-employed, exists FROM 2016.
#   Ssyk4_2012 is SSYK 2012 at three and four digits from 2014, and
#   SSYK96 before that, which is a DIFFERENT CLASSIFICATION: its codes
#   would merge against DAIOE's SSYK 2012 keys and produce silent
#   nonsense rather than a failed merge. The cascade therefore stops at
#   2015 and check_cascade_years() refuses to start below it.
INDIVID_FIRST, INDIVID_LAST = 1990, 2023
SSYK_COLUMNS = {"Ssyk4_2012_J16": (2016, INDIVID_LAST),
                "Ssyk4_2012": (2014, INDIVID_LAST)}
# The three-digit column is a column of its own and is read as one. The
# four-digit field is NEVER truncated to get it: a truncated code would
# inherit the four-digit field's missingness, so exactly the workers the
# fallback exists for would be the ones it could not reach.
SSYK3_COLUMNS = {"Ssyk3_2012_J16": (2016, INDIVID_LAST),
                 "Ssyk3_2012": (2014, INDIVID_LAST)}
# THE THREE SCORING ARMS, AND WHY THE UNIFORM ONE IS PRIMARY.
#
# The workers who lack a four-digit code are not a random subset. Under
# a mixed rule a firm scored mostly at four digits gets a sharp score
# and a firm scored mostly at three gets one smoothed toward group means
# and so compressed away from the extremes, and quartile assignment then
# depends partly on how completely the firm's workers happen to be
# coded. That is a bias channel into the treatment variable itself, not
# merely noise in it. Under a uniform rule the smoothing is common to
# every firm and the ranking survives it. The register clinches it: from
# 2019 every coded occupation carries at least a three-digit code, so
# three-digit coverage is near complete and uniform where four-digit
# coverage is partial and non-random.
#
# Do not reverse this. The mixed arm looks more precise and is not.
SCORE_ARMS = ("uniform3", "mixed43", "four_only")
# What the coarsening costs UNWEIGHTED, computed on the released DAIOE
# panel (lab-infrastructure/daioe-pipeline/data/out/
# daioe_panel_ssyk2012.dta, year 2023, exp_change_genai ranked to a
# percentile) so the MONA figure has something to be checked against.
# Verified by recomputation on 22 September: the shares and the
# distances below reproduce exactly. The counts are the scored basis,
# 423 occupations in 145 groups of which 30 hold a single occupation;
# the file holds 429 rows at 2023, six of which carry no genai score,
# and counting before that drop gives the 429 / 148 / 32 reading. 423
# is the basis that matters here, being exactly the key set of
# daioe_quartiles.dta, which is what MONA scores from.
BENCH3 = {"share_between": 0.917, "share_within": 0.083,
          "mean_abs": 5.50, "median_abs": 3.55, "p90_abs": 13.36,
          "n_ssyk4": 423, "n_ssyk3": 145, "n_singleton": 30}
# Above this the uniform arm's smoothing is discarding enough that the
# choice of primary should be revisited rather than assumed.
WITHIN_ALARM = 0.20
MAIN_LEVEL = "uniform3"
ARM_LABEL = {
    "uniform3": "every incumbent from the three-digit book (PRIMARY)",
    "mixed43": "four digits where usable, the book otherwise (robustness)",
    "four_only": "four digits only (robustness)",
}
SSYK96_LAST = 2013               # 2013 and earlier are a different scheme
CASCADE_FLOOR_YEAR = 2015        # the hard stop, one year above SSYK96
# The year the code was actually observed and whether it matches the
# November employer. The delivery spells them with the J16 suffix
# (scripts 47h and 51 read SsykAr_J16 and SsykStatus_J16) and the
# dictionary lists them without; the probe picks whichever is there.
STALE_COLS = {"ssyk_ar": ("SsykAr_J16", "SsykAr"),
              "ssyk_status": ("SsykStatus_J16", "SsykStatus")}
# The DAIOE file carries 423 four-digit keys and not one of them begins
# with a zero, so a three-digit code padded to four can never merge. It
# cannot be silently rescued and it cannot silently corrupt a cell; it is
# counted as three-digit-only and reported as such.

# The incumbent floor. Five, as script 47j's, and on the same unit when
# the 2019 monthly counts can be read; the sensitivity is reported at
# all three either way.
FLOOR_MAIN = 5
FLOORS = (1, 3, 5)

CASC_CACHE = CACHE / f"L_baseline_{BASE_YEAR}_cascade.parquet"
CASC_COLS = ["employer_id", "age_group", "ssyk4", "ssyk3",
             "source_year", "ssyk_ar", "ssyk_status", "n"]
VINT_CACHE = CACHE / f"L_baseline_{BASE_YEAR}_asof{VINTAGE}.parquet"
VINT_COLS = ["employer_id", "age_group", "ssyk4", "n"]
BASE_CACHE = CACHE / f"L_baseline_{BASE_YEAR}.parquet"
COUNTS_2019_CACHE = CACHE / f"L_counts_{BASE_YEAR}.parquet"

# What the education route prints, taken from the exports the paper
# quotes: script 68's seasonal_pooled.csv (the stock and flow steps),
# script 74's contrast_seasonal.csv (the seasonal arm of the profile) and
# script 78's gender_eq2.csv (the full sex specification). Every table
# this script writes carries the matching pair, so no comparison depends
# on a reader holding two files open.
EDU_STOCK = {"22-25": (-0.0408, 0.0150), "26-30": (-0.0394, 0.0102)}
EDU_FLOW = {"hires": (-0.0032, 0.0363), "seps": (+0.0787, 0.0203)}
EDU_PROFILE = {"22-25": (-0.0099, 0.0121), "26-30": (-0.0096, 0.0084),
               "31-34": (+0.0159, 0.0063), "35-40": (+0.0090, 0.0051),
               "41-49": (0.0, 0.0), "50+": (+0.0589, 0.0062)}
EDU_FEMALE = (-0.0746, 0.0142)

NOTES = []
FAILURES = []
# Set by incumbent_floor_series(): "person-months" when 47L's 2019 monthly
# counts are available and "November head count" when they are not.
# Printed wherever the floor is mentioned, because a floor of five means
# different things in the two units.
BASIS = "unknown"

READ_RULES = [
    "READ RULES, FIXED BEFORE THE RUN:",
    "  There is NO coefficient gate. This is a different measure of the",
    "  same object, so the estimates will differ from the education route",
    "  and that is expected. What is judged is whether the paper's",
    "  findings survive the change of route, on three questions settled",
    "  before any of this was estimated:",
    "    1. REPRODUCES if the adoption step at 22-25 is negative and",
    "       distinguishable from zero at the five per cent level on",
    "       employer clustering.",
    "    2. THE PROFILE REPRODUCES if the 50-and-over band gains against",
    "       41-49 and the young band is the lowest or the second lowest",
    "       of the six.",
    "    3. THE SEX RESULT REPRODUCES if the female differential is",
    "       negative and distinguishable from zero at the one per cent",
    "       level.",
    "  All three are read on the REPORTED score and on nothing else:",
    f"  the uniform three-digit arm ({MAIN_LEVEL}), the backward cascade,",
    f"  a floor of {FLOOR_MAIN}. The mixed43 and four_only arms, the floor",
    "  variants and the forward cascade are reported beside it and settle",
    "  nothing, however they fall.",
    "  Each verdict is reported explicitly and whatever the numbers are,",
    "  and every point estimate is reported beside the education-route",
    "  one in every table.",
    "  The education route prints: the adoption step -0.0408 (0.0150) at",
    "  22-25 and -0.0394 (0.0102) at 26-30; 22-25 against 41-49 -0.0099",
    "  (0.0121) and 50 and over +0.0589 (0.0062); the female differential",
    "  -0.0746 (0.0142); separations +0.0787 (0.0203) and hires -0.0032",
    "  (0.0363).",
    f"  Employer counts below {FLOOR} are suppressed before anything",
    "  leaves MONA, and a share is suppressed with its own numerator.",
]


def opt(label, fn, *a, **kw):
    """Run one part. A part that dies is recorded and the others still
    run; nothing partial is silently treated as a result."""
    try:
        return fn(*a, **kw)
    except BaseException as ex:
        print(f"  [optional] {label} FAILED ({type(ex).__name__}: {ex})")
        traceback.print_exc()
        FAILURES.append(label)
        return None


def _mod(fname: str, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_modules():
    """
    The scripts this one reuses rather than reimplements.

    61 builds the balanced employer by band by month skeleton, 67 the
    same skeleton with sex as a fourth dimension, 70 the six-band
    skeleton and the education exposure, 74 the per-band profile terms,
    78 the term sets of Equation (2) and of its sex split, 80 the
    description of a panel's employers, 47L the 2019 pulls, 47j the
    fixed-effect list and the incumbent bands.
    Importing them rather than copying is the point: this script must be
    68, 74, 67 and 54 with one column changed, and a copied term list
    could drift away from the estimates the paper quotes. Script 65 is
    NOT reused; its floor counts coded incumbents, which is the defect
    this script exists to remove, and occ_route_exposure replaces it.
    """
    s61 = _mod("61_redated_triple.py", "s61")
    s67 = _mod("67_gender_on_the_new_design.py", "s67")
    s74 = _mod("74_contrast_seasonal.py", "s74")
    s78 = _mod("78_final_checks.py", "s78")
    s80 = _mod("80_industry_key.py", "s80")
    l47 = _mod("47L_age_baseline_exposure.py", "l47")
    l70 = _mod("70_respecifications.py", "l70")
    j47 = s61._j47()
    # 78's module-level OUT and CACHE are its own. Point them here so that
    # anything reached through it lands with this script's exports.
    s78.OUT, s78.CACHE = OUT, CACHE
    # Five hard guards, the last of them the cascade's years and
    # columns. Each is a place where this script's docstring and read
    # rules would otherwise describe a model it is not fitting.
    if s78.POST_FROM != POST_FROM:
        raise RuntimeError(
            f"78's adoption date is {s78.POST_FROM} and this script says "
            f"{POST_FROM}; the terms come from 78, so settle it there first")
    if FLOOR_MAIN != j47.MIN_FIRM_INCUMBENTS:
        raise RuntimeError(
            f"47j's incumbent floor is {j47.MIN_FIRM_INCUMBENTS} and this "
            f"script says {FLOOR_MAIN}; the two routes would then differ in "
            f"the floor as well as in the register, and nothing here would "
            f"be a comparison")
    if list(s74.BANDS) != list(l70.CONTRAST_BANDS):
        raise RuntimeError(
            f"74 has {s74.BANDS} and 70 builds the skeleton from "
            f"{l70.CONTRAST_BANDS}; the panel and the profile terms would "
            f"not agree")
    if l70.REF_BAND != PROFILE_REF:
        raise RuntimeError(
            f"70's reference band is {l70.REF_BAND} and this script says "
            f"{PROFILE_REF}; read rule 2 names the band explicitly")
    check_cascade_years()
    return s61, s67, s74, s78, s80, l47, l70, j47


def ssyk_col(year: int) -> str:
    """
    The occupation column that year needs.

    Ssyk4_2012_J16 from 2016, which is what script 47L reads and what the
    paper's own 2019 measure is built on; Ssyk4_2012 for 2014 and 2015,
    which is SSYK 2012 at three and four digits. Below 2014 the same
    column holds SSYK96 and there is no right answer, which is why the
    caller never gets one.
    """
    for col, (lo, hi) in SSYK_COLUMNS.items():
        if lo <= year <= hi:
            return col
    raise RuntimeError(
        f"no SSYK 2012 column covers {year}: the dictionary lists "
        + "; ".join(f"{c} {lo}-{hi}" for c, (lo, hi) in SSYK_COLUMNS.items())
        + f", and {year} would be SSYK96, a different classification whose "
          f"codes merge against DAIOE's keys and give silent nonsense")


def ssyk3_col(year: int) -> str:
    """The three-digit column that year needs, on the same convention."""
    for col, (lo, hi) in SSYK3_COLUMNS.items():
        if lo <= year <= hi:
            return col
    raise RuntimeError(f"no SSYK 2012 three-digit column covers {year}")


def check_cascade_years() -> dict:
    """
    Refuse to start on a cascade the register cannot support, and return
    the column each year will use.

    Three ways to get this wrong, all of them silent: reaching below 2014
    and merging SSYK96 codes against SSYK 2012 keys; reaching 2014 or
    2015 while asking for the J16 column, which does not exist before
    2016; and letting the reported cascade cross the freeze year.
    """
    plan = {}
    for y in CASCADE_BACK + CASCADE_FWD:
        if not (INDIVID_FIRST <= y <= INDIVID_LAST):
            raise RuntimeError(
                f"Individ_{y} is outside the delivered range "
                f"{INDIVID_FIRST}-{INDIVID_LAST}")
        if y < CASCADE_FLOOR_YEAR:
            raise RuntimeError(
                f"the cascade reaches {y}, below the floor year "
                f"{CASCADE_FLOOR_YEAR}. Below {SSYK96_LAST + 1} the "
                f"occupation column holds SSYK96 and the merge would be "
                f"silent nonsense rather than a failure")
        plan[y] = (ssyk_col(y), ssyk3_col(y))
    if CASCADE_FLOOR_YEAR <= SSYK96_LAST:
        raise RuntimeError(
            f"the floor year {CASCADE_FLOOR_YEAR} is not above the last "
            f"SSYK96 year {SSYK96_LAST}")
    if max(CASCADE_BACK) > BASE_YEAR:
        raise RuntimeError(
            f"the reported cascade reaches {max(CASCADE_BACK)}, after the "
            f"freeze year {BASE_YEAR}; the paper's claim that no occupation "
            f"code recorded after {BASE_YEAR} enters anything would be false")
    return plan


def load_counts(prefix: str, years, require=None):
    out = []
    for y in years:
        c = mc.read_cache(CACHE / f"{prefix}_{y}.parquet", require=require)
        if c is None:
            return None
        out.append(c)
    return pd.concat(out, ignore_index=True) if out else None


def fit(b: pd.DataFrame, tag: str, terms: list, fes: tuple,
        cluster: str = "employer_id"):
    """
    One Poisson fit. Returns (coefficient table indexed by term, the
    clustered covariance as a DataFrame or None). A failure is recorded
    and returns (None, None), because a missing row must never be read as
    a zero.
    """
    print(f"    {tag}: {len(b):,} rows, {b['employer_id'].nunique():,} firms"
          f"{mc.mem_line(' | ')}")
    t = time.time()
    try:
        r = mc.run_fepois_multi(b, OUT, tag=f"s82_{tag}", terms=terms,
                                fes=fes, cluster=cluster)
    except BaseException as ex:
        print(f"    {tag} FAILED: {type(ex).__name__}: {ex}")
        traceback.print_exc()
        r = pd.DataFrame()
    if r.empty:
        FAILURES.append(tag)
        print(f"    {tag}: FAILED, recorded and skipped")
        return None, None
    g = r.set_index("term")
    v = None
    if "vcov" in r.attrs and Path(r.attrs["vcov"]).exists():
        v = pd.read_csv(r.attrs["vcov"]).set_index("term")
    print(f"    {tag}: done in {(time.time()-t)/60:.1f} min")
    return g, v


def cnt(v) -> str:
    """A count as the summary may print it: the export floor applies to
    the text that leaves MONA as much as to the CSV beside it."""
    if v is None or v != v:
        return "(suppressed)"
    v = int(v)
    return "(suppressed)" if 0 < v < FLOOR else f"{v:,}"


def save(rows, name: str, count_col: str = "n_firms") -> pd.DataFrame:
    """
    Write one export, with the floor applied on the way out.

    Every table this script writes passes through here, so no export can
    reach the share without the suppression having run on it, and the
    rule is enforced in one place rather than remembered at six call
    sites.
    """
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if not df.empty and count_col in df.columns:
        df = mc.enforce_min_cell(df, count_col=count_col, floor=FLOOR)
    df.to_csv(OUT / name, index=False)
    return df


def tstat(coef, se) -> float:
    return float(coef) / se if se and se == se and se > 0 else np.nan


# ----------------------------------------------------------------------
# The cascade pull
# ----------------------------------------------------------------------

def individ_catalogue(conn, years) -> dict:
    """
    What INFORMATION_SCHEMA holds for each Individ table asked for.

    A CHECK, not the basis of the design. The dictionary settles which
    years exist and which column each one needs; this says whether the
    delivery agrees, and a disagreement on a backward year stops the run
    rather than quietly shortening the cascade, because the score would
    then rest on less than the design says it does.
    """
    names = ", ".join("'" + f"Individ_{y}" + "'" for y in years)
    q = (f"SELECT TABLE_NAME, COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
         f"WHERE TABLE_NAME IN ({names})")
    cat = pd.read_sql(q, conn)
    out = {}
    for y in years:
        cols = cat[cat["TABLE_NAME"].astype(str).str.lower()
                   == f"individ_{y}"]["COLUMN_NAME"].astype(str).tolist()
        out[y] = {c.lower(): c for c in cols}
    return out


def pick_col(have: dict, *names):
    """The first of `names` the table actually carries, or None."""
    for n in names:
        if n.lower() in have:
            return have[n.lower()]
    return None


def cascade_plan(conn) -> dict:
    """
    The year-by-year plan the pull runs on: which table, which four-digit
    and three-digit occupation columns, and where the staleness columns
    come from.

    The dictionary decides; the catalogue confirms. A backward year whose
    four-digit column is missing is an error, since the design says it is
    there. A year whose THREE-digit column is missing scores at four
    digits alone and the note says so, which is a reported degradation
    rather than a silent one. A forward year whose columns are missing is
    dropped from the forward arm alone.
    """
    want = check_cascade_years()
    cat = individ_catalogue(conn, sorted(want))
    plan, dropped, no3 = {}, [], []
    for y, (c4, c3) in want.items():
        have = cat.get(y, {})
        got4, got3 = pick_col(have, c4), pick_col(have, c3)
        if got4 is None:
            if y in CASCADE_BACK:
                raise RuntimeError(
                    f"the dictionary says Individ_{y} carries {c4} and the "
                    f"catalogue does not show it (the table has "
                    f"{len(have)} columns). The reported score would rest "
                    f"on a shorter cascade than the design states, so this "
                    f"stops here rather than shortening it quietly")
            dropped.append(y)
            continue
        if got3 is None:
            no3.append(y)
        plan[y] = (got4, got3)
    if dropped:
        NOTES.append(f"cascade: the forward years {dropped} carry no usable "
                     f"occupation column in this delivery and are dropped "
                     f"from the forward arm; the reported score is "
                     f"unaffected")
    if no3:
        NOTES.append(f"cascade: {no3} carry no three-digit column, so those "
                     f"years score at four digits alone and their workers "
                     f"without a four-digit code stay unscored")
    base = cat.get(BASE_YEAR, {})
    stale = {k: pick_col(base, *names) for k, names in STALE_COLS.items()}
    NOTES.append("cascade: "
                 + ", ".join(f"{y} via {c4}/{c3 or 'no three-digit'}"
                             for y, (c4, c3) in sorted(plan.items()))
                 + "; staleness from "
                 + ", ".join(f"{k} = {v or 'ABSENT'}"
                             for k, v in stale.items()))
    return {"years": plan, "stale": stale}


def _clean(year: int, col: str, alias: str | None = None) -> str:
    """
    One vintage's code, trimmed, with Statistics Sweden's missing
    conventions removed and NOT padded.

    Two things this does that 47L's own pull does not, and both matter.
    The conventions are applied BEFORE the COALESCE, so a '****' in 2019
    does not block the 2018 code. And the value is left unpadded, so the
    number of digits survives: 47L pads to four in SQL, which turns a
    three-digit code into a four-digit one that no DAIOE key matches, and
    the difference between "three digits only" and "no code" is then
    lost. The dictionary says all occupations carry at least three digits
    from 2019 and four only from 2023, so that difference is the main
    thing a reader of the coverage table needs.
    """
    c = f"{alias or 'i' + str(year)}.{col}"
    t = f"LTRIM(RTRIM(CAST({c} AS VARCHAR(8))))"
    return (f"CASE WHEN {c} IS NULL OR {t} = '' OR LEFT({t}, 1) = '*' "
            f"THEN NULL ELSE {t} END")


def cascade_sql(plan: dict) -> str:
    """
    47L's baseline pull with the occupation resolved per PERSON by a
    cascade over vintages, at BOTH levels, the year that answered carried
    out beside the codes, and the 2019 register's own staleness columns
    beside all of it.

    The year is chosen once per person, by the first vintage that holds
    any usable code at either level, and both codes are then taken from
    that same year: a four-digit code from one year beside a three-digit
    code from another would be two different jobs. Recency is settled
    first and precision second, which is the order the earlier ruling
    fixed; the alternative, preferring an older four-digit code to a
    nearer three-digit one, would buy precision with staleness, and
    staleness is what SsykAr is in the pull to measure.

    The population is 47L's and does not move: the November 2019
    declarations, aged 22 to 69 on the 2019 register's birth year.
    """
    years = [y for y in CASCADE_BACK + CASCADE_FWD if y in plan["years"]]
    if BASE_YEAR not in years:
        raise RuntimeError(
            f"Individ_{BASE_YEAR} is not usable, so the population of the "
            f"score cannot be defined as 47L defines it; refusing to pull a "
            f"different population under the same name")
    age_case = "\n".join(
        f"             WHEN {BASE_YEAR} - TRY_CAST(b.FodelseAr AS INT) "
        f"BETWEEN {lo} AND {hi} THEN '{lab}'"
        for lab, (lo, hi) in mc.AGE_GROUPS.items())
    age_case = f"CASE\n{age_case}\n             ELSE NULL END"

    def c4(y):
        return _clean(y, plan["years"][y][0])

    def c3(y):
        col = plan["years"][y][1]
        return _clean(y, col) if col else "NULL"

    def has(y):
        return f"({c4(y)} IS NOT NULL OR {c3(y)} IS NOT NULL)"

    src = ("CASE " + " ".join(f"WHEN {has(y)} THEN '{y}'" for y in years)
           + " ELSE 'none' END")
    code4 = ("CASE " + " ".join(
        f"WHEN {has(y)} THEN COALESCE({c4(y)}, '____')" for y in years)
        + " ELSE '____' END")
    code3 = ("CASE " + " ".join(
        f"WHEN {has(y)} THEN COALESCE({c3(y)}, '___')" for y in years)
        + " ELSE '___' END")
    ar = plan["stale"].get("ssyk_ar")
    st = plan["stale"].get("ssyk_status")
    ar_e = f"LTRIM(RTRIM(CAST(b.{ar} AS VARCHAR(8))))" if ar else "'unknown'"
    st_e = f"LTRIM(RTRIM(CAST(b.{st} AS VARCHAR(8))))" if st else "'unknown'"
    joins = "\n".join(
        f"    LEFT JOIN dbo.Individ_{y} i{y}\n"
        f"      ON agi.P1207_LOPNR_PERSONNR = i{y}.{PERSON_COL}"
        for y in years)
    return f"""
    SELECT agi.P1207_LOPNR_PEORGNR AS employer_id,
           {age_case} AS age_group,
           {code4} AS ssyk4,
           {code3} AS ssyk3,
           {src} AS source_year,
           {ar_e} AS ssyk_ar,
           {st_e} AS ssyk_status,
           COUNT(DISTINCT agi.P1207_LOPNR_PERSONNR) AS n
    FROM dbo.Arb_AGIIndivid{BASE_YEAR}11_def agi
    LEFT JOIN dbo.Individ_{BASE_YEAR} b
      ON agi.P1207_LOPNR_PERSONNR = b.{PERSON_COL}
{joins}
    WHERE {BASE_YEAR} - TRY_CAST(b.FodelseAr AS INT) BETWEEN 22 AND 69
    GROUP BY agi.P1207_LOPNR_PEORGNR, {age_case}, {code4}, {code3}, {src},
             {ar_e}, {st_e}
    """


def baseline_cascade() -> pd.DataFrame:
    """The cascade frame, cached the moment it is built so that three
    jobs building it at once cost one pull and the write is atomic."""
    b = mc.read_cache(CASC_CACHE, require=CASC_COLS)
    if b is not None:
        print(f"  cascade baseline: cached ({len(b):,} rows)")
        return b
    t = time.time()
    conn = mc.connect()
    try:
        plan = cascade_plan(conn)
        b = pd.read_sql(cascade_sql(plan), conn)
    finally:
        try:
            conn.close()
        except Exception:
            pass
    for c in ("ssyk3", "source_year", "ssyk_ar", "ssyk_status"):
        b[c] = b[c].astype(str)
    mc.write_cache(b, CASC_CACHE)
    print(f"  cascade baseline: {len(b):,} rows ({time.time()-t:.0f}s)")
    return b


def baseline_vintage_sql(vintage: int, col: str) -> str:
    """
    The same query with ONE vintage supplying the code and the 2019
    register supplying the birth year.

    The population must not move. 47L reads the band and the code from
    one Individ table, so simply joining a later one would drop every
    worker the later register does not hold, and the vintage check would
    then measure attrition as well as re-coding. Here the birth year,
    and therefore the age band and the sample filter, come from the 2019
    register in every arm; a worker the later register does not hold
    keeps his place and loses his code to the '____' convention.

    The column is the one the dictionary gives that year, which for any
    year from 2016 is the J16 column the reported score uses.
    """
    age_case = "\n".join(
        f"             WHEN {BASE_YEAR} - TRY_CAST(b.FodelseAr AS INT) "
        f"BETWEEN {lo} AND {hi} THEN '{lab}'"
        for lab, (lo, hi) in mc.AGE_GROUPS.items())
    age_case = f"CASE\n{age_case}\n             ELSE NULL END"
    code = f"COALESCE({_clean(vintage, col, alias='v')}, '____')"
    return f"""
    SELECT agi.P1207_LOPNR_PEORGNR AS employer_id,
           {age_case} AS age_group,
           {code} AS ssyk4,
           COUNT(DISTINCT agi.P1207_LOPNR_PERSONNR) AS n
    FROM dbo.Arb_AGIIndivid{BASE_YEAR}11_def agi
    LEFT JOIN dbo.Individ_{BASE_YEAR} b
      ON agi.P1207_LOPNR_PERSONNR = b.{PERSON_COL}
    LEFT JOIN dbo.Individ_{vintage} v
      ON agi.P1207_LOPNR_PERSONNR = v.{PERSON_COL}
    WHERE {BASE_YEAR} - TRY_CAST(b.FodelseAr AS INT) BETWEEN 22 AND 69
    GROUP BY agi.P1207_LOPNR_PEORGNR, {age_case}, {code}
    """


def baseline_vintage(vintage: int) -> pd.DataFrame:
    """The same incumbents, coded as the register of `vintage` has them."""
    b = mc.read_cache(VINT_CACHE, require=VINT_COLS)
    if b is None:
        t = time.time()
        conn = mc.connect()
        try:
            col = ssyk_col(vintage)
            have = individ_catalogue(conn, [vintage]).get(vintage, {})
            got = pick_col(have, col)
            if got is None:
                raise RuntimeError(
                    f"the dictionary says Individ_{vintage} carries {col} "
                    f"and the catalogue does not show it")
            b = pd.read_sql(baseline_vintage_sql(vintage, got), conn)
        finally:
            try:
                conn.close()
            except Exception:
                pass
        mc.write_cache(b, VINT_CACHE)
        print(f"  baseline as of {vintage}: {len(b):,} rows "
              f"({time.time()-t:.0f}s)")
    else:
        print(f"  baseline as of {vintage}: cached ({len(b):,} rows)")
    b = b.copy()
    b["source_year"] = str(vintage)
    return b


def cascade_audit(casc: pd.DataFrame) -> None:
    """
    The head count per employer and band must equal 47L's own baseline.

    The cascade joins seven register tables to one declaration table. A
    vintage holding two rows for one person would split him across two
    codes and, because the count is a COUNT(DISTINCT person), inflate the
    employer's total. That is the one way this pull can differ from 47L's
    without failing, so it is checked against 47L's frame when that frame
    is on the share rather than assumed away.
    """
    ref = mc.read_cache(BASE_CACHE, require=["employer_id", "age_group", "n"])
    if ref is None:
        NOTES.append("cascade audit: L_baseline_2019 is not on the share, so "
                     "the head counts are not checked against 47L's own pull")
        return
    a = casc.groupby(["employer_id", "age_group"], observed=True)["n"].sum()
    b = ref.groupby(["employer_id", "age_group"], observed=True)["n"].sum()
    j = pd.concat([a.rename("casc"), b.rename("ref")], axis=1).fillna(0)
    bad = int((j["casc"] != j["ref"]).sum())
    msg = (f"cascade audit: {len(j):,} employer-band cells, {bad:,} disagree "
           f"with 47L's head count"
           + ("" if not bad else
              f"; total {int(j['casc'].sum()):,} against "
              f"{int(j['ref'].sum()):,}. A disagreement means a vintage holds "
              f"more than one row for a person and the totals are NOT the "
              f"head count; say so before quoting any coverage figure"))
    print(f"  {msg}")
    NOTES.append(msg)


def floor_counts_2019(l47):
    """
    47L's monthly counts for the base year, which is what the education
    route's floor is summed from. Read from the cache, pulled through
    47L's own query when it is not there, and None when neither works.
    """
    need = ["employer_id", "year_month", "age_group", "n_emp"]
    c = mc.read_cache(COUNTS_2019_CACHE, require=need)
    if c is not None:
        return c
    try:
        t = time.time()
        conn = mc.connect()
        try:
            c = l47.q_counts(BASE_YEAR, conn)
        finally:
            try:
                conn.close()
            except Exception:
                pass
        mc.write_cache(c, COUNTS_2019_CACHE)
        print(f"  {BASE_YEAR} monthly counts: {len(c):,} cells "
              f"({time.time()-t:.0f}s)")
        return c
    except BaseException as ex:
        print(f"  {BASE_YEAR} monthly counts FAILED ({type(ex).__name__}: "
              f"{str(ex)[:120]})")
        return None


def incumbent_floor_series(l47, casc: pd.DataFrame, j47) -> pd.Series:
    """
    The per-employer quantity the floor and the quartile cuts use, and
    the unit it is in.

    The education route floors on incumbent person-months summed over
    2019, so that is what this one floors on too. If those counts cannot
    be had, the November head count is used, the summary says so in
    those words, and the sensitivity at 1, 3 and 5 is the answer rather
    than a reassurance: the two are not the same unit and a floor of
    five means different things in them.
    """
    global BASIS
    c = floor_counts_2019(l47)
    if c is not None:
        BASIS = "person-months"
        NOTES.append(f"floor: on incumbent PERSON-MONTHS in {BASE_YEAR}, the "
                     f"same unit as the education route's, so the two floors "
                     f"are commensurable")
        d = c[c["age_group"].astype(str).isin(j47.INCUMBENT_BANDS)]
        s = d.groupby("employer_id", observed=True)["n_emp"].sum()
    else:
        BASIS = "November head count"
        NOTES.append(f"floor: the {BASE_YEAR} monthly counts could not be "
                     f"read, so the floor is on the NOVEMBER HEAD COUNT and "
                     f"is NOT commensurable with the education route's "
                     f"person-months; the sensitivity at "
                     + ", ".join(str(f) for f in FLOORS)
                     + " is the answer to that")
        d = casc[casc["age_group"].astype(str).isin(j47.INCUMBENT_BANDS)]
        s = d.groupby("employer_id", observed=True)["n"].sum()
    return s.rename("n_floor")


# ----------------------------------------------------------------------
# The score
# ----------------------------------------------------------------------

def build_book3(casc: pd.DataFrame, daioe: pd.DataFrame) -> tuple:
    """
    The three-digit score book: for each three-digit group, the 2019
    NATIONAL employment-weighted mean of the four-digit DAIOE scores in
    it, and what the coarsening costs.

    THE WEIGHTS ARE NATIONAL AND THE BOOK IS FIXED. A book built inside
    the firm would embed the firm's own occupational composition and the
    score would stop being an external measure of the work and start
    being a description of the employer. The weights are the head count
    by four-digit code in the November 2019 employed population of this
    project's own data, all ages 22 to 69, on the 2019 code alone, and
    the book is computed once and used everywhere.

    The DAIOE release publishes no three-digit product (the v1.1.0
    scores carry ssyk2012 at four digits, isco08, soc2010 and
    onetsoc2010), so it is built here and exported so the appendix can
    cite it.

    Returns (book, cost). `book` is ssyk3, score3, n_weight and the
    number of four-digit codes behind it; `cost` carries the
    employment-weighted variance of the four-digit score decomposed into
    between and within three-digit groups, the mean absolute difference
    between a worker's four-digit score and his group's book value,
    which is the error the coarsening makes on the workers who carry a
    four-digit code, and the three-digit groups holding a single
    four-digit occupation, where the arms are identical by construction
    and nothing is discarded at all.
    """
    b = casc.copy()
    b["n"] = pd.to_numeric(b["n"], errors="coerce").fillna(0).astype(int)
    raw = b["ssyk4"].astype(str).str.strip()
    b = b[(raw.str.len() == 4) & (b["source_year"].astype(str)
                                  == str(BASE_YEAR)) & (b["n"] > 0)]
    b = b.assign(ssyk4=raw[b.index]).merge(daioe, on="ssyk4", how="inner")
    if b.empty:
        return (pd.DataFrame(columns=["ssyk3", "score3", "n_weight",
                                      "n_ssyk4"]), {})
    w = (b.groupby("ssyk4", observed=True)
         .agg(n_weight=("n", "sum"), score=("score", "first")).reset_index())
    w["ssyk3"] = w["ssyk4"].str[:3]
    w["ws"] = w["score"] * w["n_weight"]
    book = (w.groupby("ssyk3", observed=True)
            .agg(ws=("ws", "sum"), n_weight=("n_weight", "sum"),
                 n_ssyk4=("ssyk4", "nunique")).reset_index())
    book["score3"] = book["ws"] / book["n_weight"].clip(lower=1)
    book = book[["ssyk3", "score3", "n_weight", "n_ssyk4"]]
    # What the coarsening discards, employment weighted over the same
    # population: the between share is the information the three-digit
    # book keeps and the within share is what it throws away.
    j = w.merge(book[["ssyk3", "score3"]], on="ssyk3", how="left")
    tot = float(j["n_weight"].sum())
    mean = float((j["score"] * j["n_weight"]).sum() / max(tot, 1))
    v_tot = float((j["n_weight"] * (j["score"] - mean) ** 2).sum()
                  / max(tot, 1))
    v_bet = float((j["n_weight"] * (j["score3"] - mean) ** 2).sum()
                  / max(tot, 1))
    mad = float((j["n_weight"] * (j["score"] - j["score3"]).abs()).sum()
                / max(tot, 1))
    single = book[book["n_ssyk4"] == 1]
    cost = {"n_workers": int(tot), "n_ssyk4": int(len(w)),
            "n_ssyk3": int(len(book)),
            "n_ssyk3_single_occupation": int(len(single)),
            "employment_share_in_single_groups":
                float(single["n_weight"].sum()) / max(tot, 1),
            "mean_score": mean,
            "variance_total": v_tot, "variance_between": v_bet,
            "variance_within": v_tot - v_bet,
            "share_between": v_bet / v_tot if v_tot > 0 else np.nan,
            "share_within": (v_tot - v_bet) / v_tot if v_tot > 0 else np.nan,
            "mean_abs_difference": mad}
    return book, cost


def classify_codes(b: pd.DataFrame, daioe: pd.DataFrame,
                   book: pd.DataFrame) -> pd.DataFrame:
    """
    Which level scores each incumbent, and with what.

    Both levels are computed for every incumbent and neither is
    preferred here: the four-digit DAIOE score where the code is four
    digits and in the file, and the three-digit book value from the
    register's own three-digit column. A worker no level can score
    leaves the mix; his employer still counts toward the floor, because
    the floor is on the firm's incumbents and not on its coded ones.

    THE THREE-DIGIT CASE IS A LEVEL, NOT A MISSING VALUE. The dictionary
    says all occupations carry at least a three-digit code from 2019 and
    a four-digit one only from 2023, so the earlier cascade years lose on
    the four-digit field and not on the register. Padding cannot confuse
    the levels: not one of DAIOE's 423 keys begins with a zero, so a
    three-digit value padded to four matches nothing.

    Adds: digits, coded, four_digit, scorable (four-digit and in DAIOE),
    three_digit_only, lost_at_merge, ssyk3, score4, score3, has4, has3.
    Which of the two a given arm uses is arm_score's business, not this
    function's: all three arms read the same classification.
    """
    b = b.copy()
    raw4 = b["ssyk4"].astype(str).str.strip()
    b["digits"] = np.where(raw4 == "____", 0, raw4.str.len())
    b["ssyk4"] = raw4.str.zfill(4)
    codes = set(daioe["ssyk4"].astype(str))
    b["coded"] = (raw4 != "____").astype(int)
    four = (b["coded"] == 1) & (b["digits"] == 4)
    b["scorable"] = (four & b["ssyk4"].isin(codes)).astype(int)
    b["four_digit"] = four.astype(int)
    b["lost_at_merge"] = (four & (b["scorable"] == 0)).astype(int)
    b["three_digit_only"] = ((b["coded"] == 1) & (b["digits"] < 4)
                             & (b["scorable"] == 0)).astype(int)
    # the second level, from the register's OWN three-digit column. The
    # four-digit field is never truncated to get it: a truncated code
    # would inherit that field's missingness, so exactly the workers the
    # fallback exists for would be the ones it could not reach.
    raw3 = (b["ssyk3"].astype(str).str.strip() if "ssyk3" in b.columns
            else pd.Series("___", index=b.index))
    b["ssyk3"] = np.where(raw3.isin(["", "___", "nan", "None"]), "___",
                          raw3.str.zfill(3))
    b["score4"] = np.where(b["scorable"] == 1,
                           b["ssyk4"].map(dict(zip(daioe["ssyk4"].astype(str),
                                                   daioe["score"]))), np.nan)
    b3 = (dict(zip(book["ssyk3"].astype(str), book["score3"]))
          if len(book) else {})
    b["score3"] = b["ssyk3"].map(b3)
    b["has4"] = b["score4"].notna().astype(int)
    b["has3"] = b["score3"].notna().astype(int)
    return b


def arm_score(inc: pd.DataFrame, arm: str) -> tuple:
    """
    The score each incumbent carries under one arm, and at which level.

    uniform3  every incumbent from the three-digit book, so the
              smoothing is the same for every firm and the ranking
              survives it. This is the reported score.
    mixed43   four digits where they are usable and the book otherwise.
              Sharper where the coding is complete and smoothed where it
              is not, which is a firm characteristic and therefore a
              bias channel; reported as robustness only.
    four_only the four-digit score alone, the workers without one
              leaving the mix.
    """
    if arm not in SCORE_ARMS:
        raise RuntimeError(f"unknown scoring arm {arm!r}; the arms are "
                           f"{SCORE_ARMS}")
    if arm == "uniform3":
        score = inc["score3"]
        level = np.where(inc["has3"] == 1, "three_digit", "unscored")
    elif arm == "four_only":
        score = inc["score4"]
        level = np.where(inc["has4"] == 1, "four_digit", "unscored")
    else:
        score = inc["score4"].where(inc["has4"] == 1, inc["score3"])
        level = np.where(inc["has4"] == 1, "four_digit",
                         np.where(inc["has3"] == 1, "three_digit",
                                  "unscored"))
    return score, pd.Series(level, index=inc.index)


def incumbent_frame(casc: pd.DataFrame, daioe: pd.DataFrame,
                    book: pd.DataFrame, j47) -> pd.DataFrame:
    """
    One row per employer, band, code pair, source year and staleness
    cell, restricted to the incumbents, with the scoring level attached.
    """
    b = casc.copy()
    b["source_year"] = b["source_year"].astype(str)
    b["n"] = pd.to_numeric(b["n"], errors="coerce").fillna(0).astype(int)
    b = b[b["age_group"].astype(str).isin(j47.INCUMBENT_BANDS) & (b["n"] > 0)]
    return classify_codes(b, daioe, book)


def occ_route_exposure(inc: pd.DataFrame, nfloor: pd.Series,
                       floor: int = FLOOR_MAIN, years=None,
                       arm: str = MAIN_LEVEL) -> pd.DataFrame:
    """
    The firm score and quartile, on 47j's rules with the occupation
    register in place of the education register.

    THE FLOOR IS ON THE FIRM'S INCUMBENTS, not on its coded incumbents:
    an employer qualifies on the same basis as on the education route,
    and the mix is then formed over whatever share of those incumbents
    carries a usable code at either level. An employer no level can
    score at all cannot have a mix and is not scored; that is a
    different loss from the floor and Part A counts the two separately.

    `arm` is the scoring rule: uniform3 (the reported one), mixed43 or
    four_only. The floor, the window and the quartile logic do not
    depend on it, so the three arms differ in the score and in nothing
    else.

    The quartile cut points are weighted by the same incumbent
    employment the floor is applied to, so the top quartile holds a
    quarter of incumbent employment rather than a quarter of employers.

    Returns employer_id, fq, mix, n (the weight), n_coded, n_nov,
    coverage, share_not_2019 and share_three_digit.
    """
    years = [str(y) for y in (years or ARM_YEARS[MAIN_ARM])]
    cols = ["employer_id", "fq", "mix", "n", "n_coded", "n_nov",
            "coverage", "share_not_2019", "share_three_digit"]
    tot = inc.groupby("employer_id", observed=True)["n"].sum().rename("n_nov")
    score, level = arm_score(inc, arm)
    keep = score.notna() & inc["source_year"].isin(years)
    u = inc[keep].assign(score=score[keep], level=level[keep])
    if u.empty:
        return pd.DataFrame(columns=cols)
    u = u.assign(ws=u["score"] * u["n"])
    fy = (u.groupby("employer_id", observed=True)
          .agg(ws=("ws", "sum"), n_coded=("n", "sum")).reset_index())
    for name, mask in (("n_not_2019", u["source_year"] != str(BASE_YEAR)),
                       ("n_three", u["level"] == "three_digit")):
        add = (u[mask].groupby("employer_id", observed=True)["n"].sum()
               .rename(name))
        fy = fy.merge(add, on="employer_id", how="left")
        fy[name] = fy[name].fillna(0)
    fy = fy.merge(tot, on="employer_id", how="left")
    fy = fy.merge(nfloor.rename("n"), on="employer_id", how="left")
    # An employer with no floor quantity at all is one the 2019 counts do
    # not hold; it cannot pass a floor it has no value for, and dropping
    # it silently would hide that, so it is dropped and counted.
    lost = int(fy["n"].isna().sum())
    if lost:
        NOTES.append(f"score: {lost:,} employers carry a scored incumbent "
                     f"but no 2019 {BASIS}, and cannot be put to the floor")
    fy = fy[fy["n"].notna()]
    fy = fy[fy["n"] >= floor]
    if fy.empty:
        return pd.DataFrame(columns=cols)
    fy["mix"] = fy["ws"] / fy["n_coded"]
    fy["coverage"] = fy["n_coded"] / fy["n_nov"].clip(lower=1)
    fy["share_not_2019"] = fy["n_not_2019"] / fy["n_coded"].clip(lower=1)
    fy["share_three_digit"] = fy["n_three"] / fy["n_coded"].clip(lower=1)
    o = np.argsort(fy["mix"].to_numpy(), kind="stable")
    v, w = fy["mix"].to_numpy()[o], fy["n"].to_numpy()[o]
    cum = np.cumsum(w) / w.sum()
    cuts = [float(v[np.searchsorted(cum, q, side="left")])
            for q in (0.25, 0.5, 0.75)]
    fy["fq"] = np.searchsorted(np.asarray(cuts), fy["mix"].to_numpy(),
                               side="right") + 1
    return fy[cols].reset_index(drop=True)


def old_rule_scored(inc: pd.DataFrame) -> set:
    """
    The employers script 65's rule scores: at least five CODED
    incumbents in November 2019 on the 2019 code alone. Kept only so
    that Part A can decompose what that rule lost.
    """
    u = inc[(inc["scorable"] == 1)
            & (inc["source_year"] == str(BASE_YEAR))]
    if u.empty:
        return set()
    s = u.groupby("employer_id", observed=True)["n"].sum()
    return set(s[s >= FLOOR_MAIN].index)


def build_exposure(l47, l70, j47, arm: str = MAIN_LEVEL,
                   floor: int = FLOOR_MAIN, years=None,
                   daioe: pd.DataFrame = None,
                   audit: bool = True) -> dict:
    """
    THE OCCUPATION-ROUTE SCORE, BUILT ONCE AND IN ONE PLACE.

    The question it answers: which employers are highly exposed, when
    the exposure comes from the occupations the employer's own
    incumbents aged 31 to 69 held in 2019 and from no education record
    at all.

    WHY THIS IS A FUNCTION AND NOT A BLOCK INSIDE main(). Script 83 puts
    the rest of the paper's exhibits on this same score, and a second
    script that rebuilt it would be a second definition of the treatment
    variable: the two would drift apart the first time either was
    corrected, and a published table would then mix two scores under one
    name. So the whole chain lives here and both scripts call it.

    What it runs, in order: the cascade pull (cached the moment it is
    built); the head-count audit against 47L's own baseline; the
    three-digit book and what the coarsening costs; the incumbent frame;
    the floor series and the unit it is in; and the firm score and
    quartile. It prints what each step found, because both callers want
    the same lines in their own log.

    Inputs are the modules load_modules() returns, plus the arm
    (uniform3 is the reported one), the floor and the cascade years.
    Returns a dict: daioe, casc, book, cost, inc, nfloor, exposure,
    basis, arm and floor.
    """
    if daioe is None:
        daioe = l70.daioe_scores()
    casc = baseline_cascade()
    if audit:
        opt("cascade audit", cascade_audit, casc)
    book, cost = build_book3(casc, daioe)
    print(f"  three-digit book: {len(book):,} groups from "
          f"{cost.get('n_ssyk4', 0):,} four-digit codes; "
          f"{cost.get('share_within', float('nan')):.1%} of the "
          f"employment-weighted variance is WITHIN groups and is what the "
          f"book discards (unweighted benchmark "
          f"{BENCH3['share_within']:.1%}); mean absolute difference "
          f"{cost.get('mean_abs_difference', float('nan')):.2f} against "
          f"{BENCH3['mean_abs']:.2f}")
    if cost.get("share_within", 0) > WITHIN_ALARM:
        print(f"  *** the within-group share is above {WITHIN_ALARM:.0%}: "
              f"the choice of {MAIN_LEVEL} as the primary arm should be "
              f"revisited, not assumed")
    inc = incumbent_frame(casc, daioe, book, j47)
    nfloor = incumbent_floor_series(l47, casc, j47)
    print(f"  floor basis: {BASIS}; {len(nfloor):,} employers carry one")
    expo = occ_route_exposure(inc, nfloor, floor,
                              years or ARM_YEARS[MAIN_ARM], arm=arm)
    if expo.empty:
        raise RuntimeError("no employer could be scored on the occupation "
                           "route; there is nothing to estimate")
    shares = (expo.groupby("fq")["n"].sum() / expo["n"].sum())
    # The arm is named only when it is not the reported one, so the line
    # this script has always printed is unchanged for the default call.
    msg = (f"occupation route{'' if arm == MAIN_LEVEL else f' ({arm})'}: "
           f"{len(expo):,} employers scored at a "
           f"floor of {floor} {BASIS}, median coverage of the code "
           f"{expo['coverage'].median():.1%}, "
           f"{expo['share_not_2019'].mean():.1%} of coded incumbents from a "
           f"year before {BASE_YEAR}; quartile shares of incumbent "
           f"employment " + " ".join(f"Q{int(k)} {v:.2f}"
                                     for k, v in shares.items()))
    print(f"  {msg}")
    NOTES.append(msg)
    return {"daioe": daioe, "casc": casc, "book": book, "cost": cost,
            "inc": inc, "nfloor": nfloor, "exposure": expo, "basis": BASIS,
            "arm": arm, "floor": floor}


# ----------------------------------------------------------------------
# Part A: the score and what it covers. No fit.
# ----------------------------------------------------------------------

def cascade_rows(inc: pd.DataFrame, plan_cols: dict, j47) -> list:
    """
    What each step of the cascade resolves, and what survives the merge.

    Coverage gets worse going back, and for reasons that have to be
    separated rather than assumed. Per cascade year: the incumbents it
    resolves, how many of those carry a four-digit code, how many are
    scorable, how many are three-digit only, and how many are lost at
    the DAIOE merge rather than at the register. The column the year
    used travels with the row, because 2015 reads a different column
    from 2016 onwards and a reader must be able to see which.
    """
    tot = int(inc["n"].sum())
    rows = []
    for y in [str(x) for x in CASCADE_BACK + CASCADE_FWD] + ["none"]:
        d = inc[inc["source_year"] == y]
        k = int(d["n"].sum())
        if k == 0 and y not in (str(BASE_YEAR), "none"):
            continue
        col = plan_cols.get(int(y)) if y != "none" else None
        for item, val in (
                ("incumbents_resolved", k),
                ("four_digit", int((d["n"] * d["four_digit"]).sum())),
                ("scorable", int((d["n"] * d["scorable"]).sum())),
                ("three_digit_only",
                 int((d["n"] * d["three_digit_only"]).sum())),
                ("lost_at_daioe_merge",
                 int((d["n"] * d["lost_at_merge"]).sum()))):
            rows.append({"panel": "cascade", "block": "step", "group": y,
                         "item": item, "n_employers": np.nan, "n_obs": val,
                         "share": val / max(tot, 1), "value": float(val),
                         "source_column": col})
    rows.append({"panel": "cascade", "block": "step", "group": "all",
                 "item": "incumbents", "n_employers": np.nan, "n_obs": tot,
                 "share": 1.0, "value": float(tot), "source_column": None})
    return rows


def staleness_rows(inc: pd.DataFrame) -> list:
    """
    How old the 2019 codes already are, and whether they match the
    November employer.

    SsykAr is the year the occupation was actually observed and
    SsykStatus whether the code agrees with the November job, both read
    from the 2019 register. This is direct evidence on staleness and it
    is better than the vintage re-scoring, which can only bound it; the
    re-scoring arm is kept as well, in Part C.
    """
    tot = int(inc["n"].sum())
    rows = []
    for col, block in (("ssyk_ar", "ssyk_ar"),
                       ("ssyk_status", "ssyk_status")):
        if col not in inc.columns:
            continue
        g = inc.groupby(inc[col].astype(str), observed=True)["n"].sum()
        for k, v in g.sort_index().items():
            rows.append({"panel": "baseline 2019", "block": block,
                         "group": str(k), "item": "incumbents",
                         "n_employers": np.nan, "n_obs": int(v),
                         "share": int(v) / max(tot, 1), "value": float(v),
                         "source_column": None})
    return rows


def coverage_by_band(casc: pd.DataFrame, daioe: pd.DataFrame,
                     book: pd.DataFrame, j47) -> list:
    """
    The share of incumbent head count carrying a usable occupation code,
    by age band, BEFORE and AFTER the cascade.

    Three shares per band and per arm, not one. `coded` is the register
    having given a code at all; `scored` is that code being four digits
    and in the DAIOE file; `three_digit` is the gap between them that is
    the register's three-digit coding rather than a missing record. The
    dictionary says all occupations carry at least three digits from
    2019 and four only from 2023, so the middle share is exactly what
    the cascade's earlier years are expected to lose on.
    """
    b = classify_codes(casc.copy(), daioe, book)
    b["source_year"] = b["source_year"].astype(str)
    b["n"] = pd.to_numeric(b["n"], errors="coerce").fillna(0).astype(int)
    b = b[b["age_group"].notna() & (b["n"] > 0)]
    arms = {"2019_only": [str(BASE_YEAR)],
            "backward": [str(y) for y in CASCADE_BACK],
            "forward": [str(y) for y in CASCADE_BACK + CASCADE_FWD]}
    order = [a for a in mc.AGE_GROUPS] + ["31-69 incumbents"]
    rows = []
    for band in order:
        d = (b[b["age_group"] == band] if band != "31-69 incumbents"
             else b[b["age_group"].astype(str).isin(j47.INCUMBENT_BANDS)])
        if d.empty:
            continue
        tot = int(d["n"].sum())
        firms = int(d["employer_id"].nunique())
        for arm, yrs in arms.items():
            m = d["source_year"].isin(yrs)
            for what, col in (("coded", "coded"), ("scored", "scorable"),
                              ("three_digit", "three_digit_only")):
                k = int((d["n"] * d[col] * m).sum())
                rows.append({"panel": "baseline 2019", "block": "coverage",
                             "group": band, "item": f"{what}_share_{arm}",
                             "n_employers": firms, "n_obs": tot,
                             "share": k / max(tot, 1), "value": float(k),
                             "source_column": None})
    return rows


def loss_rows(panel: str, emp_ids, inc: pd.DataFrame, nfloor: pd.Series,
              scored_old: set, scored_new: set) -> tuple:
    """
    The employers the old rule lost, split by what lost them.

    The old rule (script 65's) needed five CODED November incumbents on
    the 2019 code alone. The new one needs the firm's incumbents to reach
    the floor and one coded incumbent after the backward cascade. Every
    employer that the old rule failed is placed in exactly one group:
    recovered because the floor moved, recovered because the cascade
    found a code, still lost to the floor, still lost to missing codes,
    or lost to both. The five groups partition the loss, so a reader can
    see which of the three causes is doing the work rather than take it
    on trust.
    """
    ids = pd.Index(sorted(set(emp_ids)))
    nf = nfloor.reindex(ids).fillna(0)
    sc = (inc[inc["scorable"] == 1]
          .groupby(["employer_id", "source_year"], observed=True)["n"].sum()
          .unstack(fill_value=0))
    back = [str(y) for y in CASCADE_BACK if str(y) in sc.columns]
    c19 = (sc[str(BASE_YEAR)] if str(BASE_YEAR) in sc.columns
           else pd.Series(0, index=sc.index)).reindex(ids).fillna(0)
    cbk = (sc[back].sum(axis=1) if back
           else pd.Series(0, index=sc.index)).reindex(ids).fillna(0)
    old = pd.Series(ids.isin(list(scored_old)), index=ids)
    new = pd.Series(ids.isin(list(scored_new)), index=ids)
    big = nf >= FLOOR_MAIN
    groups = {
        "scored_by_both_rules": old & new,
        "recovered_by_the_floor": (~old) & new & (c19 >= 1),
        "recovered_by_the_cascade": (~old) & new & (c19 < 1),
        "lost_to_the_floor_only": (~old) & (~new) & (~big) & (cbk >= 1),
        "lost_to_missing_codes_only": (~old) & (~new) & big & (cbk < 1),
        "lost_to_both": (~old) & (~new) & (~big) & (cbk < 1),
        "scored_old_but_not_new": old & (~new),
    }
    n = int(len(ids))
    rows = [{"panel": panel, "block": "loss", "group": "all",
             "item": "employers", "n_employers": n, "n_obs": n,
             "share": 1.0, "value": float(n)}]
    for name, m in groups.items():
        k = int(m.sum())
        rows.append({"panel": panel, "block": "loss", "group": name,
                     "item": "n_employers", "n_employers": k, "n_obs": n,
                     "share": k / max(n, 1), "value": float(k)})
    summ = {"panel": panel, "n": n,
            **{k: int(v.sum()) for k, v in groups.items()}}
    return rows, summ


def floor_rows(panel: str, inc: pd.DataFrame, nfloor: pd.Series,
               emp_ids=None) -> tuple:
    """
    How many employers the score reaches at each floor, and how much
    incumbent employment they hold.

    The floor is a choice, the two routes measure it in different units
    when the 2019 monthly counts are missing, and a reader is entitled to
    see what the choice costs rather than to be told it is small.
    """
    rows, summ = [], {}
    for f in FLOORS:
        e = occ_route_exposure(inc, nfloor, floor=f)
        if emp_ids is not None:
            e = e[e["employer_id"].isin(set(emp_ids))]
        k = int(len(e))
        rows.append({"panel": panel, "block": "floor", "group": f"floor_{f}",
                     "item": "n_employers", "n_employers": k,
                     "n_obs": int(nfloor.shape[0]),
                     "share": k / max(int(nfloor.shape[0]), 1),
                     "value": float(e["n"].sum()) if k else 0.0})
        rows.append({"panel": panel, "block": "floor", "group": f"floor_{f}",
                     "item": "median_coverage", "n_employers": k,
                     "n_obs": k, "share": np.nan,
                     "value": float(e["coverage"].median()) if k else np.nan})
        summ[f] = k
    return rows, summ


def route_rows(occ: pd.DataFrame, edu: pd.DataFrame) -> tuple:
    """How many employers each route can score, and the overlap."""
    o = set(occ["employer_id"])
    e = set(edu["employer_id"])
    both, oo, eo = o & e, o - e, e - o
    rows = []
    for name, s in (("occupation_route", o), ("education_route", e),
                    ("both_routes", both), ("occupation_only", oo),
                    ("education_only", eo)):
        rows.append({"panel": "all scored employers", "block": "route",
                     "group": name, "item": "n_employers",
                     "n_employers": len(s), "n_obs": len(o | e),
                     "share": len(s) / max(len(o | e), 1),
                     "value": float(len(s))})
    return rows, both


def crosstab_rows(occ: pd.DataFrame, edu: pd.DataFrame,
                  both: set) -> tuple:
    """
    The occupation quartile against the education quartile, among the
    employers both routes score.

    The diagonal share says how often the two routes put a firm in the
    same quartile, and the rank correlation says whether they agree about
    the ordering when they disagree about the cut. Both are reported: a
    high diagonal with a low correlation would mean the agreement is an
    artefact of where the cuts fall, and the converse would mean the two
    orderings agree and the quartile boundaries do not.
    """
    j = (occ[occ["employer_id"].isin(both)][["employer_id", "fq", "mix"]]
         .rename(columns={"fq": "fq_occ", "mix": "mix_occ"})
         .merge(edu[edu["employer_id"].isin(both)][["employer_id", "fq",
                                                    "mix"]]
                .rename(columns={"fq": "fq_edu", "mix": "mix_edu"}),
                on="employer_id", how="inner"))
    rows = []
    if j.empty:
        return rows, {}
    n = len(j)
    for (qo, qe), d in j.groupby(["fq_occ", "fq_edu"], observed=True):
        rows.append({"panel": "both routes", "block": "crosstab",
                     "group": f"occ_Q{int(qo)}_edu_Q{int(qe)}",
                     "item": "n_employers", "n_employers": int(len(d)),
                     "n_obs": int(len(d)), "share": len(d) / n,
                     "value": float(len(d))})
    diag = float((j["fq_occ"] == j["fq_edu"]).mean())
    top = float(((j["fq_occ"] == 4) & (j["fq_edu"] == 4)).sum())
    n_top_occ = int((j["fq_occ"] == 4).sum())
    n_top_edu = int((j["fq_edu"] == 4).sum())
    rho = float(j["mix_occ"].corr(j["mix_edu"], method="spearman"))
    pear = float(j["mix_occ"].corr(j["mix_edu"], method="pearson"))
    for item, val, nfirm in (("share_on_diagonal", diag, n),
                             ("spearman_rank_correlation", rho, n),
                             ("pearson_correlation", pear, n),
                             ("n_top_quartile_both", top, int(top)),
                             ("n_top_quartile_occupation", n_top_occ,
                              n_top_occ),
                             ("n_top_quartile_education", n_top_edu,
                              n_top_edu)):
        rows.append({"panel": "both routes", "block": "crosstab",
                     "group": "summary", "item": item,
                     "n_employers": nfirm, "n_obs": n,
                     "share": np.nan, "value": val})
    return rows, {"n": n, "diag": diag, "rho": rho,
                  "top_both": int(top), "top_occ": n_top_occ,
                  "top_edu": n_top_edu}


def panel_rows(band: str, counts, occ, edu, inc, nfloor, scored_old,
               s61, s80, j47) -> tuple:
    """
    The two routes on the panel the fits themselves run on, the loss
    decomposition on those same employers, and the size and longevity of
    the employers one route places and the other does not.

    The panel is rebuilt rather than approximated by an employer list,
    because the question is how many of THOSE employers each route can
    score, and any other population answers a different one.
    """
    skel = s61.build_skeleton(counts, band, j47)
    if skel.empty:
        return [], {}
    emp = s80.panel_employers(skel)
    del skel
    gc.collect()
    ids = emp["employer_id"]
    in_o = ids.isin(set(occ["employer_id"]))
    in_e = ids.isin(set(edu["employer_id"]))
    n = int(len(emp))
    groups = {"scored_by_both": emp[(in_o & in_e).to_numpy()],
              "occupation_only": emp[(in_o & ~in_e).to_numpy()],
              "education_only": emp[(~in_o & in_e).to_numpy()],
              "scored_by_neither": emp[(~in_o & ~in_e).to_numpy()]}
    panel = f"{band} stock"
    rows = [{"panel": panel, "block": "panel", "group": "all",
             "item": "employers_on_panel", "n_employers": n, "n_obs": n,
             "share": 1.0, "value": float(n)},
            {"panel": panel, "block": "panel", "group": "occupation_route",
             "item": "n_employers", "n_employers": int(in_o.sum()),
             "n_obs": n, "share": float(in_o.mean()),
             "value": float(in_o.sum())},
            {"panel": panel, "block": "panel", "group": "education_route",
             "item": "n_employers", "n_employers": int(in_e.sum()),
             "n_obs": n, "share": float(in_e.mean()),
             "value": float(in_e.sum())}]
    for name, d in groups.items():
        rows.append({"panel": panel, "block": "panel", "group": name,
                     "item": "n_employers", "n_employers": int(len(d)),
                     "n_obs": n, "share": len(d) / max(n, 1),
                     "value": float(len(d))})
        for r in s80.size_rows(panel, name, d):
            r["n_obs"] = int(len(d))
            rows.append(r)
    lr, ls = loss_rows(panel, ids, inc, nfloor, scored_old,
                       set(occ["employer_id"]))
    rows += lr
    summ = {"panel": panel, "n": n,
            "n_occ": int(in_o.sum()), "n_edu": int(in_e.sum()),
            "n_both": int(len(groups["scored_by_both"])),
            "n_occ_only": int(len(groups["occupation_only"])),
            "n_edu_only": int(len(groups["education_only"])),
            "loss": ls}
    for name, key in (("occupation_only", "occ_only"),
                      ("education_only", "edu_only"),
                      ("scored_by_both", "both")):
        d = groups[name]
        summ[f"med_{key}"] = (float(d["mean_headcount"].median())
                              if len(d) >= FLOOR else np.nan)
        summ[f"months_{key}"] = (float(d["months_active"].mean())
                                 if len(d) >= FLOOR else np.nan)
    del emp
    gc.collect()
    return rows, summ


def appendix_rows(inc: pd.DataFrame, occ_by_arm: dict, j47) -> list:
    """
    The coverage table the online appendix prints, in a shape a table
    generator reads directly.

    One row per (year, unit, age band, metric). `year` is "used" for the
    cascade as applied and then each cascade year on its own. The
    incumbent metrics partition twice over: once as the mixed rule would
    score them (four digits, three digits, neither) and once per arm
    (scored, unscored), so a reader can see both what the register
    offers and what each arm takes. Employers are counted in total only,
    since an employer is not in an age band; the age breakdown is the
    incumbents'.

    The last block is the comparison between the arms: how many
    employers change quartile between the primary and each robustness
    arm, and the rank correlation of the three firm scores. That is the
    number that says whether the choice of arm matters at all.
    """
    bands = list(j47.INCUMBENT_BANDS) + ["total"]
    rows = []
    years = ["used"] + [str(y) for y in CASCADE_BACK + CASCADE_FWD]
    for y in years:
        d = inc if y == "used" else inc[inc["source_year"] == y]
        if d.empty:
            continue
        for band in bands:
            g = d if band == "total" else d[d["age_group"] == band]
            if g.empty:
                continue
            tot = int(g["n"].sum())
            m4 = g["has4"] == 1
            m3 = g["has3"] == 1
            counts = {
                "resolved": int((g["n"] * (g["coded"] == 1)).sum()),
                "four_digit_available": int(g.loc[m4, "n"].sum()),
                "three_digit_available": int(g.loc[m3, "n"].sum()),
                "scored_four_digit": int(g.loc[m4, "n"].sum()),
                "scored_three_digit": int(g.loc[~m4 & m3, "n"].sum()),
                "unscored": int(g.loc[~m4 & ~m3, "n"].sum()),
                "scored_uniform3": int(g.loc[m3, "n"].sum()),
                "unscored_uniform3": int(g.loc[~m3, "n"].sum()),
                "scored_mixed43": int(g.loc[m4 | m3, "n"].sum()),
                "unscored_mixed43": int(g.loc[~m4 & ~m3, "n"].sum()),
                "scored_four_only": int(g.loc[m4, "n"].sum()),
                "unscored_four_only": int(g.loc[~m4, "n"].sum()),
                "incumbents": tot,
            }
            for k, v in counts.items():
                rows.append({"year": y, "unit": "incumbents",
                             "age_band": band, "metric": k, "n": v,
                             "share": v / max(tot, 1), "value": np.nan})
        emp_tot = int(d["employer_id"].nunique())
        m4 = d["has4"] == 1
        m3 = d["has3"] == 1
        for k, mask in (("resolved", d["coded"] == 1),
                        ("four_digit_available", m4),
                        ("three_digit_available", m3),
                        ("scored_uniform3", m3),
                        ("scored_mixed43", m4 | m3),
                        ("scored_four_only", m4)):
            v = int(d.loc[mask, "employer_id"].nunique())
            rows.append({"year": y, "unit": "employers", "age_band": "total",
                         "metric": k, "n": v, "share": v / max(emp_tot, 1),
                         "value": np.nan})
        rows.append({"year": y, "unit": "employers", "age_band": "total",
                     "metric": "employers", "n": emp_tot, "share": 1.0,
                     "value": np.nan})
    # ---- the arms against each other ---------------------------------
    base = occ_by_arm.get(MAIN_LEVEL)
    if base is not None and not base.empty:
        for arm, e in occ_by_arm.items():
            n_e = int(len(e))
            rows.append({"year": "used", "unit": "employers",
                         "age_band": "total", "metric": f"scored_arm_{arm}",
                         "n": n_e, "share": np.nan, "value": float(n_e)})
            if arm == MAIN_LEVEL or e.empty:
                continue
            j = base[["employer_id", "fq", "mix"]].merge(
                e[["employer_id", "fq", "mix"]], on="employer_id",
                suffixes=("_p", "_o"))
            if j.empty:
                continue
            moved = int((j["fq_p"] != j["fq_o"]).sum())
            rows.append({"year": "used", "unit": "employers",
                         "age_band": "total",
                         "metric": f"quartile_changed_vs_{arm}",
                         "n": moved, "share": moved / max(len(j), 1),
                         "value": float(len(j))})
            rows.append({"year": "used", "unit": "employers",
                         "age_band": "total",
                         "metric": f"spearman_{MAIN_LEVEL}_{arm}",
                         "n": int(len(j)), "share": np.nan,
                         "value": float(j["mix_p"].corr(j["mix_o"],
                                                        method="spearman"))})
    if "mixed43" in occ_by_arm and "four_only" in occ_by_arm:
        a, b_ = occ_by_arm["mixed43"], occ_by_arm["four_only"]
        if not a.empty and not b_.empty:
            j = a[["employer_id", "mix"]].merge(b_[["employer_id", "mix"]],
                                                on="employer_id",
                                                suffixes=("_m", "_f"))
            if not j.empty:
                rows.append({"year": "used", "unit": "employers",
                             "age_band": "total",
                             "metric": "spearman_mixed43_four_only",
                             "n": int(len(j)), "share": np.nan,
                             "value": float(j["mix_m"].corr(
                                 j["mix_f"], method="spearman"))})
    return rows


def part_a(counts, occ, edu, casc, inc, nfloor, daioe, book, plan_cols,
           s61, s80, j47) -> tuple:
    """What the occupation route scores, where the employers went, and
    how the score lines up with the education route. No fit runs here."""
    scored_old = old_rule_scored(inc)
    rows = cascade_rows(inc, plan_cols, j47)
    rows += staleness_rows(inc)
    rows += coverage_by_band(casc, daioe, book, j47)
    # the appendix table, and the two robustness arms it compares against
    occ_by_arm = {MAIN_LEVEL: occ}
    for lvl in SCORE_ARMS:
        if lvl != MAIN_LEVEL:
            occ_by_arm[lvl] = occ_route_exposure(
                inc, nfloor, FLOOR_MAIN, ARM_YEARS[MAIN_ARM], arm=lvl)
    ap = mc.enforce_min_cell(pd.DataFrame(appendix_rows(inc, occ_by_arm,
                                                        j47)),
                             count_col="n", floor=FLOOR)
    ap.loc[ap["n"].isna(), ["share", "value"]] = np.nan
    ap.to_csv(OUT / "occ_route_appendix_coverage.csv", index=False)
    arm_summ = {a: int(len(e)) for a, e in occ_by_arm.items()}
    moved = {r["metric"]: (r["n"], r["share"], r["value"]) for r in
             appendix_rows(inc, occ_by_arm, j47)
             if str(r["metric"]).startswith(("quartile_changed", "spearman"))}
    del occ_by_arm
    gc.collect()
    all_ids = sorted(set(inc["employer_id"]))
    lr, ls_all = loss_rows("all employers", all_ids, inc, nfloor,
                           scored_old, set(occ["employer_id"]))
    rows += lr
    fr, fs = floor_rows("all employers", inc, nfloor)
    rows += fr
    r, both = route_rows(occ, edu)
    rows += r
    r, cross = crosstab_rows(occ, edu, both)
    rows += r
    summaries = []
    for band in BANDS:
        r, s = panel_rows(band, counts, occ, edu, inc, nfloor, scored_old,
                          s61, s80, j47)
        if not r:
            FAILURES.append(f"A/{band}/empty")
            continue
        rows += r
        summaries.append(s)
        print(f"  A: {band} stock, {cnt(s['n'])} employers, "
              f"{cnt(s['n_occ'])} scorable on the occupation route, "
              f"{cnt(s['n_edu'])} on the education route, "
              f"{cnt(s['n_both'])} on both")
    tab = mc.enforce_min_cell(pd.DataFrame(rows), count_col="n_obs",
                              floor=FLOOR)
    tab = mc.enforce_min_cell(tab, count_col="n_employers", floor=FLOOR)
    # A share and a published denominator reproduce a suppressed count, so
    # the share and the value go with their own numerator.
    gone = tab["n_employers"].isna()
    tab.loc[gone & tab["block"].isin(["route", "crosstab", "panel", "loss",
                                      "floor"]),
            ["share", "value"]] = np.nan
    # The person-count blocks carry no employer count, so their own
    # suppressed cell is n_obs, and a share with a published denominator
    # would reproduce it.
    gone_n = tab["n_obs"].isna()
    tab.loc[gone_n & tab["block"].isin(["step", "coverage", "ssyk_ar",
                                        "ssyk_status"]),
            ["share", "value"]] = np.nan
    tab = save(tab, "occ_route_coverage.csv", count_col="n_employers")
    return tab, summaries, cross, ls_all, fs, arm_summ, moved


# ----------------------------------------------------------------------
# Part B: the headline and the profile
# ----------------------------------------------------------------------

def edu_pair(d: dict, key: str) -> tuple:
    c, s = d.get(key, (np.nan, np.nan))
    return c, s


def stock_fit(counts, expo, band, arm, floor, s61, s78, j47, head,
              level: str = MAIN_LEVEL) -> None:
    """One stock fit of Equation (2), appended to the headline table."""
    skel = s61.build_skeleton(counts, band, j47)
    if skel.empty:
        FAILURES.append(f"B/{band}/{arm}/empty")
        return
    b = s78.with_exposure(skel, expo)
    del skel
    gc.collect()
    if b.empty:
        FAILURES.append(f"B/{band}/{arm}/no exposure")
        return
    n_firms = int(b["employer_id"].nunique())
    on = expo[expo["employer_id"].isin(set(b["employer_id"]))]
    share_old = float(on["share_not_2019"].mean()) \
        if "share_not_2019" in on.columns and len(on) else np.nan
    share_3 = float(on["share_three_digit"].mean()) \
        if "share_three_digit" in on.columns and len(on) else np.nan
    b, terms = s78.eq2_terms(b)
    tag = (f"stock_{band.replace('-', '_')}_{arm}_f{floor}"
           + ("" if level == MAIN_LEVEL else f"_{level}"))
    g, _ = fit(b, tag, terms, j47.FES)
    del b
    gc.collect()
    if g is None:
        return
    ec, es = edu_pair(EDU_STOCK, band)
    for r in s78.rows_of(g, terms, young_band=band, outcome="stock",
                         arm=arm, floor=floor, level=level,
                         n_firms=n_firms, share_not_2019=share_old,
                         share_three_digit=share_3):
        r["t"] = tstat(r["coef"], r["se"])
        post = r["term"] == "post_x_high_x_young"
        r["edu_coef"] = ec if post else np.nan
        r["edu_se"] = es if post else np.nan
        head.append(r)
    save(head, "occ_route_headline.csv")
    p = [h for h in head if h["young_band"] == band and h["arm"] == arm
         and h["floor"] == floor and h["level"] == level
         and h["term"] == "post_x_high_x_young"]
    if p:
        print(f"  B: {band} {arm} floor {floor} {level} adoption step "
              f"{p[0]['coef']:+.4f} ({p[0]['se']:.4f}) t {p[0]['t']:+.2f}"
              f"   education route {ec:+.4f} ({es:.4f})")


def part_b(counts, inc, nfloor, s61, s74, s78, l70, j47) -> tuple:
    """
    Equation (2) on the stock at both young bands, the six-band profile
    against 41-49, the floor sensitivity and the forward-cascade arm.

    The terms are 78's, which are 68's; the profile terms are 74's
    seasonal arm on 70's six-band skeleton. Only the column that says
    which employers are highly exposed differs from the paper's route.
    """
    head, prof = [], []
    occ = occ_route_exposure(inc, nfloor, FLOOR_MAIN, ARM_YEARS[MAIN_ARM])
    if occ.empty:
        FAILURES.append("B/no exposure")
        return head, prof
    # The three scoring arms side by side, the primary first. They share
    # the panel, the floor and the window and differ in the score alone.
    for lvl in SCORE_ARMS:
        e = (occ if lvl == MAIN_LEVEL
             else occ_route_exposure(inc, nfloor, FLOOR_MAIN,
                                     ARM_YEARS[MAIN_ARM], arm=lvl))
        if e.empty:
            FAILURES.append(f"B/{lvl}/no exposure")
            continue
        for band in BANDS:
            stock_fit(counts, e, band, MAIN_ARM, FLOOR_MAIN, s61, s78, j47,
                      head, level=lvl)
        if lvl != MAIN_LEVEL:
            del e
            gc.collect()
    # ---- the six-band profile ----------------------------------------
    skel = l70.all_band_skeleton(counts)
    if skel.empty:
        FAILURES.append("B/profile/empty")
    else:
        b = s78.with_exposure(skel, occ)
        del skel
        gc.collect()
        if b.empty:
            FAILURES.append("B/profile/no exposure")
        else:
            n_firms = int(b["employer_id"].nunique())
            b, terms = s74.build_terms(b, l70, seasonal=True)
            g, _ = fit(b, "profile_six_band", terms, j47.FES)
            del b
            gc.collect()
            if g is not None:
                for band in s74.BANDS:
                    ec, es = edu_pair(EDU_PROFILE, band)
                    if band == PROFILE_REF:
                        prof.append({"band": band, "coef": 0.0, "se": 0.0,
                                     "t": np.nan, "n_firms": n_firms,
                                     "n_obs": int(g["n_obs"].max()),
                                     "status": "reference", "edu_coef": ec,
                                     "edu_se": es})
                        continue
                    t_ = l70.band_col("gpt_x_high", band)
                    if t_ not in g.index:
                        continue
                    prof.append({"band": band,
                                 "coef": float(g.loc[t_, "coef"]),
                                 "se": float(g.loc[t_, "se"]),
                                 "t": tstat(float(g.loc[t_, "coef"]),
                                            float(g.loc[t_, "se"])),
                                 "n_firms": n_firms,
                                 "n_obs": int(g.loc[t_, "n_obs"]),
                                 "status": str(g.loc[t_].get("status", "ok")),
                                 "edu_coef": ec, "edu_se": es})
                if prof:
                    save(prof, "occ_route_profile.csv")
                    for r in prof:
                        print(f"  B: profile {r['band']:<6} {r['coef']:+.4f} "
                              f"({r['se']:.4f})   education route "
                              f"{r['edu_coef']:+.4f} ({r['edu_se']:.4f})")
    del occ
    gc.collect()
    # ---- the floor sensitivity, at 22-25 ------------------------------
    for f in FLOORS:
        if f == FLOOR_MAIN:
            continue
        e = occ_route_exposure(inc, nfloor, f, ARM_YEARS[MAIN_ARM])
        if e.empty:
            FAILURES.append(f"B/floor{f}/no exposure")
            continue
        stock_fit(counts, e, BANDS[0], MAIN_ARM, f, s61, s78, j47, head)
        del e
        gc.collect()
    # ---- the forward-cascade arm, reported and never the score --------
    e = occ_route_exposure(inc, nfloor, FLOOR_MAIN,
                           ARM_YEARS["forward"])
    if e.empty:
        FAILURES.append("B/forward/no exposure")
    else:
        stock_fit(counts, e, BANDS[0], "forward", FLOOR_MAIN, s61, s78, j47,
                  head)
        del e
        gc.collect()
    return head, prof


# ----------------------------------------------------------------------
# Part C: the sex split, the margins and the vintage check
# ----------------------------------------------------------------------

def part_c_gender(sexcounts, occ, s67, s78, j47) -> tuple:
    """The sex specification of Equation (2) at 22-25."""
    if sexcounts is None:
        FAILURES.append("C/gender/no L_counts_sex cache")
        print("  C: L_counts_sex_* missing, the sex specification is skipped")
        return [], {}
    skel = s67.build_skeleton_sex(sexcounts, SEX_BAND, j47, "n_emp")
    if skel.empty:
        FAILURES.append("C/gender/empty")
        return [], {}
    b = s78.with_exposure(skel, occ)
    del skel
    gc.collect()
    if b.empty:
        FAILURES.append("C/gender/no exposure")
        return [], {}
    n_firms = int(b["employer_id"].nunique())
    b, terms = s78.gender_eq2_terms(b)
    g, v = fit(b, f"gender_{SEX_BAND.replace('-', '_')}", terms, j47.FES)
    del b
    gc.collect()
    if g is None:
        return [], {}
    rows = []
    for r in s78.rows_of(g, terms, block="term", young_band=SEX_BAND,
                         n_firms=n_firms):
        r["t"] = tstat(r["coef"], r["se"])
        diff = r["term"] == "post_x_high_x_young_x_female"
        r["edu_coef"] = EDU_FEMALE[0] if diff else np.nan
        r["edu_se"] = EDU_FEMALE[1] if diff else np.nan
        rows.append(r)
    # The three numbers a reader wants. The women's step is the male step
    # plus the differential, so its standard error comes from the
    # covariance of the fit and not from adding two standard errors.
    m, d = "post_x_high_x_young", "post_x_high_x_young_x_female"
    c = {t_: float(g.loc[t_, "coef"]) for t_ in terms if t_ in g.index}
    e = {t_: float(g.loc[t_, "se"]) for t_ in terms if t_ in g.index}
    steps = {}
    if m in c and d in c:
        steps["male_step"] = (c[m], e[m])
        steps["female_differential"] = (c[d], e[d])
        steps["female_step"] = (c[m] + c[d], s78.lincomb(v, {m: 1, d: 1}))
    for k, (cc, ss) in steps.items():
        rows.append({"block": "step", "young_band": SEX_BAND, "term": k,
                     "coef": cc, "se": (np.nan if ss is None else ss),
                     "t": tstat(cc, ss), "n_obs": int(g["n_obs"].max()),
                     "n_firms": n_firms, "status": "derived",
                     "edu_coef": (EDU_FEMALE[0]
                                  if k == "female_differential" else np.nan),
                     "edu_se": (EDU_FEMALE[1]
                                if k == "female_differential" else np.nan)})
    save(rows, "occ_route_gender.csv")
    if "female_differential" in steps:
        cc, ss = steps["female_differential"]
        print(f"  C: female differential {cc:+.4f} ({ss:.4f}) t "
              f"{tstat(cc, ss):+.2f}   education route "
              f"{EDU_FEMALE[0]:+.4f} ({EDU_FEMALE[1]:.4f})")
    return rows, steps


def part_c_flows(flows, occ, s61, s78, j47) -> list:
    """Hires and separations at 22-25, on the terms the stock uses."""
    if flows is None:
        FAILURES.append("C/flows/no flows cache")
        print("  C: flows_* missing, the margins are skipped")
        return []
    rows = []
    for outcome, col in (("hires", "n_hire"), ("seps", "n_sep")):
        src = flows.rename(columns={col: "n_emp"})
        skel = s61.build_skeleton(src, FLOW_BAND, j47)
        del src
        gc.collect()
        if skel.empty:
            FAILURES.append(f"C/{outcome}/empty")
            continue
        b = s78.with_exposure(skel, occ)
        del skel
        gc.collect()
        if b.empty:
            FAILURES.append(f"C/{outcome}/no exposure")
            continue
        n_firms = int(b["employer_id"].nunique())
        b, terms = s78.eq2_terms(b)
        g, _ = fit(b, f"{outcome}_{FLOW_BAND.replace('-', '_')}", terms,
                   j47.FES)
        del b
        gc.collect()
        if g is None:
            continue
        ec, es = edu_pair(EDU_FLOW, outcome)
        for r in s78.rows_of(g, terms, outcome=outcome,
                             young_band=FLOW_BAND, n_firms=n_firms):
            r["t"] = tstat(r["coef"], r["se"])
            post = r["term"] == "post_x_high_x_young"
            r["edu_coef"] = ec if post else np.nan
            r["edu_se"] = es if post else np.nan
            rows.append(r)
        p = [r for r in rows if r["outcome"] == outcome
             and r["term"] == "post_x_high_x_young"]
        if p:
            print(f"  C: {outcome} {p[0]['coef']:+.4f} ({p[0]['se']:.4f}) "
                  f"t {p[0]['t']:+.2f}   education route {ec:+.4f} ({es:.4f})")
    if rows:
        save(rows, "occ_route_flows.csv")
    return rows


def vintage_stability(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """How far a different vintage moves the 2019 classifier."""
    j = a.merge(b, on="employer_id", suffixes=("_t", "_a"))
    if j.empty:
        return {}
    return {"n_both": int(len(j)),
            "share_keeping_quartile": float((j["fq_t"] == j["fq_a"]).mean()),
            "share_keeping_top": float(((j["fq_t"] == 4)
                                        == (j["fq_a"] == 4)).mean()),
            "mean_relative_mix_shift": float(np.mean(
                np.abs(j["mix_a"] - j["mix_t"])
                / j["mix_t"].abs().clip(lower=1e-9))),
            "spearman": float(j["mix_t"].corr(j["mix_a"], method="spearman")),
            "n_true_only": int(len(set(a["employer_id"])
                                   - set(b["employer_id"]))),
            "n_asof_only": int(len(set(b["employer_id"])
                                   - set(a["employer_id"])))}


def part_c_vintage(counts, inc, daioe, book, nfloor, j47, s61,
                   s78) -> tuple:
    """
    Three scores on one panel: the reported backward cascade, the 2019
    code alone, and the 2019 incumbents re-scored from the 2021 register.

    Three and not two, because the as-of arm has no cascade: comparing
    it with the reported score would sum the re-coding with the loss of
    the cascade's extra coverage and call the total an artefact. With
    the 2019-only arm in between, the re-coding artefact is the as-of
    arm against it and the cascade's contribution is the reported score
    against it, and the two are reported separately.

    All three fit on the employers all three can score, so the
    difference between the arms is the score and not the sample.
    """
    vint = baseline_vintage(VINTAGE)
    inc_v = incumbent_frame(vint, daioe, book, j47)
    del vint
    gc.collect()
    expos = {
        f"cascade_back_{BASE_YEAR}": occ_route_exposure(
            inc, nfloor, FLOOR_MAIN, ARM_YEARS[MAIN_ARM]),
        f"code_{BASE_YEAR}_only": occ_route_exposure(
            inc, nfloor, FLOOR_MAIN, ARM_YEARS["2019_only"]),
        f"asof_{VINTAGE}": occ_route_exposure(
            inc_v, nfloor, FLOOR_MAIN, [VINTAGE]),
    }
    del inc_v
    gc.collect()
    # THE THREE ARMS MUST RUN ON THE SAME EMPLOYERS. They do not score
    # the same ones: the cascade reaches firms the 2019 code alone
    # cannot, and the later register reaches others again. Comparing
    # them on their own samples would report a sample change as an
    # artefact. The fits are therefore restricted to the employers ALL
    # THREE can score, so the difference between them is the score and
    # nothing else; how many each can score is Part A's question and is
    # answered there. The quartile is NOT recomputed on the restriction:
    # each arm classifies on its own whole distribution, as it would in
    # its own fit, and only the comparison sample is narrowed.
    live = [e for e in expos.values() if not e.empty]
    common = (set.intersection(*[set(e["employer_id"]) for e in live])
              if live else set())
    msg = ("vintage: the three arms score "
           + ", ".join(f"{k} {len(v):,}" for k, v in expos.items())
           + f"; the {len(common):,} employers all three score are the "
             f"sample of every fit")
    print(f"  {msg}")
    NOTES.append(msg)
    expos = {k: v[v["employer_id"].isin(common)] for k, v in expos.items()}
    rows, stab = [], {}
    base = expos[f"code_{BASE_YEAR}_only"]
    for other in (f"asof_{VINTAGE}", f"cascade_back_{BASE_YEAR}"):
        if base.empty or expos[other].empty:
            continue
        s = vintage_stability(base, expos[other])
        stab[other] = s
        rows += [{"block": "stability", "arm": f"{BASE_YEAR}_only vs {other}",
                  "item": k, "coef": np.nan, "se": np.nan, "t": np.nan,
                  "value": v, "n_firms": s.get("n_both", np.nan),
                  "n_obs": s.get("n_both", np.nan), "status": "descriptive"}
                 for k, v in s.items()]
        print(f"  C: {other} against the 2019 code alone: "
              f"{s['share_keeping_quartile']:.1%} of employers keep their "
              f"quartile, mean relative shift in the score "
              f"{s['mean_relative_mix_shift']:.2%}")
    skel = s61.build_skeleton(counts, SEX_BAND, j47)
    if skel.empty:
        FAILURES.append("C/vintage/empty panel")
        save(rows, "occ_route_vintage.csv")
        return rows, stab
    got = {}
    ec, es = edu_pair(EDU_STOCK, SEX_BAND)
    for arm, e in expos.items():
        if e.empty:
            FAILURES.append(f"C/vintage/{arm}/no exposure")
            continue
        b = s78.with_exposure(skel, e)
        if b.empty:
            FAILURES.append(f"C/vintage/{arm}/no overlap")
            continue
        n_firms = int(b["employer_id"].nunique())
        b, terms = s78.eq2_terms(b)
        g, _ = fit(b, f"vintage_{arm}", terms, j47.FES)
        del b
        gc.collect()
        if g is None:
            continue
        for r in s78.rows_of(g, terms, block="fit", arm=arm,
                             n_firms=n_firms):
            r["item"] = r.pop("term")
            r["t"] = tstat(r["coef"], r["se"])
            r["value"] = np.nan
            rows.append(r)
        post = "post_x_high_x_young"
        if post in g.index:
            got[arm] = float(g.loc[post, "coef"])
            print(f"  C: vintage {arm:<22} adoption step {got[arm]:+.4f} "
                  f"({float(g.loc[post, 'se']):.4f})   education route "
                  f"{ec:+.4f} ({es:.4f})")
    del skel
    gc.collect()
    b19 = f"code_{BASE_YEAR}_only"
    for other, label in ((f"asof_{VINTAGE}", "recoding_artefact"),
                         (f"cascade_back_{BASE_YEAR}", "cascade_effect")):
        if b19 in got and other in got:
            a = got[other] - got[b19]
            rows.append({"block": "artefact", "arm": f"{other} minus {b19}",
                         "item": label, "coef": a, "se": np.nan, "t": np.nan,
                         "value": a, "n_firms": np.nan, "n_obs": np.nan,
                         "status": "derived"})
            stab.setdefault(other, {})["artefact"] = a
            print(f"  C: {label} on the adoption step {a:+.4f}")
    save(rows, "occ_route_vintage.csv")
    return rows, stab


def part_c(counts, sexcounts, flows, inc, daioe, book, nfloor, s61, s67,
           s78, j47) -> tuple:
    occ = occ_route_exposure(inc, nfloor, FLOOR_MAIN,
                             ARM_YEARS[MAIN_ARM])
    g_rows, steps = part_c_gender(sexcounts, occ, s67, s78, j47)
    f_rows = part_c_flows(flows, occ, s61, s78, j47)
    del occ
    gc.collect()
    v_rows, stab = part_c_vintage(counts, inc, daioe, book, nfloor, j47,
                                  s61, s78)
    return g_rows, steps, f_rows, v_rows, stab


# ----------------------------------------------------------------------
# The three verdicts
# ----------------------------------------------------------------------

def _main_rows(head: list) -> list:
    """The reported arm only: the backward cascade at the main floor."""
    return [r for r in head
            if r.get("arm", MAIN_ARM) == MAIN_ARM
            and r.get("floor", FLOOR_MAIN) == FLOOR_MAIN
            and r.get("level", MAIN_LEVEL) == MAIN_LEVEL]


def verdict_headline(head: list) -> tuple:
    """Read rule 1, at 22-25, with the 26-30 band reported beside it."""
    L, verdict = [], "NO VERDICT"
    main = _main_rows(head)
    p = [r for r in main if r["young_band"] == "22-25"
         and r["term"] == "post_x_high_x_young"]
    if not p:
        return verdict, ["  NO VERDICT on rule 1: the 22-25 stock fit did "
                         "not come back, and a missing fit is not a null."]
    c, s = float(p[0]["coef"]), float(p[0]["se"])
    t = tstat(c, s)
    ec, es = EDU_STOCK["22-25"]
    verdict = ("REPRODUCES" if (c < 0 and abs(t) >= SIG5)
               else "DOES NOT REPRODUCE")
    L.append(f"  1. THE ADOPTION STEP AT 22-25: {verdict}")
    L.append(f"     occupation route {c:+.4f} ({s:.4f}) t {t:+.2f}; "
             f"education route {ec:+.4f} ({es:.4f})")
    if verdict == "DOES NOT REPRODUCE":
        L.append("     The sign, the size and the t are reported above "
                 "whichever way they fall; a step that is negative but not "
                 "distinguishable from zero is a failure of this rule and "
                 "not a zero.")
    for band in BANDS[1:]:
        q = [r for r in main if r["young_band"] == band
             and r["term"] == "post_x_high_x_young"]
        if q:
            bc, bs = float(q[0]["coef"]), float(q[0]["se"])
            e2 = EDU_STOCK.get(band, (np.nan, np.nan))
            L.append(f"     {band}, not part of the rule: {bc:+.4f} "
                     f"({bs:.4f}) t {tstat(bc, bs):+.2f}; education route "
                     f"{e2[0]:+.4f} ({e2[1]:.4f})")
    other = [r for r in head if r not in main
             and r["term"] == "post_x_high_x_young"]
    for r in other:
        L.append(f"     {r['young_band']} {r.get('arm', '?')} floor "
                 f"{r.get('floor', '?')} {r.get('level', '?')}, reported "
                 f"and settling nothing: {float(r['coef']):+.4f} "
                 f"({float(r['se']):.4f}) t "
                 f"{tstat(r['coef'], r['se']):+.2f}, "
                 f"{cnt(r.get('n_firms'))} employers")
    return verdict, L


def verdict_profile(prof: list) -> tuple:
    """Read rule 2, on the six-band profile against 41-49."""
    if not prof:
        return "NO VERDICT", ["  NO VERDICT on rule 2: the profile fit did "
                              "not come back."]
    d = {r["band"]: float(r["coef"]) for r in prof}
    if "50+" not in d or "22-25" not in d or len(d) < 6:
        return "NO VERDICT", ["  NO VERDICT on rule 2: the profile is "
                              "incomplete, so the ranking cannot be read."]
    order = sorted(d, key=lambda b: d[b])
    rank = order.index("22-25") + 1
    gains = d["50+"] > 0
    verdict = ("THE PROFILE REPRODUCES" if (gains and rank <= 2)
               else "THE PROFILE DOES NOT REPRODUCE")
    L = [f"  2. THE AGE PROFILE: {verdict}",
         f"     50 and over against 41-49 {d['50+']:+.4f} "
         f"(education route {EDU_PROFILE['50+'][0]:+.4f}), so the older band "
         f"{'gains' if gains else 'does NOT gain'}",
         f"     22-25 is ranked {rank} of six from the bottom "
         + ("(lowest or second lowest, as the rule requires)" if rank <= 2
            else "(the rule requires first or second)"),
         "     the whole profile, lowest first:"]
    for b in order:
        e = EDU_PROFILE.get(b, (np.nan, np.nan))
        r = next(x for x in prof if x["band"] == b)
        L.append(f"       {b:<6} {d[b]:+.4f} ({float(r['se']):.4f})   "
                 f"education route {e[0]:+.4f} ({e[1]:.4f})")
    return verdict, L


def verdict_gender(steps: dict) -> tuple:
    """Read rule 3, on the female differential."""
    if "female_differential" not in steps:
        return "NO VERDICT", ["  NO VERDICT on rule 3: the sex fit did not "
                              "come back."]
    c, s = steps["female_differential"]
    t = tstat(c, s)
    verdict = ("THE SEX RESULT REPRODUCES" if (c < 0 and abs(t) >= SIG1)
               else "THE SEX RESULT DOES NOT REPRODUCE")
    L = [f"  3. THE FEMALE DIFFERENTIAL: {verdict}",
         f"     occupation route {c:+.4f} ({s:.4f}) t {t:+.2f}; "
         f"education route {EDU_FEMALE[0]:+.4f} ({EDU_FEMALE[1]:.4f})",
         "     the rule is the one per cent level, so the threshold is "
         f"|t| >= {SIG1:.2f}"]
    for k, nm in (("male_step", "young men, adoption step"),
                  ("female_step", "young women, adoption step")):
        if k in steps:
            cc, ss = steps[k]
            L.append(f"     {nm:<28} {cc:+.4f} "
                     + ("(no SE)" if ss is None or ss != ss
                        else f"({ss:.4f})"))
    return verdict, L


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main():
    mc.Tee(OUT / "82_log.txt")
    t0 = time.time()
    print("=" * 70)
    print(f"82: THE OCCUPATION ROUTE, WITH NO EDUCATION ANYWHERE   "
          f"parts {PARTS}")
    print("=" * 70)
    print("\n".join(READ_RULES))
    print(mc.mem_line("  "))

    s61, s67, s74, s78, s80, l47, l70, j47 = load_modules()
    daioe = l70.daioe_scores()
    # The dictionary's year-to-column map, which needs no connection and
    # is what Part A reports; the pull confirms it against the catalogue.
    plan_cols = check_cascade_years()
    print("  cascade: " + ", ".join(f"{y} via {c}"
                                    for y, c in sorted(plan_cols.items())))
    # The whole chain, in the one function script 83 also calls, so that
    # the score has a single definition rather than two that drift.
    built = build_exposure(l47, l70, j47, daioe=daioe)
    casc, book, cost = built["casc"], built["book"], built["cost"]
    inc, nfloor, occ = built["inc"], built["nfloor"], built["exposure"]
    if len(book):
        bk = book.copy()
        bk["item"] = "book"
        cst = pd.DataFrame([{"ssyk3": "ALL", "item": k, "score3": v,
                             "n_weight": cost.get("n_workers"),
                             "n_ssyk4": cost.get("n_ssyk4")}
                            for k, v in cost.items()])
        save(pd.concat([bk, cst], ignore_index=True),
             "occ_route_ssyk3_book.csv", count_col="n_weight")

    edu = None
    if "A" in PARTS:
        edu = l70.edu_exposure(j47, l70.DESIGN, l70.ARM)
        print(f"  education route: {len(edu):,} employers scored")

    counts = load_counts("L_counts", s61.PANEL_YEARS)
    if counts is None:
        raise RuntimeError("L_counts_* missing: run 47L first.")
    last = str(counts["year_month"].max())
    if last < POST_FROM:
        raise RuntimeError(f"the counts end at {last} and the adoption "
                           f"window opens at {POST_FROM}; refusing to run.")
    print(f"  counts: {len(counts):,} employer-age-months, ending {last}")
    sexcounts = load_counts("L_counts_sex", s61.PANEL_YEARS,
                            require=["employer_id", "year_month",
                                     "age_group", "gender", "n_emp"]) \
        if "C" in PARTS else None
    flows = load_counts("flows", s61.PANEL_YEARS,
                        require=["employer_id", "year_month", "age_group",
                                 "n_hire", "n_sep"]) if "C" in PARTS else None

    cov_tab, cov_summ, cross, loss_all, floor_summ = None, [], {}, {}, {}
    arm_summ, moved = {}, {}
    head, prof, g_rows, steps, f_rows, v_rows, stab = [], [], [], {}, [], [], {}
    if "A" in PARTS:
        r = opt("Part A", part_a, counts, occ, edu, casc, inc, nfloor, daioe,
                book, plan_cols, s61, s80, j47)
        if r:
            (cov_tab, cov_summ, cross, loss_all, floor_summ, arm_summ,
             moved) = r
    del casc, edu, occ
    gc.collect()
    if "B" in PARTS:
        r = opt("Part B", part_b, counts, inc, nfloor, s61, s74, s78,
                l70, j47)
        if r:
            head, prof = r
    if "C" in PARTS:
        r = opt("Part C", part_c, counts, sexcounts, flows, inc, daioe,
                book, nfloor, s61, s67, s78, j47)
        if r:
            g_rows, steps, f_rows, v_rows, stab = r
    del counts, sexcounts, flows, inc
    gc.collect()

    # ---- summary ------------------------------------------------------
    L = ["THE OCCUPATION ROUTE, WITH NO EDUCATION ANYWHERE", "=" * 52, "",
         "The paper routes exposure through education: an education group",
         "carries the mean DAIOE percentile of the occupations its holders",
         "worked in during 2019, and an employer is ranked by the mean over",
         "its incumbents aged 31 to 69. Here the intermediate step is",
         "deleted: an employer is ranked by the mean DAIOE percentile of",
         "the 2019 four-digit occupations of its OWN incumbents aged 31 to",
         "69. Same freeze year, same incumbent restriction, same quartile",
         "weighting, no education record anywhere.", "",
         "THE FLOOR AND ITS UNIT. The floor is on the firm's INCUMBENTS and",
         f"not on its coded incumbents, at {FLOOR_MAIN}, applied to the",
         f"{BASIS}."] + ([
             "That is the education route's own unit, so the two floors are",
             "commensurable."] if BASIS == "person-months" else [
             "That is NOT the education route's unit, which is person-months",
             "summed over 2019, so a floor of five means something stricter",
             "here. The floor sensitivity below is the answer to that rather",
             "than a reassurance."]) + ["",
         "THE SCORING LEVEL, AND WHY THE UNIFORM ONE IS PRIMARY. Every",
         "incumbent is scored from the THREE-DIGIT book, built once as the",
         f"{BASE_YEAR} national employment-weighted mean of the four-digit",
         "DAIOE scores within each three-digit group. The workers lacking a",
         "four-digit code are not a random subset, so under a mixed rule a",
         "firm scored mostly at four digits gets a sharp score and one",
         "scored mostly at three gets a score smoothed toward group means",
         "and compressed away from the extremes: quartile assignment would",
         "then depend partly on the firm's coding completeness, which is a",
         "bias channel into the treatment variable and not merely noise.",
         "Under the uniform rule the smoothing is common to every firm and",
         "the ranking survives it, and from 2019 every coded occupation",
         "carries at least three digits, so the uniform level is the",
         "near-complete one. mixed43 and four_only are fitted beside it as",
         "robustness and settle nothing. DO NOT REVERSE THIS: the mixed arm",
         "looks more precise and is not.", "",
         "THE CASCADE. Each incumbent takes the occupation recorded for him",
         f"in {BASE_YEAR}, failing that "
         + ", ".join(str(y) for y in CASCADE_BACK[1:])
         + ". Backward only:",
         "a code recorded after the freeze year would break the paper's",
         "claim that none enters anything. The forward arm (2020, 2021) is",
         "reported and is never the score.", ""]
    L += ["WHAT THE COARSENING COSTS, which is the evidence for the",
          "primary arm and belongs in the appendix:"]
    if cost:
        L += [f"  {'':<34} {'weighted':>10}  {'unweighted benchmark':>21}",
              f"  {'between-group share of variance':<34} "
              f"{cost['share_between']:9.1%}  {BENCH3['share_between']:20.1%}",
              f"  {'within-group share (discarded)':<34} "
              f"{cost['share_within']:9.1%}  {BENCH3['share_within']:20.1%}",
              f"  {'mean |four-digit minus book|':<34} "
              f"{cost['mean_abs_difference']:9.2f}  {BENCH3['mean_abs']:20.2f}",
              "  The weighted column is ours, on the "
              f"{BASE_YEAR} Swedish employed population, and is the number",
              "  that matters. The benchmark column is the same statistics "
              "UNWEIGHTED on the released",
              f"  DAIOE panel at 2023 ({BENCH3['n_ssyk4']} occupations in "
              f"{BENCH3['n_ssyk3']} groups; median distance "
              f"{BENCH3['median_abs']:.2f}, ninetieth",
              f"  percentile {BENCH3['p90_abs']:.2f}). They can differ "
              "legitimately: if the larger occupations sit",
              "  further from their group means, the weighted within-share "
              "is the higher one.",
              f"  Built on {cnt(cost['n_workers'])} workers, "
              f"{cost['n_ssyk4']:,} four-digit codes and "
              f"{cost['n_ssyk3']:,} three-digit groups, of which "
              f"{cnt(cost['n_ssyk3_single_occupation'])} hold a single",
              f"  occupation and carry "
              f"{cost['employment_share_in_single_groups']:.1%} of the "
              f"employment: for those workers the arms are identical by",
              f"  construction and nothing is discarded "
              f"(benchmark: {BENCH3['n_singleton']} such groups).",
              "  The DAIOE release publishes no three-digit product, so the "
              "book is built here and exported."]
        if cost["share_within"] > WITHIN_ALARM:
            L += ["",
                  "  *** THE WITHIN-GROUP SHARE IS ABOVE "
                  f"{WITHIN_ALARM:.0%}. The three-digit book is discarding "
                  "enough of the",
                  "  *** variation that the choice of uniform3 as the "
                  "primary arm SHOULD BE REVISITED",
                  "  *** rather than assumed. Read the mixed43 and "
                  "four_only arms beside it before",
                  "  *** quoting anything, and say in the paper which arm "
                  "the estimate comes from."]
        L.append("")
    else:
        L += ["  the book could not be built", ""]
    if "A" in PARTS:
        L += ["A. THE SCORE AND WHAT IT COVERS (no fit):"]
        if cov_tab is not None:
            st = cov_tab[cov_tab["block"] == "step"]
            if len(st):
                L.append("  where the cascade resolves each incumbent, and "
                         "what survives the four-digit merge:")
                L.append(f"    {'year':<6} {'column':<16} {'resolved':>9} "
                         f"{'4-digit':>9} {'scorable':>9} {'3-digit':>9} "
                         f"{'lost merge':>11}")
                piv = st.pivot_table(index="group", columns="item",
                                     values="share", aggfunc="first")
                for y in [str(x) for x in CASCADE_BACK + CASCADE_FWD] \
                        + ["none"]:
                    if y not in piv.index:
                        continue
                    col = st[(st["group"] == y)]["source_column"].dropna()
                    cname = str(col.iloc[0]) if len(col) else ""

                    def _p(i):
                        v = piv.loc[y].get(i, np.nan)
                        return "         " if v != v else f"{v:8.2%} "
                    L.append(f"    {y:<6} {cname:<16} "
                             f"{_p('incumbents_resolved')}"
                             f"{_p('four_digit')}{_p('scorable')}"
                             f"{_p('three_digit_only')}"
                             f"{_p('lost_at_daioe_merge')}")
                L.append("    Shares of all incumbent head count. 2015 reads "
                         "a different column from the")
                L.append("    years above it, and the coverage falling away "
                         "as the cascade goes back is")
                L.append("    what the dictionary says to expect: three "
                         "digits everywhere only from 2019,")
                L.append("    four digits only from 2023.")
            for blk, nm in (("ssyk_ar", "the year the 2019 code was actually "
                                        "observed (SsykAr)"),
                            ("ssyk_status", "whether it matches the November "
                                            "employer (SsykStatus)")):
                d = cov_tab[cov_tab["block"] == blk]
                if not len(d):
                    continue
                L.append(f"  {nm}:")
                for _, r_ in d.sort_values("group").iterrows():
                    sh = r_["share"]
                    L.append(f"    {str(r_['group']):<10} "
                             + ("       " if sh != sh else f"{sh:7.2%}"))
            cv = cov_tab[cov_tab["block"] == "coverage"]
            L.append("  coverage of the code among incumbents, before and")
            L.append("  after the cascade, coded and scored:")
            order = [a for a in mc.AGE_GROUPS
                     if a in set(cv["group"])] + ["31-69 incumbents"]
            for grp in order:
                d = cv[cv["group"] == grp].set_index("item")
                if d.empty:
                    continue

                def _s(i):
                    return (f"{float(d.loc[i, 'share']):6.1%}"
                            if i in d.index and d.loc[i, "share"] ==
                            d.loc[i, "share"] else "      ")
                L.append(f"    {grp:<18} coded {_s('coded_share_2019_only')}"
                         f" -> {_s('coded_share_backward')}   scored "
                         f"{_s('scored_share_2019_only')} -> "
                         f"{_s('scored_share_backward')}")
            if loss_all:
                L.append("  where the employers the OLD rule lost went "
                         "(all employers with an incumbent):")
                for k in ("scored_by_both_rules", "recovered_by_the_floor",
                          "recovered_by_the_cascade",
                          "lost_to_the_floor_only",
                          "lost_to_missing_codes_only", "lost_to_both",
                          "scored_old_but_not_new"):
                    if k in loss_all:
                        L.append(f"    {k:<28} {cnt(loss_all[k]):>12}   "
                                 f"{loss_all[k] / max(loss_all['n'], 1):7.2%}")
                L.append("    The first three are the employers the new rule "
                         "scores. The next two name")
                L.append("    the causes separately and the third is both "
                         "at once.")
            if arm_summ:
                L.append("  employers scored by each arm, and what the arm "
                         "changes:")
                for a_ in SCORE_ARMS:
                    if a_ in arm_summ:
                        L.append(f"    {a_:<10} {cnt(arm_summ[a_]):>12}   "
                                 f"{ARM_LABEL[a_]}")
                for k, (n_, sh_, v_) in sorted(moved.items()):
                    if not k.startswith("quartile_changed"):
                        continue
                    L.append(f"    {k:<34} {cnt(n_):>10} employers move"
                             + ("" if sh_ != sh_ else f", {sh_:6.2%}"))
                for k, (n_, sh_, v_) in sorted(moved.items()):
                    if not k.startswith("spearman"):
                        continue
                    L.append(f"    {k:<34} "
                             + ("(no value)" if v_ != v_ else f"{v_:+10.3f}")
                             + f"   on {cnt(n_)} employers")
                L.append("    A rank correlation near one and few employers "
                         "moving quartile means the")
                L.append("    choice of arm changes little; the numbers say "
                         "which, and they are in the")
                L.append("    appendix table beside the coverage they come "
                         "from.")
            if floor_summ:
                L.append("  the floor sensitivity, employers scored:")
                for f_, k in sorted(floor_summ.items()):
                    L.append(f"    floor {f_:<3} {cnt(k):>12}")
            rt = cov_tab[cov_tab["block"] == "route"].set_index("group")
            if len(rt):
                L.append("  employers each route can score:")
                for g_ in ("occupation_route", "education_route",
                           "both_routes", "occupation_only",
                           "education_only"):
                    if g_ in rt.index:
                        L.append(f"    {g_:<18} "
                                 f"{cnt(rt.loc[g_, 'n_employers']):>12}")
            if cross:
                L.append(f"  the two quartiles agree for "
                         f"{cross['diag']:.1%} of the "
                         f"{cnt(cross['n'])} employers both routes score; "
                         f"Spearman rank correlation of the underlying "
                         f"scores {cross['rho']:+.3f}")
                L.append(f"  top quartile: {cnt(cross['top_occ'])} on the "
                         f"occupation route, {cnt(cross['top_edu'])} on the "
                         f"education route, {cnt(cross['top_both'])} on both")
            for s in cov_summ:
                L.append(f"  {s['panel']}: {cnt(s['n'])} employers, "
                         f"{cnt(s['n_occ'])} scorable on the occupation "
                         f"route ({s['n_occ'] / max(s['n'], 1):.1%}), "
                         f"{cnt(s['n_edu'])} on the education route "
                         f"({s['n_edu'] / max(s['n'], 1):.1%}), "
                         f"{cnt(s['n_both'])} on both")
                ls = s.get("loss", {})
                if ls:
                    L.append(f"    of the employers the old rule lost here: "
                             f"{cnt(ls.get('lost_to_the_floor_only'))} to the "
                             f"floor, "
                             f"{cnt(ls.get('lost_to_missing_codes_only'))} to "
                             f"missing codes after the cascade, "
                             f"{cnt(ls.get('lost_to_both'))} to both")
                if s["med_occ_only"] == s["med_occ_only"] \
                        and s["med_edu_only"] == s["med_edu_only"]:
                    L.append(f"    median monthly headcount "
                             f"{s['med_occ_only']:.1f} in the employers only "
                             f"the occupation route scores, "
                             f"{s['med_edu_only']:.1f} in those only the "
                             f"education route scores, {s['med_both']:.1f} in "
                             f"those both score; months employing anybody "
                             f"{s['months_occ_only']:.1f}, "
                             f"{s['months_edu_only']:.1f} and "
                             f"{s['months_both']:.1f}")
        else:
            L.append("  Part A produced nothing")
        L.append("")
    verdicts = []
    if "B" in PARTS:
        L += ["B. THE HEADLINE:"]
        main = _main_rows(head)
        if main:
            L.append(f"  {'band':<6} {'term':<28} {'coef':>9} {'se':>9} "
                     f"{'t':>7}")
            for r in main:
                if r["term"] not in ("rb_x_high_x_young",
                                     "interim_x_high_x_young",
                                     "post_x_high_x_young"):
                    continue
                L.append(f"  {r['young_band']:<6} {r['term']:<28} "
                         f"{r['coef']:+9.4f} {r['se']:9.4f} {r['t']:+7.2f}")
        else:
            L.append("  no stock fit came back on the reported arm")
        v, lines = verdict_headline(head)
        verdicts.append(("1 adoption step at 22-25", v))
        L += [""] + lines
        v, lines = verdict_profile(prof)
        verdicts.append(("2 age profile", v))
        L += [""] + lines + [""]
    if "C" in PARTS:
        L += ["C. THE SEX SPLIT, THE MARGINS AND THE VINTAGE CHECK:"]
        v, lines = verdict_gender(steps)
        verdicts.append(("3 female differential", v))
        L += lines
        if f_rows:
            L.append("  the margins at 22-25, adoption step:")
            for outcome in ("hires", "seps"):
                q = [r for r in f_rows if r["outcome"] == outcome
                     and r["term"] == "post_x_high_x_young"]
                if q:
                    e = EDU_FLOW[outcome]
                    L.append(f"    {outcome:<6} {q[0]['coef']:+.4f} "
                             f"({q[0]['se']:.4f}) t {q[0]['t']:+.2f}   "
                             f"education route {e[0]:+.4f} ({e[1]:.4f})")
        else:
            L.append("  no flow fit came back")
        if stab:
            L.append("  the vintage check, every arm against the 2019 code "
                     "alone:")
            for arm, s in stab.items():
                if "share_keeping_quartile" in s:
                    L.append(f"    {arm:<22} "
                             f"{s['share_keeping_quartile']:.1%} keep their "
                             f"quartile, mean relative shift "
                             f"{s['mean_relative_mix_shift']:.2%}, Spearman "
                             f"{s['spearman']:+.3f}")
                if "artefact" in s:
                    L.append(f"    {arm:<22} moves the adoption step "
                             f"{s['artefact']:+.4f}")
            L.append("    The education route's own re-scoring moved it "
                     "+0.0113 (-0.0408 to -0.0295).")
        else:
            L.append("  the vintage check produced nothing")
        L.append("")
    if verdicts:
        L += ["THE THREE VERDICTS:"]
        L += [f"  {k:<28} {v}" for k, v in verdicts]
        L.append("")
    if NOTES:
        L += ["NOTES:"] + [f"  {n}" for n in NOTES] + [""]
    if FAILURES:
        L += ["WHAT FAILED: " + "; ".join(FAILURES),
              "A missing row is a missing fit or a missing pull, never a "
              "zero.", ""]
    L += READ_RULES + [
        "",
        "WHAT TO EXPECT, SO IT IS NOT READ AS A BUG:",
        "  1. This route still scores fewer employers than the education",
        "     route. The occupation register samples about half the",
        "     workforce and imputes the rest, so a firm with incumbents",
        "     can have no coded incumbent at all. What Part A settles is",
        "     how much of the gap is that and how much was the floor:",
        "     script 65's rule counted CODED November persons, which is",
        "     roughly an order of magnitude stricter than the education",
        "     route's five person-months, and script 70's ladder (172,396",
        "     to 60,704) is that rule and not this one.",
        "  2. A carried-forward code is noisier than a contemporaneous",
        "     one. That attenuates towards zero and cannot manufacture a",
        "     result, and the share of coded incumbents from before the",
        "     freeze year is on every fit's row.",
        "  3. The two routes are standardised on their own distributions",
        "     and rank employers differently, so agreement in SIZE is not",
        "     expected and is not what any of the three rules asks for.",
        "  4. The forward arm and the floor variants are reported and",
        "     settle nothing. The read rules are on the backward cascade",
        f"     at a floor of {FLOOR_MAIN} and on nothing else.",
        "", f"Runtime {(time.time()-t0)/60:.1f} min. " + mc.mem_line()]
    (OUT / "82_summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n" + "\n".join(L))
    mc.runlog("82_occupation_route", 0, (time.time() - t0) / 60)
    print("\n82 done.")


if __name__ == "__main__":
    main()
