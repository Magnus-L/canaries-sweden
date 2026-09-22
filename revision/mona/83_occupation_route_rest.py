#!/usr/bin/env python3
"""
83_occupation_route_rest.py -- everything else in the paper that uses the
                               exposure quartile, re-estimated on lane
                               28's occupation-route score.

======================================================================
  RUNS IN MONA. Parts are chosen with the environment variable
  CANARIES_83_PARTS (default ABCD) and the folder with CANARIES_83_OUT
  (default output_83); the lane runners set both. Part A also needs a
  database connection, for the survey tables and the November 2024
  declarations; Parts B and C need none once the caches are on the
  share, except the one balance-sheet read Part C makes.
======================================================================

QUESTION
Script 82 re-scored the employer from the occupations its own incumbents
held in 2019 and showed that the headline, the age profile, the margins
and the sex split survive the change of route. Everything ELSE in the
paper that uses the exposure quartile is still on the education
definition: the first stage against reported AI use, the descriptive
counterpart of the headline, the reference-window specification, the
pre-launch drift, the industry standard errors, the industry-by-age-by-
month test and the credit test. Until those move too, a table in the
paper can mix two definitions of the same treatment. This script closes
that gap, so that no published number mixes two exposure definitions.

THE SCORE IS LANE 28'S AND IS NOT REBUILT HERE
Every fit in this script takes its quartile from
82_occupation_route.build_exposure(), the primary arm (uniform3, the
backward cascade, a floor of five incumbent person-months). There is one
definition of the treatment variable in one place: a second construction
would drift from the first the moment either was corrected, and a
published table would then carry two scores under one name. The cascade
pull is cached by whichever job reaches it first, so a second part reads
it rather than pulling again.

WHAT IS ESTIMATED

Part A (part_a). The most important part, and the cheapest, so it runs
first. Two arms, neither of them a panel fit.
  (i) The first stage of script 71, on the occupation quartile: the
      firm-level association between the quartile and reported AI use in
      the 2023 ITFtg wave, the individual-level association in the 2024
      BITA wave, and the same association in every earlier firm wave the
      delivery holds, which is how the pre-ChatGPT gap of 2019, 2021 and
      2023 is read. 71's own arms are called with BOTH routes on the SAME
      tables, so the comparison is a contrast within one regression
      sample rather than between two runs, and the education route's
      published figures are printed beside as a check on that.
  (ii) The descriptive counterpart of script 66: the mean headcount per
      employer-band-month cell by exposure quartile and age band, in the
      pre window and the adoption window, and the change between them.
      Raw means. Nothing is controlled for.

Part B (part_b). The remaining rows of Table 1.
  (i) Script 75's reference-window specification at both young bands:
      Equation (2) with the tightening term entered as a WINDOW (April to
      November 2022) rather than a cumulative switch, so that the interim
      and adoption terms read directly against January 2021 to March
      2022. The adoption term is then the LEVEL after adoption and has a
      standard error of its own.
  (ii) Script 78's pre-launch drift at both young bands: a linear monthly
      trend interacted with High x Young, estimated on January 2021 to
      November 2022 with the calendar terms and the tightening window in.
  (iii) The industry clustering of script 80's Part B applied to this
      quartile: the pooled stock fits at both bands and the sex fit,
      clustered on the COMPLETED three-digit industry key that 80 builds
      and caches. 80's key builder and 80's Part B are imported and run;
      neither is rebuilt.

Part C (part_c). The two robustness tests the Results section names.
  (i) Industry by age band by month at both bands, each beside a baseline
      on the same firms, which is script 80's Part C.
  (ii) The credit test of script 73's Part B on the employers carrying a
      2019 balance sheet: a median leverage split, the adoption step
      interacted with it, and a baseline re-estimated on the same
      balance-sheet sample so that the comparison is a specification
      change and not a sample change.

Part D (part_d). The firm-size robustness, asked for after lane 28a
landed. A firm's exposure is a MEAN over its incumbents aged 31 to 69, so
its sampling variance falls with their number: a firm with one incumbent
is classified by one worker's occupation, and that classification then
assigns treatment for all of its hiring and separations for years. The
floor never protected against this, and lane 28a shows it barely binds at
all, 271,047 employers at a floor of one against 262,089 at five.
Misclassified binary treatment attenuates towards zero, so the error is
conservative for the size of the step; but a noisier score has fatter
tails, which over-represents small firms at BOTH extremes of the score
distribution, and that is a composition distortion in the treatment group
that count-weighting does not remove. Three pieces.
  (i) The reliability of the firm score as a function of the number of
      incumbents behind it. No fit. The employment-weighted variance of
      the worker-level score is decomposed into between-firm and
      within-firm, the between component is netted of the sampling noise
      it contains, and the implied reliability is reported at the
      quantiles of the incumbent-count distribution, together with the
      share of employers and the share of incumbent employment whose
      score falls below a reliability of one half. It is REPORTED AND
      NEVER USED TO CORRECT AN ESTIMATE: the error here is not classical,
      the right answer to a low reliability is a design that raises it,
      and an attenuation correction would manufacture precision the data
      do not have.
  (ii) The adoption step at a floor of sixty incumbent person-months,
      which is five workers employed all year, at both young bands and
      beside the reported floor of five. THE QUARTILE IS HELD FIXED at
      the national floor-of-five one and is NOT recut on the survivors;
      see below.
  (iii) The adoption step by firm-size tercile at both young bands,
      terciles cut on the number of incumbents aged 31 to 69 in 2019, so
      the split is pre-treatment and fixed. The quartile is again the
      national one.

WHY THE QUARTILE IS NEVER RECUT IN PART D
Recomputing the cut points inside a subsample changes the treatment
definition with the subsample, and the comparison then confounds a
different treatment with a different population. In the tercile arm that
is obvious. It is equally true of the floor arm, whose question is
whether the noisiest scores distort the treatment group: the test is to
drop those employers while leaving every surviving employer's treatment
exactly as it was. Script 82's own floor sensitivity recuts, because it
asks a different question, how many employers a floor reaches. Both the
recut and the fixed-cut populations are described in the export; only the
fixed-cut arm is fitted.

WHY 80'S PARTS ARE CALLED AND 73'S PART A IS NOT
Script 80's Part C is script 73's industry test redone on the completed
industry key and beside a same-sample baseline, and the paper's retained
shares are 80's, not 73's. Running 73's own version here would change the
industry key as well as the exposure route and the comparison would carry
two differences at once. So the industry work in Parts B and C is 80's,
with one column changed, which is what a comparison requires. The credit
test has no such successor and is 73's, through 73's own leverage builder
and its own coverage gate.

ONE THING IS SWITCHED OFF IN 80 AND IT MATTERS
80's Part B reads lane 25's exports to get the employer-clustered run it
checks itself against. Those exports are the EDUCATION route's, so read
on this quartile they would compare our coefficients with someone else's
and the four-decimal gate would fail for a reason that is not a defect.
PRIOR_DIRS is therefore emptied before Part B is called, which sends 80
down its own documented fallback and makes it refit the employer-
clustered run on this panel. That costs three fits and buys a gate that
means what it says.

READ RULES
Fixed before the run, printed at the start and in the summary, with the
education-route figure beside every estimate. There is no coefficient
gate anywhere except the clustering fits. The first stage decides whether
anything in this lane or in lane 28 may be quoted at all, and its verdict
is stated at the TOP of the summary whichever way it falls.

INPUTS AND OUTPUTS
Reads, through the modules it imports: the caches L_counts_2021 to 2025
and L_baseline_2019 (script 47L), L_counts_sex_2021 to 2025 (script 67),
edu_hr_weights_2019 to 2021 and edu_hr_2019 (script 47h, for the
education route the first stage and the descriptives compare against),
L_baseline_2019_cascade (script 82, pulled and cached here if absent),
I_industry_key (script 80, built and cached here if absent) and the input
file daioe_quartiles.dta. Pulls, in Part A, the survey catalogue and the
AI survey tables and the November declarations of the BITA year; in Part
C, the 2019 balance sheets.

Writes to output_83/: occ_rest_firststage.csv, occ_rest_descriptive.csv,
occ_rest_window.csv, occ_rest_drift.csv, occ_rest_cluster.csv,
occ_rest_industry.csv, occ_rest_credit.csv, occ_rest_size.csv,
occ_rest_reliability.csv, the vcov_*.csv files and 83_summary.txt.

IN THE PAPER
Table 1's remaining rows and Online Appendix III.2's robustness of the
exposure route: the claim that no number in the paper depends on the
exposure running through the education register.
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
OUT = HERE / os.environ.get("CANARIES_83_OUT", "output_83")
PARTS = os.environ.get("CANARIES_83_PARTS", "ABCD").upper()
OUT.mkdir(exist_ok=True)
CACHE = mc.CACHE_DIR
# Script 82 is imported for its score. Its own OUT is pointed here before
# the import so that a run of this script cannot leave a stray output_82
# folder behind, and so anything it writes lands with these exports.
os.environ.setdefault("CANARIES_82_OUT", str(OUT))

POST_FROM = "2024-01"            # adoption, as in 68, 75, 78 and 80
BANDS = ["22-25", "26-30"]       # the two young bands the paper reports
SEX_BAND = "22-25"               # the band the paper's sex result is on
PRE = ("2022-01", "2023-12")     # the descriptive comparison window, as 66
FLOOR = 5                        # the export floor, as in mona_common
MATCH_DP = 4                     # the clustering gate, decimals, as in 80
SIG5 = 1.959963984540054         # two-sided five per cent
DRIFT_RULE_SE = 2.0              # 78's own flatness rule, in SEs
HALF = 0.50                      # the first stage's "at least half"
RETAIN_RULE = 0.50               # the industry test's retained share
CREDIT_RULE = 0.80               # the credit test's retained share
FLOOR_BIG = 60                   # five workers employed all year
N_TERCILES = 3                   # the firm-size split of Part D
RELIABILITY_ALARM = 0.50         # below this a firm's score is thin
SIZE_RULE_SE = 1.0               # Part D's rule, in standard errors
# A tercile "carries the whole step" if the other two together are within
# this share of zero relative to it; fixed here rather than argued later.
TERCILE_CARRY = 0.25

# The education route, from the exports the paper quotes. Every table this
# script writes carries the matching pair, so no comparison depends on a
# reader holding two files open.
#
# The first stage (script 71): the coefficient on the top-quartile dummy
# in a linear probability model of reported AI use, in percentage points.
EDU_FIRST = {"itftg_2023": 21.5, "bita_2024": 23.5}
# and the same firm-level gap in the three waves the paper calls flat.
EDU_ANY_GAP = {"2019": 20.3, "2021": 19.8, "2023": 21.5}
# The descriptive (script 66), as a shape rather than a number: on the
# education route the exposed employers grew in every band and grew the
# oldest band fastest.
EDU_DESCRIPTIVE = ("exposed employers grew in every band and grew the "
                   "oldest band fastest")
# The reference window (script 75): the level after adoption against the
# pre-hike months.
EDU_WINDOW = {"22-25": (-0.0194, 0.0184), "26-30": (-0.0172, 0.0121)}
# The pre-launch drift (script 78, Part A(ii)): the monthly trend.
EDU_DRIFT = {"22-25": (+0.0006, 0.0008), "26-30": (+0.0019, 0.0004)}
# The industry-clustered standard errors (script 80, Part B).
EDU_CLUSTER_SE = {("pooled", "22-25", "post_x_high_x_young"): 0.0302,
                  ("pooled", "26-30", "post_x_high_x_young"): 0.0200,
                  ("gender", SEX_BAND, "post_x_high_x_young"): 0.0353,
                  ("gender", SEX_BAND,
                   "post_x_high_x_young_x_female"): 0.0179,
                  ("gender", SEX_BAND, "female_step"): 0.0285}
# The industry test (script 80, Part C): the retained share of the
# adoption step against the same-sample baseline.
EDU_RETAINED = {"22-25": 0.85, "26-30": 0.47}
# The credit test (script 73, Part B): the adoption step as a share of the
# baseline re-estimated on the balance-sheet sample, and that baseline.
EDU_CREDIT = {"22-25": (1.02, -0.0695), "26-30": (1.05, -0.0516)}

NOTES = []
FAILURES = []
# Set by part_a; read by the summary, which refuses to let anything be
# quoted until the first stage has spoken.
FIRST_STAGE = "NOT RUN"

READ_RULES = [
    "READ RULES, FIXED BEFORE THE RUN:",
    "  There is NO coefficient gate anywhere in this lane EXCEPT the",
    "  clustering fits of Part B, where the coefficients must reproduce",
    f"  their own employer-clustered run to {MATCH_DP} decimals, exactly",
    "  as script 80 gates: the cluster changes the covariance and nothing",
    "  else, so a moved coefficient means a moved panel. Everywhere else",
    "  this is a different measure of the same object, the estimates will",
    "  differ from the education route, and that is expected. Every",
    "  estimate is printed beside the education-route one.",
    "",
    "  1. THE FIRST STAGE, and it decides whether anything in this lane",
    "     or in lane 28 may be quoted at all. THE FIRST STAGE REPRODUCES",
    "     if the occupation-route quartile predicts reported AI use with",
    "     the SAME SIGN as the education route and an association AT",
    "     LEAST HALF its size, on the 2023 firm-level ITFtg gap",
    f"     (education route +{EDU_FIRST['itftg_2023']:.1f} points) and on",
    f"     the 2024 individual BITA gap (+{EDU_FIRST['bita_2024']:.1f}",
    "     points). Both must hold; if one does it is PARTLY and the",
    "     summary names which. IF IT DOES NOT REPRODUCE, NOTHING IN LANE",
    "     29 OR LANE 28 IS QUOTED, and the summary says so at the top in",
    "     those words: a score that does not predict who uses AI is not",
    "     an AI exposure measure, whatever its coefficients do.",
    "     71's own sample gates travel with 71's code (800 firms and 100",
    "     exposed for ITFtg, 600 and 80 for BITA). This route scores",
    "     fewer employers, so BELOW THRESHOLD is a live outcome and is",
    "     accepted as one: it is reported as NO VERDICT, never as a",
    "     failure and never as a pass.",
    "  2. The pre-ChatGPT any-AI gap in 2019, 2021 and 2023 is a",
    "     DIAGNOSTIC and settles nothing. The education route reports it",
    f"     flat at {EDU_ANY_GAP['2019']:.1f}, {EDU_ANY_GAP['2021']:.1f}",
    f"     and {EDU_ANY_GAP['2023']:.1f} points; ours is printed beside",
    "     it whatever it is. No threshold is set on it, because none can",
    "     be read off three coefficients with no interaction test behind",
    "     them.",
    "  3. The descriptives carry NO verdict. They are raw means, with",
    "     composition, firm size and the business cycle inside them. The",
    "     education route's shape is stated (" + EDU_DESCRIPTIVE + ")",
    "     and ours is described beside it in the same words.",
    "  4. THE REFERENCE WINDOW AGREES IN DIRECTION if the level after",
    "     adoption is negative at both bands. No significance rule is",
    "     set, because neither education-route figure is distinguishable",
    f"     from zero ({EDU_WINDOW['22-25'][0]:+.4f} "
    f"({EDU_WINDOW['22-25'][1]:.4f}) at 22-25 and",
    f"     {EDU_WINDOW['26-30'][0]:+.4f} ({EDU_WINDOW['26-30'][1]:.4f}) at",
    "     26-30), so a rule on significance would be one the education",
    "     route itself fails.",
    "  5. THE PRE-LAUNCH DRIFT is read as script 78 reads it: FLAT if the",
    f"     monthly trend is within {DRIFT_RULE_SE:.0f} of its own standard",
    "     errors of zero. The education route is FLAT at 22-25",
    f"     ({EDU_DRIFT['22-25'][0]:+.4f} ({EDU_DRIFT['22-25'][1]:.4f})) and",
    f"     NOT FLAT at 26-30 ({EDU_DRIFT['26-30'][0]:+.4f} "
    f"({EDU_DRIFT['26-30'][1]:.4f}), four",
    "     standard errors from zero), so a drift at 26-30 here is",
    "     agreement with the education route and not a defect of this",
    "     route. Both are reported whichever way they fall.",
    "  6. THE CLUSTERING. The gate above, and then: THE INFERENCE",
    "     SURVIVES INDUSTRY CLUSTERING if the adoption step at 22-25 and",
    "     the female differential keep their sign and stay",
    "     distinguishable from zero at the five per cent level with the",
    "     industry standard error. The education route's industry",
    "     standard errors are 0.0302 at 22-25, 0.0200 at 26-30, 0.0353",
    "     for young men, 0.0179 for the female differential and 0.0285",
    "     for young women.",
    "  7. THE INDUSTRY TEST. No gate, as script 80 has none. The retained",
    "     share of the adoption step against the same-sample baseline is",
    "     reported whatever it is, beside the education route's, which is",
    "     85 per cent at 22-25 and 47 per cent at 26-30, and beside the",
    "     share of firms coded from a source other than Ftg_2019, since a",
    "     carried-forward code absorbs less and so flatters the test.",
    "     Fixed in advance: the step SURVIVES at a band if it keeps its",
    f"     sign and retains at least {RETAIN_RULE:.0%} of the same-sample",
    "     baseline. At 26-30 the education route retains 47 per cent and",
    "     so would not itself pass, which is said here rather than",
    "     discovered afterwards.",
    "  8. THE CREDIT TEST. THE STEP IS NOT A CREDIT EFFECT if it keeps",
    f"     its sign and at least {CREDIT_RULE:.0%} of the baseline",
    "     re-estimated on the balance-sheet sample, at both bands. The",
    "     education route keeps 102 and 105 per cent of -0.0695 and",
    "     -0.0516. 73's own coverage gate travels with 73's code (500",
    "     firms and 30 per cent of the panel), and BELOW THRESHOLD is",
    "     again an accepted outcome reported as NO VERDICT.",
    "  9. THE FIRM-SIZE ROBUSTNESS. A firm's score is a mean over its",
    "     incumbents, so its sampling variance falls with their number,",
    "     and the floor barely binds: lane 28a scores 271,047 employers",
    "     at a floor of one and 262,089 at five. A noisier score has",
    "     fatter tails, which over-represents small employers at BOTH",
    "     ends of the score distribution, and that is a composition",
    "     distortion in the treatment group that count-weighting does",
    "     not remove. THE SIZE ROBUSTNESS PASSES if the adoption step at",
    f"     22-25 at a floor of {FLOOR_BIG} incumbent person-months keeps",
    f"     its sign and sits within {SIZE_RULE_SE:.0f} standard error of",
    "     the reported floor-of-five step, AND no size tercile carries",
    "     the whole step. Both halves are reported whichever way they",
    "     fall, and if the step is carried by the smallest tercile the",
    "     summary says so in those words: the concern is then NOT",
    "     answered.",
    "     The quartile is the NATIONAL one in every arm of Part D and is",
    "     never recut inside a subsample, because that would change the",
    "     treatment definition with the sample and the comparison would",
    "     mean nothing.",
    "     The reliability table is DESCRIPTIVE and carries no verdict.",
    "     It is reported and NEVER used to correct an estimate: the",
    "     measurement error here is not classical, and the answer to a",
    "     low reliability is a design that raises it, not a correction",
    "     that manufactures precision the data do not have.",
    f"  Employer, firm and person counts below {FLOOR} are suppressed",
    "  before anything leaves MONA, and a share is suppressed with its",
    "  own numerator.",
]


def opt(label, fn, *a, **kw):
    """Run one arm. An arm that dies is recorded and the others still
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
    rule is enforced in one place rather than remembered at seven call
    sites.
    """
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if not df.empty and count_col in df.columns:
        df = mc.enforce_min_cell(df, count_col=count_col, floor=FLOOR)
    df.to_csv(OUT / name, index=False)
    return df


def tstat(coef, se) -> float:
    return float(coef) / se if se and se == se and se > 0 else np.nan


def fmt(x) -> str:
    return "(no SE)" if x is None or x != x else f"({x:.4f})"


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
        r = mc.run_fepois_multi(b, OUT, tag=f"s83_{tag}", terms=terms,
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


def load_modules():
    """
    The scripts this one reuses rather than reimplements.

    82 builds the occupation-route score and is the ONLY place it is
    built; 61 the balanced employer by band by month skeleton, 67 the
    same with sex as a fourth dimension, 71 the two first-stage arms, 66
    the descriptive cells, 73 the identifier normalisation, the
    catalogue probe, the leverage builder and its coverage gate, 75 the
    reference-window term set, 78 the term sets of Equation (2), of its
    sex split and of the pre-launch drift, 80 the completed industry key
    and the clustered and industry-absorbed fits, 47L the 2019 pulls,
    47j the fixed-effect list and the incumbent bands.

    Importing them rather than copying is the point: this script must be
    71, 66, 75, 78 and 80 with one column changed, and a copied term list
    could drift away from the estimates the paper quotes.

    Three modules have a module-level OUT of their own. They are pointed
    here so that anything reached through them lands with this script's
    exports rather than in an earlier lane's folder.
    """
    s82 = _mod("82_occupation_route.py", "s82")
    s82.OUT = OUT
    s61, s67, s74, s78, s80, l47, l70, j47 = s82.load_modules()
    s71 = _mod("71_adoption_validation.py", "s71")
    s66 = _mod("66_plain_magnitudes.py", "s66")
    s73 = _mod("73_industry_and_credit.py", "s73")
    s75 = _mod("75_reference_window.py", "s75")
    h47 = j47._h47()
    for m_ in (s71, s66, s73, s75, s80):
        m_.OUT, m_.CACHE = OUT, CACHE
    # 80's Part B checks itself against the employer-clustered run it
    # finds in an earlier lane's folder. Those folders hold the EDUCATION
    # route's exports, and reading them here would compare our
    # coefficients with someone else's and fail the four-decimal gate for
    # a reason that is not a defect. Emptying the search path sends 80
    # down its own documented fallback, which refits the employer-
    # clustered run on this panel, which is what the gate needs.
    s80.PRIOR_DIRS = ()
    NOTES.append("80's prior-export search path is emptied, so the "
                 "employer-clustered run the four-decimal gate checks "
                 "against is refitted on THIS panel rather than read from "
                 "the education route's exports")
    # Four guards. Each is a place where this script's docstring and read
    # rules would otherwise describe a model it is not fitting.
    for name, mod in (("78", s78), ("80", s80), ("75", s75)):
        if getattr(mod, "POST_FROM", POST_FROM) != POST_FROM:
            raise RuntimeError(
                f"{name}'s adoption date is {mod.POST_FROM} and this script "
                f"says {POST_FROM}; the terms come from it, so settle it "
                f"there first")
    if s73.POOLED_FROM != POST_FROM:
        raise RuntimeError(
            f"73's adoption date is {s73.POOLED_FROM} and this script says "
            f"{POST_FROM}; the credit test's terms come from 73")
    if tuple(s66.PRE) != tuple(PRE) or s66.POST_FROM != POST_FROM:
        raise RuntimeError(
            f"66's descriptive windows are {s66.PRE} and {s66.POST_FROM} "
            f"and this script says {PRE} and {POST_FROM}; the cells come "
            f"from 66, so settle it there first")
    if s80.MATCH_DP != MATCH_DP:
        raise RuntimeError(
            f"80 gates at {s80.MATCH_DP} decimals and this script's read "
            f"rules say {MATCH_DP}")
    return (s82, s61, s66, s67, s71, s73, s74, s75, s78, s80, l47, l70,
            j47, h47)


def load_counts(prefix: str, years, require=None):
    out = []
    for y in years:
        c = mc.read_cache(CACHE / f"{prefix}_{y}.parquet", require=require)
        if c is None:
            return None
        out.append(c)
    return pd.concat(out, ignore_index=True) if out else None


# ----------------------------------------------------------------------
# Part A(i): the first stage
# ----------------------------------------------------------------------

def wave_year(source: str) -> str:
    """The survey year in a table name, for grouping the waves."""
    import re
    m = re.search(r"(\d{4})", str(source))
    return m.group(1) if m else "unknown"


def first_stage(occ: pd.DataFrame, edu: pd.DataFrame, s71, s73) -> tuple:
    """
    Does the occupation-route quartile predict reported AI use?

    Script 71's two arms, run on BOTH routes over the SAME survey tables.
    Running both is not decoration: a coefficient from this run beside a
    figure recorded from an earlier one would differ in the sample as
    well as in the route, and the question is which route predicts
    adoption among the same firms. The education route's published
    figures are printed beside its re-estimate here as a check that the
    two agree.

    71's arms take a dict of routes and read only employer_id and fq from
    each, and they normalise their own key but not the exposure's, so the
    identifiers are normalised here before they are handed over.

    Returns (rows, counts, waves): the coefficient table, the overlap
    counts, and the firm-level any-AI gap per wave and route.
    """
    routes = {}
    for name, e in (("occupation", occ), ("education", edu)):
        if e is None or e.empty:
            continue
        d = e[["employer_id", "fq"]].copy()
        d["employer_id"] = s73.norm_id(d["employer_id"])
        d["fq"] = pd.to_numeric(d["fq"], errors="coerce").astype("Int64")
        routes[name] = d.dropna(subset=["fq"])
    if "occupation" not in routes:
        FAILURES.append("A/first stage/no occupation exposure")
        return [], [], []
    conn = mc.connect()
    sink, counts = [], []
    try:
        schema = s71.discover(conn)
        if schema.empty:
            NOTES.append("first stage: the catalogue returned no survey "
                         "table, so no first stage could be run")
            return [], [], []
        base = mc.read_cache(CACHE / "L_baseline_2019.parquet",
                             require=["employer_id", "n"])
        if base is None:
            # 71 controls the firm arm for log 2019 size. Without the
            # baseline the control cannot be built, and dropping it
            # silently would change the specification rather than the
            # sample, so the arm is refused and says so.
            NOTES.append("first stage: L_baseline_2019 is not on the share, "
                         "so the log-size control of 71's firm arm cannot "
                         "be built; the ITFtg arm is NOT run")
            FAILURES.append("A/first stage/no L_baseline_2019")
        else:
            size = s71.firm_size(base)
            size["employer_id"] = s73.norm_id(size["employer_id"])
            del base
            gc.collect()
            opt("ITFtg arm", s71.itftg_arm, conn, schema, routes, size,
                sink, counts)
        opt("BITA arm", s71.bita_arm, conn, schema, routes, sink, counts)
    finally:
        try:
            conn.close()
        except Exception:
            pass
    drain(s71, "71")
    rows = (pd.concat(sink, ignore_index=True) if sink else pd.DataFrame())
    if rows.empty:
        NOTES.append("first stage: no arm returned an estimate; the "
                     "thresholds fixed in 71 are in the notes above")
        return [], counts, []
    # 71 returns every term of every model. The top-quartile dummy is the
    # one the read rule is on, and it reads in percentage points because
    # the outcome is a zero-one indicator.
    hi = rows[rows["term"] == "high"].copy()
    hi["coef_points"] = hi["coef"] * 100.0
    hi["se_points"] = hi["se"] * 100.0
    hi["t"] = hi.apply(lambda r: tstat(r["coef"], r["se"]), axis=1)
    hi["wave"] = hi["source"].map(wave_year)
    waves = []
    for _, r in hi.iterrows():
        if r["outcome"] != "ai_any" or not str(r["source"]).lower().startswith(
                ("itftg", "ai_")):
            continue
        waves.append({"route": r["route"], "wave": r["wave"],
                      "source": r["source"], "points": r["coef_points"],
                      "se_points": r["se_points"], "n": int(r["n"]),
                      "edu_points": EDU_ANY_GAP.get(str(r["wave"]), np.nan)})
    return hi.to_dict("records"), counts, waves


def headline_gap(rows: list, route: str, kind: str) -> dict:
    """
    The one association each half of read rule 1 is read on.

    `kind` is "itftg" for the 2023 firm wave, whose outcome is the
    any-AI indicator, or "bita" for the 2024 individual wave, whose
    outcome is the generative-AI question. The most recent matching table
    is taken, so a delivery holding several waves does not need the table
    name written into this script.
    """
    want = ("ai_any", ("itftg", "ai_")) if kind == "itftg" \
        else ("genai", ("bita",))
    out_col, prefixes = want
    cand = [r for r in rows
            if r.get("route") == route and r.get("outcome") == out_col
            and str(r.get("source", "")).lower().startswith(prefixes)]
    if not cand:
        return {}
    cand.sort(key=lambda r: str(r.get("wave", "")))
    r = cand[-1]
    return {"source": r["source"], "wave": r["wave"],
            "points": float(r["coef_points"]), "se": float(r["se_points"]),
            "t": float(r["t"]), "n": int(r["n"])}


# ----------------------------------------------------------------------
# Part A(ii): the descriptive counterpart
# ----------------------------------------------------------------------

def post_over_pre(g: pd.DataFrame, route: str) -> list:
    """
    The mean headcount per employer-band-month cell, by quartile and age
    band, in the pre window and in the adoption window, and the change.

    The unit is 66's: an employer, an age band and a month. `mean_value`
    is the arithmetic mean over those cells, so the statistic is what an
    employer of that quartile held in an average month, and the change is
    its log ratio between the two windows. It is a raw mean and it
    controls for nothing.
    """
    if g is None or g.empty:
        return []
    w = g.pivot_table(index=["fq", "age_group"], columns="period",
                      values="mean_value", aggfunc="first")
    f = g.pivot_table(index=["fq", "age_group"], columns="period",
                      values="n_firms", aggfunc="min")
    rows = []
    for idx in w.index:
        pre = w.loc[idx].get("pre", np.nan)
        post = w.loc[idx].get("post", np.nan)
        nf = f.loc[idx].min() if idx in f.index else np.nan
        rows.append({
            "route": route, "fq": int(idx[0]), "age_group": str(idx[1]),
            "mean_pre": float(pre) if pre == pre else np.nan,
            "mean_post": float(post) if post == post else np.nan,
            "log_change": (float(np.log(post / pre))
                           if pre == pre and post == post and pre > 0
                           else np.nan),
            "n_firms": (int(nf) if nf == nf else np.nan)})
    return rows


def describe_shape(rows: list, route: str) -> str:
    """
    The education route's claim, in the same words, tested on one route.

    The claim is that the exposed employers grew in every band and grew
    the oldest band fastest. Both halves are read off the top quartile's
    rows and reported whichever way they fall; neither is a verdict.
    """
    d = {r["age_group"]: r["log_change"] for r in rows
         if r["route"] == route and r["fq"] == 4
         and r["log_change"] == r["log_change"]}
    if not d:
        return f"{route}: the top quartile has no complete band"
    grew = [b for b, v in d.items() if v > 0]
    fastest = max(d, key=lambda b: d[b])
    every = len(grew) == len(d)
    return (f"{route}: the exposed quartile grew in "
            + ("EVERY" if every else f"{len(grew)} of {len(d)}")
            + f" band{'' if every else 's'} and grew {fastest} fastest "
            + f"({d[fastest]:+.4f}); the bands, lowest first: "
            + ", ".join(f"{b} {d[b]:+.4f}" for b in sorted(d, key=d.get)))


def part_a(counts, occ, edu, s66, s71, s73, l47, j47) -> tuple:
    """
    The first stage and the descriptive counterpart. No panel fit runs
    here, which is why this part is first: it is the cheapest of the
    three and it is the one that decides whether the rest may be quoted.
    """
    rows, ov, waves = [], [], []
    r = opt("Part A first stage", first_stage, occ, edu, s71, s73)
    if r:
        rows, ov, waves = r
    if rows:
        f = pd.DataFrame(rows)
        f["edu_recorded_points"] = np.nan
        for i, rr in f.iterrows():
            if rr["route"] != "education":
                continue
            k = ("itftg_2023" if rr["outcome"] == "ai_any"
                 and rr["wave"] == "2023" else
                 ("bita_2024" if rr["outcome"] == "genai"
                  and rr["wave"] == "2024" else None))
            if k:
                f.at[i, "edu_recorded_points"] = EDU_FIRST[k]
        save(f, "occ_rest_firststage.csv", count_col="n")
        for _, rr in f.iterrows():
            print(f"  A: {str(rr['source']):<20} {str(rr['route']):<11} "
                  f"{str(rr['outcome']):<11} {rr['coef_points']:+7.2f} "
                  f"points ({rr['se_points']:.2f}) t {rr['t']:+.2f} "
                  f"n {cnt(rr['n'])}")
    if ov:
        c = pd.DataFrame(ov)
        # 71 drops a thin overlap row; the floor is applied again on the
        # way out so the rule is enforced here as everywhere else.
        c = mc.enforce_min_cell(c, count_col="with_outcome", floor=FLOOR)
        c = mc.enforce_min_cell(c, count_col="high_with_outcome", floor=FLOOR)
        c.to_csv(OUT / "occ_rest_firststage_overlap.csv", index=False)
    # ---- the descriptive counterpart ---------------------------------
    drows = []
    for name, e in (("occupation", occ), ("education", edu)):
        if e is None or e.empty:
            continue
        g = opt(f"Part A descriptive/{name}", s66.describe, counts, e,
                "n_emp", j47.YOUNG_BANDS, j47.INCUMBENT_BANDS)
        if g is None or g.empty:
            FAILURES.append(f"A/descriptive/{name}")
            continue
        drows += post_over_pre(g, name)
    if drows:
        d = pd.DataFrame(drows)
        d["edu_route_shape"] = EDU_DESCRIPTIVE
        save(d, "occ_rest_descriptive.csv")
        for name in ("occupation", "education"):
            line = describe_shape(drows, name)
            print(f"  A: {line}")
            NOTES.append(f"descriptive {line}")
    return rows, ov, waves, drows


# ----------------------------------------------------------------------
# Part B: the remaining rows of Table 1
# ----------------------------------------------------------------------

def drain(mod, tag: str) -> None:
    """
    Move an imported script's own notes and failures into ours.

    A note 80 or 73 writes about the sample it fitted belongs in this
    script's summary, not in a list nobody prints. They are drained
    rather than copied so that a second call does not repeat the first
    call's notes.
    """
    for n in list(getattr(mod, "NOTES", [])):
        NOTES.append(f"{tag}: {n}")
    for f in list(getattr(mod, "FAILURES", [])):
        FAILURES.append(f"{tag}/{f}")
    if hasattr(mod, "NOTES"):
        mod.NOTES.clear()
    if hasattr(mod, "FAILURES"):
        mod.FAILURES.clear()


def part_b_window(counts, occ, s61, s75, s78, j47) -> list:
    """
    The level after adoption, against the months before the rate rise.

    Script 75's specification: Equation (2) with the tightening term
    entered as a WINDOW covering April to November 2022 rather than as a
    cumulative switch, so the interim and adoption terms read directly
    against January 2021 to March 2022 and the adoption term is a level
    with a standard error of its own. The panel, the exposure merge and
    the fixed effects are the paper's; only the column that says which
    employers are highly exposed differs.

    75's own reconciliation rule is NOT applied. It checks the window
    coefficient against script 68's cumulative pair, which are the
    EDUCATION route's, and on this quartile the two specifications
    describe different fitted means, so the check would fail for a reason
    that is not a defect. There is therefore no gate here, as the read
    rules say.
    """
    rows = []
    for band in BANDS:
        skel = s61.build_skeleton(counts, band, j47)
        if skel.empty:
            FAILURES.append(f"B/window/{band}/empty")
            continue
        b = s78.with_exposure(skel, occ)
        del skel
        gc.collect()
        if b.empty:
            FAILURES.append(f"B/window/{band}/no exposure")
            continue
        n_firms = int(b["employer_id"].nunique())
        b, terms = s75.add_window_terms(b)
        g, _ = fit(b, f"window_{band.replace('-', '_')}", terms, j47.FES)
        del b
        gc.collect()
        if g is None:
            continue
        ec, es = EDU_WINDOW.get(band, (np.nan, np.nan))
        for r in s78.rows_of(g, terms, young_band=band, outcome="stock",
                             n_firms=n_firms):
            r["t"] = tstat(r["coef"], r["se"])
            post = r["term"] == "post_x_high_x_young"
            r["edu_coef"] = ec if post else np.nan
            r["edu_se"] = es if post else np.nan
            rows.append(r)
        save(rows, "occ_rest_window.csv")
        p = [r for r in rows if r["young_band"] == band
             and r["term"] == "post_x_high_x_young"]
        if p:
            print(f"  B: window {band} level after adoption against the "
                  f"pre-hike months {p[0]['coef']:+.4f} ({p[0]['se']:.4f}) "
                  f"t {p[0]['t']:+.2f}   education route {ec:+.4f} ({es:.4f})")
    return rows


def part_b_drift(counts, occ, s61, s78, j47) -> list:
    """
    Whether the exposed employers' young workers were already drifting
    away before the launch.

    Script 78's Part A(ii): the pre-launch months only, from January 2021
    to November 2022, with the calendar cycle, the tightening window and
    a LINEAR MONTHLY TREND, each interacted with High x Young. The trend
    is the testable direction: a joint test of every pre-launch quarter
    net of the cycle is not identified from two years of months, which is
    why 78 tests this one and why this script does not invent another.

    The skeleton is 78's own band list, the young band beside the four
    incumbent bands, so the drift is estimated on the panel 78 estimated
    it on and the two figures are comparable.
    """
    rows = []
    for band in BANDS:
        skel = s78.build_skeleton_bands(counts, [band] + j47.INCUMBENT_BANDS,
                                        band, j47, s61.PANEL_FROM)
        if skel.empty:
            FAILURES.append(f"B/drift/{band}/empty")
            continue
        b = s78.with_exposure(skel, occ)
        del skel
        gc.collect()
        if b.empty:
            FAILURES.append(f"B/drift/{band}/no exposure")
            continue
        ym = b["year_month"].astype(str)
        pre = b[(ym >= s78.PRE_TREND_FROM)
                & (ym < s78.PRE_LAUNCH_END)].copy()
        del b
        gc.collect()
        # The window cut leaves cells that are zero in every remaining
        # month. They carry no information and a Poisson fit should not
        # be asked to carry them, so they go, exactly as 78 drops them.
        pre = j47._drop_dead_cells(pre)
        if pre.empty:
            FAILURES.append(f"B/drift/{band}/empty after the window")
            continue
        n_firms = int(pre["employer_id"].nunique())
        months = sorted(pre["year_month"].astype(str).unique())
        pre, terms = s78.drift_terms(pre)
        g, _ = fit(pre, f"drift_{band.replace('-', '_')}", terms, j47.FES)
        del pre
        gc.collect()
        if g is None:
            continue
        ec, es = EDU_DRIFT.get(band, (np.nan, np.nan))
        for r in s78.rows_of(g, terms, young_band=band, n_firms=n_firms,
                             first_month=months[0], last_month=months[-1],
                             n_months=len(months)):
            r["t"] = tstat(r["coef"], r["se"])
            tr = r["term"] == "trend_x_high_x_young"
            r["edu_coef"] = ec if tr else np.nan
            r["edu_se"] = es if tr else np.nan
            r["flat_within_2se"] = (bool(abs(r["coef"])
                                         <= DRIFT_RULE_SE * r["se"])
                                    if tr and r["se"] else None)
            rows.append(r)
        save(rows, "occ_rest_drift.csv")
        p = [r for r in rows if r["young_band"] == band
             and r["term"] == "trend_x_high_x_young"]
        if p:
            print(f"  B: drift {band} monthly trend {p[0]['coef']:+.4f} "
                  f"({p[0]['se']:.4f}) t {p[0]['t']:+.2f} over "
                  f"{p[0]['n_months']} months to {p[0]['last_month']}"
                  f"   education route {ec:+.4f} ({es:.4f})")
    return rows


def part_b_cluster(counts, sexcounts, occ, key, s61, s67, s73, s78, s80,
                   j47) -> tuple:
    """
    The same estimates, with the standard errors clustered on the
    employer's three-digit industry rather than on the employer.

    This is script 80's Part B with one column changed, and it is CALLED
    rather than copied: 80 builds the completed industry key, attaches it
    as one integer cluster with the unresolved employers sharing a single
    residual group, fits the pooled stock specification at both young
    bands and the sex specification at 22-25, and checks the coefficients
    against its own employer-clustered run.

    THE ONE GATE IN THIS SCRIPT IS 80'S. Clustering changes the
    covariance and nothing else, so a coefficient that moves means the
    panel moved, and nothing from this arm may then be quoted. Because
    the employer-clustered run 80 checks against is refitted here on this
    panel (see load_modules), the check is against this lane's own numbers
    and not against the education route's.

    Returns (rows, summ) and writes occ_rest_cluster.csv with the
    education route's industry standard errors beside ours.
    """
    r = s80.part_b(counts, sexcounts, occ, key, s61, s67, s78, s73, j47)
    drain(s80, "80")
    if not r:
        return [], {}
    rows, summ = r
    if not rows:
        FAILURES.append("B/cluster/no fit")
        return [], summ
    d = pd.DataFrame(rows)
    d["edu_se_industry"] = [
        EDU_CLUSTER_SE.get((sp, bd, tm), np.nan)
        for sp, bd, tm in zip(d["spec"], d["young_band"], d["term"])]
    d["t_industry"] = [tstat(c, s) for c, s in
                       zip(d["coef"], d["se_industry_complete"])]
    save(d, "occ_rest_cluster.csv")
    # 80 writes its own file under its own name. One folder should carry
    # one name per table, and this lane's name is the one the upload list
    # gives, so the duplicate goes.
    stray = OUT / "cluster_industry_v2.csv"
    if stray.exists():
        stray.unlink()
    ok = d["coef_match_4dp"].dropna()
    passed = bool(len(ok) and ok.all())
    print(f"  B: clustering, coefficients reproduce the employer-clustered "
          f"run to {MATCH_DP} decimals: {'YES' if passed else 'NO'}")
    for _, rr in d[d["term"].isin(["post_x_high_x_young",
                                   "post_x_high_x_young_x_female"])].iterrows():
        print(f"     {rr['spec']:<7} {rr['young_band']:<6} {rr['term']:<30} "
              f"{rr['coef']:+.4f}  employer {rr['se_employer']:.4f}  "
              f"industry {rr['se_industry_complete']:.4f}   education route "
              f"industry {rr['edu_se_industry']:.4f}")
    return d.to_dict("records"), summ


def part_b(counts, sexcounts, occ, key, s61, s67, s73, s75, s78, s80,
           j47) -> tuple:
    """The reference window, the pre-launch drift and the industry
    clustering, in that order: the two cheap fits before the six."""
    win = opt("B/window", part_b_window, counts, occ, s61, s75, s78, j47) or []
    dri = opt("B/drift", part_b_drift, counts, occ, s61, s78, j47) or []
    clu, summ = (opt("B/cluster", part_b_cluster, counts, sexcounts, occ,
                     key, s61, s67, s73, s78, s80, j47) or ([], {}))
    return win, dri, clu, summ


# ----------------------------------------------------------------------
# Part C: the two robustness tests Results names
# ----------------------------------------------------------------------

def part_c_industry(counts, occ, key, s61, s73, s78, s80, j47) -> list:
    """
    How much of the adoption step survives absorbing industry by age band
    by month.

    Script 80's Part C, called rather than copied: the stock
    specification of Equation (2) fitted twice on exactly the same
    employers, once with the paper's three effects and once with the
    month-by-age effect replaced by three-digit industry by age band by
    month, which nests it. An employer with no industry code leaves BOTH
    fits, so the difference between them is the specification and not the
    sample.

    80's version and not 73's, deliberately: 80 runs it on the COMPLETED
    industry key and beside a same-sample baseline, and the retained
    shares the paper quotes are 80's. Running 73's version here would
    change the industry key as well as the exposure route, and the
    comparison would then carry two differences at once.
    """
    rows = s80.part_c(counts, occ, key, s61, s78, s73, j47)
    drain(s80, "80")
    if not rows:
        FAILURES.append("C/industry/no fit")
        return []
    d = pd.DataFrame(rows)
    d["edu_retained_share"] = [EDU_RETAINED.get(b, np.nan)
                               for b in d["young_band"]]
    d["t"] = [tstat(c, s) for c, s in zip(d["coef"], d["se"])]
    save(d, "occ_rest_industry.csv")
    stray = OUT / "industry_seasonal_v2.csv"
    if stray.exists():
        stray.unlink()
    for band in BANDS:
        g = d[(d["young_band"] == band)
              & (d["term"] == "post_x_high_x_young")]
        if g.empty:
            continue
        base = g[g["spec"] == "baseline_same_sample"]
        ind = g[g["spec"] == "industry_age_month"]
        if base.empty or ind.empty:
            continue
        sh = float(ind["retained_share"].iloc[0])
        print(f"  C: industry {band} baseline "
              f"{float(base['coef'].iloc[0]):+.4f} "
              f"({float(base['se'].iloc[0]):.4f}) -> with industry x age x "
              f"month {float(ind['coef'].iloc[0]):+.4f} "
              f"({float(ind['se'].iloc[0]):.4f}), retained {sh:.0%}"
              f"   education route {EDU_RETAINED.get(band, float('nan')):.0%}")
    return d.to_dict("records")


def part_c_credit(counts, occ, s61, s73, j47) -> list:
    """
    Whether the adoption step is a credit effect rather than an AI one.

    Script 73's Part B, called through 73's own run_band: the employers
    carrying a 2019 balance sheet are split at the median of leverage,
    the adoption step is interacted with the split, and a BASELINE IS
    RE-ESTIMATED ON THAT SAMPLE FIRST, so the comparison is a
    specification change and not a sample change. 73's coverage gate
    travels with it: an arm needs 500 panel employers and 30 per cent of
    them to carry the covariate, and below that it reports nothing rather
    than a thin estimate.

    run_band also fits the full-panel baseline at each band before it
    reaches the credit arm. That is two fits this part did not have to
    ask for and they are worth having: on this quartile the full-panel
    adoption step is the quantity lane 28 estimated, so the two are a
    check that this lane and that one are fitting the same panel.

    run_band normalises the skeleton's identifier and merges the exposure
    on it, so the exposure's identifier is normalised here first.
    """
    lev = pd.DataFrame()
    conn = mc.connect()
    try:
        schema = s73.discover(conn)
        if schema.empty:
            NOTES.append("credit: the catalogue returned no firm table, so "
                         "no balance sheet could be read")
            return []
        lev = s73.firm_leverage(conn, schema)
    finally:
        try:
            conn.close()
        except Exception:
            pass
    drain(s73, "73")
    if lev is None or lev.empty:
        NOTES.append("credit: no 2019 balance sheet answered, so the credit "
                     "test cannot run; this is a data outcome and not a "
                     "result")
        FAILURES.append("C/credit/no leverage")
        return []
    e = occ[["employer_id", "fq"]].copy()
    e["employer_id"] = s73.norm_id(e["employer_id"])
    sinks = {"ind": [], "indq": [], "lev": [], "bank": []}
    s73.PARTS = "B"
    for band in BANDS:
        opt(f"C/credit/{band}", s73.run_band, counts, e, pd.DataFrame(),
            lev, set(), band, j47, sinks)
        drain(s73, "73")
        gc.collect()
    rows = []
    for r in sinks["lev"]:
        band = r["band"]
        share, base = EDU_CREDIT.get(band, (np.nan, np.nan))
        rows.append({**r, "spec": "credit", "t": tstat(r["coef"], r["se"]),
                     "edu_share_of_baseline": share,
                     "edu_baseline_on_balance_sheet_sample": base})
    for r in sinks["ind"]:
        rows.append({"band": r["band"], "term": "full_panel_baseline",
                     "coef": r["coef"], "se": r["se"], "spec": "full panel",
                     "t": tstat(r["coef"], r["se"]),
                     "edu_share_of_baseline": np.nan,
                     "edu_baseline_on_balance_sheet_sample": np.nan})
    if not rows:
        FAILURES.append("C/credit/no fit")
        return []
    save(rows, "occ_rest_credit.csv")
    for band in BANDS:
        b = [r for r in rows if r["band"] == band]
        step = next((r for r in b if r["term"] == "post_x_high_x_young"
                     and r["spec"] == "credit"), None)
        base = next((r for r in b
                     if r["term"] == "baseline_on_balance_sheet_sample"),
                    None)
        if step and base and base["coef"]:
            sh = step["coef"] / base["coef"]
            es, eb = EDU_CREDIT.get(band, (np.nan, np.nan))
            print(f"  C: credit {band} baseline on the balance-sheet sample "
                  f"{base['coef']:+.4f} ({base['se']:.4f}); with the "
                  f"leverage split {step['coef']:+.4f} ({step['se']:.4f}), "
                  f"{sh:.0%} of it   education route {es:.0%} of {eb:+.4f}")
    return rows


def part_c(counts, occ, key, s61, s73, s78, s80, j47) -> tuple:
    ind = opt("C/industry", part_c_industry, counts, occ, key, s61, s73,
              s78, s80, j47) or []
    cre = opt("C/credit", part_c_credit, counts, occ, s61, s73, j47) or []
    return ind, cre


# ----------------------------------------------------------------------
# Part D: the firm-size robustness
# ----------------------------------------------------------------------

def reliability_rows(inc: pd.DataFrame, occ: pd.DataFrame, s82) -> tuple:
    """
    How well measured is a firm's score, as a function of how many
    incumbents stand behind it?

    The firm score is the employment-weighted mean of its incumbents'
    occupation scores, so it is a sample mean and carries sampling error
    that falls with the number of incumbents. Decomposing the
    employment-weighted variance of the WORKER-level score into a
    between-firm and a within-firm part gives both quantities at once:
    the between part is how much firms really differ, the within part is
    how much workers differ inside a firm, and a firm with n coded
    incumbents has a score whose reliability is

        lambda(n) = var_between / (var_between + var_within / n).

    THE BETWEEN COMPONENT IS NETTED OF ITS OWN SAMPLING NOISE. The
    variance of the OBSERVED firm means is the variance of the true means
    plus var_within / n, so using it raw would overstate how much firms
    differ and so overstate the reliability. Both are exported; the
    reliability is computed from the netted one, which is the
    conservative reading.

    IT IS REPORTED AND NEVER USED TO CORRECT AN ESTIMATE. The measurement
    error here is not classical: it is larger for small employers, which
    are also the employers over-represented at the tails of the score.
    The answer to a low reliability is a design that raises it, which is
    what the floor arm below tests, not a correction that would
    manufacture precision the data do not have.

    Returns (rows, summary dict).
    """
    ids = set(occ["employer_id"])
    score, _ = s82.arm_score(inc, s82.MAIN_LEVEL)
    u = inc.assign(score=score)
    u = u[u["score"].notna() & u["employer_id"].isin(ids)]
    if u.empty:
        return [], {}
    u = u.merge(occ[["employer_id", "mix", "n_coded", "n_nov"]],
                on="employer_id", how="inner")
    w = u["n"].astype(float)
    tot = float(w.sum())
    mu = float((u["score"] * w).sum() / max(tot, 1.0))
    var_within = float((w * (u["score"] - u["mix"]) ** 2).sum()
                       / max(tot, 1.0))
    f = occ[occ["employer_id"].isin(set(u["employer_id"]))].copy()
    f["n_coded"] = pd.to_numeric(f["n_coded"], errors="coerce").fillna(0)
    f = f[f["n_coded"] > 0]
    fw = f["n_coded"].astype(float)
    var_between_raw = float((fw * (f["mix"] - mu) ** 2).sum()
                            / max(float(fw.sum()), 1.0))
    # E[1/n] under the same employment weights is simply the number of
    # employers over the total coded head count.
    noise = var_within * len(f) / max(float(fw.sum()), 1.0)
    var_between = max(var_between_raw - noise, 0.0)

    def lam(n):
        n = max(float(n), 1.0)
        d = var_between + var_within / n
        return float(var_between / d) if d > 0 else np.nan

    f["reliability"] = [lam(n) for n in f["n_coded"]]
    thin = f["reliability"] < RELIABILITY_ALARM
    emp = f["n_nov"].astype(float)
    rows = [{"block": "variance", "item": k, "n_incumbents": np.nan,
             "value": v, "n_firms": int(len(f)), "share_firms": np.nan,
             "share_employment": np.nan}
            for k, v in (("mean_score", mu),
                         ("variance_within_firm", var_within),
                         ("variance_between_firms_observed",
                          var_between_raw),
                         ("sampling_noise_removed", noise),
                         ("variance_between_firms_net", var_between),
                         ("share_between_net",
                          var_between / max(var_between + var_within,
                                            1e-12)))]
    qs = [0.10, 0.25, 0.50, 0.75, 0.90]
    for q in qs:
        n_q = float(f["n_coded"].quantile(q))
        rows.append({"block": "reliability_by_firm_quantile",
                     "item": f"p{int(q * 100)}_of_employers",
                     "n_incumbents": n_q, "value": lam(n_q),
                     "n_firms": int(len(f)), "share_firms": q,
                     "share_employment": np.nan})
    o = f.sort_values("n_coded")
    cum = (o["n_nov"].astype(float).cumsum()
           / max(float(o["n_nov"].astype(float).sum()), 1.0)).to_numpy()
    for q in qs:
        i = int(np.searchsorted(cum, q, side="left"))
        i = min(max(i, 0), len(o) - 1)
        n_q = float(o["n_coded"].iloc[i])
        rows.append({"block": "reliability_by_employment_quantile",
                     "item": f"p{int(q * 100)}_of_incumbent_employment",
                     "n_incumbents": n_q, "value": lam(n_q),
                     "n_firms": int(len(f)), "share_firms": np.nan,
                     "share_employment": q})
    for n_ in (1, 2, 5, 10, 25, 100):
        rows.append({"block": "reliability_at_n", "item": f"n_{n_}",
                     "n_incumbents": float(n_), "value": lam(n_),
                     "n_firms": int((f["n_coded"] == n_).sum()),
                     "share_firms": float((f["n_coded"] == n_).mean()),
                     "share_employment":
                         float(emp[f["n_coded"] == n_].sum()
                               / max(float(emp.sum()), 1.0))})
    rows.append({"block": "thin", "item":
                 f"reliability_below_{RELIABILITY_ALARM:.2f}",
                 "n_incumbents": np.nan, "value": RELIABILITY_ALARM,
                 "n_firms": int(thin.sum()),
                 "share_firms": float(thin.mean()),
                 "share_employment": float(emp[thin].sum()
                                           / max(float(emp.sum()), 1.0))})
    summ = {"var_within": var_within, "var_between_raw": var_between_raw,
            "var_between": var_between, "noise": noise,
            "lambda_1": lam(1), "lambda_median": lam(f["n_coded"].median()),
            "median_n": float(f["n_coded"].median()),
            "n_thin": int(thin.sum()), "share_thin": float(thin.mean()),
            "employment_thin": float(emp[thin].sum()
                                     / max(float(emp.sum()), 1.0)),
            "n_firms": int(len(f))}
    return rows, summ


def size_fit(counts, expo, band, spec, s61, s78, j47, sink,
             extra: dict = None) -> None:
    """One stock fit of Equation (2) on a subset of the employers, with
    the NATIONAL quartile carried in unchanged."""
    skel = s61.build_skeleton(counts, band, j47)
    if skel.empty:
        FAILURES.append(f"D/{spec}/{band}/empty")
        return
    b = s78.with_exposure(skel, expo)
    del skel
    gc.collect()
    if b.empty:
        FAILURES.append(f"D/{spec}/{band}/no exposure")
        return
    n_firms = int(b["employer_id"].nunique())
    n_high = int(b.loc[b["high"] == 1, "employer_id"].nunique())
    b, terms = s78.eq2_terms(b)
    g, _ = fit(b, f"size_{spec}_{band.replace('-', '_')}", terms, j47.FES)
    del b
    gc.collect()
    if g is None:
        return
    for r in s78.rows_of(g, terms, young_band=band, spec=spec,
                         n_firms=n_firms, n_exposed_firms=n_high,
                         **(extra or {})):
        r["t"] = tstat(r["coef"], r["se"])
        sink.append(r)
    save(sink, "occ_rest_size.csv")
    p = [r for r in sink if r["spec"] == spec and r["young_band"] == band
         and r["term"] == "post_x_high_x_young"]
    if p:
        print(f"  D: {spec:<12} {band} adoption step {p[0]['coef']:+.4f} "
              f"({p[0]['se']:.4f}) t {p[0]['t']:+.2f} on "
              f"{cnt(n_firms)} employers, {cnt(n_high)} of them exposed")


def part_d(counts, occ, inc, nfloor, s61, s78, s82, j47) -> tuple:
    """
    The firm-size robustness: the reliability of the score, the headline
    at a floor five workers employed all year would clear, and the
    headline by firm-size tercile.

    THE QUARTILE IS THE NATIONAL ONE IN EVERY ARM. Every subset below is
    a restriction of the frame `occ`, which already carries the national
    quartile, and no arm recomputes a cut point. Recutting inside a
    subsample would change the treatment definition with the sample and
    the comparison would confound a different treatment with a different
    population.
    """
    rel, rsumm = [], {}
    r = opt("D/reliability", reliability_rows, inc, occ, s82)
    if r:
        rel, rsumm = r
    if rel:
        save(rel, "occ_rest_reliability.csv")
        print(f"  D: reliability of the firm score: between-firm variance "
              f"{rsumm['var_between']:.2f} net of "
              f"{rsumm['noise']:.2f} of sampling noise, within-firm "
              f"{rsumm['var_within']:.2f}; a one-incumbent score has "
              f"reliability {rsumm['lambda_1']:.2f} and the median "
              f"employer, with {rsumm['median_n']:.0f} coded incumbents, "
              f"{rsumm['lambda_median']:.2f}")
        print(f"  D: {cnt(rsumm['n_thin'])} employers "
              f"({rsumm['share_thin']:.1%}) are below a reliability of "
              f"{RELIABILITY_ALARM:.2f}, holding "
              f"{rsumm['employment_thin']:.1%} of incumbent employment")
    sink = []
    # ---- (ii) the floor of sixty person-months ------------------------
    big = occ[occ["n"] >= FLOOR_BIG]
    share_f = len(big) / max(len(occ), 1)
    share_e = (float(big["n_nov"].sum()) / max(float(occ["n_nov"].sum()), 1)
               if "n_nov" in occ.columns else np.nan)
    # What a RECUT would have done, reported and not fitted, so a reader
    # can see how much of any difference is the population and how much
    # would have been the moved cut points.
    recut = s82.occ_route_exposure(inc, nfloor, FLOOR_BIG,
                                   s82.ARM_YEARS[s82.MAIN_ARM])
    moved = np.nan
    if not recut.empty:
        j = big[["employer_id", "fq"]].merge(
            recut[["employer_id", "fq"]], on="employer_id",
            suffixes=("_fixed", "_recut"))
        moved = (float((j["fq_fixed"] != j["fq_recut"]).mean())
                 if len(j) else np.nan)
    msg = (f"floor {FLOOR_BIG}: {len(big):,} of {len(occ):,} employers "
           f"clear it ({share_f:.1%} of employers, {share_e:.1%} of "
           f"incumbent employment); the quartile is the national one and "
           f"is NOT recut, and recutting on the survivors would move "
           + ("an unknown share" if moved != moved else f"{moved:.1%}")
           + " of them to a different quartile")
    print(f"  {msg}")
    NOTES.append(msg)
    del recut
    gc.collect()
    for band in BANDS:
        size_fit(counts, occ, band, "floor_5", s61, s78, j47, sink,
                 {"floor": FLOOR, "n_scored": len(occ)})
        if big.empty:
            FAILURES.append(f"D/floor_{FLOOR_BIG}/{band}/no employer")
            continue
        size_fit(counts, big, band, f"floor_{FLOOR_BIG}", s61, s78, j47,
                 sink, {"floor": FLOOR_BIG, "n_scored": len(big)})
    # ---- (iii) the firm-size terciles ---------------------------------
    # Cut on the number of incumbents aged 31 to 69 in 2019, over the
    # scored employers and unweighted, so each tercile holds a third of
    # the employers. Pre-treatment by construction: the count is the 2019
    # one the score itself was built from.
    n_inc = pd.to_numeric(occ["n_nov"], errors="coerce").fillna(0)
    cuts = [float(n_inc.quantile(q)) for q in (1 / 3, 2 / 3)]
    terc = np.searchsorted(np.asarray(cuts), n_inc.to_numpy(),
                           side="right") + 1
    occ = occ.assign(size_tercile=terc)
    for t_ in range(1, N_TERCILES + 1):
        d = occ[occ["size_tercile"] == t_]
        msg = (f"size tercile {t_}: {len(d):,} employers, "
               f"{int(n_inc[occ['size_tercile'] == t_].min()) if len(d) else 0}"
               f" to "
               f"{int(n_inc[occ['size_tercile'] == t_].max()) if len(d) else 0}"
               f" incumbents, "
               f"{int((d['fq'] == 4).sum()):,} of them in the NATIONAL top "
               f"quartile, which is not recut here")
        print(f"  {msg}")
        NOTES.append(msg)
        if d.empty:
            FAILURES.append(f"D/tercile_{t_}/no employer")
            continue
        for band in BANDS:
            size_fit(counts, d, band, f"tercile_{t_}", s61, s78, j47, sink,
                     {"floor": FLOOR, "n_scored": len(d),
                      "tercile_low": float(n_inc[occ["size_tercile"]
                                                 == t_].min()),
                      "tercile_high": float(n_inc[occ["size_tercile"]
                                                  == t_].max())})
    return sink, rel, rsumm


# ----------------------------------------------------------------------
# The verdicts, every one of them fixed before the run
# ----------------------------------------------------------------------

def verdict_size(rows: list, rsumm: dict) -> tuple:
    """
    Read rule 9, in two halves, both reported whichever way they fall.

    The floor half asks whether dropping the employers whose score rests
    on too few incumbents moves the step; the tercile half asks whether
    the step is carried by the small employers whose scores are the
    noisy ones. The quartile is the national one in every arm, so a
    difference between arms is a difference in the employers and never
    in the treatment.
    """
    if not rows:
        return "NO VERDICT", ["  9. THE FIRM-SIZE ROBUSTNESS: NO VERDICT, "
                              "no fit came back."]
    d = pd.DataFrame(rows)
    d = d[d["term"] == "post_x_high_x_young"]
    L, half = [], {}
    if rsumm:
        L.append(f"     the score's reliability: a one-incumbent score is "
                 f"{rsumm['lambda_1']:.2f} and the median employer, with "
                 f"{rsumm['median_n']:.0f} coded incumbents, "
                 f"{rsumm['lambda_median']:.2f}; "
                 f"{cnt(rsumm['n_thin'])} employers "
                 f"({rsumm['share_thin']:.1%}) sit below "
                 f"{RELIABILITY_ALARM:.2f} and hold "
                 f"{rsumm['employment_thin']:.1%} of incumbent employment")
        L.append("     Descriptive. It is never used to correct an "
                 "estimate; the answer to a low")
        L.append("     reliability is a design that raises it, which is "
                 "what the floor arm below is.")
    ref = d[(d["spec"] == "floor_5") & (d["young_band"] == "22-25")]
    big = d[(d["spec"] == f"floor_{FLOOR_BIG}")
            & (d["young_band"] == "22-25")]
    if ref.empty or big.empty:
        half["floor"] = None
        L.append(f"     the floor of {FLOOR_BIG}: NO VERDICT, one of the "
                 f"two fits did not come back")
    else:
        c0, s0 = float(ref["coef"].iloc[0]), float(ref["se"].iloc[0])
        c1, s1 = float(big["coef"].iloc[0]), float(big["se"].iloc[0])
        half["floor"] = bool((c0 < 0) == (c1 < 0)
                             and abs(c1 - c0) <= SIZE_RULE_SE * s0)
        L.append(f"     22-25 at the reported floor of {FLOOR} "
                 f"{c0:+.4f} ({s0:.4f}) on "
                 f"{cnt(ref['n_firms'].iloc[0])} employers")
        L.append(f"     22-25 at a floor of {FLOOR_BIG} person-months "
                 f"{c1:+.4f} ({s1:.4f}) on "
                 f"{cnt(big['n_firms'].iloc[0])} employers; the difference "
                 f"is {c1 - c0:+.4f}, which is "
                 f"{abs(c1 - c0) / s0:.2f} of the reported standard error "
                 f"and the rule asks for at most {SIZE_RULE_SE:.0f}")
        for band in BANDS[1:]:
            q0 = d[(d["spec"] == "floor_5") & (d["young_band"] == band)]
            q1 = d[(d["spec"] == f"floor_{FLOOR_BIG}")
                   & (d["young_band"] == band)]
            if not q0.empty and not q1.empty:
                L.append(f"     {band}, not part of the rule: "
                         f"{float(q0['coef'].iloc[0]):+.4f} "
                         f"({float(q0['se'].iloc[0]):.4f}) -> "
                         f"{float(q1['coef'].iloc[0]):+.4f} "
                         f"({float(q1['se'].iloc[0]):.4f})")
    ter = {}
    for t_ in range(1, N_TERCILES + 1):
        g = d[(d["spec"] == f"tercile_{t_}") & (d["young_band"] == "22-25")]
        if not g.empty:
            ter[t_] = (float(g["coef"].iloc[0]), float(g["se"].iloc[0]),
                       g["n_firms"].iloc[0])
    if len(ter) < N_TERCILES:
        half["tercile"] = None
        L.append("     the size terciles: NO VERDICT, a tercile did not "
                 "come back")
    else:
        for t_, (c_, s_, n_) in sorted(ter.items()):
            L.append(f"     22-25 in size tercile {t_} "
                     f"{'(the smallest)' if t_ == 1 else ''} {c_:+.4f} "
                     f"({s_:.4f}) t {tstat(c_, s_):+.2f} on "
                     f"{cnt(n_)} employers")
        carrier = None
        for t_, (c_, _, _) in ter.items():
            others = [abs(v[0]) for k, v in ter.items() if k != t_]
            if abs(c_) > 0 and max(others) <= TERCILE_CARRY * abs(c_):
                carrier = t_
        half["tercile"] = carrier is None
        if carrier is None:
            L.append("     no tercile carries the whole step: every "
                     "tercile's step is a material")
            L.append(f"     fraction of the largest, on the "
                     f"{TERCILE_CARRY:.0%} rule fixed before the run")
        else:
            L.append(f"     TERCILE {carrier} CARRIES THE WHOLE STEP: the "
                     f"other two are within {TERCILE_CARRY:.0%} of zero "
                     f"beside it")
            if carrier == 1:
                L.append("     AND IT IS THE SMALLEST TERCILE, whose "
                         "scores are the noisy ones, so the")
                L.append("     concern this part was written to answer is "
                         "NOT ANSWERED.")
    live = [v for v in half.values() if v is not None]
    if len(live) < 2:
        verdict = "NO VERDICT"
    elif all(live):
        verdict = "THE SIZE ROBUSTNESS PASSES"
    else:
        verdict = "THE SIZE ROBUSTNESS DOES NOT PASS"
    return verdict, [f"  9. THE FIRM-SIZE ROBUSTNESS: {verdict}"] + L


def verdict_first_stage(rows: list) -> tuple:
    """
    Read rule 1, and the one that decides whether anything else may be
    quoted.

    REPRODUCES if the occupation quartile predicts reported AI use with
    the same sign as the education route and an association at least half
    its size, on the firm wave and on the individual wave alike. Both
    halves are reported whichever way they fall, and a half that 71's own
    sample gate refused is NO VERDICT rather than a failure: a rule
    cannot be read on an estimate that was never made.
    """
    L, got = [], {}
    for kind, key, label in (
            ("itftg", "itftg_2023",
             "the firm-level gap in reported AI use"),
            ("bita", "bita_2024",
             "the individual-level gap in generative-AI use")):
        ours = headline_gap(rows, "occupation", kind)
        theirs = headline_gap(rows, "education", kind)
        recorded = EDU_FIRST[key]
        if not ours:
            got[kind] = None
            L.append(f"     {label}: NO VERDICT. 71 returned no estimate "
                     f"for the occupation route, which its own sample gate "
                     f"does when the overlap is thin. The notes say which "
                     f"table and which threshold.")
            continue
        # The comparison is the education route's estimate ON THE SAME
        # TABLE where this run made one, because that differs from ours
        # in the route and in nothing else. The recorded figure is
        # printed beside it as a check that the two agree.
        base = theirs["points"] if theirs else recorded
        src = ("this run's own education-route estimate on the same table"
               if theirs else "the recorded education-route figure")
        L.append(f"     {label} ({ours['source']}, {cnt(ours['n'])} "
                 f"observations): occupation route "
                 f"{ours['points']:+.2f} points ({ours['se']:.2f}) t "
                 f"{ours['t']:+.2f}")
        if not base:
            got[kind] = None
            L.append(f"       NO VERDICT: {src} is {base:+.2f} points, so "
                     f"neither half of the rule can be read against it.")
            continue
        same_sign = (ours["points"] > 0) == (base > 0)
        share = abs(ours["points"]) / abs(base)
        got[kind] = bool(same_sign and share >= HALF)
        L.append(f"       against {base:+.2f} points on {src}"
                 + (f" (recorded {recorded:+.2f})" if theirs else "")
                 + f"; the rule asks for the same sign and at least "
                   f"{HALF:.0%} of it, and ours is "
                   f"{'the same sign' if same_sign else 'THE OPPOSITE SIGN'}"
                   f" at {share:.0%} of it, so the rule is "
                   f"{'MET' if got[kind] else 'NOT MET'}")
    live = [v for v in got.values() if v is not None]
    if not live:
        verdict = "NO VERDICT"
    elif all(live) and len(live) == 2:
        verdict = "THE FIRST STAGE REPRODUCES"
    elif any(live):
        verdict = "THE FIRST STAGE PARTLY REPRODUCES"
    else:
        verdict = "THE FIRST STAGE DOES NOT REPRODUCE"
    head = [f"  1. THE FIRST STAGE: {verdict}"]
    if verdict == "THE FIRST STAGE DOES NOT REPRODUCE":
        head += ["     NOTHING IN LANE 29 OR LANE 28 IS QUOTED. A score "
                 "that does not predict who",
                 "     uses AI is not an AI exposure measure, whatever its "
                 "coefficients do."]
    elif verdict == "THE FIRST STAGE PARTLY REPRODUCES":
        ok = [k for k, v in got.items() if v]
        no = [k for k, v in got.items() if v is False]
        head += [f"     The rule is met on {', '.join(ok)} and NOT on "
                 f"{', '.join(no)}. Read the half that failed",
                 "     before quoting anything from either lane."]
    elif verdict == "NO VERDICT":
        head += ["     No estimate was made, so the rule cannot be read. "
                 "That is not a pass:",
                 "     nothing in either lane is quoted until it is."]
    return verdict, head + L


def verdict_waves(waves: list) -> list:
    """
    The pre-ChatGPT gap, as a diagnostic and not as a test.

    The education route reports it flat across 2019, 2021 and 2023, which
    is evidence that the score picks up firms already doing AI work
    rather than firms that adopted after the launch. Ours is printed
    beside it. No threshold is set: three coefficients with no
    interaction test behind them cannot carry one.
    """
    L = ["  2. THE PRE-CHATGPT ANY-AI GAP (a diagnostic; it settles "
         "nothing):"]
    if not waves:
        return L + ["     no firm-level wave returned an estimate"]
    for route in ("occupation", "education"):
        d = sorted([w for w in waves if w["route"] == route],
                   key=lambda w: str(w["wave"]))
        if not d:
            continue
        L.append(f"     {route:<11} "
                 + "  ".join(f"{w['wave']} {w['points']:+.1f} "
                             f"({w['se_points']:.1f})" for w in d))
    L.append("     the education route is recorded flat at "
             + ", ".join(f"{k} {v:+.1f}" for k, v in
                         sorted(EDU_ANY_GAP.items())) + " points")
    return L


def verdict_descriptive(rows: list) -> list:
    """Read rule 3: no verdict, the shape of each route in the same
    words, and the reminder that these are raw means."""
    L = ["  3. THE DESCRIPTIVE COUNTERPART (no verdict; raw means):"]
    if not rows:
        return L + ["     no cell came back"]
    for route in ("occupation", "education"):
        L.append("     " + describe_shape(rows, route))
    L.append(f"     the education route is recorded as: {EDU_DESCRIPTIVE}")
    L.append("     Composition, firm size and the business cycle are inside "
             "every one of these")
    L.append("     numbers, which is what the regression exists to remove. "
             "Never present one")
    L.append("     as though it were the other.")
    return L


def verdict_window(rows: list) -> tuple:
    """Read rule 4: the level after adoption, negative at both bands."""
    if not rows:
        return "NO VERDICT", ["  4. THE REFERENCE WINDOW: NO VERDICT, no "
                              "fit came back."]
    got = {}
    L = []
    for band in BANDS:
        p = [r for r in rows if r["young_band"] == band
             and r["term"] == "post_x_high_x_young"]
        if not p:
            continue
        c, s = float(p[0]["coef"]), float(p[0]["se"])
        got[band] = c
        ec, es = EDU_WINDOW.get(band, (np.nan, np.nan))
        L.append(f"     {band}: {c:+.4f} ({s:.4f}) t {tstat(c, s):+.2f}; "
                 f"education route {ec:+.4f} ({es:.4f})")
    if len(got) < len(BANDS):
        verdict = "NO VERDICT"
    elif all(v < 0 for v in got.values()):
        verdict = "THE REFERENCE WINDOW AGREES IN DIRECTION"
    else:
        verdict = "THE REFERENCE WINDOW DOES NOT AGREE IN DIRECTION"
    return verdict, [f"  4. THE REFERENCE WINDOW: {verdict}"] + L + [
        "     This is the LEVEL after adoption relative to the months "
        "before the rate rise,",
        "     not the adoption step. It goes into Table 1 as that and "
        "nowhere as 'the effect'."]


def verdict_drift(rows: list) -> tuple:
    """Read rule 5: 78's flatness rule on the monthly trend."""
    if not rows:
        return "NO VERDICT", ["  5. THE PRE-LAUNCH DRIFT: NO VERDICT, no "
                              "fit came back."]
    L, flat = [], {}
    for band in BANDS:
        p = [r for r in rows if r["young_band"] == band
             and r["term"] == "trend_x_high_x_young"]
        if not p:
            continue
        c, s = float(p[0]["coef"]), float(p[0]["se"])
        flat[band] = abs(c) <= DRIFT_RULE_SE * s if s else False
        ec, es = EDU_DRIFT.get(band, (np.nan, np.nan))
        n_m = int(p[0].get("n_months", 0) or 0)
        L.append(f"     {band}: {c:+.4f} ({s:.4f}) t {tstat(c, s):+.2f}, "
                 f"{'FLAT' if flat[band] else 'NOT FLAT'} by the two-SE "
                 f"rule; over {n_m} pre-launch months that is "
                 f"{c * max(n_m - 1, 0):+.4f} in total")
        L.append(f"       education route at {band} {ec:+.4f} ({es:.4f}), "
                 f"{'FLAT' if abs(ec) <= DRIFT_RULE_SE * es else 'NOT FLAT'} "
                 f"by the same rule")
    if len(flat) < len(BANDS):
        verdict = "NO VERDICT"
    elif all(flat.values()):
        verdict = "FLAT AT BOTH BANDS"
    elif any(flat.values()):
        verdict = "FLAT AT " + ", ".join(b for b, v in flat.items() if v) \
            + " AND NOT AT " + ", ".join(b for b, v in flat.items() if not v)
    else:
        verdict = "NOT FLAT AT EITHER BAND"
    return verdict, [f"  5. THE PRE-LAUNCH DRIFT: {verdict}"] + L + [
        "     The education route is flat at 22-25 and four standard "
        "errors from zero at",
        "     26-30, so a drift at 26-30 here is agreement with it and "
        "not a defect of",
        "     this route."]


def verdict_cluster(rows: list, summ: dict) -> tuple:
    """
    Read rule 6: the gate first, then whether the inference survives.

    The gate is 80's and is absolute: the coefficients must reproduce
    the employer-clustered run to four decimals, because clustering moves
    the covariance and nothing else. If it fails, nothing from this arm
    is quoted and the size of the standard errors is beside the point.
    """
    if not rows:
        return "NO VERDICT", ["  6. THE INDUSTRY CLUSTERING: NO VERDICT, no "
                              "fit came back."]
    d = pd.DataFrame(rows)
    ok = d["coef_match_4dp"].dropna()
    passed = bool(len(ok) and ok.all())
    L = [f"     the coefficients reproduce the employer-clustered run to "
         f"{MATCH_DP} decimals: {'YES' if passed else 'NO'}"]
    if not passed:
        L.append("     THE RULE SAYS NOTHING FROM THE CLUSTERING ARM IS "
                 "QUOTED: a moved coefficient")
        L.append("     means a moved panel, since the cluster changes the "
                 "covariance and nothing else.")
        return "THE GATE FAILS", ["  6. THE INDUSTRY CLUSTERING: THE GATE "
                                  "FAILS"] + L
    L.append(f"     {'spec':<7} {'band':<6} {'term':<32} {'coef':>9} "
             f"{'employer':>10} {'industry':>10} {'education':>10}")
    checks = {}
    for spec, band, term, name in (
            ("pooled", "22-25", "post_x_high_x_young", "adoption step"),
            ("pooled", "26-30", "post_x_high_x_young", "adoption step"),
            ("gender", SEX_BAND, "post_x_high_x_young", "young men"),
            ("gender", SEX_BAND, "post_x_high_x_young_x_female",
             "female differential")):
        g = d[(d["spec"] == spec) & (d["young_band"] == band)
              & (d["term"] == term)]
        if g.empty:
            continue
        r = g.iloc[0]
        ci, se = float(r["coef"]), float(r["se_industry_complete"])
        checks[(spec, band, term)] = (ci, se)
        L.append(f"     {spec:<7} {band:<6} {term:<32} {ci:+9.4f} "
                 f"{float(r['se_employer']):10.4f} {se:10.4f} "
                 f"{float(r['edu_se_industry']):10.4f}")
    steps = summ.get("gender", {}).get("steps", {})
    for k, nm in (("female_step", "young women, adoption step"),
                  ("male_step", "young men, adoption step")):
        if k in steps:
            c_, se_e, _, se_i = steps[k]
            L.append(f"     {'derived':<7} {SEX_BAND:<6} {nm:<32} "
                     f"{c_:+9.4f} "
                     f"{fmt(se_e):>10} {fmt(se_i):>10} "
                     + (f"{EDU_CLUSTER_SE[('gender', SEX_BAND, k)]:10.4f}"
                        if ('gender', SEX_BAND, k) in EDU_CLUSTER_SE
                        else f"{'':>10}"))
    keep = []
    for key in (("pooled", "22-25", "post_x_high_x_young"),
                ("gender", SEX_BAND, "post_x_high_x_young_x_female")):
        if key not in checks:
            keep.append(None)
            continue
        c_, s_ = checks[key]
        keep.append(bool(c_ < 0 and s_ and abs(c_ / s_) >= SIG5))
    if any(k is None for k in keep):
        verdict = "NO VERDICT"
    elif all(keep):
        verdict = "THE INFERENCE SURVIVES INDUSTRY CLUSTERING"
    else:
        verdict = "THE INFERENCE DOES NOT SURVIVE INDUSTRY CLUSTERING"
    return verdict, [f"  6. THE INDUSTRY CLUSTERING: {verdict}"] + L


def verdict_industry(rows: list) -> tuple:
    """Read rule 7: the retained share, with the non-2019 share beside
    it, and no gate."""
    if not rows:
        return "NO VERDICT", ["  7. THE INDUSTRY TEST: NO VERDICT, no fit "
                              "came back."]
    d = pd.DataFrame(rows)
    L, keeps = [], {}
    for band in BANDS:
        g = d[(d["young_band"] == band)
              & (d["term"] == "post_x_high_x_young")]
        b = g[g["spec"] == "baseline_same_sample"]
        i = g[g["spec"] == "industry_age_month"]
        if b.empty or i.empty:
            continue
        cb, ci = float(b["coef"].iloc[0]), float(i["coef"].iloc[0])
        sh = float(i["retained_share"].iloc[0])
        keeps[band] = bool((cb < 0) == (ci < 0) and abs(sh) >= RETAIN_RULE)
        L.append(f"     {band}: baseline {cb:+.4f} "
                 f"({float(b['se'].iloc[0]):.4f}) on the same firms -> "
                 f"with industry x age x month {ci:+.4f} "
                 f"({float(i['se'].iloc[0]):.4f}), retained {sh:.0%}; "
                 f"education route {EDU_RETAINED.get(band, float('nan')):.0%}")
        L.append(f"       {cnt(i['n_firms'].iloc[0])} employers in both "
                 f"fits, {float(i['share_not_from_2019'].iloc[0]):.1%} of "
                 f"them coded from a source other than Ftg_2019, in "
                 f"{int(i['n_groups'].iloc[0]):,} industry groups")
    if len(keeps) < len(BANDS):
        verdict = "NO VERDICT"
    elif all(keeps.values()):
        verdict = "THE STEP SURVIVES AT BOTH BANDS"
    elif any(keeps.values()):
        verdict = ("THE STEP SURVIVES AT "
                   + ", ".join(b for b, v in keeps.items() if v)
                   + " AND NOT AT "
                   + ", ".join(b for b, v in keeps.items() if not v))
    else:
        verdict = "THE STEP SURVIVES AT NEITHER BAND"
    return verdict, [f"  7. THE INDUSTRY TEST: {verdict}"] + L + [
        "     A carried-forward industry code is noisier than a "
        "contemporaneous one and so",
        "     absorbs less, which raises the retained share for a "
        "mechanical reason. That",
        "     is why the non-2019 share is printed beside it. At 26-30 "
        "the education route",
        f"     itself retains {EDU_RETAINED['26-30']:.0%} and so would not "
        f"pass this rule."]


def verdict_credit(rows: list) -> tuple:
    """Read rule 8: the step against the baseline on the same
    balance-sheet sample."""
    if not rows:
        return "NO VERDICT", ["  8. THE CREDIT TEST: NO VERDICT, no fit "
                              "came back."]
    L, keeps = [], {}
    for band in BANDS:
        b = [r for r in rows if r["band"] == band]
        step = next((r for r in b if r.get("term") == "post_x_high_x_young"
                     and r.get("spec") == "credit"), None)
        base = next((r for r in b if r.get("term")
                     == "baseline_on_balance_sheet_sample"), None)
        lev = next((r for r in b if r.get("term")
                    == "post_x_high_x_young_x_lev"), None)
        full = next((r for r in b if r.get("term")
                     == "full_panel_baseline"), None)
        if not (step and base and base["coef"]):
            continue
        sh = step["coef"] / base["coef"]
        keeps[band] = bool((step["coef"] < 0) == (base["coef"] < 0)
                           and sh >= CREDIT_RULE)
        es, eb = EDU_CREDIT.get(band, (np.nan, np.nan))
        L.append(f"     {band}: baseline on the balance-sheet sample "
                 f"{base['coef']:+.4f} ({base['se']:.4f}); the step with "
                 f"the leverage split {step['coef']:+.4f} "
                 f"({step['se']:.4f}), {sh:.0%} of it; education route "
                 f"{es:.0%} of {eb:+.4f}")
        if lev:
            L.append(f"       the exposed-and-levered term "
                     f"{lev['coef']:+.4f} ({lev['se']:.4f}) t "
                     f"{tstat(lev['coef'], lev['se']):+.2f}")
        if full:
            L.append(f"       the full-panel baseline at this band is "
                     f"{full['coef']:+.4f} ({full['se']:.4f}), which is "
                     f"lane 28's quantity and is here as a check that the "
                     f"two lanes fit the same panel")
    if len(keeps) < len(BANDS):
        verdict = "NO VERDICT"
    elif all(keeps.values()):
        verdict = "THE STEP IS NOT A CREDIT EFFECT"
    else:
        verdict = "THE STEP DOES NOT SURVIVE THE CREDIT TEST AT BOTH BANDS"
    return verdict, [f"  8. THE CREDIT TEST: {verdict}"] + L + [
        "     The sample is the employers with a 2019 balance sheet, which "
        "in this delivery",
        "     means limited companies: the public sector and the "
        "unincorporated are not in",
        "     it, and the paper must say so rather than leave it implicit."]


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main():
    global FIRST_STAGE
    mc.Tee(OUT / "83_log.txt")
    t0 = time.time()
    print("=" * 70)
    print(f"83: THE REST OF THE PAPER ON THE OCCUPATION ROUTE   "
          f"parts {PARTS}")
    print("=" * 70)
    print("\n".join(READ_RULES))
    print(mc.mem_line("  "))

    (s82, s61, s66, s67, s71, s73, s74, s75, s78, s80, l47, l70, j47,
     h47) = load_modules()

    # THE SCORE. Lane 28's, built in lane 28's own function, and not
    # rebuilt here: one definition of the treatment variable, in one
    # place. The cascade pull behind it is cached the moment it is built.
    built = s82.build_exposure(l47, l70, j47)
    occ = built["exposure"]
    inc, nfloor = built["inc"], built["nfloor"]
    drain(s82, "82")
    print(f"  the score: {len(occ):,} employers on the {built['arm']} arm "
          f"at a floor of {built['floor']} {built['basis']}")

    counts = load_counts("L_counts", s61.PANEL_YEARS)
    if counts is None:
        raise RuntimeError("L_counts_* missing: run 47L first.")
    last = str(counts["year_month"].max())
    if last < POST_FROM:
        raise RuntimeError(f"the counts end at {last} and the adoption "
                           f"window opens at {POST_FROM}; refusing to run.")
    print(f"  counts: {len(counts):,} employer-age-months, ending {last}")

    edu = None
    if "A" in PARTS:
        edu = opt("education route", l70.edu_exposure, j47, l70.DESIGN,
                  l70.ARM)
        print(f"  education route: "
              f"{0 if edu is None else len(edu):,} employers scored")
    sexcounts = load_counts("L_counts_sex", s61.PANEL_YEARS,
                            require=["employer_id", "year_month",
                                     "age_group", "gender", "n_emp"]) \
        if "B" in PARTS else None
    key = None
    if "B" in PARTS or "C" in PARTS:
        key = opt("industry key", s80.industry_key, s73)
        drain(s80, "80")
        if key is None or key.empty:
            FAILURES.append("industry key")
            print("  the industry key could not be built; the clustering "
                  "and the industry test cannot run")

    fs_rows, ov, waves, drows = [], [], [], []
    win, dri, clu, csumm = [], [], [], {}
    ind, cre, siz, rel, rsumm = [], [], [], [], {}
    if "A" in PARTS:
        r = opt("Part A", part_a, counts, occ, edu, s66, s71, s73, l47, j47)
        if r:
            fs_rows, ov, waves, drows = r
    del edu
    gc.collect()
    if "B" in PARTS and key is not None and not key.empty:
        r = opt("Part B", part_b, counts, sexcounts, occ, key, s61, s67,
                s73, s75, s78, s80, j47)
        if r:
            win, dri, clu, csumm = r
    elif "B" in PARTS:
        r = opt("Part B without the key", part_b_window, counts, occ, s61,
                s75, s78, j47)
        win = r or []
        r = opt("Part B drift", part_b_drift, counts, occ, s61, s78, j47)
        dri = r or []
    if "C" in PARTS and key is not None and not key.empty:
        r = opt("Part C", part_c, counts, occ, key, s61, s73, s78, s80, j47)
        if r:
            ind, cre = r
    elif "C" in PARTS:
        cre = opt("Part C credit", part_c_credit, counts, occ, s61, s73,
                  j47) or []
    if "D" in PARTS:
        r = opt("Part D", part_d, counts, occ, inc, nfloor, s61, s78, s82,
                j47)
        if r:
            siz, rel, rsumm = r
    del counts, sexcounts, occ, inc, nfloor
    gc.collect()

    # ---- summary ------------------------------------------------------
    verdicts = []
    v_fs, fs_lines = (verdict_first_stage(fs_rows) if "A" in PARTS
                      else ("NOT RUN", ["  1. THE FIRST STAGE: NOT RUN in "
                                        "this part. Nothing in lane 29 or "
                                        "lane 28",
                                        "     is quoted until Part A has "
                                        "run and reported it."]))
    FIRST_STAGE = v_fs
    L = ["THE REST OF THE PAPER ON THE OCCUPATION ROUTE", "=" * 52, ""]
    # The first thing a reader sees, whichever way it fell.
    if v_fs == "THE FIRST STAGE REPRODUCES":
        L += ["THE FIRST STAGE REPRODUCES. The occupation-route quartile "
              "predicts reported AI",
              "use with the same sign as the education route and at least "
              f"{HALF:.0%} of its size, so",
              "the estimates below and those of lane 28 may be read.", ""]
    elif v_fs == "THE FIRST STAGE DOES NOT REPRODUCE":
        L += ["*** THE FIRST STAGE DOES NOT REPRODUCE. ***",
              "*** NOTHING IN THIS LANE OR IN LANE 28 IS QUOTED. The "
              "occupation-route quartile",
              "*** does not predict who reports using AI, and a score that "
              "does not do that is",
              "*** not an AI exposure measure, whatever its coefficients "
              "do. Everything below is",
              "*** printed so the failure can be read, and for no other "
              "purpose.", ""]
    else:
        L += [f"*** {v_fs}. Read rule 1 before anything else below, and "
              f"before anything",
              "*** from lane 28. Until it is settled, nothing from either "
              "lane is quoted.", ""]
    L += ["THE SCORE. Every fit here uses lane 28's firm score, built by "
          "that script's own",
          "build_exposure(): the employment-weighted mean DAIOE percentile "
          "of the 2019",
          "three-digit occupations of the employer's own incumbents aged 31 "
          "to 69, the",
          "backward cascade, a floor of five. It is not rebuilt here, so "
          "there is one",
          "definition of the treatment variable and not two.", ""]
    if "A" in PARTS:
        L += ["A. THE FIRST STAGE AND THE DESCRIPTIVE COUNTERPART:"]
        L += fs_lines + [""]
        L += verdict_waves(waves) + [""]
        L += verdict_descriptive(drows) + [""]
        verdicts.append(("1 the first stage", v_fs))
        if ov:
            L.append("  the overlap each arm had to work with:")
            for r in ov:
                L.append(f"    {str(r['source']):<20} {str(r['route']):<11} "
                         f"matched {cnt(r['matched']):>9}  with outcome "
                         f"{cnt(r['with_outcome']):>9}  top quartile "
                         f"{cnt(r['high_with_outcome']):>8}")
            L.append("")
    if "B" in PARTS:
        L += ["B. THE REMAINING ROWS OF TABLE 1:"]
        v, lines = verdict_window(win)
        verdicts.append(("4 the reference window", v))
        L += lines + [""]
        v, lines = verdict_drift(dri)
        verdicts.append(("5 the pre-launch drift", v))
        L += lines + [""]
        v, lines = verdict_cluster(clu, csumm)
        verdicts.append(("6 the industry clustering", v))
        L += lines + [""]
    if "C" in PARTS:
        L += ["C. THE TWO ROBUSTNESS TESTS:"]
        v, lines = verdict_industry(ind)
        verdicts.append(("7 the industry test", v))
        L += lines + [""]
        v, lines = verdict_credit(cre)
        verdicts.append(("8 the credit test", v))
        L += lines + [""]
    if "D" in PARTS:
        L += ["D. THE FIRM-SIZE ROBUSTNESS:"]
        v, lines = verdict_size(siz, rsumm)
        verdicts.append(("9 the firm-size robustness", v))
        L += lines + [""]
    if verdicts:
        L += ["THE VERDICTS:"]
        L += [f"  {k:<28} {v}" for k, v in verdicts]
        if FIRST_STAGE != "THE FIRST STAGE REPRODUCES":
            L += ["  and none of the others is quoted until the first "
                  "stage is settled."]
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
        "  1. The estimates will differ from the education route's. Two",
        "     measures of the same object standardised on their own",
        "     distributions rank employers differently, and no rule here",
        "     asks them to agree in SIZE.",
        "  2. This route scores fewer employers, so every panel here is a",
        "     SUBSET of the education route's and every standard error is",
        "     larger for that reason alone before anything else is said.",
        "  3. 71's sample gates and 73's coverage gate travel with their",
        "     code. A BELOW THRESHOLD outcome is reported as NO VERDICT",
        "     and is neither a pass nor a failure.",
        "  4. The only coefficient gate in this lane is the clustering's,",
        f"     and it is a reproduction check to {MATCH_DP} decimals, not a",
        "     judgement about the estimate.",
        "  5. A higher floor scores fewer employers and a different set",
        "     of them, so Part D's floor arm has larger standard errors",
        "     than the reported one for that reason before anything else",
        "     is said. What the rule reads is the point estimate against",
        "     the REPORTED standard error, not against its own.",
        "  6. The reliability of a firm score falls with the number of",
        "     incumbents behind it by construction, so a low figure at",
        "     one incumbent is arithmetic and not a finding. What the",
        "     table is for is the share of the sample sitting there.",
        "", f"Runtime {(time.time()-t0)/60:.1f} min. " + mc.mem_line()]
    (OUT / "83_summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n" + "\n".join(L))
    mc.runlog("83_occupation_route_rest", 0, (time.time() - t0) / 60)
    print("\n83 done.")


if __name__ == "__main__":
    main()
