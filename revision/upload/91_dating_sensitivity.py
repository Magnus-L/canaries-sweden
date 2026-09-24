#!/usr/bin/env python3
"""
91_dating_sensitivity.py -- does the headline depend on where the
                            adoption window is judged to begin?

======================================================================
  RUNS IN MONA. It reuses script 82's score and script 61's skeleton
  and rebuilds neither, so there is one definition of the treatment
  variable and one definition of the panel. Nothing is pulled that 82
  has not already cached.
======================================================================

THE QUESTION
The paper dates the adoption window to January 2024 on two Swedish
surveys and not on the employment panel: Statistics Sweden records firms
using at least one AI technique rising from 10 per cent for 2023 to 25
for 2024, and a panel of professional employees shows a majority using
AI at work by May 2024. Both are annual or single-wave readings, so they
date a year and not a month.

Script 84's quarterly path then showed that the decline is GRADUAL and is
already distinguishable from the tightening level in 2023Q4, before the
boundary. Its own read rule, written before that run, recorded the
consequence in as many words: "THE DATING DOES NOT REPRODUCE ... quarters
before 2024 that are already negative and significant: 2023Q4".

So the boundary does not sit on a break, and a referee will ask what
happens if it moves. This script answers that, and only that.

WHAT IT ESTIMATES
Equation (2) exactly as the paper reports it, with ONE constant changed:
the month from which the adoption indicator switches on. The interim
indicator always runs from the launch (December 2022) to the month
before the boundary, and the adoption indicator from the boundary to the
end of the panel, so the two always partition the post-launch period and
no month is ever dropped.

  boundaries  2023-07, 2023-10, 2024-01 (reported), 2024-04, 2024-07
  bands       22-25 and 26-30
  ten fits, Poisson PML, employer-by-month, employer-by-age and
  month-by-age effects, three calendar-quarter terms, clustered by
  employer.

The terms are built by script 78's own eq2_terms, called with its
POST_FROM set per boundary, rather than by a term list copied into this
file. A copied list is how two scripts drift apart, and the whole value
of this exercise is that nothing but the boundary differs.

THE GATE, FIXED BEFORE THE RUN
The 2024-01 arm must reproduce the reported estimates to four decimals:

    22-25   post -0.0578 (0.0155)   step from 2023 -0.0399 (0.0102)
    26-30   post -0.0482 (0.0104)   step from 2023 -0.0403 (0.0067)

If it does not, this script has changed something it should not have,
and NOTHING from the run is quoted. The gate is checked before any other
arm is read, and a failure stops the script rather than warning.

THE READ RULE, AND WHAT STABILITY WOULD AND WOULD NOT MEAN
Revised 23 September 2026 after a cross-vendor review and after the
fixture below demonstrated the point on synthetic data.

STABILITY HERE IS CLOSE TO MECHANICAL, AND THE RULE MUST NOT PRETEND
OTHERWISE. If the underlying adjusted path is linear in t over the 31
post-launch months, the difference between the two period means is
(31/2)*slope whatever the boundary: the comparison period and the
adoption window slide together and the cutoff cancels. The synthetic
GRADUAL world in test_91 reproduces exactly that, a spread of 0.013
across the five boundaries on a step of -0.156. So an invariant sweep is
evidence that THE NUMBER WE PRINT does not hinge on the partition, and it
is NOT evidence about when the decline began or what caused it.

The rule is therefore descriptive and the report is a range:

  Report the five estimates with their intervals and state the range.
  Say in terms: "the contrast varies from X to Y across these
  partitions; this checks sensitivity to the reporting boundary, not the
  timing or the cause of the decline."

The earlier draft of this rule accepted "within one standard error" as
equivalence. One standard error is not an equivalence margin, overlapping
intervals do not establish equivalence, and a formal test would need the
cross-fit covariance, which this run does not produce. Nothing here is
reported as a test.

For every arm the export carries the dates, both durations, g0, g2,
g2-g0 AND g1+g2, the late period against the reference. The components
matter: two moving averages can shift a long way together while their
difference barely moves, and printing only the difference would hide it.

THE SWEEP IS NO LONGER THE PRIMARY ANSWER. Script l45 reports contrasts
taken from the flexible quarterly path on periods of equal length and
identical calendar composition, calendar 2024 against calendar 2023 among
them, which depend on no boundary at all and are more precisely estimated
than the headline. That table answers the dating objection; this sweep
shows that the printed number does not move much when the partition does.

WHAT THIS RUN IS NOT
It is not a test of whether the effect is real. Moving a boundary inside
a period that is already declining says how the estimand responds to the
partition, not what caused the decline. The rivals are tested elsewhere:
the industry absorption and the credit channel in script 83, the two
clusterings in script 80, the sex contrast in script 82.

INPUTS AND OUTPUTS
Reads the L_counts caches script 61 writes and the occupation cascade
script 82 caches; builds the score through 82.build_exposure. Writes
occ_route_dating.csv and the clustered covariance of every fit to the
output folder, plus 91_summary.txt.

    python3 91_dating_sensitivity.py
    CANARIES_91_OUT=output_91 python3 91_dating_sensitivity.py
"""
import gc
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mona_common as mc  # noqa: E402


def _mod(fname: str, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


OUT = HERE / os.environ.get("CANARIES_91_OUT", "output_91")
OUT.mkdir(exist_ok=True)

# The boundaries. 2024-01 is the reported one and is listed in place so
# that the sweep reads as one series and not as a baseline plus variants.
BOUNDARIES = ["2023-07", "2023-10", "2024-01", "2024-04", "2024-07"]
REPORTED = "2024-01"
BANDS = ["22-25", "26-30"]
FLOOR = 5

# The gate. These are the estimates the paper prints, from lane
# round3_20260923-0655-lanes28b-29bcd (occ_route_headline.csv and
# occ_rest_window.csv). Four decimals, as stated in the docstring.
GATE = {
    "22-25": {"post": (-0.0578, 0.0155), "step": (-0.0399, 0.0102)},
    "26-30": {"post": (-0.0482, 0.0104), "step": (-0.0403, 0.0067)},
}
GATE_TOL = 5e-5

RB = "rb_x_high_x_young"
INTERIM = "interim_x_high_x_young"
POST = "post_x_high_x_young"

FAILURES: list = []


def fit(b: pd.DataFrame, tag: str, terms: list, fes: tuple,
        cluster: str = "employer_id"):
    """One Poisson fit; returns (coefficients indexed by term, clustered
    covariance as a DataFrame or None). A failure returns (None, None)
    and is recorded, because a missing row must never read as a zero."""
    print(f"    {tag}: {len(b):,} rows, {b['employer_id'].nunique():,} firms"
          f"{mc.mem_line(' | ')}")
    t = time.time()
    try:
        r = mc.run_fepois_multi(b, OUT, tag=f"s91_{tag}", terms=terms,
                                fes=fes, cluster=cluster)
    except BaseException as ex:
        print(f"    {tag} FAILED: {type(ex).__name__}: {ex}")
        traceback.print_exc()
        r = pd.DataFrame()
    if r.empty:
        FAILURES.append(tag)
        return None, None
    g = r.set_index("term")
    v = None
    if "vcov" in r.attrs and Path(r.attrs["vcov"]).exists():
        v = pd.read_csv(r.attrs["vcov"]).set_index("term")
    print(f"    {tag}: done in {(time.time() - t) / 60:.1f} min")
    return g, v


def step_from_interim(g: pd.DataFrame, v: pd.DataFrame) -> tuple:
    """post minus interim, with the standard error from the covariance of
    the two terms: Var(a-b) = Vaa + Vbb - 2Vab. Never from adding
    variances, which would ignore a covariance that is not small here."""
    c = float(g.loc[POST, "coef"]) - float(g.loc[INTERIM, "coef"])
    if v is None or POST not in v.index or INTERIM not in v.index:
        return c, np.nan
    var = (float(v.loc[POST, POST]) + float(v.loc[INTERIM, INTERIM])
           - 2.0 * float(v.loc[POST, INTERIM]))
    return c, float(np.sqrt(var)) if var > 0 else np.nan


def check_gate(rows: list) -> None:
    """The reported arm must reproduce what the paper prints, or nothing
    from this run is quotable. Stops the script; does not warn."""
    bad = []
    for band, want in GATE.items():
        r = [x for x in rows
             if x["young_band"] == band and x["boundary"] == REPORTED]
        if len(r) != 1:
            bad.append(f"{band}: the reported arm produced {len(r)} rows")
            continue
        r = r[0]
        for key, (wc, ws) in want.items():
            gc_, gs = r[f"{key}_coef"], r[f"{key}_se"]
            if abs(gc_ - wc) > GATE_TOL or abs(gs - ws) > GATE_TOL:
                bad.append(f"{band} {key}: this run gives {gc_:+.4f} "
                           f"({gs:.4f}), the paper prints {wc:+.4f} ({ws:.4f})")
    if bad:
        print("\n  THE GATE FAILED. Nothing from this run is quotable.")
        for b in bad:
            print(f"    {b}")
        raise SystemExit("91: the reported arm does not reproduce; stopping.")
    print(f"  the gate passes: the {REPORTED} arm reproduces the paper's "
          f"estimates at both bands to four decimals")


def main() -> int:
    t0 = time.time()
    print("91: the headline under five adoption boundaries\n")

    s82 = _mod("82_occupation_route.py", "s82")
    s61, s67, s74, s78, s80, l47, l70, j47 = s82.load_modules()

    counts = s82.load_counts("L_counts", s61.PANEL_YEARS)
    if counts is None:
        raise SystemExit("91: the L_counts caches are not present; run 61 "
                         "or 82 part B first.")

    built = s82.build_exposure(l47, l70, j47)
    if built["arm"] != s82.MAIN_LEVEL or built["floor"] != s82.FLOOR_MAIN:
        raise SystemExit(f"91: the score came back as arm {built['arm']} "
                         f"floor {built['floor']}, not the reported arm")
    expo = built["exposure"]
    print(f"  score: {expo['employer_id'].nunique():,} employers, "
          f"the reported arm\n")

    keep = s78.POST_FROM        # restored before returning, so that an
    rows = []                   # import of 78 elsewhere is unaffected
    try:
        for band in BANDS:
            skel = s61.build_skeleton(counts, band, j47)
            if skel.empty:
                FAILURES.append(f"{band}/skeleton empty")
                continue
            base = s78.with_exposure(skel, expo)
            del skel
            gc.collect()
            if base.empty:
                FAILURES.append(f"{band}/no exposure")
                continue
            n_firms = int(base["employer_id"].nunique())
            print(f"  {band}: {n_firms:,} employers")
            for boundary in BOUNDARIES:
                s78.POST_FROM = boundary
                b, terms = s78.eq2_terms(base.copy())
                tag = f"{band.replace('-', '_')}_{boundary.replace('-', '_')}"
                g, v = fit(b, tag, terms, j47.FES)
                del b
                gc.collect()
                if g is None:
                    continue
                sc, ss = step_from_interim(g, v)
                # The estimation sample must be the same across arms. Only
                # the indicators change, so fepois should separate out the
                # same cells every time; if it does not, the arms are not
                # comparable and the sweep is measuring two things at once.
                n_obs = int(g.loc[POST, "n_obs"]) if "n_obs" in g.columns \
                    else -1
                seen = [r["n_obs"] for r in rows
                        if r["young_band"] == band and r["n_obs"] > 0]
                if n_obs > 0 and seen and n_obs != seen[0]:
                    FAILURES.append(
                        f"{band}/{boundary}: the estimation sample is "
                        f"{n_obs:,} cells against {seen[0]:,} at the first "
                        f"boundary, so the arms are not the same sample")
                rows.append({
                    "young_band": band, "boundary": boundary,
                    "is_reported": boundary == REPORTED,
                    "rb_coef": float(g.loc[RB, "coef"]),
                    "rb_se": float(g.loc[RB, "se"]),
                    "interim_coef": float(g.loc[INTERIM, "coef"]),
                    "interim_se": float(g.loc[INTERIM, "se"]),
                    "post_coef": float(g.loc[POST, "coef"]),
                    "post_se": float(g.loc[POST, "se"]),
                    "step_coef": sc, "step_se": ss,
                    "step_t": sc / ss if ss and ss == ss and ss > 0 else np.nan,
                    # the level against the reference period, reported
                    # beside the step and not read as the sweep's answer
                    "level_vs_reference": float(g.loc[RB, "coef"])
                                          + float(g.loc[POST, "coef"]),
                    "n_firms": n_firms,
                    "n_obs": int(g.loc[POST, "n_obs"])
                        if "n_obs" in g.columns else np.nan,
                    "status": "ok",
                })
            del base
            gc.collect()
    finally:
        s78.POST_FROM = keep

    if not rows:
        raise SystemExit("91: no fit produced a row; nothing to gate.")

    check_gate(rows)

    df = pd.DataFrame(rows)
    df = mc.enforce_min_cell(df, count_col="n_firms", floor=FLOOR)
    df.to_csv(OUT / "occ_route_dating.csv", index=False)
    print(f"\n  wrote {OUT / 'occ_route_dating.csv'}")

    # The summary, in the shape the read rule is written on.
    lines = ["THE HEADLINE UNDER FIVE ADOPTION BOUNDARIES",
             "=" * 52, "",
             "The step is post minus interim: the change from the period",
             "immediately before the boundary to the period after it. Its",
             "standard error comes from the covariance of the two terms.", ""]
    for band in BANDS:
        d = df[df.young_band == band]
        if d.empty:
            continue
        lines.append(f"  {band}:")
        lines.append("    boundary   tightening        interim          "
                     "adoption         STEP FROM BEFORE")
        for _, r in d.iterrows():
            mark = "  <- reported" if r.is_reported else ""
            lines.append(
                f"    {r.boundary}   {r.rb_coef:+.4f} ({r.rb_se:.4f})  "
                f"{r.interim_coef:+.4f} ({r.interim_se:.4f})  "
                f"{r.post_coef:+.4f} ({r.post_se:.4f})  "
                f"{r.step_coef:+.4f} ({r.step_se:.4f}){mark}")
        lo, hi = float(d.step_coef.min()), float(d.step_coef.max())
        lines.append("")
        lines.append(f"    RANGE: the contrast runs from {lo:+.4f} to "
                     f"{hi:+.4f} across the five partitions, a spread of "
                     f"{hi - lo:.4f}.")
        lines.append("    This checks sensitivity to the reporting boundary,")
        lines.append("    NOT the timing or the cause of the decline. Under a")
        lines.append("    smooth path the contrast is invariant to the cutoff")
        lines.append("    by construction, so a small spread is not evidence")
        lines.append("    about when the decline began.")
        lines.append("")
    lines.append("THE GATE: the reported arm reproduced the paper's estimates "
                 "at both bands to four decimals.")
    if FAILURES:
        lines.append("")
        lines.append("FAILURES (a missing row is not a zero):")
        lines += [f"  {f}" for f in FAILURES]
    text = "\n".join(lines)
    (OUT / "91_summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"\n91 done in {(time.time() - t0) / 60:.1f} min.")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
