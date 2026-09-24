#!/usr/bin/env python3
"""
93_uncounted_occupation.py -- the payslips the panel never counts, cut on
                              the occupation route's quartiles.

======================================================================
  RUNS IN MONA. It fits nothing and, if the caches are present, issues
  no SQL. It reuses script 79's cached declaration counts and script
  82's score, and rebuilds neither.
======================================================================

THE QUESTION
Online Appendix Table tab:uncounted reports the share of employer-
declaration person-months whose worker reaches none of the 2023, 2021
and 2019 individual registers, for all employers and for the top
exposure quartile against the lower three. Script 79 (lane 26, part C)
produced it before the design moved onto occupations, so its quartiles
are the 2019 EDUCATION-mix quartiles. It was the last quantity in the
appendix resting on the education route. This script re-cuts the same
counts on the 2019 OCCUPATION-mix quartiles the paper treats on.

WHAT CHANGES AND WHAT DOES NOT
Only the map from employer to quartile. The counts are 79's own caches,
U_uncounted_{year}_noage.parquet (the declaration carries no age, so 79
cached under the "noage" tag), and the table and the export floor are
79's own functions, imported rather than copied. The all-employer column
does not depend on the quartiles at all, so it must come back exactly as
79 printed it; that is the gate.

If a cache is missing, 79's own puller is called and caches the year, so
the script still completes, at the cost of an SQL slot.

THE GATE
  1. The all-employer no-register share must reproduce 79's printed
     figure for every year to three decimals (half a unit of the last
     digit). A miss means a different population, and nothing is quoted.
  2. Q1..Q4 plus unscored must sum to the all-employer total, and Q1..Q3
     to the Q1-Q3 row, exactly, before the floor.

READ RULE, FIXED BEFORE THE RUN
The table replaces the education-route one whatever it shows. If the
top-quartile gap MOVES between 2020-22 and 2024-25 by more than a tenth
of a percentage point, the bound in the appendix no longer holds at
"a thirtieth" and the paragraph is rewritten around the new movement;
it is stated plainly either way.

OUTPUT (output_93/)
  uncounted_share.csv   same columns as 79's, so l34 reads it unchanged
  93_summary.txt        per-year shares, the gap, its movement, the bound

    CANARIES_93_OUT=output_93 python3 93_uncounted_occupation.py
"""

import os
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# 79 creates its own output folder at import; point it at ours so the
# lane leaves no stray output_79 behind.
os.environ.setdefault("CANARIES_79_OUT", os.environ.get("CANARIES_93_OUT",
                                                        "output_93"))
os.environ.setdefault("CANARIES_82_OUT", os.environ.get("CANARIES_93_OUT",
                                                        "output_93"))
import mona_common as mc  # noqa: E402


def _mod(fname: str, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


OUT = HERE / os.environ.get("CANARIES_93_OUT", "output_93")
OUT.mkdir(exist_ok=True)
CACHE = mc.CACHE_DIR
TAG = "noage"                       # 79's cache tag: no age in the declaration

# The gate: 79's all-employer no-register share, per cent, as printed in
# round3_20260922-lane26c-uncounted/79_summary.txt and in the appendix.
GATE_ALL = {2019: 0.612, 2020: 0.205, 2021: 0.187, 2022: 0.188,
            2023: 0.204, 2024: 0.195, 2025: 0.210}
GATE_TOL = 0.0005                   # half a unit of the third decimal

PRE = [2020, 2021, 2022]            # 2019's gap takes the opposite sign
NEW = [2024, 2025]
# Workers aged 22-25 as a share of top-quartile employment, occupation
# route (Table tab:descriptive_bands, Panel A). Used only for the bound.
YOUNG_SHARE_Q4 = 0.046
MOVE_ALARM = 0.10                   # percentage points, the read rule

FAILURES: list = []


def load_frames(s79) -> dict:
    """79's cached counts; 79's own puller for any year that is missing."""
    frames, missing = {}, []
    for y in GATE_ALL:
        c = mc.read_cache(CACHE / f"U_uncounted_{y}_{TAG}.parquet",
                          require=["employer_id", "status", "decl_band",
                                   "n_personmonths"])
        if c is None:
            missing.append(y)
        else:
            frames[y] = c
    if missing:
        print(f"  caches missing for {missing}: pulling with 79's own query")
        conn = mc.connect()
        try:
            age_col, age_kind, note = s79.probe_declaration_age(conn, missing)
            if age_col is not None:
                raise SystemExit("93: the declaration now carries an age "
                                 "column; 79's cache tag no longer applies")
            frames.update(s79.uncounted_counts(missing, age_col, age_kind,
                                               conn))
        finally:
            try:
                conn.close()
            except Exception:
                pass
    return frames


def check(tab: pd.DataFrame) -> None:
    """Both gates, on the unfloored table."""
    t = tab[tab["decl_band"] == "all"].set_index(["year", "quartile_group"])
    for y, want in GATE_ALL.items():
        got = 100 * t.loc[(y, "all"), "share_no_register"]
        ok = abs(got - want) <= GATE_TOL
        print(f"  gate 1 {y}: all-employer {got:.4f}% against 79's "
              f"{want:.3f}%  {'ok' if ok else 'FAIL'}")
        if not ok:
            FAILURES.append(f"gate 1 {y}: {got:.4f} vs {want:.3f}")
        parts = [g for g in ("Q1", "Q2", "Q3", "Q4", "unscored")
                 if (y, g) in t.index]
        s = int(t.loc[[(y, g) for g in parts], "n_total"].sum())
        if s != int(t.loc[(y, "all"), "n_total"]):
            FAILURES.append(f"gate 2 {y}: quartiles sum to {s:,}, all is "
                            f"{int(t.loc[(y, 'all'), 'n_total']):,}")
        low = int(t.loc[[(y, g) for g in ("Q1", "Q2", "Q3")
                         if (y, g) in t.index], "n_total"].sum())
        if (y, "Q1-Q3") in t.index and low != int(t.loc[(y, "Q1-Q3"),
                                                        "n_total"]):
            FAILURES.append(f"gate 2 {y}: Q1-Q3 does not recompose")


def summarise(tab: pd.DataFrame, notes: list) -> list:
    t = (tab[tab["decl_band"] == "all"]
         .pivot(index="year", columns="quartile_group",
                values="share_no_register") * 100)
    L = ["93: THE PAYSLIPS THE PANEL NEVER COUNTS, ON THE OCCUPATION ROUTE",
         "",
         "Script 79's counts (lane 26, part C), re-cut on script 82's 2019",
         "occupation-mix quartiles, the reported arm. Nothing is fitted.",
         "",
         "  per cent of person-months whose worker is in none of the three",
         "  individual registers:",
         "    year     all      Q4   Q1-Q3  unscored   gap Q4-(Q1-Q3)"]
    for y, r in t.iterrows():
        L.append(f"    {y}  {r.get('all'):6.3f}  {r.get('Q4'):6.3f}  "
                 f"{r.get('Q1-Q3'):6.3f}  {r.get('unscored', float('nan')):8.3f}"
                 f"   {r.get('Q4') - r.get('Q1-Q3'):+.3f}")
    gap = t["Q4"] - t["Q1-Q3"]
    g_pre, g_new = gap.loc[PRE].mean(), gap.loc[NEW].mean()
    move = g_new - g_pre
    L += ["",
          f"  mean gap 2020-2022 {g_pre:+.4f} pp; 2024-2025 {g_new:+.4f} pp;",
          f"  movement {move:+.4f} pp.",
          f"  bound: if every additional payslip were a 22-25 worker "
          f"({YOUNG_SHARE_Q4:.1%} of top-quartile employment), the implied "
          f"fall in measured young employment is "
          f"{abs(move) / YOUNG_SHARE_Q4:.2f} per cent.",
          f"  the education-route run gave gaps of +0.065 and +0.071, "
          f"movement +0.006, bound 0.13 per cent.",
          ""]
    if abs(move) > MOVE_ALARM:
        L.append(f"  STATED PLAINLY: the gap moves by more than "
                 f"{MOVE_ALARM} pp; the appendix bound must be rewritten.")
    else:
        L.append(f"  the gap moves by less than {MOVE_ALARM} pp; the "
                 f"appendix bound stands with the new figures.")
    L += ["", "NOTES:"] + [f"  {n}" for n in notes]
    L += ["", "GATE: " + ("PASS" if not FAILURES else "FAIL")]
    L += [f"  {f}" for f in FAILURES]
    return L


def main() -> int:
    t0 = time.time()
    print("93: the uncounted payslips on the occupation route\n")
    s79 = _mod("79_last_gaps.py", "s79")
    s73 = _mod("73_industry_and_credit.py", "s73")
    s82 = _mod("82_occupation_route.py", "s82")
    s61, s67, s74, s78, s80, l47, l70, j47 = s82.load_modules()

    built = s82.build_exposure(l47, l70, j47, audit=False)
    if built["arm"] != s82.MAIN_LEVEL or built["floor"] != s82.FLOOR_MAIN:
        raise SystemExit(f"93: the score came back as arm {built['arm']} "
                         f"floor {built['floor']}, not the reported arm")
    expo = built["exposure"][["employer_id", "fq"]]
    print(f"  score: {expo['employer_id'].nunique():,} employers\n")

    frames = load_frames(s79)
    tab = s79.uncounted_table(frames, expo, s73)
    del frames
    check(tab)
    out = s79.floor_table(tab)
    out.to_csv(OUT / "uncounted_share.csv", index=False)

    L = summarise(tab, s79.NOTES + s82.NOTES)
    (OUT / "93_summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n93 done in {(time.time() - t0) / 60:.1f} min.")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
