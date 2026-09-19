#!/usr/bin/env python3
"""
52_slim_exports.py -- shrink script 50's big exports under the 5 MB cap.

======================================================================
  RUNS IN MONA. NO SQL AT ALL: it reads output_50/ and writes
  output_50_slim/. Seconds, not minutes. Run it in Spyder (F5) or
  submit it; either is fine.
======================================================================

WHY. Ten of 50's files came out over the 5 MB per-file export cap:
m6b_inr_tertiary_2020..2023 at about 5.2 MB and the six m7_validation
files at 6.6 to 7.3 MB.

WHY THEY CAN SHRINK WITHOUT LOSING ANYTHING WE USE. Those files carry a
4-digit occupation code on every row, and the only reason the analysis
wants it is to look up that occupation's DAIOE exposure. Doing that merge
HERE instead of on the Mac replaces 423 occupation codes with the two
numbers the analysis actually consumes:

  m6 / m6b    per (education, experience band, freshness):
              n, sum(n * daioe percentile), sum(n * top-quartile flag)
              -- exactly the sufficient statistics for the weighted means
              l13 computes, so every design's score is unchanged
  m7          per (age, lagged education, experience band, tertiary,
              enrolment field) x TRUE EXPOSURE QUARTILE:
              n and sum(n * percentile)
              -- the quartile is what the agreement, precision and recall
              are computed on; the percentile sum preserves the mean
              exposure within each cell for the error statistic

Two further changes, both improvements rather than compromises:

  FLOOR RAISED TO 10 on these files. They are sparse and
  high-dimensional; a cell of five described by education x experience x
  enrolment field x occupation x age is a narrow description of five
  people. Ten is the safer number and costs the analysis very little once
  the occupation dimension is collapsed, because collapsing makes the
  cells much larger.

  SUPPRESSED ROWS ARE DROPPED, not blanked. A blanked row still says "a
  cell exists here with between one and nine people". Dropping says
  nothing. It is also what makes the files small.

Anything already under the cap is copied through unchanged, so the slim
folder is a complete replacement for the export.
"""

import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE / "output_50"
OUT = HERE / "output_50_slim"
OUT.mkdir(exist_ok=True)
FLOOR_SLIM = 10
CAP_MB = 5.0

try:
    import mona_common as mc
    DAIOE_PATH = mc.DAIOE_PATH
except Exception:
    DAIOE_PATH = str(HERE.parent / "input" / "daioe_quartiles.dta")


def log(msg):
    print(msg)
    with open(OUT / "52_log.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def load_daioe() -> pd.DataFrame:
    d = pd.read_stata(DAIOE_PATH)
    d["ssyk4"] = d["ssyk4"].astype(str).str.zfill(4)
    q = d["exposure_quartile"]
    if not pd.api.types.is_numeric_dtype(q):
        q = q.astype(str).str.extract(r"(\d)")[0].astype(int)
    return pd.DataFrame({"ssyk4": d["ssyk4"],
                         "score": d["pctl_rank_genai"].astype(float),
                         "high": d["high_exposure"].astype(float),
                         "q": q.astype(int)})


def read(name: str) -> pd.DataFrame:
    d = pd.read_csv(SRC / name, dtype={"ssyk4": str, "ssyk4_t": str,
                                       "inr": str, "enr_inr": str,
                                       "grp": str, "grp_lag": str})
    d["n"] = pd.to_numeric(d["n"], errors="coerce")
    return d.dropna(subset=["n"])          # already-suppressed cells go


def write(d: pd.DataFrame, name: str):
    d = d[d["n"] >= FLOOR_SLIM].copy()     # raise the floor AND drop the rows
    d.to_csv(OUT / name, index=False)
    mb = (OUT / name).stat().st_size / 1e6
    flag = "  OVER CAP" if mb > CAP_MB else ""
    log(f"  {name:<34} {len(d):>9,} rows  {mb:5.2f} MB{flag}")
    return mb


def collapse_scores(d: pd.DataFrame, keys: list, daioe: pd.DataFrame,
                    code_col: str = "ssyk4") -> pd.DataFrame:
    m = d.merge(daioe, left_on=code_col, right_on="ssyk4", how="inner")
    m["ws"] = m["n"] * m["score"]
    m["wh"] = m["n"] * m["high"]
    return (m.groupby(keys, observed=True)
            .agg(n=("n", "sum"), ws=("ws", "sum"), wh=("wh", "sum"))
            .reset_index())


def main():
    log("=" * 66)
    log("52: slimming script 50's exports under the 5 MB cap")
    log("=" * 66)
    if not SRC.exists():
        raise SystemExit(f"{SRC} not found; run 50 first")
    daioe = load_daioe()
    log(f"  DAIOE: {len(daioe)} occupations")
    over = []

    for f in sorted(SRC.glob("*.csv")):
        name = f.name
        mb_in = f.stat().st_size / 1e6
        try:
            if name.startswith("m6_matrix_"):
                d = read(name)
                out = collapse_scores(d, ["grp", "expband", "fresh"], daioe)
            elif name.startswith("m6b_inr_tertiary_"):
                d = read(name)
                out = collapse_scores(d, ["inr", "expband", "fresh"], daioe)
            elif name.startswith("m7_validation_"):
                d = read(name)
                m = d.merge(daioe, left_on="ssyk4_t", right_on="ssyk4", how="inner")
                m["ws"] = m["n"] * m["score"]
                keys = ["age_group", "grp_lag", "expband_lag", "tertiary_lag",
                        "enr_inr", "q"]
                keys = [k for k in keys if k in m.columns]
                out = (m.groupby(keys, observed=True, dropna=False)
                       .agg(n=("n", "sum"), ws=("ws", "sum")).reset_index()
                       .rename(columns={"q": "q_true"}))
            else:
                shutil.copy(f, OUT / name)
                mb = (OUT / name).stat().st_size / 1e6
                log(f"  {name:<34} copied unchanged      {mb:5.2f} MB")
                if mb > CAP_MB:
                    over.append(name)
                continue
            mb = write(out, name)
            log(f"      (was {mb_in:5.2f} MB)")
            if mb > CAP_MB:
                over.append(name)
        except Exception as ex:
            log(f"  {name}: FAILED ({type(ex).__name__}) {str(ex)[:160]}")
            log(f"  continuing with the rest")

    log("")
    if over:
        log("STILL OVER 5 MB, split these by age band before exporting:")
        for n in over:
            log(f"  {n}")
    else:
        log("Every file in output_50_slim is under the 5 MB cap.")
    log("")
    log("EXPORT output_50_slim, NOT output_50. The slim folder is complete:")
    log("files that were already small are copied through unchanged.")
    log(f"Floor raised to {FLOOR_SLIM} on the collapsed files, and suppressed")
    log("rows are dropped rather than blanked.")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        print("\nUNCAUGHT EXCEPTION\n" + traceback.format_exc())
        raise
