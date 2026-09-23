#!/usr/bin/env python3
"""
test_88_occupation_route_contrast_by_track.py -- the score must be lane
                                                 28's, the terms and the
                                                 panel must be 77's, and
                                                 a shortfall planted in
                                                 one track must appear
                                                 in that track alone.

One synthetic world: 400 employers carry a 2019 occupation mix, 300 also
carry the monthly panel by sex and education across five broad tracks.
A SHORTFALL AT 22-25 IS PLANTED IN ONE TRACK ONLY, in the exposed
employers from January 2024. A cut that works must find it in that track
and not in the others, and the all-worker base must sit between them.
That is the mechanism, planted before anything is fitted.

Also tested: that the exposure is 82's score and not 77's education
exposure; that the term set is 77's own; that the lane opens no database
connection and stops when the counts are not cached; that 77's export
name is not reused; and that every track is fitted and reported.

    CANARIES_DRYRUN=1 python3 revision/local/test_88_occupation_route_contrast_by_track.py
"""
import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["CANARIES_DRYRUN"] = "1"
os.environ["CANARIES_ECHO_LIMIT"] = "100000000"
HERE = Path(__file__).resolve().parent
MONA, UPLOAD = HERE.parent / "mona", HERE.parent / "upload"
TMP = Path(tempfile.mkdtemp(prefix="canaries88_"))
SHARE = TMP / "input"; SHARE.mkdir()
for f in ("daioe_quartiles.dta", "eloundou_ssyk4.dta",
          "utb_grupp2_sun2020_niva3_inr4_nyckel.dta"):
    shutil.copy(UPLOAD / f, SHARE / f)
os.environ["CANARIES_SHARE"] = str(SHARE)
os.environ["CANARIES_88_OUT"] = str(TMP / "out")
os.environ["CANARIES_82_OUT"] = str(TMP / "out")
sys.path.insert(0, str(MONA)); sys.path.insert(0, str(HERE))
import mona_common as mc  # noqa: E402
mc.SHARE = str(SHARE); mc.CACHE_DIR = TMP / "cache"; mc.CACHE_DIR.mkdir()
_LOCAL = str(SHARE / "daioe_quartiles.dta")
mc.DAIOE_PATH = _LOCAL
_ld = mc.load_daioe
mc.load_daioe = lambda path=_LOCAL: _ld(path)

CONNECTED = []


def _no_sql():
    CONNECTED.append(1)
    raise AssertionError("this lane performs no SQL and must not connect")


mc.connect = _no_sql

FITS = []
_real_multi = mc.run_fepois_multi


def counting_multi(panel, workdir, tag, *a, **kw):
    FITS.append((tag, tuple(kw.get("terms", ()))))
    return _real_multi(panel, workdir, tag, *a, **kw)


mc.run_fepois_multi = counting_multi


def load(n, a):
    sp = importlib.util.spec_from_file_location(a, MONA / n)
    m = importlib.util.module_from_spec(sp); sys.modules[a] = m
    sp.loader.exec_module(m); return m


s88 = load("88_occupation_route_contrast_by_track.py", "s88")
FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name
          + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


AGES = ["22-25", "26-30", "31-34", "35-40", "41-49", "50+"]
N_SCORE, N_PANEL = 400, 300
SCORED, PANEL = list(range(1, N_SCORE + 1)), list(range(1, N_PANEL + 1))
UNSCORED = "9999"
D = pd.read_stata(_LOCAL)
D["ssyk4"] = D["ssyk4"].astype(str).str.zfill(4)
_d = D.sort_values("pctl_rank_genai").reset_index(drop=True)
TIER_CODE, seen = [], set()
for q in (0.10, 0.40, 0.65, 0.92):
    for i in range(int(q * len(_d)), len(_d)):
        c = _d.loc[i, "ssyk4"]
        if c[:3] not in seen:
            TIER_CODE.append(c); seen.add(c[:3]); break
TIER_OF = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 3}
tier = lambda e: TIER_OF[e % 10]
size_mult = lambda e: 1 + (e % 9)
EXPOSED = {e for e in SCORED if tier(e) == 3}

rows = []
for emp in SCORED:
    t_, m_ = tier(emp), size_mult(emp)
    c, far = TIER_CODE[t_], TIER_CODE[3 - t_]
    for age in AGES:
        rows.append((emp, age, c, c[:3], "2019", 6 * m_))
        rows.append((emp, age, far, far[:3], "2019", 1 * m_))
        rows.append((emp, age, UNSCORED, UNSCORED[:3], "2019", 2))
CASC = pd.DataFrame(rows, columns=["employer_id", "age_group", "ssyk4",
                                   "ssyk3", "source_year", "n"])
CASC["ssyk_ar"] = "2019"; CASC["ssyk_status"] = "1"
CASC.to_parquet(mc.CACHE_DIR / "L_baseline_2019_cascade.parquet", index=False)
(CASC.groupby(["employer_id", "age_group", "ssyk4"], observed=True)["n"].sum()
 .reset_index()).to_parquet(mc.CACHE_DIR / "L_baseline_2019.parquet",
                            index=False)
pd.DataFrame([(e, f"2019-{m:02d}", a, size_mult(e))
              for e in SCORED for m in range(1, 13) for a in AGES],
             columns=["employer_id", "year_month", "age_group", "n_emp"]
             ).to_parquet(mc.CACHE_DIR / "L_counts_2019.parquet", index=False)

(s82, s61, s76, s77, l47, l70, j47) = s88.load_modules()
for m_ in (s82, s61, s76, s77, l47, l70, j47):
    m_.OUT, m_.CACHE = s88.OUT, mc.CACHE_DIR
s88.OUT.mkdir(parents=True, exist_ok=True)
h47 = j47._h47()

print("\n--- the lane refuses to run without the counts ---")
try:
    s88.main()
    check("a missing counts cache stops the lane", False, "it ran anyway")
except RuntimeError as ex:
    check("a missing counts cache stops the lane",
          "L_counts_sex_edu" in str(ex), str(ex)[:60])
check("no database connection was opened", not CONNECTED)

FIELD = {"ict": "48", "engineering": "52", "business_law_social": "34",
         "health_education_care": "72", "other": "99"}
TRACKS = list(FIELD)
HIT_TRACK = "ict"
MONTHS = [f"{y}-{m:02d}" for y in s76.YEARS
          for m in range(1, 13 if y < 2025 else 7)]
FALL = float(np.log(0.80))


def counts_sex_edu() -> pd.DataFrame:
    """A shortfall at 22-25 planted in ONE track of the exposed firms."""
    rng = np.random.default_rng(88)
    out = []
    for emp in PANEL:
        hit = emp in EXPOSED
        for ym in MONTHS:
            for age in ("22-25", "26-30", "41-49"):
                for sex in ("1", "2"):
                    for tr in TRACKS:
                        x = 10.0
                        if hit and tr == HIT_TRACK and age == "22-25" \
                                and ym >= "2024-01":
                            x *= np.exp(FALL)
                        out.append((emp, ym, age, sex, "4", FIELD[tr],
                                    int(rng.poisson(x)) + 1))
    return pd.DataFrame(out, columns=s76.EDU_COLS + ["n_emp"])


C = counts_sex_edu()
for y in s76.YEARS:
    C[C["year_month"].str.slice(0, 4) == str(y)].to_parquet(
        mc.CACHE_DIR / f"L_counts_sex_edu_{y}.parquet", index=False)

print("\n--- the score and the terms are the right ones ---")
built = s82.build_exposure(l47, l70, j47)
check("the score is 82's primary arm",
      built["arm"] == s82.MAIN_LEVEL and built["floor"] == s82.FLOOR_MAIN,
      f"{built['arm']} floor {built['floor']}")
frame = s76.tag_frame(C[C["year_month"].str.slice(0, 4) == "2024"], h47)
sk = s77.three_band_skeleton(s76.collapse(frame))
b = sk.merge(built["exposure"][["employer_id", "fq"]], on="employer_id",
             how="inner")
b["high"] = (b["fq"] == 4).astype(int)
_, terms = s77.contrast_terms(b)
check("the panel holds the three bands and no others",
      set(sk["age_group"].astype(str)) == set(s77.BANDS),
      str(sorted(set(sk["age_group"].astype(str)))))
check("the terms are 77's own, per young band",
      all(any(band.replace("-", "_") in t for t in terms)
          for band in s77.YOUNG_BANDS), f"{len(terms)} terms")
del frame, sk, b

print("\n--- main() ---")
rc = s88.main()
check("main returns 0", rc == 0, str(rc))
check("still no database connection", not CONNECTED)
T = s88.OUT / "occ_route_contrast_by_track.csv"
check("the export exists under this route's name", T.exists())
check("77's export name is not reused",
      not (s88.OUT / "contrast_by_track.csv").exists())
check("six fits, all workers and each of the five tracks",
      len([t for t, _ in FITS if t.startswith("s88_")]) == 1 + len(TRACKS),
      str([t for t, _ in FITS if t.startswith("s88_")]))

if T.exists():
    d = pd.read_csv(T)
    check("every track is fitted and reported",
          set(d["track"]) == {"all"} | set(TRACKS),
          str(sorted(set(d["track"]))))
    y = d[d.band_vs_ref == "22-25"].set_index("track")
    if HIT_TRACK in y.index:
        hit = float(y.loc[HIT_TRACK, "coef"])
        others = [float(y.loc[g, "coef"]) for g in TRACKS
                  if g != HIT_TRACK and g in y.index]
        check("the planted shortfall lands in the track it was planted in",
              hit < 0 and abs(hit / float(y.loc[HIT_TRACK, "se"])) > 2,
              f"{hit:+.4f}")
        check("the other tracks do not carry it",
              all(o > hit + 0.05 for o in others),
              f"hit {hit:+.4f} against {[round(o, 4) for o in others]}")
        if "all" in y.index:
            check("the all-worker base sits between the tracks",
                  hit < float(y.loc["all", "coef"]) < max(others) + 0.05,
                  f"base {float(y.loc['all', 'coef']):+.4f}")

summ = (s88.OUT / "88_summary.txt").read_text(encoding="utf-8")
check("the summary names the base as the figure tracks are read against",
      "THE BASE, fitted here" in summ)
check("the summary says the cut stays education and why",
      "classifies no young worker by occupation" in summ)
check("the summary prints the read rules",
      "READ RULES, FIXED BEFORE THE RUN" in summ)
check("the summary claims no external reproduction",
      "not a gate" in summ)

print("\n" + "=" * 60)
print(f"{len(FAILS)} FAILED" if FAILS else "ALL PASS")
for f in FAILS:
    print("  " + f)
shutil.rmtree(TMP, ignore_errors=True)
raise SystemExit(1 if FAILS else 0)
