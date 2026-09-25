#!/usr/bin/env python3
"""
l54_rti_ssyk4.py: the Autor and Dorn (2013) routine-task intensity (RTI)
on SSYK 2012 four-digit occupations, for script 94's horse race of
generative-AI exposure against routine-task intensity.

QUESTION
Is the young workers' decline specific to what generative AI can do, or
would any ranking that loads on routine work reproduce it? The standard
measure of routine work in the task literature is Autor and Dorn's RTI,
RTI = ln(routine) - ln(manual) - ln(abstract) task input, from the 1977
Dictionary of Occupational Titles. This script puts it on the same SSYK
2012 codes as DAIOE, by the route every other US occupation score in the
paper takes, and reports how far it is from DAIOE before any register run
is prepared: if the two were nearly collinear across occupations
(|r| > 0.8), a horse race could not separate them and script 94 would
not be worth a MONA slot.

SOURCES (all read from data/raw/; none is fetched at run time)
  1. RTI by occ1990dd: `RTIa` in dta/occ1990dd_data2012.dta inside
     Autor-Dorn-LowSkillServices-FileArchive.zip, the authors' own file
     archive for the paper. https://www.ddorn.net/data/Autor-Dorn-LowSkillServices-FileArchive.zip
     `RTIa` is the occupation-level RTI the paper uses; it is NOT a
     function of the three task means in file [B1] alone (a naive
     ln R - ln M - ln A on those means fits it with R2 0.96 and cannot be
     formed at all for the 20 occupations with a zero task mean), so the
     authors' own variable is taken and the naive one is reported only
     as a check.
  2. The three task means, file [B1] occ1990dd_task_alm.zip,
     https://www.ddorn.net/data/occ1990dd_task_alm.zip (for the check).
  3. Census 2010 occupation code -> occ1990dd, file [A8]
     occ2010_occ1990dd.zip, https://www.ddorn.net/data/occ2010_occ1990dd.zip
     (Autor 2015, JEP; the crosswalk Dorn's page publishes for 2010 codes).
  4. Census 2010 occupation code -> SOC 2010, the Census Bureau's 2010
     Occupation Code List (sheet 2010OccCodeList),
     https://www2.census.gov/programs-surveys/demo/guidance/industry-occupation/2010-occ-codes-with-crosswalk-from-2002-2011.xls
     saved as data/raw/census_2010_occ_codes_with_crosswalk.xls.
  5. SOC 2010 -> ISCO-08 (BLS) and ISCO-08 -> SSYK 2012 (SCB): the two
     files, and the two functions of src/09, that built
     dingel_neiman_ssyk4.dta.
  Files 1 to 3 downloaded 25 September 2026 from ddorn.net; file 4 the
  same day from census.gov. SHA-256 pinned below.

CITATION (verified against Crossref, DOI 10.1257/aer.103.5.1553)
  Autor, D. H., and Dorn, D. (2013). The growth of low-skill service jobs
  and the polarization of the US labor market. American Economic Review,
  103(5), 1553-1597.
  Crosswalk [A8]: Autor, D. H. (2015). Why are there still so many jobs?
  Journal of Economic Perspectives, 29(3), 3-30.

REDISTRIBUTION. Dorn's page asks users to cite the source and states no
licence. The raw files and the derived SSYK file are therefore kept OUT of
the public repository (.gitignore) until the authors' permission or a
licence is confirmed; this script, which rebuilds both from the public
URLs, is committed.

THE ROUTE, STEP BY STEP (unweighted means at every many-to-one step, as
for Dingel and Neiman)
  occ1990dd --[A8], inverted--> Census 2010 code: every Census code takes
      the RTI of the occ1990dd it maps to.
  Census 2010 --[Census list]--> SOC 2010 detailed: a detailed SOC code
      (last digit 1-9) is taken as is; a broad or minor group (trailing
      zeros) or an X wildcard is expanded to every detailed code in the
      BLS SOC list under it, but only to codes no Census code names
      exactly. A SOC code reached by several Census codes takes their mean.
  SOC 2010 --[BLS]--> ISCO-08 --[SCB, inverted]--> SSYK 2012: src/09's
      build_ssyk_telework, imported, with the score renamed in and out.
  GATE: the same src/09 path fed Dingel and Neiman must reproduce
      revision/upload/dingel_neiman_ssyk4.dta to 1e-9, so the route is the
      one the paper already uses and not a new one.

OUTPUTS
  revision/upload/rti_ssyk4.dta   ssyk4 (int32), rti (float64): the
      file script 94 reads from the share's input folder. Same two-column
      shape and dtypes as dingel_neiman_ssyk4.dta. LOCAL ONLY (see above).
  revision/tables/l54_rti_coverage.csv       coverage at each step
  revision/tables/l54_rti_correlations.csv   RTI against DAIOE
  Neither table carries an occupation's score.

    python3 revision/local/l54_rti_ssyk4.py
"""

import hashlib
import importlib.util
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REV = Path(__file__).resolve().parents[1]
ROOT = REV.parent
RAW = ROOT / "data" / "raw"
AD = RAW / "autor_dorn"
T = REV / "tables"

ARCHIVE = AD / "Autor-Dorn-LowSkillServices-FileArchive.zip"
ARCHIVE_MEMBER = ("Autor-Dorn-LowSkillServices-FileArchive.zip Folder/dta/"
                  "occ1990dd_data2012.dta")
TASKS = AD / "occ1990dd_task_alm.zip"
OCC2010 = AD / "occ2010_occ1990dd.zip"
CENSUS = RAW / "census_2010_occ_codes_with_crosswalk.xls"
PINNED = {
    ARCHIVE: "a4c9b8200a39d357c1141cfe3cb8b54b7ed586dcf9768e3b1c6689d79c45be0b",
    TASKS: "c713c17655865655b5d8cf499b775af421b1ef63c6c4c5720110ccf9dd84c24f",
    OCC2010: "454cf8d712e2266ff656a50b106194241aedff98c31045c14d33c89f4751084b",
    CENSUS: "ebf7e6f31c0cda16c7c8518a23bfa13ca537105ba41c427f99dff155a68c2cc6",
}
DAIOE = REV / "upload" / "daioe_quartiles.dta"
DN_UP = REV / "upload" / "dingel_neiman_ssyk4.dta"
OUT_DTA = REV / "upload" / "rti_ssyk4.dta"
STOP_R = 0.80           # the stop rule set before the run: |r| above this
                        # and the horse race is not worth a MONA slot


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Two modules called `config` exist (revision/ and src/). l40 needs the
# first and src/09 the second, so each is loaded with its own on the path
# and the cached module is dropped in between.
sys.path.insert(0, str(REV))
l40 = _load("l40", REV / "local" / "l40_tab_occ_mix_by_sex.py")
sys.modules.pop("config", None)
sys.path.remove(str(REV))
# The paper's crosswalk code, imported rather than copied (as l52 does).
sys.path.insert(0, str(ROOT / "src"))
x09 = _load("x09", ROOT / "src" / "09_remote_work_robustness.py")
x09.BLS_CROSSWALK = RAW / "isco_soc_crosswalk2.xls"
x09.SCB_CROSSWALK = RAW / "ssyk2012_isco08.xlsx"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check_pins() -> None:
    for p, want in PINNED.items():
        if not p.exists():
            raise SystemExit(f"missing {p}; download it from the URL in the "
                             f"docstring")
        got = sha(p)
        if got != want:
            raise SystemExit(f"{p.name}: sha256 {got[:12]} is not the pinned "
                             f"{want[:12]}; the source has changed")
    print(f"  {len(PINNED)} source files match their pinned SHA-256")


def read_zip_dta(zpath: Path, member: str | None = None) -> pd.DataFrame:
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if n.endswith(".dta")
                 and "__MACOSX" not in n]
        name = member or names[0]
        return pd.read_stata(io.BytesIO(z.read(name)))


# ---------------------------------------------------------------------------
# 1. RTI by occ1990dd
# ---------------------------------------------------------------------------
def rti_occ1990dd() -> pd.DataFrame:
    d = read_zip_dta(ARCHIVE, ARCHIVE_MEMBER)
    d = d[["occ1990dd", "RTIa"]].rename(columns={"RTIa": "rti"})
    d["occ1990dd"] = d["occ1990dd"].astype(int)
    assert d["rti"].notna().all() and d["occ1990dd"].is_unique
    # The check: the authors' variable against the naive formula on the
    # published task means, where the naive one can be formed.
    t = read_zip_dta(TASKS)
    t["occ1990dd"] = t["occ1990dd"].astype(int)
    m = d.merge(t, on="occ1990dd", how="inner")
    ok = (m["task_manual"] > 0) & (m["task_abstract"] > 0)
    naive = (np.log(m.loc[ok, "task_routine"]) - np.log(m.loc[ok, "task_manual"])
             - np.log(m.loc[ok, "task_abstract"]))
    r = float(np.corrcoef(naive, m.loc[ok, "rti"])[0, 1])
    print(f"  RTI: {len(d)} occ1990dd occupations; the authors' RTIa against "
          f"ln R - ln M - ln A on [B1]'s means: r = {r:.3f} on {int(ok.sum())} "
          f"occupations ({int((~ok).sum())} have a zero task mean)")
    return d, r


# ---------------------------------------------------------------------------
# 2. occ1990dd -> Census 2010 -> SOC 2010 detailed
# ---------------------------------------------------------------------------
def census_rti(rti: pd.DataFrame) -> pd.DataFrame:
    c = read_zip_dta(OCC2010)
    c["census"] = c["occ"].astype(str).str.strip().str.zfill(4)
    c["occ1990dd"] = pd.to_numeric(c["occ1990dd"], errors="coerce")
    out = c.merge(rti, on="occ1990dd", how="left")
    return out[["census", "occ1990dd", "rti"]]


def census_soc() -> pd.DataFrame:
    c = pd.read_excel(CENSUS, "2010OccCodeList", header=None, dtype=str)
    c = c.iloc[:, [1, 2, 3]]
    c.columns = ["title", "census", "soc"]
    c = c.dropna(subset=["census", "soc"])
    c["census"] = c["census"].str.strip()
    c["soc"] = c["soc"].str.strip()
    c = c[c["census"].str.fullmatch(r"\d{4}")
          & c["soc"].str.fullmatch(r"\d\d-[\dX]{4}")]
    return c[["census", "soc", "title"]]


def soc_pattern(code: str) -> str:
    """A Census SOC entry as a regex over detailed SOC 2010 codes. X is a
    wildcard digit; a broad or minor group is written with trailing zeros
    (detailed SOC codes never end in zero), and those zeros are wildcards."""
    head, tail = code[:3], code[3:]
    if re.fullmatch(r"\d{3}[1-9]", tail):
        return re.escape(code)
    tail = re.sub(r"0+$", lambda m: "X" * len(m.group(0)), tail)
    return re.escape(head) + tail.replace("X", r"\d")


def soc_rti(cr: pd.DataFrame, cs: pd.DataFrame, soc_universe: set) -> tuple:
    j = cs.merge(cr, on="census", how="left")
    exact = j[j["soc"].str.fullmatch(r"\d\d-\d{3}[1-9]")]
    named = set(exact["soc"])
    rows = [(r.soc, r.census, r.rti) for r in exact.itertuples()]
    wild = j[~j.index.isin(exact.index)]
    n_expanded = 0
    for r in wild.itertuples():
        pat = re.compile(soc_pattern(r.soc))
        hits = [s for s in soc_universe if pat.fullmatch(s) and s not in named]
        n_expanded += len(hits)
        rows += [(s, r.census, r.rti) for s in hits]
    s = pd.DataFrame(rows, columns=["soc2010", "census", "rti"])
    s = s[s["rti"].notna()]
    soc = s.groupby("soc2010", as_index=False)["rti"].mean()
    info = {"census_codes": int(len(cs)),
            "census_codes_with_rti": int(j["rti"].notna().sum()),
            "census_codes_exact_soc": int(len(exact)),
            "census_codes_group_soc": int(len(wild)),
            "soc_detailed_from_groups": int(n_expanded),
            "soc2010_scored": int(len(soc)),
            "soc2010_in_bls_list": int(len(soc_universe)),
            "soc2010_bls_scored": int(len(set(soc["soc2010"]) & soc_universe))}
    return soc, info


# ---------------------------------------------------------------------------
# 3. SOC 2010 -> ISCO-08 -> SSYK 2012, and the gate
# ---------------------------------------------------------------------------
def soc_to_ssyk(soc2010_scores: pd.DataFrame, col: str) -> pd.DataFrame:
    s = soc2010_scores.rename(columns={col: "teleworkable"})
    out = x09.build_ssyk_telework(s, x09.load_soc_to_isco(),
                                  x09.load_isco_to_ssyk())
    return out.rename(columns={"teleworkable": col})


def crosswalk_gate() -> None:
    """The same path fed Dingel and Neiman reproduces the upload file."""
    dn = pd.read_csv(RAW / "dingel_neiman_telework.csv")
    dn["soc2010"] = dn["onetsoccode"].str[:7]
    soc = dn.groupby("soc2010")["teleworkable"].mean().reset_index()
    mine = soc_to_ssyk(soc, "teleworkable")
    mine["ssyk4"] = mine["ssyk4"].astype(int)
    up = pd.read_stata(DN_UP)
    m = up.merge(mine, on="ssyk4", how="outer", suffixes=("_up", "_mine"),
                 indicator=True)
    if not ((m["_merge"] == "both").all() and len(m) == len(up)):
        raise SystemExit("GATE FAILED: the DN code sets differ")
    diff = float((m["teleworkable_up"] - m["teleworkable_mine"]).abs().max())
    if diff > 1e-9:
        raise SystemExit(f"GATE FAILED: DN scores differ by {diff:.2e}")
    print(f"  GATE crosswalk: the src/09 path reproduces dingel_neiman_ssyk4.dta "
          f"({len(up)} codes, max abs diff {diff:.1e})")


# ---------------------------------------------------------------------------
# 4. coverage and correlation with DAIOE
# ---------------------------------------------------------------------------
def wcorr(x, y, w):
    x, y, w = (np.asarray(v, dtype=float) for v in (x, y, w))
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    c = np.average((x - mx) * (y - my), weights=w)
    return float(c / np.sqrt(np.average((x - mx) ** 2, weights=w)
                             * np.average((y - my) ** 2, weights=w)))


def ranks(v):
    return pd.Series(v).rank().to_numpy()


def main() -> int:
    print("l54: Autor-Dorn RTI on SSYK 2012\n")
    check_pins()
    crosswalk_gate()
    rti, r_naive = rti_occ1990dd()
    cr = census_rti(rti)
    cs = census_soc()
    universe = set(x09.load_soc_to_isco()["soc2010"])
    soc, info = soc_rti(cr, cs, universe)
    print(f"  Census 2010: {info['census_codes']} codes, "
          f"{info['census_codes_with_rti']} carry an RTI through [A8]; "
          f"{info['census_codes_exact_soc']} name a detailed SOC code, "
          f"{info['census_codes_group_soc']} a group, expanded to "
          f"{info['soc_detailed_from_groups']} detailed codes")
    print(f"  SOC 2010: {info['soc2010_bls_scored']} of the "
          f"{info['soc2010_in_bls_list']} codes in the BLS list scored")

    ss = soc_to_ssyk(soc, "rti")
    ss["ssyk4"] = ss["ssyk4"].astype(str).str.zfill(4)

    d = pd.read_stata(DAIOE)[["ssyk4", "pctl_rank_genai"]]
    d["ssyk4"] = d["ssyk4"].astype(str).str.zfill(4)
    e = l40.load()
    emp = e.groupby("ssyk4", as_index=False)["n"].sum()
    m = d.merge(ss, on="ssyk4", how="left").merge(emp, on="ssyk4", how="left")
    m["n"] = m["n"].fillna(0.0)
    n_d, n_r = len(d), int(m["rti"].notna().sum())
    extra = sorted(set(ss["ssyk4"]) - set(d["ssyk4"]))
    emp_tot = float(m["n"].sum())
    emp_r = float(m.loc[m["rti"].notna(), "n"].sum())
    # the three-digit coverage is the one that binds on MONA: the uniform3
    # book scores an incumbent when ANY four-digit code in his group has a
    # score, so a three-digit group with no scored code is where
    # incumbents are lost
    m["ssyk3"] = m["ssyk4"].str[:3]
    g3 = m.groupby("ssyk3").agg(has=("rti", lambda s: s.notna().any()),
                                n=("n", "sum"))
    emp3 = float(g3.loc[g3["has"], "n"].sum())
    missing = m[m["rti"].isna()].sort_values("n", ascending=False)
    print(f"\n  COVERAGE: {n_r} of the {n_d} DAIOE occupations carry an RTI "
          f"({n_r / n_d:.1%}), holding {emp_r / emp_tot:.1%} of 2024 "
          f"employment aged 16-64; {int(g3['has'].sum())} of {len(g3)} "
          f"three-digit groups ({emp3 / emp_tot:.1%} of employment) hold at "
          f"least one scored code")
    if len(missing):
        print("  the largest DAIOE occupations without an RTI: "
              + ", ".join(f"{r.ssyk4} ({r.n:,.0f})"
                          for r in missing.head(8).itertuples()))
    if extra:
        print(f"  {len(extra)} SSYK codes carry an RTI but no DAIOE score "
              f"(not used): {', '.join(extra[:10])}")

    c = m[m["rti"].notna()]
    r_u = float(np.corrcoef(c["rti"], c["pctl_rank_genai"])[0, 1])
    rho_u = float(np.corrcoef(ranks(c["rti"]), ranks(c["pctl_rank_genai"]))[0, 1])
    cw = c[c["n"] > 0]
    r_w = wcorr(cw["rti"], cw["pctl_rank_genai"], cw["n"])
    rho_w = wcorr(ranks(cw["rti"]), ranks(cw["pctl_rank_genai"]), cw["n"])
    # off the diagonal at the occupation level, employment weighted: the
    # share of employment that is high on one score and low on the other
    # at the employment-weighted medians
    def wmed(v, w):
        o = np.argsort(v.to_numpy())
        cum = np.cumsum(w.to_numpy()[o]) / w.sum()
        return float(v.to_numpy()[o][np.searchsorted(cum, 0.5)])
    ma, mr = wmed(cw["pctl_rank_genai"], cw["n"]), wmed(cw["rti"], cw["n"])
    hi_a, hi_r = cw["pctl_rank_genai"] > ma, cw["rti"] > mr
    off = float(cw.loc[hi_a != hi_r, "n"].sum() / cw["n"].sum())
    print(f"\n  RTI AGAINST DAIOE genAI, across the {len(c)} occupations both "
          f"score:")
    print(f"    Pearson  unweighted {r_u:+.3f}   employment-weighted {r_w:+.3f}")
    print(f"    Spearman unweighted {rho_u:+.3f}   employment-weighted {rho_w:+.3f}")
    print(f"    {off:.1%} of employment is off the diagonal of the two "
          f"weighted-median cuts")
    verdict = ("STOP: |r| above the rule, separation doubtful"
               if max(abs(r_u), abs(r_w)) > STOP_R else
               f"PROCEED: |r| at most {max(abs(r_u), abs(r_w)):.2f}, below "
               f"the stop rule of {STOP_R}")
    print(f"    {verdict}")

    cov = pd.DataFrame(
        [{"item": k, "value": v} for k, v in info.items()]
        + [{"item": "occ1990dd_with_rti", "value": len(rti)},
           {"item": "r_RTIa_vs_naive_formula", "value": r_naive},
           {"item": "ssyk4_daioe", "value": n_d},
           {"item": "ssyk4_daioe_with_rti", "value": n_r},
           {"item": "employment_share_2024_with_rti", "value": emp_r / emp_tot},
           {"item": "ssyk3_groups", "value": len(g3)},
           {"item": "ssyk3_groups_with_rti", "value": int(g3["has"].sum())},
           {"item": "employment_share_2024_in_ssyk3_with_rti",
            "value": emp3 / emp_tot},
           {"item": "ssyk4_with_rti_not_in_daioe", "value": len(extra)}])
    cov.to_csv(T / "l54_rti_coverage.csv", index=False)
    pd.DataFrame([
        {"stat": "pearson", "weight": "none", "value": r_u, "n_occ": len(c)},
        {"stat": "pearson", "weight": "employment_2024", "value": r_w,
         "n_occ": len(cw)},
        {"stat": "spearman", "weight": "none", "value": rho_u, "n_occ": len(c)},
        {"stat": "spearman", "weight": "employment_2024", "value": rho_w,
         "n_occ": len(cw)},
        {"stat": "offdiag_employment_share", "weight": "employment_2024",
         "value": off, "n_occ": len(cw)},
    ]).to_csv(T / "l54_rti_correlations.csv", index=False)

    # The upload file, in dingel_neiman_ssyk4.dta's shape: the DAIOE key
    # set only, ssyk4 as int32 and the score as float64.
    up = c[["ssyk4", "rti"]].copy()
    up["ssyk4"] = up["ssyk4"].astype(int).astype("int32")
    up["rti"] = up["rti"].astype("float64")
    up = up.sort_values("ssyk4").reset_index(drop=True)
    up.to_stata(OUT_DTA, write_index=False, version=118)
    back = pd.read_stata(OUT_DTA)
    dn = pd.read_stata(DN_UP)
    assert list(back.columns) == ["ssyk4", "rti"]
    assert str(back["ssyk4"].dtype) == str(dn["ssyk4"].dtype) == "int32"
    assert str(back["rti"].dtype) == str(dn["teleworkable"].dtype) == "float64"
    assert back["ssyk4"].is_unique and back["rti"].notna().all()
    assert set(back["ssyk4"]) <= set(pd.read_stata(DAIOE)["ssyk4"])
    size = OUT_DTA.stat().st_size
    assert size < 9.5e6
    print(f"\n  wrote {OUT_DTA.relative_to(ROOT)}: {len(back)} codes, "
          f"{size:,} bytes, sha256 {sha(OUT_DTA)}")
    print(f"  wrote {T.relative_to(ROOT)}/l54_rti_coverage.csv and "
          f"l54_rti_correlations.csv")
    return 1 if verdict.startswith("STOP") else 0


if __name__ == "__main__":
    raise SystemExit(main())
