#!/usr/bin/env python3
"""
l50_remote_work_horserace.py: realised remote work against AI exposure on
the posting margin, from public Platsbanken advertisements only.

QUESTION
Lambert and Schindler (2026, "The Broken Ladder", SSRN 6787638) find that
realised remote work, not generative-AI exposure, accounts for the fall
in junior hiring after 2022 in four English-speaking countries: estimated
jointly, the remote-work coefficient survives and the AI coefficient goes
to about zero. Their main measure (their p. 10) is Hansen et al. (2023):
the share of an occupation's 2021-2022 postings that offer one or more
days of remote or hybrid work; a firm-level version of the same share is
their measure of actual adoption. The paper so far controls only for
Dingel and Neiman (2020) teleworkability, which says whether a job COULD
be done at home. This script builds the Swedish analogue of their main
measure from the advertisements themselves and runs their horse race on
our two posting designs.

THE REMOTE-WORK FIELD, AND WHY THE MEASURE IS A TEXT RULE
The historical JobTech schema carries a structured boolean `remote_work`
(the advertiser's "distansarbete" tick box). It is ABSENT from every
advertisement published in the 2020, 2021 and 2022 files and present from
2023, where it is ticked on well under one per cent of advertisements.
It therefore cannot measure 2021-2022 adoption. The measure is a
transparent keyword rule on the advertisement text (headline, text,
requirements and conditions: the text the Monitor's entry-level rule
reads), in six named groups (Swedish distance work, Swedish home work,
Swedish hybrid work, English remote, English work from home, English
hybrid). Phrases that name a task done at a distance, a hybrid car, a
hybrid cloud or a hybrid role combining two functions, and "utgår
hemifrån" (field staff who drive out from home), are deliberately not
matched, and a match negated in its own clause ("kan inte utföras på
distans", "no remote work", "jobbar långt hemifrån") is discarded, as is
a match whose clause describes how a service is delivered (teaching,
training, meetings, counselling, care, support) rather than where the
employee works. The rule is validated against the structured field where both
exist (2023 to June 2026) and by a hand-read sample of matches, both
written out.

DESIGNS
(1) Occupation x month panel, Equation (1) exactly as l08 estimates it:
    ln postings, occupation and month effects, PostRB x High and PostGPT x
    High, clustered by occupation, N = 28,084, January 2020 to June 2026.
    Gate: -0.1271 and -0.0593. Then (i) PostRB x RemoteHigh and PostGPT x
    RemoteHigh added, RemoteHigh = top quartile of the occupation remote
    share across the 369 panel occupations; (ii) continuous standardised
    scores; (iii) Lambert and Schindler's form, one post-launch dummy
    times each standardised score alone and then jointly; (iv) Poisson of
    (i). (i) and the AI-only baseline also on entry-level advertisements
    (the Monitor's keyword rule, l09's classifier), counted per
    occupation and month on the same deduplication.
(2) Within employer, l09's OA V design (Poisson; employer x quartile and
    employer x month effects; clustered by employer; January 2021 to June
    2026). Gate: the all-employer PostGPT x High of -0.158. Then the same
    panel cut one level finer, employer x DAIOE quartile x remote-high x
    month, with employer x cell and employer x month effects, so that
    PostGPT x RemoteHigh is identified within the employer-month; and the
    l09 specification run separately on employers above and below the
    median of their own 2021-2022 remote share.

STAGES
    python3 revision/local/l50_remote_work_horserace.py --extract
        One streaming pass over data/raw/2020..2025 and the cached
        2026-Q1/Q2 files (four worker processes), one parquet of ad-level
        flags per file in data/processed/l50_ads/ (gitignored), plus a
        random sample of matched snippets for the hand check. ~50 min.
    python3 revision/local/l50_remote_work_horserace.py
        Everything else, from the parquet files. About 15 minutes.

OUTPUTS
revision/tables/l50_remote_coverage.csv, l50_remote_validation.csv,
l50_remote_monthly.csv, l50_remote_occ_measure.csv,
l50_remote_correlations.csv, l50_remote_horserace.csv, l50_remote_within.csv;
kept out of the public repository, in data/processed/l50_ads/: the
employer measure (l50_remote_employer_measure.csv, the firm dimension is
internal) and the hand-check snippets (l50_remote_snippets.csv, raw ad
text);
canaries-sweden-paper/tables/tableA_remote_horserace.tex (not \\input
anywhere).
"""

import argparse
import importlib.util
import json
import random
import re
import sys
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REV = Path(__file__).resolve().parents[1]
_cfg_spec = importlib.util.spec_from_file_location("v2config", REV / "config.py")
_cfg = importlib.util.module_from_spec(_cfg_spec)
_cfg_spec.loader.exec_module(_cfg)

MON = REV.parents[2] / "lab-infrastructure" / "ai-monitor"
CACHE = Path.home() / ".cache" / "aiel-jobads"
EXTRACT = _cfg.PROCESSED / "l50_ads"
PAPER_TAB = REV.parents[1] / "canaries-sweden-paper" / "tables"
FILES = [str(y) for y in range(2020, 2026)] + ["2026-Q1", "2026-Q2"]

RB_YM, GPT_YM = "2022-04", "2022-12"
MEAS_LO, MEAS_HI = "2021-01", "2022-12"     # Lambert-Schindler's 2021-2022

# ---------------------------------------------------------------------------
# The text rule. Applied to the lower-cased ad_text (headline + text +
# requirements + conditions). Each group is a list of alternatives; an ad is
# remote if any group fires. Bits identify the group in the audit.
# ---------------------------------------------------------------------------
_V = r"(?:arbeta|jobba|arbetar|jobbar|arbetet|arbete|jobbet|tjänsten|utföras|utförs|sitta|sker)"
GROUPS = {
    # Swedish distance work: "distansarbete", "arbeta/jobba ... på/från
    # distans", "möjlighet till ... distans", "N dagar på distans".
    # "distans" must end the word, so distansutbildning, distansundervisning
    # and distanskurs never match.
    "sv_distans": (1, [
        r"\bdistans(?:arbet\w*|jobb\w*)\b",
        rf"\b{_V}\b[^.\n!?]{{0,40}}?\b(?:på|från) distans\b",
        r"\b(?:ske|utföra|arbetar man|jobbar man)\b[^.\n!?]{0,30}?\b(?:på|från) distans\b",
        r"\bdistans eller (?:på )?kontor\w*\b",
        r"\bmöjlighet\w*\s+(?:till|att)\b[^.\n!?]{0,30}?\bdistans\b",
        r"\bdelvis på distans\b",
        r"\b\d+\s*(?:dag|dagar)\w*\s+(?:i veckan\s+|per vecka\s+)?på distans\b",
        r"\bhelt på distans\b",
        r"\b(?:placering|placeringsort|arbetsort|ort|tjänsteställe)\s*:[^.\n]{0,40}?\bdistans\b",
        r"\b(?:eller|och|och/eller)\s+(?:på\s+|från\s+)?distans\b",
    ]),
    # Swedish home work: a work verb then "hemifrån" (so "utgår hemifrån",
    # "kör hemifrån", "vara borta hemifrån" do not match), hemarbete,
    # hemmakontor; "arbeta hemma" but not "arbeta hemma hos" (home care).
    "sv_hem": (2, [
        r"\b(?:arbeta|jobba|arbetar|jobbar|arbete|jobb|arbetet|jobbet|sitta|utföras|utförs|sker|ske)\s+(?:\w+\s+){0,3}?hemifrån\b",
        r"\bkontor\w*\b[^.\n]{0,30}?\b(?:eller|och|och/eller)\s+hemifrån\b",
        r"\bhemma[-/ ]?kontor\w*\b",
        r"\bkontor\w*\s*,\s*hemma\b",
        r"\b(?:på kontoret|kontoret)\b[^.\n]{0,15}?\b(?:andra|övriga)\s+(?:dagar\s+)?hemma\b",
        r"\bvarifrån (?:du|man) (?:vill\s+)?(?:jobbar|arbetar|jobba|arbeta)\b",
        r"\bkontor\w*\s*(?:/|och|eller)\s*hemma\b",
        r"\b(?:en|två|tre|fyra|\d)\s+dag\w*\s+(?:i veckan\s+)?hemma\b",
        r"\bhemarbete\w*\b",
        r"\bhemmakontor\w*\b",
        r"\bhemarbetsplats\w*\b",
        r"\b(?:arbeta|jobba)\s+hemma\b(?!\s+hos)",
    ]),
    # Swedish hybrid work: work-arrangement compounds only; hybridbil,
    # hybridmoln, hybridroll (a role combining two functions), hybrida
    # miljöer (IT) do not match.
    "sv_hybrid": (4, [
        r"\bhybrid(?:arbete|arbetsplats|arbetssätt|arbetsmodell|kontor)\w*\b",
        r"\bhybrid(?:t|a)?\s+(?:arbete\w*|arbetsplats\w*|arbetssätt\w*|arbetsmodell\w*|arbetsform\w*|arbetsliv\w*|kontor\w*|upplägg\w*|distans\w*|remote)\b",
        # a hybrid solution or model, only when the same clause names home,
        # distance or on-site work (so hybrid cloud solutions do not match)
        r"\bhybrid(?:lösning\w*|modell\w*|\s+lösning\w*|\s+modell\w*|\s+solution\w*|\s+model\w*)\b[^.\n]{0,60}?\b(?:hemifrån|hemma|distans|på plats|on-?site|from home|remote\w*)\b",
        r"\bflexibel\w*\s*/\s*hybrid\b",
        r"#?\bli-(?:hybrid|remote)\b",
        r"\bhybrid-arbete\w*\b",
        r"\b(?:placering|placeringsort|arbetsort|ort|location)\s*:[^.\n]{0,30}?[-/ ]hybrid\b",
        r"\b(?:stockholm|göteborg|malmö|uppsala|solna|linköping)[-/ ]hybrid\b",
        r"\btjänsten är hybrid\b",
    ]),
    # English remote, as an arrangement for the worker, not a task done
    # remotely (remote support, remote monitoring, remote sensing).
    "en_remote": (8, [
        r"\bremote[- ]?(?:first|friendly)\b",
        r"\bfully remote\b",
        r"\b(?:work|working|works)\b[^.\n]{0,20}?\bremotely\b",
        r"^remote\s*:",
        r"\b(?:work|working|jobba|arbeta)\s+remote\b",
        r"\bremote\s*(?:&|and|och|/|or|eller)\s*(?:onsite|on-site|office|kontor\w*)\b",
        r"\b(?:onsite|on-site|office|kontor\w*)\s*(?:&|and|och|/|or|eller)\s*remote\b",
        r"\bremote[- ]flexible\b",
        r"\bremote[- ]?(?:arbetsplats\w*|upplägg\w*|anställning\w*|tjänst\w*|roll\w*)\b",
        r"\b(?:placering|placeringsort|arbetsort|ort|location|place of employment)\s*:[^.\n]{0,30}?\bremote\b",
        r"\b(?:kontoret|office)\b[^.\n]{0,20}?\boch remote\b",
        r"\bdone remotely\b",
        r"\bremote\s*(?:work|working|arbete|jobs?|positions?|options?|possibilit\w*|opportunit\w*|days?|basis|setup|arrangements?|policy)\b",
        r"\b(?:hybrid|partly|partially|part[- ]time|delvis|on|på)\s+remote\b",
        r"\bremote\s*(?:/|or|och|eller)\s*hybrid\b",
        r"\b(?:arbeta|jobba)\s+(?:på\s+|från\s+)?remote\b",
        r"\bremotearbete\w*\b",
        r"\b\d+\s*%\s*remote\b",
    ]),
    "en_wfh": (16, [
        r"\bwork\w*\b[^.\n]{0,25}?\bfrom home\b",
        r"\b(?:on-?site|office)\b[^.\n]{0,25}?\bfrom home\b",
        r"\bwfh\b",
        r"\bhome[- ]office\b",
        r"\bhome[- ]based\b",
        r"\btelework\w*\b",
        r"\btelecommut\w*\b",
    ]),
    "en_hybrid": (32, [
        r"\bhybrid[- ](?:work|working|workplace|workspace|office|setup|set-up|arrangements?|schedule|way of working|positions?|jobs?)\b",
        r"\b(?:a|our|the)\s+(?:\d+\s*:\s*\d+\s+|\d+\s*/\s*\d+\s+)?hybrid (?:model|environment|solution)\b[^.\n]{0,60}?\b(?:home|office|remote\w*|on-?site)\b",
        r"\b\d+\s*:\s*\d+\s+hybrid model\b",
    ]),
}
GROUP_RE = {g: (bit, re.compile("|".join(p))) for g, (bit, p) in GROUPS.items()}


# A match is discarded when a negation sits in the same clause just before
# it or inside it ("kan inte utföras på distans", "no remote work",
# "ej möjlighet till distansarbete").
NEG = re.compile(r"\b(?:inte|ej|icke|ingen|inga|inget|not|no|non|cannot|can't|långt|away|borta)\b")
# "på distans", "remote" and "hemifrån" also describe how a SERVICE is
# delivered (teaching, training, meetings, counselling, care visits,
# support) rather than where the employee works; a match whose clause names
# such a delivery is discarded.
TASK = re.compile(r"\b(?:undervisning\w*|utbildning\w*|föreläsning\w*|kurs\w*|möte\w*|"
                  r"samtal\w*|besök\w*|vård\w*|behandling\w*|support\w*|training|"
                  r"meetings?|assessment|studera\w*|studier)\b")


def _clause_before(low: str, m, width: int) -> str:
    pre = low[max(0, m.start() - width):m.start()]
    cut = max(pre.rfind("."), pre.rfind("\n"), pre.rfind("!"), pre.rfind("?"),
              pre.rfind("•"))
    return pre[cut + 1:] if cut >= 0 else pre


def _negated(low: str, m) -> bool:
    """Negation within 30 characters before the match (same clause) or inside
    it; a service-delivery noun within 60 characters before it, inside it,
    or as the very next word ("remote training", not "hybrid work, training
    budget")."""
    post = low[m.end():m.end() + 25]
    post = post if re.match(r"\s+\w", post) else ""
    return bool(NEG.search(_clause_before(low, m, 30)) or NEG.search(m.group(0))
                or TASK.search(_clause_before(low, m, 60)) or TASK.search(m.group(0))
                or TASK.search(post))


def remote_bits(low: str) -> tuple[int, str]:
    """Bitmask of groups that fire (un-negated) on lower-cased text, and a
    snippet around the first match (for the hand check)."""
    bits, snip = 0, ""
    for g, (bit, rx) in GROUP_RE.items():
        for m in rx.finditer(low):
            if _negated(low, m):
                continue
            bits |= bit
            if not snip:
                snip = low[max(0, m.start() - 90):m.end() + 60].replace("\n", " ")
            break
    return bits, snip


def _zip_for(stem: str) -> Path:
    p = _cfg.RAW / f"{stem}.jsonl.zip"
    return p if p.exists() else CACHE / f"{stem}.jsonl.zip"


def extract_file(stem: str) -> str:
    """Stream one file; one row per parseable ad with the flags needed."""
    sys.path.insert(0, str(MON / "scripts"))
    sys.path.insert(0, str(MON / "demo"))
    import bulk_pipeline_v11 as bp                      # read-only import
    from firm_dimension_extract import norm_orgnr       # the cube's rule
    l01_spec = importlib.util.spec_from_file_location(
        "l01", REV / "local" / "l01_postings_accounting.py")
    l01 = importlib.util.module_from_spec(l01_spec)
    l01_spec.loader.exec_module(l01)

    cols = {k: [] for k in ("ok", "ad_id", "ssyk4", "ym", "dkey", "orgnr",
                            "cube_ssyk", "rf", "bits", "entry")}
    snips = []
    rng = random.Random(20260924)
    n_hit = 0
    with zipfile.ZipFile(_zip_for(stem)) as zf:
        for name in (n for n in zf.namelist() if n.endswith(".jsonl")):
            with zf.open(name) as f:
                for line in f:
                    try:
                        ad = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    reason, rec = l01.classify_ad(ad)
                    txt = bp.ad_text(ad)
                    low = txt.lower()
                    bits, snip = remote_bits(low)
                    rw = ad.get("remote_work", "<absent>")
                    rf = -1 if rw == "<absent>" else (-2 if rw is None else int(bool(rw)))
                    e = ad.get("employer") or {}
                    org = norm_orgnr(e.get("organization_number")
                                     if isinstance(e, dict) else "")
                    cols["ok"].append(reason == "ok")
                    cols["ad_id"].append(rec["ad_id"] if rec else "")
                    cols["ssyk4"].append(rec["ssyk4"] if rec else "")
                    cols["ym"].append(rec["year_month"] if rec else
                                      (ad.get("publication_date") or "")[:7])
                    cols["dkey"].append(int.from_bytes(bp.dedup_key(ad), "big", signed=True))
                    cols["orgnr"].append(org)
                    cols["cube_ssyk"].append(bp._taxonomy_id(ad.get("occupation_group")) or "0000")
                    cols["rf"].append(rf)
                    cols["bits"].append(bits)
                    cols["entry"].append(1 if bp.ENTRY.search(low) else 0)
                    if bits:
                        n_hit += 1
                        item = {"file": stem, "ym": cols["ym"][-1],
                                "ssyk4": cols["ssyk4"][-1], "bits": bits,
                                "rf": rf, "snippet": snip}
                        if len(snips) < 300:
                            snips.append(item)
                        else:
                            j = rng.randrange(n_hit)
                            if j < 300:
                                snips[j] = item
    df = pd.DataFrame(cols)
    for c in ("rf", "entry"):
        df[c] = df[c].astype("int8")
    df["bits"] = df["bits"].astype("int16")
    EXTRACT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(EXTRACT / f"{stem}.parquet", index=False)
    pd.DataFrame(snips).to_csv(EXTRACT / f"snips_{stem}.csv", index=False)
    return f"{stem}: {len(df):,} ads, {n_hit:,} remote hits"


def run_extract(force=False):
    todo = [s for s in FILES if force or not (EXTRACT / f"{s}.parquet").exists()]
    print(f"L50 extract: {len(todo)} files to stream: {todo}")
    # Largest files first so the pool finishes together.
    todo.sort(key=lambda s: -_zip_for(s).stat().st_size)
    with Pool(4) as pool:
        for msg in pool.imap_unordered(extract_file, todo):
            print("  " + msg, flush=True)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def load_occ_ads():
    """The l08 analysis sample at ad level: l01-kept ads, ad_id deduplicated
    across 2020..2025 in file order (script 02), and within the 2026 half
    year (script l08); 2020-01..2025-12 from the annual files and
    2026-01..2026-06 from the quarters."""
    # Two tracks, as the processed files were built. Track A is script 02:
    # the annual files and then the closed quarters, one ad_id set across
    # all of them; its months to December 2025 enter the panel (the quarter
    # files carry late-2025 ads, which 02 counts in 2025). Track B is l08:
    # the two 2026 quarters deduplicated on their own; its months January
    # to June 2026 enter the panel.
    def dedup(d, seen):
        has = d["ad_id"] != ""
        keep = ~((has & d["ad_id"].duplicated()) | (has & d["ad_id"].isin(seen)))
        d = d[keep]
        seen.update(d.loc[d["ad_id"] != "", "ad_id"])
        return d

    parts = []
    seen_a, seen_b = set(), set()
    for stem in FILES:
        d = pd.read_parquet(EXTRACT / f"{stem}.parquet",
                            columns=["ok", "ad_id", "ssyk4", "ym", "rf", "bits", "entry"])
        d = d[d["ok"]].drop(columns="ok")
        da = dedup(d, seen_a)
        parts.append(da[(da["ym"] >= "2020-01") & (da["ym"] <= "2025-12")])
        if stem.startswith("2026"):
            db = dedup(d, seen_b)
            parts.append(db[(db["ym"] >= "2026-01") & (db["ym"] <= "2026-06")])
    a = pd.concat(parts, ignore_index=True)
    a["remote"] = (a["bits"] > 0).astype("int8")
    return a


def load_cube_ads():
    """The Monitor cube's population at ad level: within-file content-hash
    deduplication in file order, 2021-01..2026-06, organisation number
    present. Used for the employer remote share."""
    parts = []
    for stem in FILES[1:]:
        d = pd.read_parquet(EXTRACT / f"{stem}.parquet",
                            columns=["ym", "dkey", "orgnr", "cube_ssyk", "bits", "entry"])
        d = d[~d["dkey"].duplicated()]
        d = d[(d["ym"] >= "2021-01") & (d["ym"] <= "2026-06") & (d["orgnr"] != "")]
        parts.append(d)
    c = pd.concat(parts, ignore_index=True)
    c["remote"] = (c["bits"] > 0).astype("int8")
    return c


def stars(p):
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""


def wcorr(x, y, w=None):
    x, y = np.asarray(x, float), np.asarray(y, float)
    w = np.ones_like(x) if w is None else np.asarray(w, float)
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    return cov / np.sqrt(np.average((x - mx) ** 2, weights=w)
                         * np.average((y - my) ** 2, weights=w))


def zscore(s):
    return (s - s.mean()) / s.std(ddof=0)


def main():
    import pyfixest as pf
    T = _cfg.V2_TAB
    print("L50: realised remote work vs AI exposure on the posting margin")

    # -------- 1. field coverage and the text rule ---------------------------
    a = load_occ_ads()
    a["year"] = a["ym"].str[:4]
    cov = (a.groupby("year")
            .agg(n_ads=("rf", "size"),
                 field_present=("rf", lambda s: (s >= 0).mean()),
                 field_true=("rf", lambda s: (s == 1).mean()),
                 text_remote=("remote", "mean"))
            .reset_index())
    for g, (bit, _) in GROUP_RE.items():
        cov[f"text_{g}"] = a.groupby("year")["bits"].apply(
            lambda s, b=bit: ((s & b) > 0).mean()).values
    cov.to_csv(T / "l50_remote_coverage.csv", index=False)
    print(cov.to_string(index=False))

    # Gate: the ad-level sample reproduces the l08 panel cell by cell.
    panel = pd.read_csv(_cfg.PROCESSED / "postings_daioe_merged_extended.csv",
                        dtype={"ssyk4": str})
    panel["ssyk4"] = panel["ssyk4"].str.zfill(4)
    cnt = a.groupby(["ssyk4", "ym"]).size().rename("n_rebuilt").reset_index()
    chk = panel.merge(cnt, left_on=["ssyk4", "year_month"],
                      right_on=["ssyk4", "ym"], how="left")
    exact = (chk["n_rebuilt"] == chk["n_ads"]).mean()
    print(f"  GATE rebuild: {exact:.4%} of the {len(panel):,} panel cells match "
          f"ad for ad; panel ads {panel['n_ads'].sum():,} vs rebuilt "
          f"{chk['n_rebuilt'].sum():,.0f}")
    # The processed 2020-2025 file was built on 24 Feb 2026 with the JobStream
    # snapshot then in use, which added a few late-2025 ads that no closed
    # file carries. Every other cell must match exactly.
    bad = chk[chk["n_rebuilt"] != chk["n_ads"]]
    print(f"  mismatched cells: {len(bad)}, all in "
          f"{sorted(bad['year_month'].unique())}; net {bad['n_rebuilt'].sum() - bad['n_ads'].sum():+.0f} ads")
    assert bad["year_month"].between("2025-10", "2025-12").all(), \
        "ad-level extract departs from the l08 panel outside the JobStream months"
    assert abs(chk["n_rebuilt"].sum() / panel["n_ads"].sum() - 1) < 1e-4

    # Validation where the structured field exists (2023 to June 2026).
    v = a[a["rf"] >= 0]
    val = []
    for lab, sub in (("all 2023-2026H1", v),) + tuple(
            (y, v[v["year"] == y]) for y in sorted(v["year"].unique())):
        tp = ((sub["rf"] == 1) & (sub["remote"] == 1)).sum()
        val.append({"sample": lab, "n_ads": len(sub),
                    "field_true": int((sub["rf"] == 1).sum()),
                    "text_true": int(sub["remote"].sum()),
                    "text_share_if_field_true": tp / max((sub["rf"] == 1).sum(), 1),
                    "text_share_if_field_false": sub.loc[sub["rf"] == 0, "remote"].mean(),
                    "field_share_if_text_true": tp / max(sub["remote"].sum(), 1),
                    "field_share_if_text_false": (sub.loc[sub["remote"] == 0, "rf"] == 1).mean()})
    # occupation-level agreement of the two shares, 2023-2026H1
    og = v.groupby("ssyk4").agg(n=("rf", "size"), fs=("rf", "mean"),
                                ts=("remote", "mean"))
    og = og[og["n"] >= 100]
    val.append({"sample": "occupation-level corr (>=100 ads), unweighted",
                "n_ads": int(og["n"].sum()), "field_true": len(og),
                "text_true": np.nan,
                "text_share_if_field_true": wcorr(og["fs"], og["ts"]),
                "text_share_if_field_false": np.nan,
                "field_share_if_text_true": np.nan,
                "field_share_if_text_false": np.nan})
    val.append({"sample": "occupation-level corr (>=100 ads), ad-weighted",
                "n_ads": int(og["n"].sum()), "field_true": len(og),
                "text_true": np.nan,
                "text_share_if_field_true": wcorr(og["fs"], og["ts"], og["n"]),
                "text_share_if_field_false": np.nan,
                "field_share_if_text_true": np.nan,
                "field_share_if_text_false": np.nan})
    og["rank_f"], og["rank_t"] = og["fs"].rank(), og["ts"].rank()
    val.append({"sample": "occupation-level Spearman (>=100 ads)",
                "n_ads": int(og["n"].sum()), "field_true": len(og),
                "text_true": np.nan,
                "text_share_if_field_true": wcorr(og["rank_f"], og["rank_t"]),
                "text_share_if_field_false": np.nan,
                "field_share_if_text_true": np.nan,
                "field_share_if_text_false": np.nan})
    val = pd.DataFrame(val)
    val.to_csv(T / "l50_remote_validation.csv", index=False)
    print(val.to_string(index=False))

    mon = a.groupby("ym").agg(n_ads=("remote", "size"), text_remote=("remote", "mean"),
                              field_true=("rf", lambda s: (s == 1).mean()))
    mon.reset_index().to_csv(T / "l50_remote_monthly.csv", index=False)

    snips = pd.concat([pd.read_csv(p) for p in sorted(EXTRACT.glob("snips_*.csv"))],
                      ignore_index=True)
    s2122 = snips[(snips["ym"] >= MEAS_LO) & (snips["ym"] <= MEAS_HI)]
    # Kept out of the (public) repository: snippets are raw advertisement
    # text and can carry contact names; they live beside the extract.
    s2122.sample(n=min(120, len(s2122)), random_state=1).to_csv(
        EXTRACT / "l50_remote_snippets.csv", index=False)

    # -------- 2. the measures ------------------------------------------------
    m = a[(a["ym"] >= MEAS_LO) & (a["ym"] <= MEAS_HI)]
    occ = (m.groupby("ssyk4").agg(n_ads_2122=("remote", "size"),
                                  remote_share=("remote", "mean"))
           .reset_index())
    d = pd.read_csv(_cfg.PROCESSED / "daioe_quartiles.csv", dtype={"ssyk4": str})
    d["ssyk4"] = d["ssyk4"].str.zfill(4)
    dn = pd.read_stata(REV / "upload" / "dingel_neiman_ssyk4.dta")
    dn["ssyk4"] = dn["ssyk4"].astype(str).str.zfill(4)
    occ_ids = sorted(panel["ssyk4"].unique())
    o = (pd.DataFrame({"ssyk4": occ_ids})
         .merge(d[["ssyk4", "pctl_rank_genai", "exposure_quartile"]], on="ssyk4", how="left")
         .merge(occ, on="ssyk4", how="left")
         .merge(dn[["ssyk4", "teleworkable"]], on="ssyk4", how="left"))
    o["n_ads_2122"] = o["n_ads_2122"].fillna(0).astype(int)
    o["high"] = o["exposure_quartile"].astype(str).str.startswith("Q4").astype(int)
    print(f"  occupations: {len(o)}; with 2021-22 ads: {(o['n_ads_2122'] > 0).sum()}; "
          f"with <20 ads: {(o['n_ads_2122'] < 20).sum()}")
    assert o["remote_share"].notna().all(), "panel occupation without 2021-22 ads"
    q75 = o["remote_share"].quantile(0.75)
    o["remote_high"] = (o["remote_share"] >= q75).astype(int)
    o["z_ai"] = zscore(o["pctl_rank_genai"])
    o["z_rem"] = zscore(o["remote_share"])

    # employment weights (2024 YREG54BAS, ages 16-64, both sexes) via l40
    _s = importlib.util.spec_from_file_location("l40", REV / "local" / "l40_tab_occ_mix_by_sex.py")
    l40 = importlib.util.module_from_spec(_s)
    _s.loader.exec_module(l40)
    emp = l40.load().groupby("ssyk4")["n"].sum().rename("emp_2024").reset_index()
    emp["ssyk4"] = emp["ssyk4"].astype(str).str.zfill(4)
    o = o.merge(emp, on="ssyk4", how="left")
    o.to_csv(T / "l50_remote_occ_measure.csv", index=False)

    corr = []
    def add_corr(lab, x, y, sub):
        sub = sub[sub[x].notna() & sub[y].notna()]
        e = sub[sub["emp_2024"].fillna(0) > 0]
        corr.append({"pair": lab, "n_occ": len(sub),
                     "unweighted": wcorr(sub[x], sub[y]),
                     "ad_weighted_2122": wcorr(sub[x], sub[y], sub["n_ads_2122"]),
                     "employment_weighted_2024": wcorr(e[x], e[y], e["emp_2024"]),
                     "spearman": wcorr(sub[x].rank(), sub[y].rank())})
    add_corr("remote_share x DAIOE pctl_rank_genai", "remote_share", "pctl_rank_genai", o)
    add_corr("remote_share x Dingel-Neiman teleworkable", "remote_share", "teleworkable", o)
    add_corr("DAIOE pctl_rank_genai x Dingel-Neiman teleworkable", "pctl_rank_genai", "teleworkable", o)
    # median-cut off-diagonal share
    ai_hi = o["pctl_rank_genai"] >= o["pctl_rank_genai"].median()
    rm_hi = o["remote_share"] >= o["remote_share"].median()
    off = ai_hi != rm_hi
    ew = o["emp_2024"].fillna(0)
    corr.append({"pair": "off-diagonal share, median cuts DAIOE x remote",
                 "n_occ": len(o), "unweighted": off.mean(),
                 "ad_weighted_2122": np.average(off, weights=o["n_ads_2122"]),
                 "employment_weighted_2024": np.average(off, weights=ew),
                 "spearman": np.nan})
    # quartile overlap used in the regressions
    both = (o["high"] == 1) & (o["remote_high"] == 1)
    corr.append({"pair": "share of DAIOE Q4 occupations also RemoteHigh",
                 "n_occ": int(o["high"].sum()),
                 "unweighted": both.sum() / o["high"].sum(),
                 "ad_weighted_2122": np.average(both[o["high"] == 1],
                                                weights=o.loc[o["high"] == 1, "n_ads_2122"]),
                 "employment_weighted_2024": np.nan, "spearman": np.nan})
    dist = o["remote_share"].describe(percentiles=[.1, .25, .5, .75, .9])
    for k, val_ in dist.items():
        corr.append({"pair": f"remote_share distribution: {k}", "n_occ": len(o),
                     "unweighted": val_, "ad_weighted_2122": np.nan,
                     "employment_weighted_2024": np.nan, "spearman": np.nan})
    corr.append({"pair": "remote_share ad-weighted mean 2021-22", "n_occ": len(o),
                 "unweighted": np.average(o["remote_share"], weights=o["n_ads_2122"]),
                 "ad_weighted_2122": np.nan, "employment_weighted_2024": np.nan,
                 "spearman": np.nan})
    corr = pd.DataFrame(corr)

    # -------- 3. occupation-panel horse race --------------------------------
    p = panel.merge(o[["ssyk4", "remote_high", "z_ai", "z_rem", "n_ads_2122"]],
                    on="ssyk4", how="left")
    p["high"] = p["high_exposure"]
    for nm, dummy in (("high", "high"), ("rhigh", "remote_high")):
        p[f"rb_x_{nm}"] = ((p["year_month"] >= RB_YM) & (p[dummy] == 1)).astype(int)
        p[f"gpt_x_{nm}"] = ((p["year_month"] >= GPT_YM) & (p[dummy] == 1)).astype(int)
    p["post_rb"] = (p["year_month"] >= RB_YM).astype(int)
    p["post_gpt"] = (p["year_month"] >= GPT_YM).astype(int)
    for z in ("z_ai", "z_rem"):
        p[f"rb_x_{z}"] = p["post_rb"] * p[z]
        p[f"gpt_x_{z}"] = p["post_gpt"] * p[z]
        p[f"post_x_{z}"] = p["post_gpt"] * p[z]
    p["ln_ads"] = np.log(p["n_ads"])

    rows = []
    def fit(label, formula, data, est="OLS", sample="all ads"):
        if est == "OLS":
            r = pf.feols(formula, data=data, vcov={"CRV1": "ssyk4"})
        else:
            r = pf.fepois(formula, data=data, vcov={"CRV1": "ssyk4"})
        for t in r.coef().index:
            rows.append({"design": "occupation_panel", "sample": sample,
                         "spec": label, "estimator": est, "term": t,
                         "coef": r.coef()[t], "se": r.se()[t], "pval": r.pvalue()[t],
                         "n_obs": r._N})
        print(f"  [{sample} | {label} | {est}] N={r._N:,}  " + "  ".join(
            f"{t} {r.coef()[t]:+.4f} ({r.se()[t]:.4f})" for t in r.coef().index))
        return r

    FE = " | ssyk4 + year_month"
    base = fit("baseline", "ln_ads ~ rb_x_high + gpt_x_high" + FE, p)
    assert abs(base.coef()["gpt_x_high"] + 0.0593) < 5e-4 and \
        abs(base.coef()["rb_x_high"] + 0.1271) < 5e-4 and base._N == 28084, \
        "baseline gate failed"
    print("  GATE baseline reproduced: -0.1271 / -0.0593 on 28,084")
    fit("i_quartile", "ln_ads ~ rb_x_high + gpt_x_high + rb_x_rhigh + gpt_x_rhigh" + FE, p)
    fit("i_remote_only", "ln_ads ~ rb_x_rhigh + gpt_x_rhigh" + FE, p)
    fit("ii_continuous_ai_only", "ln_ads ~ rb_x_z_ai + gpt_x_z_ai" + FE, p)
    fit("ii_continuous", "ln_ads ~ rb_x_z_ai + gpt_x_z_ai + rb_x_z_rem + gpt_x_z_rem" + FE, p)
    fit("iii_LS_ai_alone", "ln_ads ~ post_x_z_ai" + FE, p)
    fit("iii_LS_remote_alone", "ln_ads ~ post_x_z_rem" + FE, p)
    fit("iii_LS_joint", "ln_ads ~ post_x_z_ai + post_x_z_rem" + FE, p)
    fit("iv_poisson_baseline", "n_ads ~ rb_x_high + gpt_x_high" + FE, p, est="Poisson")
    fit("iv_poisson_quartile", "n_ads ~ rb_x_high + gpt_x_high + rb_x_rhigh + gpt_x_rhigh" + FE,
        p, est="Poisson")
    p50 = p[p["n_ads_2122"] >= 50]
    fit("i_quartile_occ_ge50ads", "ln_ads ~ rb_x_high + gpt_x_high + rb_x_rhigh + gpt_x_rhigh" + FE,
        p50, sample="occupations with >=50 ads in 2021-22")

    # entry-level advertisements, same deduplication, same occupations
    ent = (a[a["entry"] == 1].groupby(["ssyk4", "ym"]).size()
           .rename("n_ads").reset_index().rename(columns={"ym": "year_month"}))
    ent = ent[ent["ssyk4"].isin(occ_ids)]
    pe = ent.merge(p.drop(columns=["n_ads", "ln_ads"]).drop_duplicates(["ssyk4", "year_month"]),
                   on=["ssyk4", "year_month"], how="inner")
    pe["ln_ads"] = np.log(pe["n_ads"])
    n_entry_ads = int(pe["n_ads"].sum())
    print(f"  entry-level: {n_entry_ads:,} ads in {len(pe):,} occupation-months "
          f"({pe['ssyk4'].nunique()} occupations); "
          f"{n_entry_ads / p['n_ads'].sum():.1%} of panel ads")
    fit("baseline", "ln_ads ~ rb_x_high + gpt_x_high" + FE, pe, sample="entry-level ads")
    fit("i_quartile", "ln_ads ~ rb_x_high + gpt_x_high + rb_x_rhigh + gpt_x_rhigh" + FE, pe,
        sample="entry-level ads")
    fit("iv_poisson_quartile", "n_ads ~ rb_x_high + gpt_x_high + rb_x_rhigh + gpt_x_rhigh" + FE,
        pe, est="Poisson", sample="entry-level ads")
    fit("iii_LS_joint", "ln_ads ~ post_x_z_ai + post_x_z_rem" + FE, pe, sample="entry-level ads")

    # -------- 4. within employer --------------------------------------------
    _s9 = importlib.util.spec_from_file_location("l09", REV / "local" / "l09_firm_within_did.py")
    l09 = importlib.util.module_from_spec(_s9)
    _s9.loader.exec_module(l09)
    cube = l09.load_cube()
    dq = d[["ssyk4", "exposure_quartile"]].copy()
    dq["exposure_quartile"] = dq["exposure_quartile"].astype(str).str.extract(r"Q(\d)").astype(int)

    wrows = []
    def wfit(label, bal, terms, fe="fe_fq + fe_ft", sample="all ads", note=""):
        r = pf.fepois(f"n_ads ~ {' + '.join(terms)} | {fe}", data=bal, vcov={"CRV1": "orgnr"})
        for t in terms:
            wrows.append({"design": "within_employer", "sample": sample, "spec": label,
                          "estimator": "Poisson", "term": t, "coef": r.coef()[t],
                          "se": r.se()[t], "pval": r.pvalue()[t], "n_obs": r._N,
                          "n_firms": bal["orgnr"].nunique(), "note": note})
        print(f"  [within | {sample} | {label}] N={r._N:,} firms={bal['orgnr'].nunique():,}  "
              + "  ".join(f"{t} {r.coef()[t]:+.4f} ({r.se()[t]:.4f})" for t in terms))
        return r

    bal_a = l09.build_panel(cube, dq, "ads")
    r0 = wfit("l09_a_baseline", bal_a, ["rb_x_high", "gpt_x_high"])
    assert abs(r0.coef()["gpt_x_high"] + 0.158) < 5e-4, "within-employer gate failed"
    print("  GATE within-employer reproduced: -0.158")

    # (a) the panel one level finer: employer x quartile x remote-high x month
    def finer_panel(outcome, employers):
        cm = cube.merge(dq, on="ssyk4", how="inner").merge(
            o[["ssyk4", "remote_high"]], on="ssyk4", how="inner")
        cm = cm[cm["orgnr"].isin(employers)]
        cell = (cm.groupby(["orgnr", "exposure_quartile", "remote_high", "month"])[outcome]
                .sum().rename("n_ads").reset_index().rename(columns={"month": "year_month"}))
        pairs = cell[cell["n_ads"] > 0][["orgnr", "exposure_quartile", "remote_high"]].drop_duplicates()
        months = sorted(cube["month"].unique())
        bal = (pairs.assign(_k=1).merge(pd.DataFrame({"year_month": months, "_k": 1}), on="_k")
               .drop(columns="_k")
               .merge(cell, on=["orgnr", "exposure_quartile", "remote_high", "year_month"], how="left"))
        bal["n_ads"] = bal["n_ads"].fillna(0).astype(int)
        bal["high"] = (bal["exposure_quartile"] == 4).astype(int)
        pr, pg = (bal["year_month"] >= RB_YM).astype(int), (bal["year_month"] >= GPT_YM).astype(int)
        bal["rb_x_high"], bal["gpt_x_high"] = pr * bal["high"], pg * bal["high"]
        bal["rb_x_rhigh"], bal["gpt_x_rhigh"] = pr * bal["remote_high"], pg * bal["remote_high"]
        bal["fe_fqr"] = (bal["orgnr"] + "_" + bal["exposure_quartile"].astype(str)
                         + "_" + bal["remote_high"].astype(str))
        bal["fe_ft"] = bal["orgnr"] + "_" + bal["year_month"]
        return bal

    emp_a = sorted(bal_a["orgnr"].unique())
    fa = finer_panel("ads", emp_a)
    wfit("finer_ai_only", fa, ["rb_x_high", "gpt_x_high"], fe="fe_fqr + fe_ft",
         note="employer x quartile x remote-high cells; l09 employers")
    wfit("finer_joint", fa, ["rb_x_high", "gpt_x_high", "rb_x_rhigh", "gpt_x_rhigh"],
         fe="fe_fqr + fe_ft", note="employer x quartile x remote-high cells; l09 employers")
    bal_c = l09.build_panel(cube, dq, "entry")
    r0c = wfit("l09_c_baseline", bal_c, ["rb_x_high", "gpt_x_high"], sample="entry-level ads")
    fc = finer_panel("entry", sorted(bal_c["orgnr"].unique()))
    wfit("finer_joint", fc, ["rb_x_high", "gpt_x_high", "rb_x_rhigh", "gpt_x_rhigh"],
         fe="fe_fqr + fe_ft", sample="entry-level ads",
         note="employer x quartile x remote-high cells; l09 entry employers")

    # (b) the employer's own 2021-22 remote share, and the median split
    ca = load_cube_ads()
    ca_ok = ca[ca["cube_ssyk"].isin(set(dq["ssyk4"]))]
    cube_2122 = cube[(cube["month"] >= MEAS_LO) & (cube["month"] <= MEAS_HI)]
    ours = ca_ok[(ca_ok["ym"] >= MEAS_LO) & (ca_ok["ym"] <= MEAS_HI)]
    cube_n = cube_2122[cube_2122["ssyk4"].isin(set(dq["ssyk4"]))]["ads"].sum()
    print(f"  GATE employer basis: 2021-22 distinct ads with orgnr and DAIOE code, "
          f"cube {cube_n:,} vs rebuilt {len(ours):,} ({len(ours) / cube_n - 1:+.3%})")
    assert abs(len(ours) / cube_n - 1) < 0.005, "employer basis does not match the cube"
    er = (ours.groupby("orgnr").agg(n_ads_2122=("remote", "size"),
                                    remote_share=("remote", "mean")).reset_index())
    er_s = er[er["orgnr"].isin(emp_a)].copy()
    med = er_s["remote_share"].median()
    emp_med = med
    er_s["remote_above_median"] = (er_s["remote_share"] > med).astype(int)
    print(f"  employers in the l09 sample: {len(emp_a):,}; with 2021-22 ads: "
          f"{len(er_s):,}; median remote share {med:.4f}; "
          f"share of employers at zero {(er_s['remote_share'] == 0).mean():.1%}")
    # Employer-level file kept out of the public repository (the firm
    # dimension is internal, ML 26 Aug 2026); sole proprietors ("EF:", a
    # keyed hash of a personal identity number) are omitted even locally.
    _emp_out = er.merge(er_s[["orgnr", "remote_above_median"]], on="orgnr", how="left")
    _emp_out[~_emp_out["orgnr"].str.startswith("EF:")].to_csv(
        EXTRACT / "l50_remote_employer_measure.csv", index=False)
    edist = er_s["remote_share"].describe(percentiles=[.1, .25, .5, .75, .9])
    for k, val_ in edist.items():
        corr = pd.concat([corr, pd.DataFrame([{
            "pair": f"employer remote_share distribution (l09 employers): {k}",
            "n_occ": len(er_s), "unweighted": val_, "ad_weighted_2122": np.nan,
            "employment_weighted_2024": np.nan, "spearman": np.nan}])], ignore_index=True)
    corr.to_csv(T / "l50_remote_correlations.csv", index=False)
    print(corr.to_string(index=False))

    # With a median at zero, "above the median" is "any remote advertisement".
    hi_set = set(er_s.loc[er_s["remote_above_median"] == 1, "orgnr"])
    lo_set = set(emp_a) - hi_set          # includes employers with no 2021-22 ads
    for lab, s in (("high-remote employers", hi_set), ("low-remote employers", lo_set)):
        wfit("split", bal_a[bal_a["orgnr"].isin(s)].copy(), ["rb_x_high", "gpt_x_high"],
             sample=f"all ads, {lab}", note=f"median employer remote share {med:.4f}")
    emp_c = set(bal_c["orgnr"].unique())
    for lab, s in (("high-remote employers", hi_set & emp_c), ("low-remote employers", emp_c - hi_set)):
        wfit("split", bal_c[bal_c["orgnr"].isin(s)].copy(), ["rb_x_high", "gpt_x_high"],
             sample=f"entry-level ads, {lab}", note=f"median employer remote share {med:.4f}")
    # the difference between the two halves in one panel
    bal_a["remote_emp"] = bal_a["orgnr"].isin(hi_set).astype(int)
    bal_a["gpt_x_high_x_rememp"] = bal_a["gpt_x_high"] * bal_a["remote_emp"]
    bal_a["rb_x_high_x_rememp"] = bal_a["rb_x_high"] * bal_a["remote_emp"]
    wfit("split_interaction", bal_a, ["rb_x_high", "gpt_x_high", "rb_x_high_x_rememp",
                                      "gpt_x_high_x_rememp"], sample="all ads",
         note="difference high-remote minus low-remote employers")

    res = pd.DataFrame(rows)
    res.to_csv(T / "l50_remote_horserace.csv", index=False)
    wres = pd.DataFrame(wrows)
    wres.to_csv(T / "l50_remote_within.csv", index=False)
    q75v = o["remote_share"].quantile(0.75)
    notes = (
        "Remote share: the share of an occupation's distinct advertisements published in 2021 and 2022 "
        "that offer remote or hybrid work, from a keyword rule on the advertisement text "
        "(Section~\\ref{sec:posting_rivals}); High remote is its top quartile across the "
        f"{len(o)} panel occupations (a share of at least {q75v*100:.1f} per cent). "
        "Columns (1), (2), (4) to (6): Equation (1) with occupation and month effects, January 2020 to June 2026, "
        "standard errors clustered by occupation; column (1) is the published specification. Column (3) replaces "
        "both dummies with the standardised DAIOE percentile and remote share, reported in the High AI and High remote rows. "
        "Columns (5) and (6) count only "
        "advertisements the Monitor's keyword rule flags as entry-level. The middle panel interacts one "
        "post-launch dummy (from December 2022) with each standardised score, alone and together. The bottom panel "
        "is the within-employer design of Online Appendix V (Poisson, January 2021 to June 2026, clustered by "
        "employer); Joint cuts each employer's cells by DAIOE quartile and by High remote; High- and low-remote "
        "split employers by their own 2021 to 2022 advertisements: high-remote employers posted at least one "
        f"advertisement offering remote or hybrid work (the median employer's share is {emp_med*100:.0f}, so this is "
        "the median split). "
        "PostRB terms are estimated throughout and reported in the CSV. "
        "$^{*}$ $p<0.10$, $^{**}$ $p<0.05$, $^{***}$ $p<0.01$.")
    write_tex(res, wres, o, corr, notes)
    print("Saved l50 tables and tableA_remote_horserace.tex")


def write_tex(res, wres, o, corr, notes):
    def c(df, sample, spec, term, est=None):
        s = df[(df["sample"] == sample) & (df["spec"] == spec) & (df["term"] == term)]
        if est:
            s = s[s["estimator"] == est]
        if s.empty:
            return "", ""
        r = s.iloc[0]
        return f"${r.coef:.3f}^{{{stars(r.pval)}}}$" if stars(r.pval) else f"${r.coef:.3f}$", f"({r.se:.3f})"

    def n(df, sample, spec):
        s = df[(df["sample"] == sample) & (df["spec"] == spec)]
        return f"{int(s.iloc[0].n_obs):,}"

    cols = [("all ads", "baseline", "OLS"), ("all ads", "i_quartile", "OLS"),
            ("all ads", "ii_continuous", "OLS"), ("all ads", "iv_poisson_quartile", "Poisson"),
            ("entry-level ads", "baseline", "OLS"), ("entry-level ads", "i_quartile", "OLS")]
    terms = [("rb_x_high", r"PostRB $\times$ High AI"), ("gpt_x_high", r"PostGPT $\times$ High AI"),
             ("rb_x_rhigh", r"PostRB $\times$ High remote"), ("gpt_x_rhigh", r"PostGPT $\times$ High remote")]
    # the continuous column reports the standardised scores in the same rows
    cont = {"rb_x_high": "rb_x_z_ai", "gpt_x_high": "gpt_x_z_ai",
            "rb_x_rhigh": "rb_x_z_rem", "gpt_x_rhigh": "gpt_x_z_rem"}
    L = [r"\begin{table}[ht!]", r"\centering",
         r"\caption{Realised remote work and AI exposure on the posting margin}",
         r"\label{tab:remote_horserace}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
         r" & \multicolumn{4}{c}{All advertisements} & \multicolumn{2}{c}{Entry-level} \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}",
         r" & (1) & (2) & (3) & (4) & (5) & (6) \\",
         r" & OLS & OLS & OLS, $z$ & Poisson & OLS & OLS \\", r"\midrule"]
    for t, lab in terms:
        cs, ss = [], []
        for smp, spec, est in cols:
            tt = cont[t] if spec == "ii_continuous" else t
            a_, b_ = c(res, smp, spec, tt, est)
            cs.append(a_), ss.append(b_)
        L.append(lab + " & " + " & ".join(cs) + r" \\")
        L.append(" & " + " & ".join(ss) + r" \\")
    L.append(r"\midrule")
    L.append("Observations & " + " & ".join(n(res, s, sp) for s, sp, _ in cols) + r" \\")
    # Lambert-Schindler form
    L += [r"\midrule", r"\multicolumn{7}{l}{\textit{One post-launch period, standardised scores (Lambert and Schindler's form)}} \\",
          r" & AI alone & Remote alone & Joint & & Joint, entry & \\"]
    a1, s1 = c(res, "all ads", "iii_LS_ai_alone", "post_x_z_ai")
    b1, t1 = c(res, "all ads", "iii_LS_joint", "post_x_z_ai")
    e1, u1 = c(res, "entry-level ads", "iii_LS_joint", "post_x_z_ai")
    L.append(rf"PostGPT $\times$ AI ($z$) & {a1} & & {b1} & & {e1} & \\")
    L.append(rf" & {s1} & & {t1} & & {u1} & \\")
    a2, s2 = c(res, "all ads", "iii_LS_remote_alone", "post_x_z_rem")
    b2, t2 = c(res, "all ads", "iii_LS_joint", "post_x_z_rem")
    e2, u2 = c(res, "entry-level ads", "iii_LS_joint", "post_x_z_rem")
    L.append(rf"PostGPT $\times$ Remote ($z$) & & {a2} & {b2} & & {e2} & \\")
    L.append(rf" & & {s2} & {t2} & & {u2} & \\")
    # within employer
    L += [r"\midrule", r"\multicolumn{7}{l}{\textit{Within employer (Poisson; employer $\times$ cell and employer $\times$ month effects)}} \\",
          r" & Baseline & Joint & High-remote & Low-remote & Entry, joint & \\"]
    w = wres
    def wc(sample, spec, term):
        return c(w, sample, spec, term)
    specs = [("all ads", "l09_a_baseline"), ("all ads", "finer_joint"),
             ("all ads, high-remote employers", "split"), ("all ads, low-remote employers", "split"),
             ("entry-level ads", "finer_joint")]
    for t, lab in (("gpt_x_high", r"PostGPT $\times$ High AI"), ("gpt_x_rhigh", r"PostGPT $\times$ High remote")):
        cs, ss = zip(*[wc(s, sp, t) for s, sp in specs])
        L.append(lab + " & " + " & ".join(cs) + r" & \\")
        L.append(" & " + " & ".join(ss) + r" & \\")
    def wn(s, sp):
        r = w[(w["sample"] == s) & (w["spec"] == sp)].iloc[0]
        return f"{int(r.n_firms):,}"
    L.append("Employers & " + " & ".join(wn(s, sp) for s, sp in specs) + r" & \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    L += [r"\begin{minipage}{0.97\textwidth}\footnotesize\vspace{4pt}", notes, r"\end{minipage}",
          r"\end{table}"]
    PAPER_TAB.mkdir(parents=True, exist_ok=True)
    (PAPER_TAB / "tableA_remote_horserace.tex").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-stream every file")
    args = ap.parse_args()
    if args.extract:
        run_extract(force=args.force)
    else:
        main()
