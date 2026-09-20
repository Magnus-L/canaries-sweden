#!/usr/bin/env python3
"""
assemble.py -- ONE command that turns MONA exports into the evidence.

    python3 revision/assemble.py <export dir> [more dirs...] [--sim] [--file]

What it does, in order:
  1 finds every known output file in the directories given AND in
    revision/output/**, newest wins, and says which run each number
    came from
  2 applies the pre-committed read rules -- written before any of these
    runs, and quoted in the report -- to every design
  3 runs the local predictive validation (l13) if script 50's export is
    present
  4 optionally runs the simulation study (--sim, about three hours)
  5 writes revision/EVIDENCE.md and .pdf: one section per claim, each
    saying what can now be claimed, on what number, and under which rule

It NEVER deletes anything and never writes to the export directories.
Filing and deleting stay with `fdr`, so a bug here cannot lose an export.
With --file it copies (never moves) what it found into
revision/output/round2_<stamp>-assembled/ and verifies by hash.

Every claim is reported with its verdict, including "not supported" and
"pending". A missing input is reported as pending, never silently skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
OUTPUT = HERE / "output"
LOCAL = HERE / "local"

HEADLINE = -0.174                       # the submitted paper's 22-25 gamma2
OCC_ART = {2021: -0.3068, 2022: -0.1627}    # script 45
EDU_ART = {2021: -0.3596, 2022: -0.2941}    # script 47b, worker-level
CLEAN, HALF = 0.05, {2021: 0.1534, 2022: 0.0814}

RULE = ("Pre-committed before the runs (47b's docstring, repeated in 47h, 47i "
        "and 47j): an artefact below 0.05 in absolute value at BOTH truncations "
        "means the design can carry register evidence; between that and half the "
        "occupation artefact (0.153 at T=2021, 0.081 at T=2022) it is usable only "
        "with the artefact stated beside every estimate; at or above half, the "
        "route is closed by lag.")


def verdict(a2021, a2022) -> str:
    a = [abs(x) if x is not None and not pd.isna(x) else np.nan for x in (a2021, a2022)]
    if any(np.isnan(a)):
        return "PENDING"
    if all(x < CLEAN for x in a):
        return "CLEAN"
    if a[0] < HALF[2021] and a[1] < HALF[2022]:
        return "USABLE WITH CAVEAT"
    return "CLOSED"


# ---------------------------------------------------------------- discovery
WANTED = {
    "45":   "asof_estimates.csv",
    "47b":  "edu_asof_estimates.csv",
    "47h":  "horserace_estimates.csv",
    "47i":  "firmmix_estimates.csv",
    "47j":  "triple_estimates.csv",
    "48":   "gender_poisson.csv",
    "41":   "vintage_es.csv",
    "44":   "decile_pooled.csv",
    "46":   "wfh_horserace.csv",
    "50":   "m7_validation_t2023_k2.csv",
    # the register-immune family, added 20 Sep once it existed
    "47k":  "settled_estimates.csv",
    "47L":  "agebase_estimates.csv",
    "47Lg": "agebase_gradient.csv",
    "53":   "fresh_pooled.csv",
    "54":   "flow_estimates.csv",
    "54g":  "flow_gradient.csv",
    "56":   "dynamics_young.csv",
    "57":   "reliability.csv",
    "57v":  "vintage_estimates.csv",
    "58":   "readrule.csv",
    "59":   "path_quarter.csv",
    "60":   "prespecified.csv",
    "60p":  "profile.csv",
}


def find(dirs: list) -> dict:
    """Newest file wins; returns {key: (path, mtime)}."""
    found = {}
    roots = [Path(d) for d in dirs] + [OUTPUT]
    for key, name in WANTED.items():
        best = None
        for root in roots:
            if not root.exists():
                continue
            for p in [root / name] + list(root.rglob(name)):
                if p.is_file() and (best is None or p.stat().st_mtime > best.stat().st_mtime):
                    best = p
        if best is not None:
            found[key] = best
    return found


def file_copies(dirs: list) -> "Path | None":
    stamp = time.strftime("%Y%m%d-%H%M")
    dest = OUTPUT / f"round2_{stamp}-assembled"
    n = 0
    for d in dirs:
        d = Path(d)
        if not d.exists():
            continue
        dest.mkdir(parents=True, exist_ok=True)
        for p in sorted(d.iterdir()):
            if not p.is_file():
                continue
            t = dest / p.name
            shutil.copy2(p, t)
            a = hashlib.sha256(p.read_bytes()).hexdigest()
            b = hashlib.sha256(t.read_bytes()).hexdigest()
            if a != b:
                raise RuntimeError(f"copy verification failed for {p}")
            n += 1
    if n:
        print(f"  filed {n} files -> {dest} (originals untouched)")
        return dest
    return None


# ------------------------------------------------------------------ readers
def arts_from_arms(df, design_col, design, age_col="age_group", age="22-25",
                   arm_col="arm", coef="gamma2", tcol="trunc"):
    """artefact = as-of minus true, per truncation."""
    out = {}
    for T in (2021, 2022):
        s = df[(df[design_col] == design) & (df[age_col] == age) & (df[tcol] == T)]
        if len(s) < 2:
            out[T] = np.nan
            continue
        try:
            tr = float(s[s[arm_col] == "true"][coef].iloc[0])
            af = float(s[s[arm_col] == "asof"][coef].iloc[0])
            out[T] = af - tr
        except (IndexError, ValueError):
            out[T] = np.nan
    return out


def read_47h(p: Path):
    df = pd.read_csv(p)
    rows = []
    for d in df["design"].dropna().unique():
        a = arts_from_arms(df, "design", d)
        s = df[(df.design == d) & (df.age_group == "22-25") & (df.arm == "true")]
        true22 = float(s["gamma2"].iloc[0]) if len(s) else np.nan
        a50 = arts_from_arms(df, "design", d, age="50+")
        rows.append(dict(design=d, a2021=a[2021], a2022=a[2022], true22=true22,
                         placebo=max(abs(a50[2021]), abs(a50[2022]))
                         if not any(pd.isna(list(a50.values()))) else np.nan,
                         verdict=verdict(a[2021], a[2022])))
    return pd.DataFrame(rows).sort_values(
        "a2021", key=lambda s: s.abs(), na_position="last")


def read_47i(p: Path):
    df = pd.read_csv(p)
    rows = []
    for d in df["design"].dropna().unique():
        for age in df["age_group"].dropna().unique():
            a = arts_from_arms(df, "design", d, age=age)
            s = df[(df.design == d) & (df.age_group == age) & (df.arm == "true")]
            rows.append(dict(design=d, age_group=age,
                             true=float(s["gamma2"].iloc[0]) if len(s) else np.nan,
                             a2021=a[2021], a2022=a[2022],
                             verdict=verdict(a[2021], a[2022])))
    return pd.DataFrame(rows)


def read_47j(p: Path):
    df = pd.read_csv(p)
    rows = []
    for d in df["design"].dropna().unique():
        for yb in df["young_band"].dropna().unique():
            a = arts_from_arms(df[df.young_band == yb], "design", d,
                               age_col="young_band", age=yb, coef="gamma3")
            s = df[(df.design == d) & (df.young_band == yb) & (df.arm == "true")]
            rows.append(dict(design=d, young_band=yb,
                             true=float(s["gamma3"].iloc[0]) if len(s) else np.nan,
                             a2021=a[2021], a2022=a[2022],
                             verdict=verdict(a[2021], a[2022])))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- report
def sec(title, body):
    return f"\n## {title}\n\n{body}\n"


def build_report(found: dict, sim_done: bool, val_txt: str) -> str:
    L = [f"# Canaries: what the evidence supports\n",
         f"*Assembled {time.strftime('%Y-%m-%d %H:%M')} by `revision/assemble.py`. "
         f"Submitted headline, ages 22-25: {HEADLINE:+.3f}.*\n",
         f"\n> {RULE}\n"]
    src = "\n".join(f"- **{k}** `{v.name}` from `{v.parent.name}` "
                    f"({time.strftime('%d %b %H:%M', time.localtime(v.stat().st_mtime))})"
                    for k, v in sorted(found.items()))
    L.append(sec("Where every number comes from", src or "_nothing found_"))

    # 1 the paper's own claim, education-based
    if "47h" in found:
        t = read_47h(found["47h"])
        win = t[t["verdict"].isin(["CLEAN", "USABLE WITH CAVEAT"])]
        head = ("| design | artefact T2021 | artefact T2022 | true 22-25 | 50+ placebo | verdict |\n"
                "|---|---|---|---|---|---|\n")
        for r in t.itertuples():
            head += (f"| {r.design} | {r.a2021:+.4f} | {r.a2022:+.4f} | {r.true22:+.4f} | "
                     f"{r.placebo:.4f} | {r.verdict} |\n")
        if len(win):
            claim = (f"**Supported, by {len(win)} of {len(t)} designs.** The paper's claim and "
                     f"its within-employer design survive with education-based exposure; the "
                     f"best is `{win.iloc[0].design}`. Report the artefact beside the estimate "
                     f"unless the verdict is CLEAN.")
        else:
            claim = ("**Not supported.** Every worker-level education design manufactures the "
                     "result out of register lag, as 47b did. The paper's original claim cannot "
                     "be re-made this way; go to 47j.")
        L.append(sec("1. The paper's claim, with education-based exposure (47h)",
                     claim + "\n\n" + head))
    else:
        L.append(sec("1. The paper's claim, with education-based exposure (47h)",
                     "**Pending** -- `horserace_estimates.csv` not found."))

    # 2 within-employer triple difference
    if "47j" in found:
        t = read_47j(found["47j"])
        tbl = ("| design | young band | gamma3 (true) | artefact T2021 | artefact T2022 | verdict |\n"
               "|---|---|---|---|---|---|\n")
        for r in t.itertuples():
            tbl += (f"| {r.design} | {r.young_band} | {r.true:+.4f} | {r.a2021:+.4f} | "
                    f"{r.a2022:+.4f} | {r.verdict} |\n")
        ok = t[(t.young_band == "22-25") & t.verdict.isin(["CLEAN", "USABLE WITH CAVEAT"])]
        neg = ok[ok["true"] < 0]
        claim = ("**Supported.** Inside the same employer in the same month, young workers fell "
                 "behind older ones after ChatGPT, and more so where the firm's incumbent staff "
                 "are AI-exposed. No young worker's own education record enters the classifier."
                 if len(neg) else
                 "**Not supported as stated.** Either the artefact fails the rule or the gap is "
                 "not negative; read the table before writing anything.")
        L.append(sec("2. The age gradient within employers (47j)", claim + "\n\n" + tbl))
    else:
        L.append(sec("2. The age gradient within employers (47j)",
                     "**Pending** -- `triple_estimates.csv` not found."))

    # 3 firm-mix
    if "47i" in found:
        t = read_47i(found["47i"])
        y = t[t.age_group == "22-25"]
        tbl = ("| design | age | gamma2 (true) | artefact T2021 | artefact T2022 | verdict |\n"
               "|---|---|---|---|---|---|\n")
        for r in t.itertuples():
            tbl += (f"| {r.design} | {r.age_group} | {r.true:+.4f} | {r.a2021:+.4f} | "
                    f"{r.a2022:+.4f} | {r.verdict} |\n")
        claim = ("**Supported**, across firms: firms whose workforce is AI-exposed employ fewer "
                 "young people after ChatGPT. Weaker identification than 47j, and the "
                 "industry-reweighted contrast is still owed."
                 if len(y) and (y["true"] < 0).any()
                 and y.verdict.isin(["CLEAN", "USABLE WITH CAVEAT"]).any()
                 else "**Not supported as stated.** See the table.")
        L.append(sec("3. Exposed firms employ fewer young workers (47i)", claim + "\n\n" + tbl))
    else:
        L.append(sec("3. Exposed firms employ fewer young workers (47i)",
                     "**Pending** -- `firmmix_estimates.csv` not found."))

    # 4 the diagnosis, which stands either way
    d = (f"Occupation design (45): artefact {OCC_ART[2021]:+.4f} at T=2021 and "
         f"{OCC_ART[2022]:+.4f} at T=2022, against a true coefficient of +0.019 in the fully "
         f"covered years, and a submitted headline of {HEADLINE:+.3f}.\n\n"
         f"Worker-level education design (47b): {EDU_ART[2021]:+.4f} and {EDU_ART[2022]:+.4f}, "
         f"worse than the occupation design, with a 50+ placebo of -0.003 and mapped shares "
         f"moving by 0.1 to 0.4 per cent between the arms.\n\n"
         "**Supported, and independent of everything above.** A design in wide use manufactures "
         "this result out of register lag; the natural fix manufactures more of it; the damage "
         "is specific to the ages whose human capital is still moving; and no coverage "
         "diagnostic detects it.")
    L.append(sec("4. Register lag manufactures the result (45 and 47b)", d))

    # 5 validation
    L.append(sec("5. Does education predict the job at all (50 + l13)",
                 (f"```\n{val_txt.strip()}\n```" if val_txt else
                  "**Pending** -- script 50's export not found, so the predictive validation "
                  "has not run.")))

    # 6 supporting
    for key, title, claim in (
        ("48", "6. Gender split (48)",
         "Whether the decline is concentrated among young women."),
        ("41", "7. Code vintage (41)",
         "Whether the decline is steeper where the occupation code is older, which would "
         "corroborate the lag mechanism directly."),
        ("44", "8. Decile gradient (44)",
         "Whether the decline rises monotonically with exposure."),
        ("46", "9. Telework horse race (46)",
         "How much of the young-worker decline telework absorbs.")):
        if key in found:
            df = pd.read_csv(found[key])
            L.append(sec(title, claim + f"\n\n_{len(df)} rows in `{found[key].name}`; "
                                        f"read it before writing the sentence._"))
        else:
            L.append(sec(title, claim + "\n\n**Pending.**"))

    # ---- the register-immune family, which is now the spine ----
    def _num(key, col, **eq):
        """One number from a found csv, or None, without ever raising."""
        if key not in found:
            return None
        try:
            d = pd.read_csv(found[key])
            for k, v in eq.items():
                d = d[d[k].astype(str) == str(v)]
            return None if d.empty else float(d[col].iloc[0])
        except Exception:
            return None

    imm = []
    g = _num("47L", "gamma", variant="floor", payroll_tax_control="False")
    if g is None:
        g = _num("47L", "gamma", variant="floor")
    se = _num("47L", "se", variant="floor")
    if g is not None:
        imm.append(f"- **47L, employment stock, exposure frozen 2019**: "
                   f"{g:+.4f}" + (f" (SE {se:.4f})" if se else "")
                   + ". Uses no occupation code after 2019 and no education "
                     "register at all.")
    lam = None
    if "57" in found:
        try:
            d = pd.read_csv(found["57"])
            d = d[(d["age_group"] == "ALL")]
            lam = float(d.sort_values("year")["lam"].iloc[-1])
        except Exception:
            lam = None
    if lam is not None:
        imm.append(f"- **57, reliability of that frozen exposure**: lambda = "
                   f"{lam:.3f} at the last measured year. Attenuation is "
                   f"{'mild, so a null is a null' if lam > 0.75 else 'severe, so a null is not informative'}.")
    h = _num("54", "gamma", outcome="hires", variant="all_months")
    if h is None:
        h = _num("54", "gamma", outcome="hires")
    if h is not None:
        imm.append(f"- **54, hiring flow**: {h:+.4f}. The fast margin, and "
                   f"the one the entry-level claim is about.")
    L.append(sec("8b. The register-immune family",
                 ("\n".join(imm) + "\n\nThese share the DAIOE measure and "
                  "differ in their register dependence and identifying "
                  "variation, so agreement between them is corroboration "
                  "only against measurement error, not against a mismeasured "
                  "exposure concept.")
                 if imm else "**Pending**: none of 47L, 54 or 57 found."))

    for key, title, claim in (
        ("53", "8c. The paper's own estimand on contemporaneous codes (53)",
         "Young against young, inside the employer, monthly, restricted to "
         "worker-months whose occupation code was assigned in the "
         "observation year. Window ends 2023. Read the fresh arm against "
         "the stale arm: the contrast is internal."),
        ("47k", "8d. The settled sample (47k)",
         "Restricted to young workers whose education record is correct."),
        ("56", "8e. By age and over time (56)",
         "Event studies on stock, hires and separations. The PRE-PERIOD is "
         "the test; quote it."),
        ("58", "8f. Seasonal control and the pre-committed rule (58)",
         "Whether the 2025H1 reading survives honest seasonal handling. The "
         "rule can refuse, and on 20 Sep it did."),
        ("59", "8g. Quarterly and monthly path (59)",
         "Whether any late movement is a drift or one odd month."),
        ("60", "8h. When did it start? (60)",
         "Four pre-specified treatment dates, anchored on SCB's measured "
         "Swedish firm adoption (10.4 per cent in 2023, 25.2 in 2024, 35.0 "
         "in 2025), plus an exploratory profile. Earlier estimates all "
         "define post as Dec 2022 and therefore average an untreated year "
         "into the post window."),
    ):
        if key in found:
            try:
                n = len(pd.read_csv(found[key]))
            except Exception:
                n = "?"
            L.append(sec(title, claim + f"\n\n_{n} rows in "
                                        f"`{found[key].name}`; read it before "
                                        f"writing the sentence._"))
        else:
            L.append(sec(title, claim + "\n\n**Pending.**"))

    L.append(sec("10. Advertisements, no register involved",
                 "**Supported, and unaffected by any of the above.** Within-employer design on "
                 "public advertisements: PostGPT x High -0.158 on 12,141 employers, -0.196 on "
                 "entry-level advertisements, event study to -0.47 by 2026H1."))
    if sim_done:
        rk = HERE / "sim" / "results" / "ranking.txt"
        L.append(sec("11. Simulation study",
                     f"```\n{rk.read_text().strip()}\n```" if rk.exists()
                     else "ran, but no ranking file was written"))
    return "".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", help="export directories (e.g. ~/Downloads/MyFiles2014)")
    ap.add_argument("--sim", action="store_true", help="also run the simulation study (~3 h)")
    ap.add_argument("--file", action="store_true", help="copy the exports into revision/output")
    ap.add_argument("--out", default=str(HERE / "EVIDENCE"))
    a = ap.parse_args()

    print("assemble: scanning")
    if a.file:
        file_copies(a.dirs)
    found = find(a.dirs)
    for k in sorted(WANTED):
        print(f"  {k:<5} {'found  ' + found[k].name if k in found else 'PENDING'}")

    val = ""
    if "50" in found:
        d = found["50"].parent
        print(f"\nassemble: predictive validation on {d}")
        r = subprocess.run([sys.executable, str(LOCAL / "l13_validate_edu_designs.py"), str(d)],
                           capture_output=True, text=True)
        val = r.stdout if r.returncode == 0 else f"l13 failed: {r.stderr[-400:]}"
        print(val)

    sim_done = False
    if a.sim and "50" in found:
        print("\nassemble: simulation study (this takes about three hours)")
        r = subprocess.run([sys.executable, str(HERE / "sim" / "run_sim.py"),
                            "--export", str(found["50"].parent), "--full", "--seeds", "5",
                            "--resume", "--out", str(HERE / "sim" / "results")])
        sim_done = r.returncode == 0
    elif a.sim:
        print("assemble: --sim ignored, script 50's export is not here")

    md = Path(a.out + ".md")
    md.write_text(build_report(found, sim_done, val))
    print(f"\nwrote {md}")
    try:
        subprocess.run(["pandoc", str(md), "-o", a.out + ".pdf",
                        "--pdf-engine=tectonic", "-V", "geometry:margin=2.4cm"],
                       check=True, capture_output=True)
        print(f"wrote {a.out}.pdf")
    except Exception as ex:
        print(f"(no pdf: {type(ex).__name__})")
    print("\n" + "=" * 70)
    print(md.read_text()[:1500])


if __name__ == "__main__":
    main()
