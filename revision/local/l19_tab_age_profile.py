#!/usr/bin/env python3
"""
l19_tab_age_profile.py: Online Appendix Table III.2 (tab:age_profile_rebuilt),
the headline design by age beside the age profile on a different exposure
construction.

WHAT THE TABLE REPORTS
Two panels that must not be read against each other.

  Panel A  The headline route: the top quartile of the 2019 education mix
           of an employer's incumbents aged 31 to 69, for the two young
           bands. Rows give the adoption step with and without the
           calendar cycle removed (scripts 68 and 61), the vintage
           re-scoring beside each, the tightening step, the step from the
           2023 level (the post-adoption minus the interim term of the
           window specification of script 75, standard error from their
           covariance), the level after adoption against the months
           before the rate rise, and the hires and separations steps at
           adoption.
  Panel B  Six age bands on the continuous occupation-scaled measures of
           script 63 (DAIOE, Eloundou et al. and teleworkability), each
           coefficient per standard deviation of the 2019 firm-age
           baseline on that measure's own scale, treatment dated at
           adoption. A different estimand, reported as a check on the
           measure.

The note quotes the 22-25 contrast against 41-49 with the cycle removed
(script 74). The vintage re-scoring is the change in the coefficient when
the 2019 incumbents are re-scored from the education register as it stood
in 2021 (the as-of arms of scripts 61 and 68); the 26-30 arm of script 68
was not re-scored, so that cell is empty.

INPUTS AND OUTPUTS
Reads, from the export directories the final-code manifest names (or one
directory given on the command line): seasonal_pooled.csv (script 68),
output_61__redated_pooled.csv (script 61), reference_window.csv and
vcov_s75_<band>_stock.csv (script 75), contrast_seasonal.csv (script 74)
and output_63__robustness_gradient.csv (script 63). Writes
revision/tables/tableA_age_profile.tex and copies it to
canaries-sweden-paper/tables/.

    python3 revision/local/l19_tab_age_profile.py [export_dir]

IN THE PAPER
Online Appendix III.2, Table tab:age_profile_rebuilt.
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

REV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REV))
from config import V2_TAB  # noqa: E402

OUT = REV / "output"
LANE14 = OUT / "round3_20260921-2152-lane14-seasonal-complete"
ROUND2_61_63 = OUT / "round2_20260920-2148-jobs616263"
LANE20 = OUT / "round3_20260922-0105-lane20-seasonal-contrast"
LANE21_22 = OUT / "round3_20260922-0712-lanes21-22"
PAPER_TAB = REV.parents[1] / "canaries-sweden-paper" / "tables"

BANDS = ["22-25", "26-30", "31-34", "35-40", "41-49", "50+"]
TERM = "post_x_high_x_young"
POOLED = "post2024_x_high_x_young"


def source(default_dir: Path, name: str) -> Path:
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else default_dir
    p = d / name
    if not p.exists():
        raise SystemExit(f"  missing input: {p}")
    return p


def fmt(c, se, p=None):
    """Estimate with a star at five per cent, from the exported p-value
    where the export carries one and from the normal threshold otherwise."""
    sig = (p < 0.05) if p is not None else (abs(c) > 1.96 * se)
    star = "^{*}" if sig else ""
    return f"${c:+.4f}{star}$ ({se:.4f})"


def panel_a():
    """Headline route: education-mix quartile, the two young bands."""
    rows = []
    d = pd.read_csv(source(LANE14, "seasonal_pooled.csv"))
    d = d[d.get("status", "ok") == "ok"]
    for band in ("22-25", "26-30"):
        t = d[(d.young_band == band) & (d.outcome == "stock")
              & (d.arm == "true") & (d.term == TERM)]
        a = d[(d.young_band == band) & (d.outcome == "stock")
              & (d.arm == "asof") & (d.term == TERM)]
        if t.empty:
            raise SystemExit(f"  no cycle-removed step for {band}")
        art = (f"${float(a.coef.iloc[0]) - float(t.coef.iloc[0]):+.4f}$"
               if not a.empty else "--")
        rows.append((f"{band}, cycle removed",
                     fmt(float(t.coef.iloc[0]), float(t.se.iloc[0])), art))
    d = pd.read_csv(source(ROUND2_61_63, "output_61__redated_pooled.csv"))
    d = d[d.get("status", "ok") == "ok"]
    for band in ("22-25", "26-30"):
        t = d[(d.young_band == band) & (d.arm == "true")
              & (d.term == POOLED) & (d.design == "OL_daioe")]
        a = d[(d.young_band == band) & (d.arm == "asof")
              & (d.term == POOLED) & (d.design == "OL_daioe")]
        if t.empty:
            raise SystemExit(f"  no pre-cycle step for {band}")
        art = (f"${float(a.coef.iloc[0]) - float(t.coef.iloc[0]):+.4f}$"
               if not a.empty else "--")
        rows.append((f"{band}, before the cycle",
                     fmt(float(t.coef.iloc[0]), float(t.se.iloc[0])), art))
    w = pd.read_csv(source(LANE21_22, "reference_window.csv"))
    w = w[w.get("status", "ok") == "ok"]

    def win(band, term):
        r = w[(w.young_band == band) & (w.outcome == "stock")
              & (w.term == term)]
        if r.empty:
            raise SystemExit(f"  no {term} for {band} in reference_window.csv")
        return fmt(float(r.coef.iloc[0]), float(r.se.iloc[0]))

    rows.append(("22-25, tightening step, April 2022",
                 win("22-25", "rbw_x_high_x_young"), "--"))
    for band in ("22-25", "26-30"):
        c, se = step23(w, band)
        rows.append((f"{band}, step from the 2023 level", fmt(c, se), "--"))
    rows.append(("22-25, level after adoption vs pre-hike months",
                 win("22-25", TERM), "--"))
    rows.append(("26-30, level after adoption vs pre-hike months",
                 win("26-30", TERM), "--"))
    # the two margins, additional step at adoption (script 68)
    d = pd.read_csv(source(LANE14, "seasonal_pooled.csv"))
    d = d[d.get("status", "ok") == "ok"]
    for band in ("22-25", "26-30"):
        for outcome, label in (("hires", "hires"), ("seps", "separations")):
            r = d[(d.young_band == band) & (d.outcome == outcome)
                  & (d.arm == "true") & (d.term == TERM)]
            if r.empty:
                raise SystemExit(f"  no {outcome} step for {band}")
            rows.append((f"{band}, {label}, step at adoption",
                         fmt(float(r.coef.iloc[0]), float(r.se.iloc[0])), "--"))
    return rows


def step23(w: pd.DataFrame, band: str) -> tuple[float, float]:
    """Post-adoption minus interim term of the window specification, which
    equals gamma_2 minus gamma_0 of Equation (2); standard error from the
    exported covariance."""
    def row(term):
        r = w[(w.young_band == band) & (w.outcome == "stock") & (w.term == term)]
        if r.empty:
            raise SystemExit(f"  no {term} for {band} in reference_window.csv")
        return float(r.coef.iloc[0])
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else LANE21_22
    v = pd.read_csv(d / f"vcov_s75_{band.replace('-', '_')}_stock.csv",
                    index_col=0)
    i = "interim_x_high_x_young"
    var = v.loc[TERM, TERM] + v.loc[i, i] - 2 * v.loc[TERM, i]
    return row(TERM) - row(i), float(var) ** 0.5


def panel_b():
    """Six bands, continuous occupation-scaled measures. Different estimand."""
    d = pd.read_csv(source(ROUND2_61_63, "output_63__robustness_gradient.csv"))
    d = d[d.dating == "adoption"]
    out = []
    for b in BANDS:
        cells = []
        for m in ("daioe", "eloundou", "telework"):
            r = d[(d.age_group == b) & (d.measure == m) & (d.outcome == "stock")]
            cells.append("--" if r.empty else
                         fmt(float(r.coef.iloc[0]), float(r.se.iloc[0]),
                             float(r.pvalue.iloc[0])))
        out.append((b, cells))
    return out


def contrast_22_25() -> str:
    """The 22-25 contrast against 41-49, cycle removed, for the note."""
    d = pd.read_csv(source(LANE20, "contrast_seasonal.csv"))
    r = d[(d.arm == "seasonal") & (d.band_vs_ref == "22_25")]
    if len(r) != 1:
        raise SystemExit("  contrast_seasonal.csv: one seasonal 22_25 row expected")
    return f"${float(r.coef.iloc[0]):+.4f}$ ({float(r.se.iloc[0]):.4f})"


def main() -> int:
    A, B = panel_a(), panel_b()
    contrast = contrast_22_25().replace("+", "")

    tex = [r"\begin{table}[ht!]", r"\centering",
           r"\caption{The headline design by age, and the age profile on a "
           r"different exposure construction.}",
           r"\label{tab:age_profile_rebuilt}", r"\footnotesize",
           r"\begin{tabular}{lccc}", r"\toprule",
           r"\multicolumn{4}{l}{\textit{Panel A. Headline route: "
           r"top-quartile 2019 education mix of incumbents aged 31+}} \\",
           r"\addlinespace[2pt]",
           r" & Estimate (SE) & Vintage re-scoring & \\", r"\midrule"]
    for lab, e, art in A:
        tex.append(f"{lab} & {e} & {art} & \\\\")
        print(f"  A  {lab:48s} {e}  art {art}")
    tex += [r"\addlinespace[6pt]",
            r"\multicolumn{4}{l}{\textit{Panel B. Continuous "
            r"occupation-scaled measures, per standard deviation}} \\",
            r"\addlinespace[2pt]",
            r"Age band & DAIOE & Eloundou & Teleworkable \\", r"\midrule"]
    for b, cells in B:
        tex.append(f"{b} & " + " & ".join(cells) + r" \\")
        print(f"  B  {b:6s} " + "  ".join(c.replace('$', '') for c in cells))
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\begin{minipage}{0.94\textwidth}\footnotesize\vspace{4pt}",
            r"Poisson pseudo-maximum likelihood on employer $\times$ age "
            r"$\times$ month counts, employer-by-month, employer-by-age and "
            r"month-by-age effects, treatment dated January 2024, standard "
            r"errors clustered by employer, $^{*}$ denotes $p<0.05$. "
            r"Exposure is frozen in 2019 throughout and no occupation code "
            r"recorded after 2019 enters the exposure, the outcome or the "
            r"sample. \textbf{The two panels are not comparable.} Panel A "
            r"is a top-quartile indicator built from the education mix of a "
            r"firm's incumbents, and it is the estimand the paper reports; "
            r"it exists for the two young bands only. The tightening step "
            r"and the two levels come from the same specification with the "
            r"Riksbank interaction as a window (April to November 2022), so "
            r"that the post-adoption term reads against January 2021 to "
            r"March 2022. Panel B scores occupations continuously, so a "
            r"coefficient there is the effect of one standard deviation of "
            r"the 2019 firm-age baseline exposure, on that measure's own "
            r"scale. The vintage re-scoring column is the change in the coefficient "
            r"when each employer's 2019 incumbents are re-scored from the "
            r"education register as it stood in 2021, the staleness the "
            r"2024--25 records inherit, and the same panel is re-estimated; "
            r"the threshold fixed before that test was 0.05. Employment at "
            r"41--49 declines on both AI measures in Panel B. The paper "
            r"does not claim the young were hit harder than the "
            r"prime-aged: tested directly with the calendar cycle removed "
            r"on all six bands, the 22--25 contrast against 41--49 is "
            + contrast + r", a null (Table~\ref{tab:profile_seasonal}).",
            r"\end{minipage}", r"\end{table}"]
    out = V2_TAB / "tableA_age_profile.tex"
    out.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"\n  wrote {out.relative_to(REV)}")
    if PAPER_TAB.exists():
        shutil.copy(out, PAPER_TAB / out.name)
        print(f"  copied to {PAPER_TAB / out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
