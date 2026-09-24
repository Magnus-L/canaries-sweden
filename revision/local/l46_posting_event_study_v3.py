#!/usr/bin/env python3
"""
l46_posting_event_study_v3.py: the posting-margin event study of Online
Appendix II.2, rebuilt on the window the paper's estimates use.

QUESTION
Online Appendix II.2 reported the monthly event study, its joint Wald
pre-tests and the Rambachan and Roth relative-magnitudes panel from
src/07_robustness.py, which estimates on January 2020 to December 2025.
The pooled posting estimates the paper quotes (Section II.6, script l08)
run to June 2026. This script re-runs the same three diagnostics on the
same panel as l08, so the two sections share one window and one sample.

WHAT IT DOES
The estimators are those of src/07_robustness.py, unchanged; only the
window is a parameter here instead of a hard-coded end month:
  R8  monthly event study: ln(postings) on month dummies x High,
      occupation and month effects (linearmodels PanelOLS), SEs
      clustered by occupation, February 2020 omitted; joint Wald test
      on all coefficients before April 2022.
  R9  the same at quarterly frequency, 2020Q1 omitted; joint Wald test
      on all quarters before 2022Q2.
  R11 the average post-ChatGPT coefficient (December 2022 onwards),
      its delta-method SE, and the sensitivity grid of 07: the
      interval theta +/- (1.96 SE + Mbar x Dmax), where Dmax is the
      largest absolute month-to-month change in the pre-April-2022
      coefficients (reference month included as zero). The breakdown
      value is the first Mbar on the 0.25 grid whose interval covers
      zero. This is 07's simplified implementation, not the HonestDiD
      linear program; see notes/event-study-rebuild_2026-09-24.md.

GATE
With --window 2025-12 the script must reproduce the published old
numbers (chi2_26 = 107.059, chi2_8 = 52.769, theta = -0.16897,
SE 0.05905, breakdown 0.25). It checks this on every run before
writing the June 2026 outputs.

INPUTS AND OUTPUTS
Reads data/processed/postings_daioe_merged_extended.csv (written by l08;
the II.6 sample, 28,084 occupation-by-month cells, all positive).
Writes revision/tables/posting_es_monthly_v3.csv,
posting_es_quarterly_v3.csv, posting_pretrend_v3.csv,
posting_rr_sensitivity_v3.csv, posting_es_summary_v3.csv, and
revision/figures/figA3_event_study_v3.{pdf,png},
figA6_rambachan_roth_v3.{pdf,png}, and
figA5_event_study_quarterly_v3.{pdf,png} (the offline appendix's
quarterly figure).

IN THE PAPER
Online Appendix II.2 (sec:posting_design_robustness), Figure
fig:posting_design, and the response to R1 where it quotes II.2.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from linearmodels.panel import PanelOLS

REV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REV / "local"))
_cfg_spec = importlib.util.spec_from_file_location("v2config", REV / "config.py")
_cfg = importlib.util.module_from_spec(_cfg_spec)
_cfg_spec.loader.exec_module(_cfg)
from _figsafe import save  # noqa: E402

# House style of src/07_robustness.py: its colours and rcParams come from
# src/config.py, so they are taken from there, not restated.
_src_spec = importlib.util.spec_from_file_location(
    "srcconfig", REV.parent / "src" / "config.py")
_src = importlib.util.module_from_spec(_src_spec)
_src_spec.loader.exec_module(_src)
DARK_BLUE, ORANGE, TEAL, GRAY = _src.DARK_BLUE, _src.ORANGE, _src.TEAL, _src.GRAY
_src.set_rcparams()

PANEL = _cfg.PROCESSED / "postings_daioe_merged_extended.csv"
TAB = _cfg.V2_TAB
FIG = _cfg.V2_FIG
START, BASE_MONTH = "2020-01", "2020-02"
RB, GPT = _cfg.RIKSBANKEN_HIKE[:7], _cfg.CHATGPT_LAUNCH[:7]

OLD = {"wald_m": 107.059, "wald_q": 52.769, "theta": -0.16897,
       "se": 0.05905, "mbar": 0.25}


def load(end):
    df = pd.read_csv(PANEL, dtype={"ssyk4": str})
    df = df[(df["year_month"] >= START) & (df["year_month"] <= end)].copy()
    df = df[df["n_ads"] > 0].copy()
    df["ln_ads"] = np.log(df["n_ads"])
    df["date"] = pd.to_datetime(df["year_month"] + "-01")
    return df


def wald(coefs, cov, cols):
    b = np.array([coefs[c] for c in cols])
    V = cov.loc[cols, cols].values
    W = float(b @ np.linalg.inv(V) @ b)
    return W, len(cols), 1 - scipy_stats.chi2.cdf(W, len(cols))


def monthly(df):
    months = sorted(df["year_month"].unique())
    excl = [m for m in months if m != BASE_MONTH]
    cols = [f"m_{m}_x_high" for m in excl]
    for m, c in zip(excl, cols):
        df[c] = ((df["year_month"] == m) & (df["high_exposure"] == 1)).astype(int)
    p = df.set_index(["ssyk4", "date"])
    res = PanelOLS(p["ln_ads"], p[cols], entity_effects=True,
                   time_effects=True).fit(cov_type="clustered",
                                          cluster_entity=True)
    return res, excl, cols


def quarterly(df):
    d = df.copy()
    d["quarter"] = d["date"].dt.to_period("Q").astype(str)
    q = (d.groupby(["ssyk4", "quarter", "high_exposure"])
         .agg(n_ads=("n_ads", "sum")).reset_index())
    q = q[q["n_ads"] > 0].copy()
    q["ln_ads"] = np.log(q["n_ads"])
    q["date"] = pd.PeriodIndex(q["quarter"], freq="Q").to_timestamp()
    excl = [x for x in sorted(q["quarter"].unique()) if x != "2020Q1"]
    cols = [f"q_{x}_x_high" for x in excl]
    for x, c in zip(excl, cols):
        q[c] = ((q["quarter"] == x) & (q["high_exposure"] == 1)).astype(int)
    p = q.set_index(["ssyk4", "date"])
    res = PanelOLS(p["ln_ads"], p[cols], entity_effects=True,
                   time_effects=True).fit(cov_type="clustered",
                                          cluster_entity=True)
    return res, excl, cols, len(q)


def analyse(end):
    df = load(end)
    n_cells = len(df)
    res, excl, cols = monthly(df)
    coefs = res.params[cols]
    ses = res.std_errors[cols]

    pre_rb = [m for m in excl if m < RB]
    pre_ref = [m for m in excl if m < BASE_MONTH]
    W_m, k_m, p_m = wald(coefs, res.cov, [f"m_{m}_x_high" for m in pre_rb])

    qres, qexcl, qcols, n_q = quarterly(df)
    qpre = [x for x in qexcl if x < "2022Q2"]
    W_q, k_q, p_q = wald(qres.params, qres.cov, [f"q_{x}_x_high" for x in qpre])

    # R11, as in 07: Dmax over pre-April-2022 coefficients, base included.
    pre_coefs = [0.0] + [coefs[f"m_{m}_x_high"] for m in sorted(pre_rb)]
    dmax = max(abs(pre_coefs[i] - pre_coefs[i - 1])
               for i in range(1, len(pre_coefs)))
    post = [f"m_{m}_x_high" for m in excl if m >= GPT]
    w = np.ones(len(post)) / len(post)
    theta = float(np.mean([coefs[c] for c in post]))
    se_theta = float(np.sqrt(w @ res.cov.loc[post, post].values @ w))
    z = scipy_stats.norm.ppf(0.975)
    grid, mbar_bd = [], None
    for mb in np.arange(0, 3.25, 0.25):
        lo = theta - z * se_theta - mb * dmax
        hi = theta + z * se_theta + mb * dmax
        inc = (lo <= 0) and (hi >= 0)
        grid.append({"mbar": mb, "theta_hat": theta, "se_theta": se_theta,
                     "bias_bound": mb * dmax, "ci_lo": lo, "ci_hi": hi,
                     "includes_zero": inc})
        if inc and mbar_bd is None:
            mbar_bd = mb

    es = pd.DataFrame({"year_month": excl,
                       "coef": [coefs[f"m_{m}_x_high"] for m in excl],
                       "se": [ses[f"m_{m}_x_high"] for m in excl]})
    es = pd.concat([es, pd.DataFrame({"year_month": [BASE_MONTH],
                                      "coef": [0.0], "se": [0.0]})])
    es["date"] = pd.to_datetime(es["year_month"] + "-01")
    es = es.sort_values("date").reset_index(drop=True)
    es["ci_lo"] = es["coef"] - 1.96 * es["se"]
    es["ci_hi"] = es["coef"] + 1.96 * es["se"]

    qes = pd.DataFrame({"quarter": qexcl,
                        "coef": [qres.params[c] for c in qcols],
                        "se": [qres.std_errors[c] for c in qcols]})

    summary = {
        "window": f"{START} to {end}", "n_cells": n_cells,
        "n_occupations": df["ssyk4"].nunique(),
        "n_months": df["year_month"].nunique(),
        "n_pre_reference_coefs": len(pre_ref),
        "n_pre_ratehike_coefs": len(pre_rb),
        "n_post_chatgpt_coefs": len(post),
        "wald_monthly": W_m, "df_monthly": k_m, "p_monthly": p_m,
        "wald_quarterly": W_q, "df_quarterly": k_q, "p_quarterly": p_q,
        "n_quarter_cells": n_q,
        "theta_hat": theta, "se_theta": se_theta,
        "p_theta": 2 * (1 - scipy_stats.norm.cdf(abs(theta / se_theta))),
        "dmax_pre": dmax, "breakdown_mbar": mbar_bd,
        "n_pre_coefs_signif_05": int(sum(
            abs(coefs[f"m_{m}_x_high"] / ses[f"m_{m}_x_high"]) > 1.96
            for m in pre_rb)),
    }
    return summary, es, qes, pd.DataFrame(grid)


def plot_es(es):
    """Panel (a), drawn exactly as src/07_robustness.py draws figA3."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(es["date"], es["ci_lo"], es["ci_hi"],
                    alpha=0.2, color=DARK_BLUE)
    ax.plot(es["date"], es["coef"], color=DARK_BLUE, linewidth=1.5,
            marker="o", markersize=3)
    ax.axhline(0, color=GRAY, linewidth=0.8, linestyle="--")
    rb_date, gpt_date = pd.Timestamp(_cfg.RIKSBANKEN_HIKE), pd.Timestamp(_cfg.CHATGPT_LAUNCH)
    ax.axvline(rb_date, color=ORANGE, linewidth=1.5, linestyle="--", alpha=0.9)
    ax.axvline(gpt_date, color=TEAL, linewidth=1.5, linestyle="--", alpha=0.9)
    ymin, ymax = ax.get_ylim()
    label_y = ymax - (ymax - ymin) * 0.08
    ax.annotate("Riksbanken hike\n(Apr 2022)", xy=(rb_date, label_y),
                fontsize=9, color=ORANGE, fontweight="bold", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=ORANGE, alpha=0.85))
    ax.annotate("ChatGPT launch\n(Nov 2022)", xy=(gpt_date, label_y),
                fontsize=9, color=TEAL, fontweight="bold", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=TEAL, alpha=0.85))
    ax.set_xlabel("")
    ax.set_ylabel("Coefficient (relative to Feb 2020)")
    ax.set_title("Event study: High vs low genAI exposure (monthly DiD coefficients)")
    fig.tight_layout()
    save(fig, "figA3_event_study_v3", __file__, FIG)
    plt.close(fig)


def plot_es_quarterly(qes):
    """The quarterly path, drawn as src/07_robustness.py draws figA5."""
    q = pd.concat([qes, pd.DataFrame({"quarter": ["2020Q1"], "coef": [0.0],
                                      "se": [0.0]})])
    q["date"] = pd.PeriodIndex(q["quarter"], freq="Q").to_timestamp()
    q = q.sort_values("date")
    q["ci_lo"] = q["coef"] - 1.96 * q["se"]
    q["ci_hi"] = q["coef"] + 1.96 * q["se"]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(q["date"], q["ci_lo"], q["ci_hi"], alpha=0.2, color=DARK_BLUE)
    ax.plot(q["date"], q["coef"], color=DARK_BLUE, linewidth=2, marker="o",
            markersize=5)
    ax.axhline(0, color=GRAY, linewidth=0.8, linestyle="--")
    rb_date, gpt_date = pd.Timestamp(_cfg.RIKSBANKEN_HIKE), pd.Timestamp(_cfg.CHATGPT_LAUNCH)
    ax.axvline(rb_date, color=ORANGE, linewidth=1.5, linestyle="--", alpha=0.9)
    ax.axvline(gpt_date, color=TEAL, linewidth=1.5, linestyle="--", alpha=0.9)
    ymin, ymax = ax.get_ylim()
    label_y = ymax - (ymax - ymin) * 0.08
    ax.annotate("Riksbanken hike\n(Apr 2022)", xy=(rb_date, label_y),
                fontsize=9, color=ORANGE, fontweight="bold", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=ORANGE, alpha=0.85))
    ax.annotate("ChatGPT launch\n(Nov 2022)", xy=(gpt_date, label_y),
                fontsize=9, color=TEAL, fontweight="bold", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=TEAL, alpha=0.85))
    ax.set_xlabel("")
    ax.set_ylabel("Coefficient (relative to 2020 Q1)")
    ax.set_title("Quarterly event study: High vs low genAI exposure")
    fig.tight_layout()
    save(fig, "figA5_event_study_quarterly_v3", __file__, FIG)
    plt.close(fig)


def plot_rr(grid, theta, mbar_bd):
    """Panel (b), drawn as src/07_robustness.py draws figA6."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(grid["mbar"], grid["ci_lo"], grid["ci_hi"],
                    alpha=0.2, color=DARK_BLUE, label="95% honest CI")
    ax.axhline(theta, color=DARK_BLUE, linewidth=2,
               label=f"$\\hat{{\\theta}}$ = {theta:.3f}")
    ax.axhline(0, color=GRAY, linewidth=0.8, linestyle="--")
    if mbar_bd is not None:
        ax.axvline(mbar_bd, color=ORANGE, linewidth=1.5, linestyle=":",
                   label=f"Breakdown $\\bar{{M}}$ = {mbar_bd:.2f}")
    ax.set_xlabel("$\\bar{M}$ (relative magnitudes)")
    ax.set_ylabel("Average post-ChatGPT effect")
    ax.set_title("Rambachan-Roth sensitivity: average post-ChatGPT effect\n"
                 "on high vs low genAI exposure occupations")
    ax.legend(loc="lower left", framealpha=0.9)
    fig.tight_layout()
    save(fig, "figA6_rambachan_roth_v3", __file__, FIG)
    plt.close(fig)


def main():
    print("L46: posting event study on the II.6 window")
    old, *_ = analyse("2025-12")
    got = {"wald_m": old["wald_monthly"], "wald_q": old["wald_quarterly"],
           "theta": old["theta_hat"], "se": old["se_theta"],
           "mbar": old["breakdown_mbar"]}
    for k, v in OLD.items():
        tol = 5e-4 if k != "mbar" else 1e-9
        assert abs(got[k] - v) < tol * max(1, abs(v)), \
            f"GATE: old window does not reproduce {k}: {got[k]} vs {v}"
    print("  gate passed: January 2020 to December 2025 reproduces 07's numbers")

    new, es, qes, grid = analyse(_cfg.POSTINGS_REGRESSION_END)
    assert new["n_cells"] == 28084, new["n_cells"]
    # The pooled II.6 coefficients on the same sample (l08's export), so
    # the reconciliation in the text reads from one file: relative to
    # February 2020 the event-study average carries both pooled terms.
    did = pd.read_csv(TAB / "postings_extended_did.csv")
    did = did[(did["window"] == "extended_to_2026-06")
              & (did["estimator"] == "OLS_ln")].set_index("term")
    assert int(did.loc["rb_x_high", "n_obs"]) == new["n_cells"]
    new["pooled_rb_x_high"] = did.loc["rb_x_high", "coef"]
    new["pooled_gpt_x_high"] = did.loc["gpt_x_high", "coef"]
    new["pooled_rb_plus_gpt"] = new["pooled_rb_x_high"] + new["pooled_gpt_x_high"]
    for k in ("pooled_rb_x_high", "pooled_gpt_x_high", "pooled_rb_plus_gpt"):
        old[k] = None
    es.to_csv(TAB / "posting_es_monthly_v3.csv", index=False)
    qes.to_csv(TAB / "posting_es_quarterly_v3.csv", index=False)
    grid.to_csv(TAB / "posting_rr_sensitivity_v3.csv", index=False)
    pd.DataFrame([{**{f"old_{k}": v for k, v in old.items()},
                   **{f"new_{k}": v for k, v in new.items()}}]).T \
        .rename(columns={0: "value"}) \
        .to_csv(TAB / "posting_es_summary_v3.csv")
    pd.DataFrame([
        {"window": new["window"], "frequency": "monthly",
         "n_pre_periods": new["df_monthly"], "wald_stat": new["wald_monthly"],
         "p_value": new["p_monthly"]},
        {"window": new["window"], "frequency": "quarterly",
         "n_pre_periods": new["df_quarterly"], "wald_stat": new["wald_quarterly"],
         "p_value": new["p_quarterly"]},
    ]).to_csv(TAB / "posting_pretrend_v3.csv", index=False)
    plot_es(es)
    plot_es_quarterly(qes)
    plot_rr(grid, new["theta_hat"], new["breakdown_mbar"])
    for k in new:
        print(f"  {k:>24}: old {old[k]!s:>24}   new {new[k]!s:>24}")


if __name__ == "__main__":
    main()
