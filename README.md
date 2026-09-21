# Same Storm, Different Boats: Generative AI and Young Workers Within Firms

Replication package for Lodefalk, Löthman, Koch, and Engberg (2026), "Same Storm, Different Boats: Generative AI and Young Workers Within Firms."

Örebro University WP 2026:2 / Ratio WP 388. The manuscript and appendices live in the sibling
repository `Magnus-L/canaries-sweden-paper`; `paper/` here holds the frozen submitted version only.

## Status

**Revise and resubmit at *Economics Letters*** (R&R received 2 August 2026; editor Eric Chyn).
Authors: **Lodefalk, Löthman, Koch, Engberg** — Lodefalk first, and Koch not Kock; an earlier
ordering circulated and reached two slide decks before it was corrected.

**The revision withdrew the submitted employment design, and this README describes the
replacement.** The submitted version compared young workers in AI-exposed occupations with
young workers in less exposed occupations inside the same firm. That needs a current
occupation code on every young worker in every month, and Sweden's occupation register is
published with a two-year lag. An as-of backtest, which imposes the 2024–25 staleness on years
where the truth is observable, showed the lag alone moves the coefficient from +0.019 to −0.288
— more than the whole of the −0.174 we had reported. We withdrew the design rather than defend
it. The revision code lives in `revision/`; the manuscript is in the sibling repository
`Magnus-L/canaries-sweden-paper`.

## Overview

We ask whether the divergence between stock prices and job postings after 2022 reflects
generative AI displacing labour demand, or monetary tightening. Sweden separates the two
because the Riksbank's first rate rise (April 2022) preceded ChatGPT (November 2022).

Using 4.9 million Platsbanken advertisements (2020 to June 2026) matched to the DAIOE
generative-AI exposure index, the aggregate posting decline tracks the rate hike and not
ChatGPT. Employment behaves differently. Comparing young workers with their older colleagues
inside the same employer, with exposure scored once in 2019 from the education mix of
incumbents aged 31 and over, employment of workers aged 22–25 falls about 4 per cent once
firms adopt AI, and the same shortfall reaches 26–30 a year later. The adjustment runs through
separations rather than reduced hiring, and is about twice as large for young women. Against
41–49 alone the contrast is not statistically distinguishable, so the finding is a shortfall of
the under-31s against the older workforce as a whole rather than the young against the
prime-aged.

## Data availability

| Dataset | Source | Access |
|---------|--------|--------|
| Platsbanken historical ads | [JobTech Development](https://data.jobtechdev.se/annonser/historiska/) | Open (CC0) |
| DAIOE genAI exposure | [Engberg et al. (2024), IZA DP 16717](https://docs.iza.org/dp16717.pdf) | Open |
| OMXS30 / OMXSPI prices | Yahoo Finance (`^OMX`, `^OMXSPI`) | Open |
| Riksbanken policy rate | [riksbank.se](https://riksbank.se) | Open |
| Indeed US postings | [Indeed Hiring Lab](https://github.com/hiring-lab/job_postings_tracker) | Open |
| S&P 500 / Nasdaq prices | Yahoo Finance (`^GSPC`, `^IXIC`) | Open |
| Dingel-Neiman teleworkability | [Dingel & Neiman (2020)](https://github.com/jdingel/DingelNeiman-workathome) | Open |
| Eloundou GPT exposure | [Eloundou et al. (2024)](https://doi.org/10.1126/science.adj0998) | Open |
| AGI employer declarations (SCB) | Statistics Sweden, MONA | Restricted |

The posting analysis (steps 1–13) uses only public data. The employment analysis (steps 14–19) uses restricted administrative microdata from Statistics Sweden, accessible via the MONA platform. See `src/MONA_INSTRUCTIONS.md` for access details.

## Computational requirements

- **Software:** Python 3.10+ with packages listed in `requirements.txt`; Stata 18+ for replication checks
- **Hardware:** Standard laptop/desktop. ~8 GB RAM recommended for processing JSONL files.
- **Time:** Full local pipeline ~20–30 minutes (dominated by download); subsequent runs ~5–10 minutes.
- **Storage:** ~6 GB for raw data, ~100 MB for processed data.

## Instructions for replicators

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full local pipeline (public data)
python src/run_all.py

# Or skip the download step if data already exists:
python src/run_all.py --skip-download

# Or start from a specific step:
python src/run_all.py --from-step 4
```

The master script (`run_all.py`) executes steps 1–13 sequentially. MONA scripts (14–19) must be run separately at SCB.

## Pipeline steps

> **Which pipeline reproduces which paper.** The numbered steps below are the
> ORIGINAL pipeline and they reproduce the **submitted** version, including the
> employment results the revision withdrew. They are kept because the posting
> analysis is unchanged and because a reader may want to see what the closed
> design produced. Step 12 is named `12_create_figure2_age_gradient.py` for the
> same historical reason.
>
> **The revision's code is in `revision/`**: `revision/mona/` for the scripts
> that run inside SCB's MONA environment, `revision/local/` for the figures and
> tables built from their exports, and `revision/output/` for the exports
> themselves, which are aggregate coefficients and contain no microdata. The
> employment results in the current manuscript come from there, not from the
> steps below.


### Local pipeline (public data)

| Step | Script | Description |
|------|--------|-------------|
| 1 | `01_download_platsbanken.py` | Download Platsbanken data (~5.4 GB) |
| 2 | `02_process_platsbanken.py` | Process JSONL → SSYK4 × month aggregates |
| 3 | `03_fetch_auxiliary.py` | Fetch OMXS30, Riksbank rate, DAIOE, crosswalks |
| 4 | `04_merge_and_classify.py` | Merge postings with DAIOE, assign quartiles |
| 5 | `05_analysis.py` | Posting DiD regression (Equation 1) |
| 6 | `06_figures_tables.py` | Main figures (scary chart, quartile panels) and tables |
| 7 | `07_robustness.py` | 8 robustness specifications, event studies, Rambachan-Roth |
| 8 | `08_interest_rate_exposure.py` | Interest rate sensitivity scatter |
| 9 | `09_remote_work_robustness.py` | Teleworkability split (Dingel-Neiman) |
| 10 | `10_eloundou_robustness.py` | Alternative AI measure (Eloundou et al.) |
| 11 | `11_riksbank_rate_figure.py` | Riksbank policy rate timeline |
| 12 | `12_create_figure2_age_gradient.py` | Figure 2: age gradient bar chart |
| 13 | `13_onepager_figure.py` | One-pager summary figure |

### MONA pipeline (restricted data, run at SCB)

| Step | Script | Description |
|------|--------|-------------|
| 14 | `14_mona_canaries_descriptive.py` | Descriptive canaries figure (Fig. 2 in paper) |
| 15 | `15_mona_employer_did.py` | Employer-level DiD + event study (Equation 2) |
| 16 | `16_mona_gender_spotlights.py` | Gender heterogeneity + spotlight occupations |
| 17 | `17_mona_pctchange_figure.py` | Percentage change figures |
| 18 | `18_mona_eventstudy_corrected.py` | Corrected event study (full FE structure) |
| 19 | `19_mona_export_csv.py` | Export aggregated results for local use |

### Offline appendix outputs (local)

| Step | Script | Description |
|------|--------|-------------|
| 20 | `20_employment_age_yreg.py` | YREG annual employment (superseded by MONA) |
| 21 | `21_halfyear_posting_es.py` | Half-year posting event study |
| 22 | `22_stock_market_comparison.py` | Nasdaq vs OMXS30 comparison |

### Stata replication

| File | Description |
|------|-------------|
| `replication_stata.do` | Independent replication of posting DiD (reghdfe) |
| `mona_canaries_regression.do` | Independent replication of employer DiD (reghdfe) |

## Directory structure

```
├── src/                    Pipeline scripts (numbered 01-22)
├── data/raw/               Downloaded files (large JSONL gitignored)
├── data/processed/         Generated by pipeline
├── data/output/            Aggregated MONA outputs (non-restricted)
├── figures/                300 dpi PNGs and PDFs
├── tables/                 CSV + LaTeX
├── paper/                  LaTeX source (main.tex, appendix.tex, appendix_offline.tex)
├── onepager/               Policy one-pager
├── requirements.txt
├── LICENSE (MIT)
└── README.md (this file)
```

## Offline appendix

`paper/appendix_offline.tex` contains supplementary material moved from the Online Appendix during revision. It is **not part of the journal submission** — it preserves analysis for co-author review and replication transparency. The file compiles standalone with tectonic/pdflatex.

## License

Code: MIT License. Data: see individual source licences.

## Citation

```bibtex
@techreport{lodefalk2026samestorm,
  title={Same Storm, Different Boats: Generative AI and Young Workers Within Firms},
  author={Lodefalk, Magnus and L{\"o}thman, Lydia and Koch, Michael and Engberg, Erik},
  year={2026},
  type={Working Paper}
}
```

## AI disclosure

During the preparation of this work, the authors used Claude (Anthropic) for code development, data processing, and editorial suggestions. The authors reviewed and edited all content and take full responsibility for the publication.
