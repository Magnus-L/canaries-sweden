# Same Storm, Different Boats: Generative AI and the Age Gradient in Hiring

Replication package for Lodefalk, Löthman, Koch, and Engberg (2026), "Same Storm, Different Boats: Generative AI and the Age Gradient in Hiring."

## Overview

We examine whether the widely discussed divergence between stock prices and job postings (the "scary chart") reflects AI displacement of labour demand, or macroeconomic tightening. Using 4.6 million job ads from Sweden's Platsbanken (2020–2026) matched to the DAIOE generative AI exposure index, we find that the posting decline aligns with the Riksbank's rate hike rather than AI. However, an employer-level difference-in-differences using full-population register data reveals a monotonic age gradient: employment of 22–25 year olds in AI-exposed occupations fell by 6.5% after ChatGPT, while employment of workers over 50 rose by 1.5%.

## Data availability

| Dataset | Source | Access |
|---------|--------|--------|
| Platsbanken historical ads | [JobTech Development](https://data.jobtechdev.se/annonser/historiska/) | Open (CC0) |
| DAIOE genAI exposure | [Engberg et al. (2024)](https://doi.org/XXX) | Open |
| OMXS30 / OMXSPI prices | Yahoo Finance (`^OMX`, `^OMXSPI`) | Open |
| Riksbanken policy rate | [riksbank.se](https://riksbank.se) | Open |
| Indeed US postings | [Indeed Hiring Lab](https://github.com/hiring-lab/job_postings_tracker) | Open |
| S&P 500 / Nasdaq prices | Yahoo Finance (`^GSPC`, `^IXIC`) | Open |
| Dingel-Neiman teleworkability | [Dingel & Neiman (2020)](https://github.com/jdingel/DingelNeiman-workathome) | Open |
| Eloundou GPT exposure | [Eloundou et al. (2024)](https://doi.org/XXX) | Open |
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
  title={Same Storm, Different Boats: Generative AI and the Age Gradient in Hiring},
  author={Lodefalk, Magnus and L{\"o}thman, Lydia and Koch, Michael and Engberg, Erik},
  year={2026},
  type={Working Paper}
}
```

## AI disclosure

During the preparation of this work, the authors used Claude (Anthropic) for code development, data processing, and editorial suggestions. The authors reviewed and edited all content and take full responsibility for the publication.
