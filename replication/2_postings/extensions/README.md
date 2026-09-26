# Extensions: the two remote-work estimation scripts and their results

Online Appendix Table A4 (`22_tab_remote_measures.py`) and the remote-work paragraphs of
Section 3 of the paper and of Online Appendix II.3 rest on two estimation scripts that ran in
the authors' research repository, not in this package, because their inputs are not shipped.
They are kept here as they ran, with the occupation-level results they wrote, so that every
number the table and the text carry can be traced to a file and the estimation can be
repeated by a replicator who assembles the inputs.

| Script | Estimates | Inputs it needs beyond the package |
|---|---|---|
| `l50_remote_work_horserace.py` | the share of an occupation's 2021 to 2022 Platsbanken advertisements that offer remote or hybrid work, from a keyword rule on the advertisement text; Lambert and Schindler's horse race across occupations and within employers with that measure; the split of employers by their own 2021 to 2022 remote advertising | the full advertisement text of every archive 2020 to 2026-Q2 (`1_data_public/01` downloads them); the employer-by-month advertisement counts of Part V (`CANARIES_FIRM_CUBE`, see `2_postings/README.md`); two modules of the AI-Econ Lab's advertisement monitor (`bulk_pipeline_v11.ad_text`, `dedup_key`, `ENTRY`; `firm_dimension_extract.norm_orgnr`), which give the text, the content-hash de-duplication and the organisation-number rule the Part V counts were built with |
| `l52_hansen_wfh_horserace.py` | the same horse race with the measure Lambert and Schindler (2026) use, the share of remote or hybrid postings in the United States (Hansen et al. 2023, WFH Map), crosswalked SOC 2018 to SOC 2010 to ISCO-08 to SSYK 2012 | the WFH Map public release `remote_work_in_job_ads_public_data.xlsx` (<https://wfhmap.com/data/>, Category A, fetched 24 September 2026, SHA-256 `82fd3174…4108f`), whose terms do not grant redistribution; the BLS SOC 2010 to 2018 crosswalk; the SOC to ISCO and ISCO to SSYK crosswalks of `data/raw/`; and the ad-level extract of `l50` for the entry-level rows |

The paths in both scripts are those of the research repository (`revision/local/`,
`revision/config.py`, `lab-infrastructure/ai-monitor/`); they are not routed through
`config.py`, and `run_public.sh` does not run them. Both check, before anything is written, that
their AI-only baselines reproduce the paper's: Equation (1) on the occupation panel (`03`,
$-0.1271$ and $-0.0593$ on 28,084 cells) and the within-employer design (`14`, $-0.158$).
`22_tab_remote_measures.py` repeats those two checks on the result files before it builds the
table.

## Results shipped (`results/`)

Occupation-level and estimate-level files only. The employer-level remote-work measure and
the hand-check snippets of advertisement text that `l50` also wrote are not shipped (the first
is a firm-level derived file, the second raw advertisement text).

| File | Content | Read by |
|---|---|---|
| `l50_remote_horserace.csv` | Equation (1) and its variants with the Platsbanken measure, all and entry-level advertisements | `22`; `MANIFEST.csv` |
| `l50_remote_within.csv` | the within-employer design with the Platsbanken measure, the finer panel and the employer split | `22`; `MANIFEST.csv` |
| `l50_remote_correlations.csv` | correlations of the measures across the 369 panel occupations, and the distribution of the remote share | `22`; `MANIFEST.csv` |
| `l50_remote_occ_measure.csv` | the occupation-level remote share, 2021 to 2022, with the DAIOE score and 2024 employment | record |
| `l50_remote_coverage.csv`, `l50_remote_validation.csv`, `l50_remote_monthly.csv` | coverage of the structured remote-work field by year, the text rule's agreement with it where both exist (2023 to June 2026), and the monthly series | record (the hand-read precision and recall quoted in II.3 are not in a file; `MANIFEST.csv` marks them pending) |
| `l52_hansen_horserace.csv`, `l52_hansen_within.csv`, `l52_hansen_correlations.csv` | the same three files with the Hansen et al. measure | `22`; `MANIFEST.csv` |
| `l52_hansen_occ_measure.csv`, `l52_hansen_coverage.csv` | the crosswalked Hansen shares by SSYK occupation and their coverage of the panel | record |
