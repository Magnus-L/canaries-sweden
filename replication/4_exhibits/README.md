# 4_exhibits: the register tables and figures, built from the exports

The employment results of the paper are estimated inside Statistics Sweden's
MONA environment (pack `3_register_mona/`). What leaves MONA are aggregates:
coefficients, standard errors, clustered covariance matrices, cell counts and
each run's own summary. The builders in this pack turn those exports into every
register table and figure of the paper and the online appendix. They need no
register access and no estimation software beyond Python with pandas and
Matplotlib; the whole pack runs in about half a minute.

```
python 4_exhibits/run_all.py        # every builder in paper order
python 4_exhibits/01_table1_headline.py [export_dir]   # one builder
```

Each builder reads the export folder named in its header (under
`3_register_mona/exports/`, listed with SHA-256 digests in
`EXPORT_RUNS.csv`) and writes to `output/tables/` or `output/figures/` under
the file name the manuscript includes. A builder may be pointed at another
export folder by giving it as the first argument.

## What the builders check before writing

No number in these tables is typed from a log. Each builder reads its estimates
from the export CSVs and, before writing, checks them against a second record
of the same fit: that every standard error is the square root of its own
diagonal in the exported covariance, that every printed cell matches the run's
own summary file where one exists, and that the identities a derived row must
satisfy hold (tau equals the later-period term minus the interim term; young
women equal the male term plus the female differential; a stacked difference
equals the difference of its arms). Table 1 is checked in addition against
script 97's own re-estimation of every tau it prints, a separate MONA run. A
builder that finds a disagreement stops and writes nothing. The typed values
that remain are listed under "Typed values" below; each is either a check or a
published statistic, and each has a row in `MANIFEST.csv`.

## Builders, in paper order

Exhibit numbers are those of the online appendix as compiled on 26 September
2026.

| Builder | Exhibit | Exports read (folder under `3_register_mona/exports/`) | Output | Time |
|---|---|---|---|---|
| `01_table1_headline.py` | Table 1 | `2026-09-23_0655_s82-partB_s83-partsBCD` (scripts 83, 80, 82); `2026-09-25_2250_s97` (the second record) | `table1_headline_v3.tex` | 1 s |
| `02_figure2_age_profile.py` | Figure 2 | `2026-09-25_1832_s95-s96-s98` (script 95) | `fig2_age_profile_v4.pdf` | 2 s |
| `03_figure3_quarterly_path.py` | Figure 3 | `2026-09-23_0917_s84`; `2026-09-21_2152_s68` (quarter axis) | `fig2_spreading_v3.pdf` | 2 s |
| `03_figure3_quarterly_path.py --monthly` | offline appendix (was OA Figure A6 until 26 September) | `2026-09-23_0917_s84` | `fig2_spreading_monthly_v3.pdf` | 2 s |
| `04_tab_estimation_sample.py` | OA Table A2 | `2026-09-21_0812_s68` (log); `2026-09-22_2232_s82-partA`; `2026-09-23_0655_s82-partB_s83-partsBCD`; `2026-09-23_1125_s85`; `2026-09-23_1407_s88` | `tableI2_sumstats_employment.tex` | 1 s |
| `05_tab_descriptive_bands.py` | OA Table A12 | `2026-09-23_1125_s85` | `tableA_descriptive_bands.tex` | 1 s |
| `06_fig_prepath.py` | OA Figure A7 | `2026-09-23_1234_s86` | `fig_prepath.pdf` | 2 s |
| `07_tab_prepath.py` | OA Table A23 | `2026-09-23_1234_s86`; `2026-09-23_0655_s82-partB_s83-partsBCD`; `2026-09-26_2103_s105` (script 105, bottom panel) | `tableA_prepath.tex` | 1 s |
| `08_tab_profile_bands.py` | OA Table A17 | `2026-09-23_0655_s82-partB_s83-partsBCD`; `2026-09-23_1125_s85` | `tableA_profile_split65.tex` | 1 s |
| `09_tab_industry_credit.py` | OA Table A24 | `2026-09-23_0655_s82-partB_s83-partsBCD` (scripts 80, 73 within 83; tau from the exported covariances) | `tableA_industry_credit.tex` | 1 s |
| `10_tab_cluster_industry.py` | OA Table A26 | `2026-09-23_0655_s82-partB_s83-partsBCD` (script 80 within 83) | `tableA_cluster_industry.tex` | 1 s |
| `11_tab_continuous_profile.py` | OA Table A18 | `2026-09-20_2148_s61-s63` (script 63) | `tableA_age_profile.tex` | 1 s |
| `12_fig_first_stage.py` | OA Figure A5 | `2026-09-22_2333_s83-partA` | `figA2_first_stage_v3.pdf` | 2 s |
| `13_tab_fixed_contrasts.py` | OA Table A16 | `2026-09-23_0917_s84` | `tableA_fixed_contrasts.tex` | 1 s |
| `14_tab_window.py` | OA Table A14 | `2026-09-23_0655_s82-partB_s83-partsBCD` | `tableA_window.tex` | 1 s |
| `15_tab_gender_split.py` | OA Table A19 | `2026-09-23_1352_s87` | `tableA_gender_split.tex` | 1 s |
| `16_tab_education_mix.py` | OA Table A20 | `2026-09-23_1352_s87` | `tableA_education_mix.tex` | 1 s |
| `17_tab_contrast_by_track.py` | OA Table A25 | `2026-09-23_1407_s88`; `2026-09-23_0655_s82-partB_s83-partsBCD`; `2026-09-23_1352_s87` | `tableA_contrast_by_track.tex` | 1 s |
| `18_tab_score_precision_and_coverage.py` | OA Tables A27 and A34 | `2026-09-22_2232_s82-partA`; `2026-09-23_0655_s82-partB_s83-partsBCD` (tau from the exported covariances) | `tableA_size_reliability.tex`, `tableA_occ_coverage.tex` | 1 s |
| `19_tab_register_coverage.py` | OA Tables A30, A31, A32 | `2026-09-18_1640_s40-s47`; `2026-09-20_1735_s39-s41`; `2026-09-18_1736_s45`; `2026-09-22_2327_s82-partC` | `tableIV1_coverage.tex`, `tableIV2_vintage.tex`, `tableIV3_backtest.tex` | 1 s |
| `20_fig_backtest.py` | OA Figure A8 | `2026-09-18_1736_s45` | `figA1_asof_backtest.pdf` | 2 s |
| `21_tab_uncounted.py` | OA Table A40 | `2026-09-24_1037_s93` | `tableA_uncounted.tex` | 1 s |
| `22_tab_headline_components.py` | OA Table A15 | `2026-09-23_0655_s82-partB_s83-partsBCD` (scripts 82, 83, 80); `2026-09-23_1352_s87`; `2026-09-26_2103_s105` (script 105, industry SEs) | `tableA_headline_components.tex` | 1 s |
| `23_tab_final_checks.py` | OA Table A21 | `2026-09-25_1832_s95-s96-s98` (scripts 95, 96); `2026-09-25_2250_s97` | `tableA_final_checks.tex` | 1 s |
| `24_tab_unlinked.py` | OA Table A41 | `2026-09-26_0827_s100-s101` (script 100) | `tableA_unlinked.tex` | 1 s |
| `25_tab_nonmatch.py` | OA Table A42 | `2026-09-25_1459_s99` (script 99) | `tableA_nonmatch.tex` | 1 s |
| `26_tab_backtest_arms.py` | OA Table A33 | `2026-09-26_1040_s98b`; `2026-09-25_1832_s95-s96-s98` (the same fits, exported first) | `tableIV4_backtest_arms.tex` | 1 s |
| `27_fig_prepath_female.py` | OA Figure A6 | `2026-09-25_2250_s97` (script 97) | `fig_prepath_female.pdf` | 2 s |
| `28_tab_payment_rule.py` | OA Table A39 | `2026-09-26_1909_s104` (script 104) | `tableA_payment_rule.tex` | 1 s |

Figures are written as PDF (the file the manuscript includes) and as PNG at
300 dots per inch. The remaining exhibits of the online appendix (Figure 1,
the posting tables and figures of Parts II and V, the published-aggregates
table A29 and the occupation-mix table A22) are built from public data in
packs 2 and 5. The monthly diagnostic of `03 --monthly` moved from the online
appendix to the offline appendix on 26 September 2026 and is still built.

## Typed values

These values are written into a builder rather than read from an export:

- `03_figure3_quarterly_path.py`: Statistics Sweden's share of enterprises
  with ten or more employees using AI, 10 per cent (2023) and 25 per cent
  (2024), a published statistic drawn in the lower panel.
- `09_tab_industry_credit.py`: in the note, "under 2 per cent" of employers
  coded from another year and leverage held by "85 per cent of each panel";
  the exports give 1.7 and 1.9 per cent, and 85.0 and 84.3 per cent. The
  builder still checks the leverage coverage and median split against the
  run's summary before writing.
- `10_tab_cluster_industry.py`: the industry-cluster counts (260, 263, 260),
  checked against the export.
- `12_fig_first_stage.py`: the two first-stage rows the text quotes (20.9
  points, SE 1.5, t 14.4, 3,587 firms; 22.5 points, SE 2.5, t 9.0, 2,285
  respondents); a check only, the figure is drawn from the export.
- `14_tab_window.py`: the eight estimates the appendix paragraph states; a
  check only.
- `17_tab_contrast_by_track.py`: in the note, ICT's shares of exposed
  employers' young women and men, 2.9 and 6.8 per cent (0.029 and 0.068 in
  `occ_route_education_mix_by_sex.csv`).
- `18_tab_score_precision_and_coverage.py`: in the coverage note, 2.5 per cent
  of resolved codes from a year before 2019 (a share of codes) and 7.4 per cent
  of the average employer's codes (the figure script 82's summaries print);
  Table 1's tau at the reported floor (-0.0399 (0.0102), -0.0403 (0.0067)), a
  check only.
- `19_tab_register_coverage.py`: in the backtest note, +0.019, -0.288, -0.307
  and "about -0.17" for the submitted design; in the vintage note, the ranges
  "500 to 8,000" and "2.1 to 2.3 million" person-months, checked against the
  export to the nearest hundred and tenth of a million (the export runs from
  488 to 7,837 and from 2.10 to 2.31 million).
- `23_tab_final_checks.py`: the employer counts of the panel titles (104,333
  and 104,217), checked against the export; the leverage medians, the
  third-quarter term and the score's standard deviation in the note are read
  from the exports and summaries.
- `24_tab_unlinked.py`: Table 1's tau and the female differential's tau as
  the gate the builder requires (-0.0399 (0.0102), -0.0714 (0.0109)); the
  difference row of Panel A is the difference of the shares as printed, to
  three decimals.

The table notes are the text the online appendix prints, as it stood on
26 September 2026; a rebuilt table is identical to the printed one once the
co-author markup still carried by two printed files (`table1_headline_v3.tex`,
`tableA_headline_components.tex`'s comment header) is accepted, which
`0_verification/check_manifest.py` does before comparing.
