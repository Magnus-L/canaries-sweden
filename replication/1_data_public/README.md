# Pack 1: public data

These four scripts turn the public sources into the files every posting estimate starts
from. Nothing here needs a permission of any kind. Paths are set in `../config.py`; the
large Platsbanken archives are located through `CANARIES_JOBADS_DIR`.

## Scripts, in run order

| Script | What it produces | Serves | Runtime |
|---|---|---|---|
| `01_download_platsbanken.py` | the Platsbanken annual archives 2020 to 2025 and the closed-quarter archives 2026-Q1 and 2026-Q2 in `CANARIES_JOBADS_DIR`; `--verify` compares each with the SHA-256 of the archive the paper used | every posting result | set by the connection (about 6 GB) |
| `02_process_platsbanken.py` | `data/processed/postings_ssyk4_monthly.csv`, advertisements and vacancies by four-digit SSYK 2012 occupation and month, rebuilt from the archives, and a month-by-month comparison with the frozen counts (`output/results/postings_ssyk4_rebuild_vs_frozen.csv`) | a check on the frozen counts | 5.5 minutes |
| `03_market_and_policy_series.py` | the OMXS30 and OMX Stockholm All-Share monthly indices (100 in February 2020) and the Riksbank policy rate by month | Figure 1; Online Appendix Figure A1 | seconds |
| `04_merge_and_classify.py` | `daioe_quartiles.csv` (the 2023 DAIOE generative-AI percentile and its quartile for 423 occupations) and the posting counts matched to it, 369 occupations | the exposure quartile of every posting estimate | seconds |

Runtimes were measured on an Apple M2 laptop with 16 GB of memory.

## The frozen occupation-by-month counts

Every posting estimate starts from `data/raw/postings_ssyk4_monthly_2026-02-24.csv`, the
occupation-by-month counts built on 24 February 2026 from the annual archives 2020 to 2025.
That build also appended advertisements from JobTech's live feed (JobStream), and the
extraction it read was not kept. `02_process_platsbanken.py` rebuilds the counts from the
archives alone. Over January 2020 to December 2025 the rebuild agrees with the frozen file
in 27,964 of 28,050 occupation-months and carries 50 advertisements fewer (4,586,156 against
4,586,206): one in October 2025, one in November and 48 in December, the advertisements the
live feed added at the end of the year. The frozen file is shipped so that the published
numbers reproduce exactly; to run everything on the rebuild instead, set
`CANARIES_POSTINGS_SSYK4=data/processed/postings_ssyk4_monthly.csv`. No month of the live
feed is appended after December 2025: the window to June 2026 takes January to June 2026
from the closed-quarter archives (`2_postings/03`). The 50 advertisements the live feed added
to October to December 2025 move the estimates of Equation (1) by less than 0.0001: on the
rebuild, $\hat\beta_1$ is $-0.1271$ and $\hat\beta_2$ is $-0.0593$ on the window to June 2026,
against $-0.1271$ and $-0.0593$ on the frozen counts, with standard errors equal to four
decimals.

## The archives

JobTech republishes an archive when it revises it, so a download made later need not be
byte-identical to the one the paper used. `data/DATA-MANIFEST.csv` lists the SHA-256 of each
archive the paper read, and `01_download_platsbanken.py --verify` checks a local copy against
it. The annual 2025 archive has been republished since the paper's copy was taken (24
February 2026): a copy taken on 22 July 2026 differs in its bytes but not in its content for
this analysis. Both hold 582,241 advertisements, every one passes the sample filter, and the
two give the same set of advertisement identifiers and the same counts in every
occupation-month.

## The policy rate

The Riksbank decisions are typed into `03_market_and_policy_series.py` from the bank's
announcements. The monthly series is the rate in force on the last day of each month, so a
decision taken on 28 April 2022 sets April's value.

## The stock indices

`data/raw/omxs30_daily.csv` and `omxspi_daily.csv` are the daily closes fetched from Yahoo
Finance on 18 September 2026; the month of the fetch is incomplete and is dropped, so the
monthly series run to August 2026. `--refresh` fetches all four market series again.
