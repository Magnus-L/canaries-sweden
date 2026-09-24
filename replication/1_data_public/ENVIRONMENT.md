# Computational environment of the public tiers

Everything outside MONA (packs 0, 1, 2, 4 and 5) was run on:

- Hardware: Apple M2, 8 cores, 16 GB of memory; macOS 26.6.
- Python 3.12.9 with the packages pinned in `../requirements.txt`: pandas 3.0.3, numpy
  2.3.5, scipy 1.15.3, pyfixest 0.40.1 (the Poisson and OLS fixed-effects estimates),
  linearmodels 7.0 (the decile gradient and the event study), statsmodels 0.14.6 (the
  teleworkability split), matplotlib 3.10.8, pyarrow 24.0.0 (pandas' string backend),
  openpyxl 3.1.5 and xlrd 2.0.2 (the crosswalk workbooks), requests 2.34.2, tqdm 4.67.3,
  yfinance 1.2.0 (only for refreshing the market series), pillow 12.1.0 (only for the
  figure comparison in `0_verification`).
- R 4.6.0 with HonestDiD 0.2.8 (CRAN), for `2_postings/08_honestdid.R` alone.
- Disk: about 6 GB for the Platsbanken archives. Memory: no process exceeded 3.4 GB
  (maximum resident set size over the whole run); 16 GB is ample.
- Wall-clock time: 42 minutes for packs 1, 2 and 5 with the archives in place and the
  HonestDiD cache used; a cold computation of the HonestDiD bounds adds several hours.

No random numbers are drawn anywhere in these tiers, so no seed is set; every result is
deterministic given the inputs.
