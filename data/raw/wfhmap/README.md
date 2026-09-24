# WFH Map: remote and hybrid work in job postings (Hansen et al. 2023)

Source: WFH Map, https://wfhmap.com/data/ ("Category A", free immediate download, no sign-up).
File: `remote_work_in_job_ads_public_data.xlsx`, fetched from https://wfhmap.com/download/cat-a
on 24 Sep 2026 (curl, identifying User-Agent). HTTP Last-Modified 26 Jul 2026; the contents sheet
says "Current version of the data compiled on 25/07/2026 and covers January 2019 to Jun 2026".
SHA-256 82fd317495e9c3dcba52e45b61547781feb17d7386682e6d3167389230a4108f (387,538 bytes).
`wfhmap_data_page_2026-09-24.html` is the data page as fetched, kept as the record of the terms.

Measure: share of new job vacancy postings that explicitly offer the right to work one or more
days per week from home or another remote location (hybrid and fully remote pooled).

Sheets used by `revision/local/l52_hansen_wfh_horserace.py`:
- `us_occ_by_month`: United States, monthly, by 2018 SOC 3-digit minor group (96 groups), with
  posting counts N; January 2019 to June 2026.
- `us_occ_detailed`: United States, 2018 SOC detailed occupations (733), pooled 2017-2019 and
  2023-2026 only. There is no 2021-2022 column at the detailed level in the public file.
Also in the file: country by month (US, UK, Australia, Canada, New Zealand). Industry, city,
county and state tables and the 2-digit SOC series are "Category B" (sign-up); not fetched.

Lambert and Schindler (2026) use the detailed occupation, 2021-2022, pooled over four countries.
That cut is not in the public release; l52 uses the minor-group 2021-2022 share (headline) and
the detailed 2023-2026 share (robustness).

Citation requested by the providers: Hansen, S., Lambert, P. J., Bloom, N., Davis, S. J.,
Sadun, R., & Taska, B. (2023). Remote Work across Jobs, Companies, and Space (NBER Working Paper
No. 31007). https://doi.org/10.3386/w31007

Licence: the site states "free to download" and asks users to cite the working paper, but its
footer reads "© 2026 WFH Map. All rights reserved." and no licence grants redistribution. The
xlsx and the HTML page are therefore NOT committed to this public repository; re-download from
the URL above (the SHA-256 identifies the vintage used).
