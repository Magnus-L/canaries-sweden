# Pack 5: Statistics Sweden's published occupational statistics

Three checks that use the occupational register only as Statistics Sweden publishes it, in
aggregate, where the agency and not the authors codes each worker's occupation. All read
responses of Statistics Sweden's open API (table YREG54BAS) saved in `data/raw/`, so they
reproduce without a connection; each can query the API again.

| Script | What it produces | Exhibit | Runtime |
|---|---|---|---|
| `01_published_age_gap.py` | for each published age band, the log employment gap between top-quartile occupations and the rest, and its change since 2022 (2020 to 2024) | Online Appendix Table A24 | seconds |
| `02_occupation_mix_by_sex.py` | the employment-weighted DAIOE exposure of women's and men's occupations by age band, 2024 | Online Appendix Table A18; one sentence of Section 3 | seconds |
| `03_exposure_measure_agreement.py` | how alike DAIOE and the Eloundou score rank occupations: their correlation across the 393 occupations both score (checked against the register run's own number) and the employment-weighted agreement on the top-quartile cut, 2024 | the data section's sentence on the two measures (Online Appendix Part I) | seconds |

`01` reads the quartiles of `1_data_public/04`; `02` reads `3_register_mona/inputs/daioe_quartiles.dta`,
the same file the register scripts score from, so the exposure and the top quartile are the
ones the employment design uses. The first check carries three caveats, which its summary
file states: the register behind the series moves from RAMS to BAS at reference year 2022,
on the treatment boundary; the published series stops at 2024; and the published age bands
are not the paper's. The second is a national distribution, not a within-employer one.
