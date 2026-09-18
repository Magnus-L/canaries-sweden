# revision/sim -- the simulation study

Ranks education-exposure designs by how close they get to the truth under
register lag, rather than by how they look. Written 19 September 2026 after
script 47b showed that the education design fails the as-of backtest worse
than the occupation design it was meant to replace, and that no coverage
statistic detects the failure.

## Run it

```bash
bash revision/sim/when_export_lands.sh <path to script 50's output_50>
```

That does four things: the predictive validation of every design on real
data, the acceptance tests bound to the measured moments, the simulation
study, and the ranking. Without the export the simulator still runs on named
placeholders (`python3 run_sim.py --quick`), and the acceptance tests say
loudly which fields are guesses.

## What is where

| File | What it does |
|---|---|
| `calib.py` | turns script 50's export into generator parameters; every field is either measured or a NAMED placeholder, and `Calibration.source` says which |
| `dgp.py` | the person-level generator, the register layer, and the emitters that produce exactly the frames 47h consumes |
| `run_sim.py` | the runner: scenarios x register settings x seeds x designs, metrics, ranking |
| `test_sim.py` | acceptance tests, structural (always) and calibrated (only with the export) |
| `when_export_lands.sh` | all of the above in one command |
| `../local/l13_validate_edu_designs.py` | the real-data predictive validation, which needs no simulation at all |

## The two things a reader should know

**The truth is the oracle, not gamma.** Designs are scored against the
estimate a perfect exposure assignment produces on the same sample, because
an education measure is attenuated by construction and comparing it to the
structural gamma would penalise every design equally for something none of
them can fix. A common small-sample offset also cancels in that comparison.

**The DGP draws occupations from the measured P(ssyk4 | group, experience
band, year) matrix**, so a design scoring by (group, band) is estimating the
DGP's own conditional mean and will fit well BY CONSTRUCTION. The study
therefore ranks on robustness to lag -- artefact, false-pass rate, null
false-decline -- and never on fit. Persistent person effects keep the mapping
stochastic rather than deterministic.

## Estimates of cost

At 60,000 persons and 1,500 employers one draw is about 8 to 12 minutes
(8 designs x 5 fits plus the oracle). The full grid is 3 scenarios x 3
register settings x 5 seeds = 45 draws, so roughly 7 to 9 hours; `--resume`
makes a killed run cheap, and `--quick` is one draw. Scale with
`--n-persons`, and scale `--min-cell` with it, since the scoring floor is a
count of people.
