# Lane 6: attenuation is mild, and young hiring breaks in 2025H1

**20 September 2026, 00:45.** Digest of `57_summary.txt` and `56_summary.txt`.

## 1. The attenuation worry is largely answered, and the answer is reassuring

lambda(y), the share of the 2019 exposure signal still present:

| year | ALL | 22-25 | 26-30 | 31-34 | 35-40 | 41-49 | 50+ |
|---|---|---|---|---|---|---|---|
| 2019 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2021 | 0.917 | 0.835 | 0.885 | 0.906 | 0.921 | 0.935 | 0.937 |
| 2022 | 0.890 | 0.817 | 0.846 | 0.869 | 0.894 | 0.912 | 0.917 |
| 2023 | 0.890 | 0.829 | 0.871 | 0.870 | 0.885 | 0.906 | 0.910 |

The 2019 assignment loses about a tenth of its signal by 2021 and then
**stops decaying**: 0.890 in 2022 and 0.890 in 2023, and at 22-25 it actually
ticks back up, 0.817 to 0.829. A firm's age-specific occupation mix is far more
persistent than feared.

Consequence: the register-immune designs are NOT going blind at the end of the
window. Correcting 47L's +0.007 for attenuation at 22-25 gives +0.007/0.83 =
+0.008. The null is a null, not a washed-out effect. The standard-error penalty
of correcting is 1/0.83, about 20 per cent, not the 5.5x the education route
would have cost.

The two baselines agree on the direction of everything, and on hires the 2022
baseline finds LESS (-0.0100) than the 2019 one (-0.0167), which is the opposite
of what attenuation would produce.

| outcome | 2019 baseline | 2022 baseline |
|---|---|---|
| hires | -0.0167 | -0.0100 |
| seps  | -0.0171 | -0.0483 |
| stock | +0.0070 | -0.0005 |

## 2. The event studies: nothing in the stock, a break in young hiring

The 22-25 differential has a strong and obvious H1/H2 seasonal, because the
reference is 2022H1 and young hiring at exposed firms is relatively stronger in
H2. Every H2 coefficient therefore carries the seasonal. **Compare like with
like.**

22-25 differential, H1 half-years only:

| | hires | seps | stock |
|---|---|---|---|
| 2019H1 | +0.0057 | +0.0316 | +0.0009 |
| 2020H1 | +0.0114 | +0.0286 | +0.0068 |
| 2021H1 | +0.0018 | +0.0156 | -0.0038 |
| 2022H1 | 0 (ref) | 0 (ref) | 0 (ref) |
| 2023H1 | +0.0271 | +0.0292 | +0.0128 |
| 2024H1 | +0.0132 | +0.0198 | -0.0009 |
| **2025H1** | **-0.0689** | +0.0406 | -0.0072 |

Six H1 half-years sit between -0.000 and +0.027 on hires. The seventh is
**-0.069**, a break of roughly eight log points from the H1 norm and the only
negative value in the column across the whole panel, H1 and H2 together.

Separations show no such break: 2025H1 is +0.041, the highest H1 reading, so
the outflow did not fall. The stock shows nothing, -0.007.

**That combination is the entry-level story in its textbook form: the inflow
adjusts, the outflow does not, and the stock has barely begun to move one
half-year in.** It also explains why every stock-based design in this round
returned zero. They were measuring the slow margin, and the fast one only
turned in 2025.

## 3. What this is NOT yet

- **One half-year.** The script's own read rule says an isolated half-year is
  noise, and that rule was written before the number existed.
- **2025 is preliminary AGI** (`_prel`, six months). The definitive file may
  move it.
- **No standard errors in the summary.** `dynamics_young.csv` has them and must
  be read before this is quoted.
- **The H1-versus-H1 comparison was chosen after seeing the data.** The
  seasonality is mechanical and obvious, so the slice is defensible, but it was
  not pre-specified and the honest fix is to absorb it: add an
  `exposure x half-of-year` control so the event-study coefficients are purged
  of the exposure-specific seasonal, then re-read 2025H1 against a clean
  baseline.
- **The pooled (all-ages) hires path also ends low**, -0.0959 at 2025H1 against
  a pre-period range of -0.019 to -0.058, so part of the 2025 move is not
  specific to the young.

## 4. What to do next

1. Read the standard errors in `dynamics_young.csv` before anything else.
2. Re-run 56 with an exposure-by-half-of-year control, so the 2025H1 reading
   does not depend on a post-hoc slice. Cheap: the panels are cached.
3. Ask whether the 2025 definitive AGI delivery is available, since the whole
   finding currently rests on preliminary data.
