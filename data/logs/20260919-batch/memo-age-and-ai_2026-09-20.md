# Measuring what AI has done to workers of different ages

**A memo on the problem we ran into, what I did about it, and where it leaves us.**
20 September 2026, written against the complete MONA export.

---

## Why I am writing this

We set out to answer a narrow question with unusually good data. Within a single
employer, in a single month, do young workers in more AI-exposed work fare
differently from young workers in less exposed work? Sweden lets us ask it
properly. We observe every employment spell monthly, we can hold the employer
fixed, and we have occupational detail at four digits.

Along the way the measurement gave out. Not in a small way. Not in a way a
robustness table absorbs.

I want to set out what happened, because the episode taught me more about how to
do this kind of work than the original result did, and because most of what I
learned generalises well beyond this paper. Read it as one colleague to another.
I have tried to show the reasoning rather than only the conclusions, including
the places where I reached a conclusion confidently and had to withdraw it.

---

## 1. The problem

Our headline was that employment of 22 to 25 year olds in the most AI-exposed
quartile of occupations fell by about 17 log points relative to less exposed
young workers in the same firm, after ChatGPT. The estimate was precise, the age
profile was monotone, and the placebo on workers over 50 was near zero. It looked
like a finding. It was not.

Occupation is not observed monthly. It comes from the annual occupation register,
and that register arrives late. Our panel runs to mid 2025. Occupation codes stop
in 2023. The last two years therefore classify people by what they were doing up
to two years earlier.

This is not exotic; it is the ordinary condition of register work, and everybody
who uses occupational data lives with some version of it. The question is whether
it matters enough here to change what we conclude, and the useful feature of the
problem is that there is a clean way to find out rather than a judgement to be
made.

---

## 2. The test that settled it

Take a year T, discard every occupation code assigned after it, and let the later
years inherit stale codes exactly as 2024 and 2025 do. Then re-estimate, and
compare with the estimate using the true codes, which for those years we hold.
The gap is what the lag manufactures out of nothing.

The logic is worth carrying to other projects. We are not asking whether the
codes for 2025 are accurate, which we cannot know and will not know for two
years; we are reproducing the ailment in a period where we also hold the
diagnosis, and measuring how far the two diverge. The design converts an
unanswerable question about the present into an answerable question about the
past. That trade is available far more often than it is taken.

For 22 to 25 year olds:

| truncation | true codes | stale codes | manufactured |
|---|---|---|---|
| T = 2021 | +0.019 (0.013) | −0.288 (0.017) | **−0.307** |
| T = 2022 | +0.018 (0.011) | −0.145 (0.012) | **−0.163** |

The artefact is as large as the finding, and at the longer truncation larger.
Note also the true arms: with correct codes the estimate is positive and
insignificant at both truncations, which is a second and independent reason to
doubt the headline.

One caution, because I was sloppy about it once. You cannot subtract the artefact
from the headline and call the remainder the truth. The backtest runs on a
shorter panel with a different share of corrupted years, so the quantity does not
transfer. What it establishes is weaker and more useful: the lag alone can
produce a coefficient of this size, so the headline is not identified. That is
all it establishes. It does not tell us the true value.

---

## 3. The repair that failed, and why I am glad we tried it

The obvious substitute is education. It is recorded earlier and moves more
slowly, and there is a good recent literature assigning exposure through it. We
built eight variants, from a plain replication through to designs using the
recency of qualification and the field of current enrolment.

Then I made the mistake this project has punished more than once, and I set it
out because it is the sort of error that looks like diligence while it is
happening. I judged the designs by the size of their artefact.

The horse race, run to completion, closed all eight:

| design | artefact T2021 | artefact T2022 |
|---|---|---|
| OL_exact | −0.115 | −0.130 |
| OL_daioe | −0.162 | −0.145 |
| fresh_stock | −0.163 | −0.145 |
| entrant_share | −0.336 | −0.229 |
| entrant | −0.384 | −0.287 |
| enrol | −0.385 | −0.288 |
| expband | −0.442 | −0.407 |
| full | −0.493 | −0.355 |

Read the ordering, because it is the interesting part and I did not predict it.
**The more refined the design, the worse the artefact.** The crude one is least
bad; the design combining every refinement is worst, at −0.49. Each refinement
conditions on another finely measured characteristic, and every one of those is
also stale, so finer conditioning multiplies the misclassification rather than
reducing it. The over-fifties placebo stays clean throughout at −0.001 to −0.002,
and the artefact shrinks monotonically with age, which is what education going
stale predicts.

A simulator calibrated on moments measured in our own registers then showed why
none of this could be repaired. Under a world with no effect at all, the
education designs report an age gap between the young and the over-fifties of
about −0.31. Under a world with a true gap of −0.15, they report about −0.34.
Fit a line and you get

> measured gap = −0.31 + 0.18 × true gap

against the same estimator given correct codes, which returns

> measured gap = −0.02 + 0.72 × true gap

The second line describes a well-behaved estimator with ordinary attenuation.
The first does not describe an estimator at all in any useful sense, since it
reports roughly the same number whether the truth is zero or substantial.
However precisely we measure that number, it is not telling us about the world.

Two lessons, and the second is the one I would tattoo on a student.

A large bias and a useless estimator are different problems, and the distinction
is not academic. A bias that is constant can be differenced away, and much of
applied work quite properly does so. However, the bias here is worst precisely on
the age contrast, because misclassification pushes the young estimate down and
the old estimate up, so the gap is inflated from both ends. It lives inside the
very difference we want to take.

And **you cannot correct your way out of this.** The instinct is to divide by the
pass-through and inflate the standard errors accordingly. That is legitimate when
measurement error is classical, which is to say when a null world returns a null.
Ours returns −0.31. Dividing that by 0.18 gives roughly −1.7, which is not an
estimate of anything. Before applying an attenuation correction, simulate a null
and look at the intercept. If it is not zero, the correction amplifies the bias
instead of removing it. Check the intercept first. It takes an hour.

So the right response to a low pass-through is not to correct it. It is to
redesign until it is high.

---

## 4. The repair that worked

The principle is simple once stated. **Measure exposure once, before the shock,
using data that has already arrived, and never measure it again.**

For each employer and each age band, compute the average AI exposure of the
occupations that group actually held in 2019. Freeze it. From then on a worker
needs only a birth year and a payslip, both of which arrive monthly and on time.
The occupation register is never consulted again and the education register is
not used at all. Whatever the lag does to later vintages, it cannot reach this
measure. The measure was finished before the lag existed.

Absorb employer by month, employer by age, and month by age. The comparison is
then made inside the firm, against its own other age groups.

The design has a real cost and I want to name it plainly rather than bury it in
an appendix. Identification comes from comparing firms whose young workers were
differently exposed at baseline, so it requires the corresponding parallel-trend
assumption on the multiplicative scale.

---

## 5. What we find

**On the employment stock, a tight bound and nothing more.** Across six
specifications, including two coverage restrictions and a control for the expiry
of the youth payroll-tax reduction in April 2023, the estimate sits between
−0.002 and +0.010 with standard errors around 0.009, on panels of eleven to
thirty-eight million cells. The implied interval runs from about −1 per cent to
+2 per cent. That is not a failure to find something; it is a tight bound on how
large any effect on the stock can be, and it is tighter than anything obtainable
from survey or vendor data.

**At the paper's own estimand, a consistently negative estimate that does not
reach significance.** The within-employer triple difference compares young
workers with older ones inside the same employer in the same month, and no young
worker is ever classified: exposure comes from the education mix of the firm's
incumbents aged 31 and over in 2019.

| | estimate | SE | t | artefact |
|---|---|---|---|---|
| OL_daioe, true codes | −0.0132 | 0.0111 | −1.19 | +0.005 |
| entrant, true codes | −0.0153 | 0.0114 | −1.34 | +0.001 |

The artefact of one to five thousandths, against a threshold of 0.05, makes this
the cleanest measurement in the project by a wide margin. All six specifications
are negative, averaging −0.012. None is distinguishable from zero. The interval
rules out a decline larger than about 3.5 per cent, where the original claim was
16.

Clean and underpowered are the same property here rather than two problems. The
design absorbs employer by month, employer by age and month by age, which is why
the artefact is tiny and also why little variation remains.

**The firm-level education design passes the rule decisively and finds no
decline at any age.** Its artefact at 22-25 is −0.008, six times inside the
threshold, and under half a point at every other age. Aggregating a mismeasured
variable over a whole workforce washes the misclassification out, which is a
cheaper fix than anything else we tried. What it finds is positive everywhere,
from +0.010 at 41-49 to +0.063 at 50+, with +0.024 at 22-25.

**And here is the tension I am not going to paper over.** The frozen-exposure
design reports its own age gradient, and that gradient does not put the young
worst. At 22-25 it gives +0.008 in the main variant and −0.005 in the shrunk one,
neither distinguishable from zero, while 41-49 is significantly negative at
−0.019 and −0.031 and the over-fifties are significantly positive. Two clean
designs, two different age profiles. I do not have a reconciliation, and until
there is one, any sentence claiming the young are the affected group rests on 47j
and not on the whole family.

---

## 6. The objection you should be raising

If you have been reading carefully you should be uneasy. Exposure is frozen in
2019, so by 2025 it is a six-year-old proxy for who is exposed today, and a stale
proxy in a continuous regressor attenuates the coefficient towards zero. The
design is therefore weakest exactly where the effect is most likely to be. That
is a serious objection and not a footnote, and it is the same disease as the
register lag wearing a different coat.

So we measured it. Rebuild the exposure measure contemporaneously in every year
with its own register and regress it on the 2019 version across firm-age cells.
The slope is the share of the original signal that survives.

| | 2019 | 2021 | 2022 | 2023 |
|---|---|---|---|---|
| all ages | 1.000 | 0.917 | 0.890 | 0.890 |
| 22 to 25 | 1.000 | 0.835 | 0.817 | 0.829 |

The signal decays by about a tenth in the first two years and then stops. At the
youngest ages it recovers slightly. A firm's age-specific occupational
composition is far more persistent than I expected, and the objection does not
survive its own measurement.

This matters in three ways. The nulls are real nulls, since correcting the stock
estimate moves it from +0.007 to +0.008. The standard-error penalty for
correcting is about a fifth rather than the four hundred and fifty per cent the
education route would have demanded. Moreover, re-running the whole design on a
2022 baseline, three years closer to the outcome years and still before ChatGPT,
moved the hiring estimate towards zero rather than away from it, which is the
opposite of what attenuation would produce.

---

## 7. The timing, which is where the evidence is strongest

Every estimate above defines the post period as the ChatGPT launch. That pools
thirteen months of 2023 with eighteen of 2024 and 2025. If the labour market
responded to firms adopting rather than to a model shipping, that averages an
untreated period with a treated one and attenuates the coefficient for reasons
that have nothing to do with the world.

Sweden lets us date this rather than assert it. SCB's survey of enterprises with
ten or more employees puts AI use at **10.4 per cent in 2023, 25.2 in 2024 and
35.0 in 2025**. Adoption two and a half times over between the 2023 and 2024
observations, the largest jump in the series, almost all of it after the date we
had been calling the treatment.

Four dates, fixed in the source before the run and all reported:

| assumed start | anchor | stock | SE | t |
|---|---|---|---|---|
| 2022-12 | ChatGPT launch | −0.0017 | 0.0108 | −0.16 |
| 2023-04 | quarter after GPT-4 | −0.0047 | 0.0109 | −0.43 |
| 2023-11 | Copilot general availability | −0.0094 | 0.0105 | −0.90 |
| **2024-01** | **SCB's adoption jump** | **−0.0179** | 0.0105 | **−1.71** |

The estimate grows monotonically as the treatment is dated later while the
standard error stays flat. That is the signature of a real late effect diluted by
an early start date, and it is not something noise produces neatly in one
direction across four pre-specified anchors. Dated where the adoption data put
it, the effect is −1.8 per cent and significant at ten per cent. The hiring
profile does the same thing across all seven grid points, from −0.015 at the
launch to −0.041 by mid-2024, and separations stay flat throughout, which is an
inflow story rather than a scale effect.

Two things I will not over-read. Nothing reaches five per cent. And the profile's
minimum sits at the endpoint rather than in the interior, so a trend and a break
look identical here; the data cannot yet date the effect, only say that later
dating helps.

---

## 8. What is not settled

The 2025 half-year reading is not a finding. Our pre-committed rule required it
to be negative in both specifications, twice its standard error in both, and
larger than every pre-period coefficient. It is negative in both and larger than
every pre-period coefficient in both. It fails on precision: −0.075 against a
standard error of 0.079, and −0.049 against 0.067. The rule was written before
the numbers existed and it refused them.

Worth noting that once the seasonal is handled properly the pre-period is flat,
with a largest pre-2022H2 coefficient of 0.0125 against 0.064 in the raw series,
and 2025H1 is six times anything before it. The shape is right; the precision is
not.

2025 is the preliminary file and stops in June. There is no definitive 2025
delivery and no occupation register for 2024, both confirmed directly against the
database today. The final month, June 2025, returns −0.43 with a standard error
of 0.26 and is an incomplete-file artefact rather than a result.

The fresh-code panel, which restricts to worker-months whose occupation code was
assigned in the observation year, gives −0.033 at 22-25 with t of −3.03, the only
negative age band, and passes its own placebo. I am not going to lead with it.
The share of young workers who are freshly coded moves by −0.016 in the exposed
quartile relative to the rest, and that movement has almost exactly the same age
profile as the estimated effect, so selection into the sample and the effect
cannot be separated by anything we have.

And one loose end. Our reimplementation of the education design does not
reproduce the earlier script's as-of arm, and the difference is not the cascade
correction I had assumed, because reproducing that correction exactly changes
nothing. The earlier script is the one with the known defect and the simulator
sides with the newer one, so the analysis proceeded with the discrepancy recorded
in the output rather than quietly resolved in favour of the answer I prefer. It
should be closed before publication.

---

## 9. What I would take from this

**Design so that the measurement is finished before the treatment begins.** The
single move that rescued this project was fixing exposure in a pre-period and
never touching it again, so that everything arriving late became irrelevant by
construction rather than by assumption. That is worth more than any correction.

**Judge a design by whether it moves when the truth moves, not by the size of its
bias.** Simulate a world with no effect and a world with a known effect, then look
at the difference between what your estimator reports in the two. If it barely
moves, no amount of correction will help. A small standard error on an estimator
like that is a warning rather than a comfort.

**Aggregate a mismeasured variable before using it.** The same education data
that produced a −0.36 artefact at worker level produced −0.008 at firm level.
Nothing changed but the unit of aggregation.

**Match the margin to the mechanism.** We spent a great deal of effort measuring a
stock when the hypothesis was always about a flow, and the stock would have been
the wrong place even with perfect occupational data, because notice periods and
collective agreements stand between a demand shock and a headcount.

**Date the treatment, do not assume it.** This is the one I am least proud of
missing. Diffusion happened almost entirely after the date we called the
treatment, and every pooled estimate is attenuated accordingly.

**Triangulation requires independent failure modes.** I spent an evening pleased
that four designs agreed on a null before noticing that two of them would have
agreed on a null whatever the truth was, and that all four share the same exposure
measure. Agreement between designs that fail the same way is not corroboration.
It is one design in four hats.

A closing thought on how to present this. There is a standing temptation to write
a methodological difficulty of this kind as a confession, with the concessions
front-loaded and the contribution apologised for in advance. I think that reads
the situation backwards. Almost nobody in this literature could have detected the
problem at all: doing so requires the assignment year of every occupation code,
monthly employer-employee links and population coverage, and work built on survey
or vendor data simply has no register to truncate. We ran the test. It came back
at −0.307, and we believed it over our own published headline. That is not a
weakness of the data. It is a demonstration of what these data can do that the
alternatives cannot, and the paper should say so once, plainly, and then move on.

---

## 10. What the paper can claim

Three sentences, in descending order of how well they are supported.

Within Swedish firms, across age groups, on employment through mid-2025, the
effect of AI exposure is at most about one log point. This is a bound, it is
tight, and it contradicts a literature claiming much larger effects.

At the paper's own estimand, comparing young workers with older ones inside the
same employer, the estimate is negative in every specification at about −1.3 per
cent, with an artefact of half a percentage point, and is not distinguishable
from zero. What it licenses is an upper bound of roughly 3.5 per cent on any
decline.

Dating the treatment to when Swedish firms actually adopted AI rather than to
when ChatGPT launched, the estimate grows monotonically to −1.8 per cent and is
significant at ten per cent, with no corresponding movement in separations.

What we cannot say is that the effect is established, and the tension in section
5 between two clean designs with different age profiles has to be resolved before
the age-specific claim is made at all.

---

## 11. What I would run next

Three things, cheap, in order of value.

Re-run the within-employer design with the treatment dated at 2024-01 rather than
at the launch. That is the one specification combining our cleanest design with
our best-supported timing, and it has not been run.

Reconcile the two age gradients in section 5. They use different exposure
constructions on the same population and should not disagree.

Validate the exposure measure externally. Every design here rests on DAIOE at
four-digit occupation and none of them tests it, so if that measure is wrong they
all fail together and their agreement means nothing. Skans and Sokolow have shown
this can be done in Swedish data against survey measures of actual LLM use.
