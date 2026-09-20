# Measuring what AI has done to workers of different ages

**A memo on the problem we ran into, what I did about it, and where I think it leaves us.**
20 September 2026

---

## Why I am writing this

We set out to answer a narrow question with unusually good data. Within a single
employer, in a single month, do young workers in more AI-exposed work fare
differently from young workers in less exposed work? Sweden lets us ask it
properly. We observe every employment spell monthly, we can hold the employer
fixed, and we have occupational detail at four digits.

Along the way the measurement gave out. Not in a small way. Not in a way a
robustness table absorbs. I want to set out what happened, because the episode
taught me more about how to do this kind of work than the original result did,
and because most of what I learned generalises well beyond this paper.

Read this as one colleague to another. I have tried to show the reasoning rather
than only the conclusions, including the two places where I reached a conclusion
confidently and had to withdraw it, because those are the parts of a research
account that are usually edited out and are almost always the parts a reader
would learn most from.

---

## 1. The problem

Our headline was that employment of 22 to 25 year olds in the most AI-exposed
quartile of occupations fell by about 17 log points relative to less exposed
young workers in the same firm, after ChatGPT. The estimate was precise, the age
profile was monotone, and the placebo on workers over 50 was near zero. It looked
like a finding. It was not.

The difficulty is that occupation is not observed monthly. It comes from the
annual occupation register, and that register arrives late. Our panel runs to mid
2025. Occupation codes stop in 2023. The last two years therefore classify people
by what they were doing up to two years earlier.

This is not exotic; it is the ordinary condition of register work, and everybody
who uses occupational data lives with some version of it. The question is whether
it matters enough here to change what we conclude, and the useful feature of the
problem is that there is a clean way to find out rather than a judgement to be
made.

---

## 2. The test that settled it

Take a year T, discard every occupation code assigned after it, and let the later
years inherit stale codes exactly as 2024 and 2025 do. Then re-estimate. We can
compare that estimate with the one using the true codes, because for those years
the truth is observable. The gap between the two is what the lag manufactures out
of nothing.

I want to dwell on why this works, because the logic is worth carrying to other
projects. We are not asking whether the codes for 2025 are accurate, which we
cannot know and will not know for two years; we are reproducing the ailment in a
period where we also hold the diagnosis, and measuring how far the two diverge.
The design converts an unanswerable question about the present into an answerable
question about the past. That trade is available far more often than it is
taken.

Here is what it returned for 22 to 25 year olds:

| truncation | true codes | stale codes | manufactured |
|---|---|---|---|
| T = 2021 | +0.019 (0.013) | −0.288 (0.017) | **−0.307** |
| T = 2022 | +0.018 (0.011) | −0.145 (0.012) | **−0.163** |

The artefact is as large as the finding, and at the longer truncation it is
larger. Note also the true arms: with correct codes the estimate is positive and
insignificant at both truncations, which is a second and independent reason to
doubt the headline.

I should be careful here, because I was sloppy about it once already. You cannot
simply subtract the artefact from the headline and call the remainder the truth.
The backtest runs on a shorter panel with a different share of corrupted years, so
the quantity does not transfer. What the backtest establishes is weaker and more
useful: the lag alone can produce a coefficient of this size, so the headline is
not identified. That is all it establishes. It does not tell us the true
value. For that we need a different
design, which is the rest of this memo.

---

## 3. The repair that failed, and why I am glad we tried it

The obvious substitute is education. It is recorded earlier and moves more
slowly, and there is a good recent literature assigning exposure through it. We built eight variants, from a plain
replication of the published approach through to designs using the recency of
qualification and the field of current enrolment.

Then I made the mistake that this project has punished more than once, and I set
it out here because it is the sort of error that looks like diligence while it is
happening. I judged the designs by the size of their artefact.

That is the wrong criterion, and the Monte Carlo showed why. We built a simulator
calibrated on moments measured in our own registers: how often young workers
change occupation, how often a change moves them across an exposure boundary, how
often education level rises, how stale codes actually are. Then we generated worlds where we know the answer, and asked what each design
would report. This is the cheapest discipline in empirical work and we use it far
too rarely.

Under a world with no effect at all, the education designs report an age gap
between the young and the over-fifties of about −0.31. Under a world with a true
gap of −0.15, they report about −0.34. Fit a line through that and you get

> measured gap ≈ −0.31 + 0.18 × true gap

against the same estimator given correct codes, which returns

> measured gap ≈ −0.02 + 0.72 × true gap

The second line describes a well-behaved estimator with ordinary attenuation.
The first does not describe an estimator at all in any useful sense, since it
reports roughly the same number whether the truth is zero or substantial. However
precisely we measure that number, it is not telling us about the world.

Two lessons, and the second is the one I would tattoo on a student.

The first is that a large bias and a useless estimator are different problems,
and the distinction is not academic. A bias that is constant can be differenced
away or netted out, and much of applied work quite properly does so. However, the
bias here is worst precisely on the age contrast, because misclassification pushes
the young estimate down and the old estimate up, so the gap is inflated from both
ends. It lives inside the very difference we want to take.

The second is that **you cannot correct your way out of this.** The instinct is to
divide by the pass-through and inflate the standard errors accordingly. That is
legitimate when measurement error is classical, which is to say when a null world
returns a null. Ours returns −0.31 in a null world. Dividing that by 0.18 gives
roughly −1.7, which is not an estimate of anything. Before you ever apply an
attenuation correction, simulate a null and look at the intercept. If it is not
zero, the correction amplifies the bias instead of removing it. Check the
intercept first. It takes an hour.

So the right response to a low pass-through is not to correct it. It is to
redesign until it is high. Going from 0.18 to 0.8 takes the standard-error penalty
from five and a half times to a quarter, and costs no assumptions at all.

---

## 4. The repair that worked

The principle is simple once stated. **Measure exposure once, before the shock,
using data that has already arrived, and never measure it again.**

Concretely: for each employer and each age band, compute the average AI exposure
of the occupations that group actually held in 2019. Freeze it. From then on a
worker needs only a birth year and a payslip, both of which arrive monthly and on
time. The occupation register is never consulted again and the education register
is not used at all. Whatever the lag does to later vintages, it cannot reach this
measure. The measure was finished before the lag existed.

Absorb employer by month, employer by age, and month by age. The comparison is
then made inside the firm, against its own other age groups, among firms whose
young workers did more or less exposed work before any of this began.

The design has a real cost and I want to name it plainly rather than bury it in
an appendix. Because exposure now varies across employers rather than across
workers inside one, identification comes from comparing firms whose young workers
were differently exposed at baseline, and that requires the corresponding
parallel-trend assumption on the multiplicative scale. It is not the pure
within-firm contrast we originally wanted. Nor can it be: that contrast requires
classifying each young worker as they are today, which is precisely the thing the
register no longer lets us do.

---

## 5. What we find

**On the stock of employment, nothing.** Across six specifications, including two
coverage restrictions and a control for the expiry of the youth payroll-tax
reduction in April 2023, the estimate sits between −0.002 and +0.010 with standard
errors around 0.009, on panels of eleven to thirty-eight million cells. The
implied interval runs from about −0.01 to +0.02. That is not a failure to find
something, and I would resist writing it up as one: it is a tight bound on how
large any effect on the stock can be, and it is tighter than anything obtainable
from survey or vendor data.

**On the flow of hiring, something.** Employment stock is the slowest margin in
the Swedish labour market, with notice periods and collective agreements standing
between a demand shock and a headcount. The literature we are arguing with is
about entry-level hiring, not about stocks. So we built the flow. A hire is a person present at an employer this month and
absent last month. That needs no occupational information whatsoever.

Reading the young differential in hiring, first half-years only so that like is
compared with like:

| | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|
| hires | +0.006 | +0.011 | +0.002 | ref | +0.027 | +0.013 | **−0.069** |
| separations | +0.032 | +0.029 | +0.016 | ref | +0.029 | +0.020 | +0.041 |
| stock | +0.001 | +0.007 | −0.004 | ref | +0.013 | −0.001 | −0.007 |

Six first half-years between zero and +0.027, and then −0.069. It is the only
negative value in the entire hiring column, across both halves of every year.
Separations do not fall. If anything they are slightly up. The stock has barely
moved.

**Then the standard errors arrived and the reading collapsed.** That −0.069
carries a standard error of 0.079, and the honest seasonal handling gives
−0.049 with 0.067. Both are within one standard error of zero, and the
precision on that half-year is three to four times worse than on any other,
which is what made the point estimate look dramatic in the first place. A rule
we had written before the number existed says an effect must be twice its
standard error, and it refused. I had spent an evening comparing point
estimates across half-years without looking at their precision, which is
exactly what the rule was there to prevent.

The second half-years tell the complementary story. Through December 2024 the
young hiring differential sat between +0.044 and +0.076 with a standard deviation
of 0.010, straight through ChatGPT, with the post-2022 average marginally higher
than the pre-2022 one. Nothing was happening. Then 2025 breaks.

That combination is the entry-level hypothesis in its textbook form. The inflow
adjusts, the outflow does not, and the stock has not yet had time to drain. It
also explains something that had puzzled me for most of a week, which is why
every stock-based design in this round returned zero however we constructed it:
they were all watching the slow margin, patiently and precisely, while the fast
one did not turn until the very end of the observation window.

---

## 6. The objection you should be raising

If you have been reading carefully you should by now be uneasy, and the unease is
the right one. Exposure is frozen in 2019, so by 2025 it is a six-year-old proxy
for who is exposed today, and a stale proxy in a continuous regressor attenuates
the coefficient towards zero. The design is therefore weakest exactly where the
effect is most likely to be. That is a serious objection and not a footnote, and
it is the same disease as the register lag wearing a different coat.

So we measured it. Rebuild the exposure measure contemporaneously in every year
that has its own occupation register, and regress it on the 2019 version across
firm-age cells. The slope is the share of the original signal that survives, and
it is precisely the factor by which a 2019-based coefficient is attenuated.

| | 2019 | 2021 | 2022 | 2023 |
|---|---|---|---|---|
| all ages | 1.000 | 0.917 | 0.890 | 0.890 |
| 22 to 25 | 1.000 | 0.835 | 0.817 | 0.829 |

The signal decays by about a tenth in the first two years and then stops. At the
youngest ages it recovers slightly. A firm's age-specific occupational composition
is far more persistent than I expected. The objection does not survive its own
measurement.

This matters more than it may look, and it matters in three separate ways. It
means the nulls are real nulls, since correcting the stock estimate for
attenuation moves it from +0.007 to +0.008. It means the standard-error penalty
for correcting is about a fifth rather than the four hundred and fifty per cent
the education route would have demanded. Moreover, when we re-ran the whole design
on a 2022 baseline, three years closer to the outcome years and still before
ChatGPT, the hiring estimate moved towards zero rather than away from it, which is
the opposite of what attenuation would produce.

---

## 7. What is not settled

I would not let any of section 5 into a seminar without the following said in the
same breath.

The 2025 break is **one half-year**. Our own reading rule says an isolated half-year is noise. That rule was written
before the number existed, which is the only reason it carries weight now.

The 2025 data are **preliminary**. Every 2025 figure comes from the provisional
monthly file, and there is no 2025 second half-year at all, so the natural
confirmation does not exist in the data we hold.

The **first-half-against-first-half comparison was chosen after seeing the data.**
The series has an obvious and mechanical seasonal, so the slice is defensible, but
nobody specified it in advance. A script now running checks it two independent
ways, and one of them uses no seasonal model at all.

And one loose end I have not tied. Our careful reimplementation of the education
design does not reproduce the earlier script's as-of arm, and the difference is
not the cascade correction I had assumed, because reproducing that correction
exactly turns out to change nothing at all. The earlier script is the one with the
known defect, and the simulator sides with the newer one, so I have let the
analysis proceed while recording the discrepancy in the output itself rather than
quietly resolving it in favour of the answer I prefer. I am not comfortable with
it, and it should be closed before publication.

---

## 8. What I would take from this

**Design so that the measurement is finished before the treatment begins.** The
single move that rescued this project was fixing exposure in a pre-period and
never touching it again, so that everything arriving late became irrelevant by
construction rather than by assumption, and no reviewer could ask a question
about the 2024 register that the design had not already made unanswerable in our
favour. That is worth more than any correction.

**Judge a design by whether it moves when the truth moves, not by the size of its
bias.** Simulate a world with no effect and a world with a known effect, then look
at the difference between what your estimator reports in the two, which is the
only quantity that tells you whether the estimator is listening to the data at
all. If it barely moves, no amount of correction will help. A small standard error
on an estimator like that is a warning rather than a comfort.

**Match the margin to the mechanism.** We spent a great deal of effort measuring a
stock when the hypothesis was always about a flow, and the stock would have been
the wrong place to look even with perfect occupational data, because notice
periods and collective agreements stand between a demand shock and a headcount.
Therefore the measurement problem, painful as it was, cost us less than the
conceptual one.

**Date the treatment, do not assume it.** This is the one I am least proud of
missing. Every estimate above defines the post period as the ChatGPT launch,
which pools thirteen months of 2023 with eighteen of 2024 and 2025. If the
labour market responded to firms adopting rather than to a model shipping,
that averages an untreated period with a treated one and attenuates the
coefficient for reasons that have nothing to do with the world. SCB's own
survey puts Swedish firm adoption at 10.4 per cent in 2023, 25.2 in 2024 and
35.0 in 2025, so the diffusion happened almost entirely after the date we
called the treatment. Our tight bound is a bound on the average over the whole
window, and I had been presenting it as though it bounded the effect. It does
not.

**Measure your attenuation rather than apologising for it.** We very nearly wrote
a careful paragraph conceding that our proxy decays with distance from the
baseline, hedging the late-period results accordingly, and inviting the reader to
discount them. It decays by eleven per cent and then stops. That is a completely
different paper from the one the paragraph would have written, and the only thing
standing between the two was an afternoon's work.

**Triangulation requires independent failure modes.** I spent an evening pleased
that four designs agreed on a null before noticing that two of them would have
agreed on a null whatever the truth was, and that all four shared the same
exposure measure. Agreement between designs that fail the same way is not
corroboration. It is one design in four hats.

A closing thought on how to present this. There is a standing temptation, particularly for those of us who came up being
told to anticipate every objection, to write a methodological difficulty of this
kind as a confession, with the concessions front-loaded and the contribution
apologised for in advance. I think that reads the situation
backwards. Almost nobody in this literature could have detected the problem at
all: doing so requires the assignment year of every occupation code, monthly
employer-employee links and population coverage, and work built on survey or
vendor data simply has no register to truncate. We ran the test. It came back at −0.307, and we believed it over our own
published headline. That is not a
weakness of the data. It is a demonstration of what these data can do that the
alternatives cannot, and the paper should say so once, plainly, and then move on.

---

## 9. Where the work stands

The single most valuable thing outstanding is not another design. It is
re-estimating what we already have with the treatment dated where the adoption
data put it, which costs one parameter and a re-fit.

Three things are running or waiting. The horse race across the education designs
is re-running, though the simulator has already told us what it will conclude. The
fresh-code panel and the hiring-flow script have both completed and their results
are sitting in the secure environment unread. And the script that checks the
seasonal, reports the standard errors and establishes whether the definitive 2025
file now exists is in the queue.

The last of those is the one I care about. If SCB has delivered the definitive
2025 data, the finding in section 5 must be re-estimated on it before any of us
believes it. If it has not, then what we have is a sharp bound on the stock, a
suggestive break in the flow, and an honest account of why we can say the first
with confidence and not yet the second.
