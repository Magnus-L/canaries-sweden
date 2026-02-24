# Referee Report -- Economics Letters

**Manuscript:** "Two Economies? Stock Markets, Job Postings, and AI Exposure in Sweden"

**Reviewer:** Reviewer 1

---

## 1. Recommendation

**Minor revision.**

## 2. Summary

This paper investigates whether the widely discussed divergence between stock prices and job postings -- the "scary chart" -- reflects AI-driven labour displacement or conventional macroeconomic forces. Using 4.6 million Swedish job ads from Platsbanken (2020--2025) matched to the DAIOE generative AI exposure index, the authors exploit a difference-in-differences design in which Sweden's central bank began raising rates seven months before the ChatGPT launch. They find that the posting decline began with monetary tightening and affected all AI exposure quartiles broadly, with no statistically significant additional decline in highly exposed occupations after ChatGPT.

## 3. Main Strengths

- **Novel decomposition of the posting gap.** The paper is the first to decompose the "scary chart" by AI exposure using microdata on postings rather than employment. This is a genuine contribution, since the original debate (Thompson, Brynjolfsson et al.) was about postings, and the distinction between a leading indicator and a stock measure matters for interpretation.

- **Clean timing identification.** The seven-month gap between the Riksbanken rate hike and ChatGPT provides a useful natural experiment that other country studies lack. The event-study confirmation of parallel pre-trends is reassuring.

- **Comprehensive robustness.** Seven alternative specifications covering different exposure measures, sample restrictions, and weighting schemes lend credibility to the null result. The transparency commitment (open data and code) is commendable.

## 4. Main Concerns

1. **Confounders beyond monetary policy.** The identification strategy assumes that the Riksbanken rate hike and the ChatGPT launch are the two relevant shocks, but several other factors could confound the comparison. The energy crisis following Russia's invasion of Ukraine (February 2022) hit Sweden particularly hard, with electricity prices spiking differentially across sectors. Supply-chain disruptions, the weakening krona, and post-pandemic labour market normalisation all coincide with the treatment window. The paper should discuss why these alternative shocks would not generate a spurious differential pattern correlated with AI exposure. Even a brief argument -- for instance, that energy-intensive sectors are not systematically more or less AI-exposed -- would strengthen the design. Without this, the "monetary policy, not AI" conclusion may be overstated: the result could equally be "macro shocks, not AI."

2. **The null result framing deserves more nuance.** The ChatGPT interaction coefficient in the baseline specification is -0.062 with p = 0.11. This is not far from conventional significance, and the 95% confidence interval presumably includes economically meaningful negative effects. The paper should report and discuss the confidence interval for beta_2 explicitly. In the tercile specification, the ChatGPT coefficient is actually significant (p = 0.026), and in the all-apps specification it is also significant (p = 0.018). The paper mentions these as "robust" but does not flag that two of seven robustness checks flip the key result. This tension needs honest discussion. I would suggest framing the conclusion as "we cannot rule out modest AI effects, but the dominant driver is monetary tightening" rather than the current clean dichotomy.

3. **Choice of AI exposure measure.** The DAIOE index is developed by one of the authors, which is disclosed but nonetheless raises questions about measure selection. The paper would benefit from a brief comparison with the main alternative measures in the literature: Felten et al. (2021), Webb (2020), and Eloundou et al. (2023). If SSYK-crosswalked versions of these exist or could be approximated, running at least one as a robustness check would be valuable. If they cannot be crosswalked, a qualitative comparison of which occupations land in Q4 under DAIOE versus the alternatives would help readers assess external validity. The "language model" robustness check partially addresses this, but it is not clear whether this is substantively different from the genAI measure.

4. **Platsbanken coverage.** Not all Swedish job openings are posted on Platsbanken. The paper should discuss what share of total vacancies Platsbanken captures and whether coverage is systematically different across AI exposure quartiles. If high-exposure occupations (e.g., white-collar professional roles) are more likely to be posted on LinkedIn or company websites than on Platsbanken, this would attenuate any measurable AI effect. Even a sentence citing Statistics Sweden's vacancy survey for comparison would help.

## 5. Minor Comments

- The abstract states "4.6 million Swedish job ads (2020--2025)" but the data section mentions the download extends through February 2026. Reconcile the date range.

- Equation (1): the PostGPT dummy is coded as 1 from December 2022, but ChatGPT launched on November 30, 2022. A brief justification for December rather than November would be helpful, or a robustness check with November coding.

- The comparison with Brynjolfsson et al. (2025) could be more careful. Their study uses Current Population Survey employment data and focuses on entry-level workers, whereas this paper uses posting counts without distinguishing entry-level from experienced positions. If Platsbanken codes experience requirements, stratifying by entry-level status would strengthen the comparison considerably.

- The comparison with Kauhanen and Rouvinen (2025) is appropriate but the paper should note that Finland also tightened monetary policy in the same period (ECB rates rose from July 2022). The Finnish null result therefore faces similar identification challenges, and the two papers reinforce each other more than the current framing suggests.

- Specification (3) with occupation-specific time trends: the monetary policy interaction becomes insignificant (point estimate -0.068, no stars), which arguably undermines the "monetary policy did it" story as much as the "AI did it" story. The paper should acknowledge this rather than emphasizing only that beta_2 goes to zero in this specification.

- The paper does not report R-squared or within-R-squared values. These would be informative given the large number of fixed effects.

- The word "scary" appears frequently. While it references the original chart, toning down the language slightly would suit Economics Letters better.
