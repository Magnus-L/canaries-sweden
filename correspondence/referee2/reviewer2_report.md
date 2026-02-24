# Referee Report: "Two Economies? Stock Markets, Job Postings, and AI Exposure in Sweden"

**Journal:** Economics Letters
**Reviewer:** Anonymous (Reviewer 2)
**Date:** February 2026

---

## 1. Recommendation

**Major revision.**

## 2. Summary

This letter uses 4.6 million Swedish job ads from Platsbanken (2020--2025) matched to a generative AI exposure index to test whether the widely discussed decline in job postings is attributable to AI displacement or monetary tightening. The authors exploit the seven-month gap between the Riksbanken rate hike (April 2022) and the ChatGPT launch (November 2022) in a difference-in-differences design and conclude that postings fell due to monetary policy, not AI. The paper is timely and the Swedish institutional setting provides a useful out-of-sample test of the Brynjolfsson et al. (2025) "canaries" finding.

## 3. Main Strengths

- **Clever timing test.** The seven-month gap between the Riksbanken rate hike and ChatGPT provides genuine identifying variation that is not available in the US data. This is the paper's core contribution and it is well-motivated.
- **Comprehensive public data pipeline.** The use of the full Platsbanken population with open data and replication code sets a high transparency standard for the field.
- **Relevant policy question.** Whether the "scary chart" reflects AI or macro fundamentals matters enormously for labour market policy, and this paper adds a clean Nordic data point alongside the Finnish evidence.

## 4. Main Concerns

**1. The identification strategy conflates multiple shocks.**
The post-Riksbank dummy (April 2022 onward) does not isolate monetary policy. Russia invaded Ukraine in February 2022, triggering energy price spikes that hit Sweden's industrial base hard. European supply chain disruptions were also ongoing. The authors need to explain why their Post-RB dummy captures monetary policy specifically, rather than the entire constellation of macro shocks in early-to-mid 2022. At minimum, the paper should include energy-intensive occupation controls or industry-level proxies for exposure to the energy crisis. The current interpretation -- that beta_1 "is" monetary policy -- is too strong without ruling out these competing explanations.

**2. The economic mechanism linking monetary policy to high-AI-exposure occupations differentially is unclear.**
The DiD design assumes that high-AI-exposure occupations respond more strongly to rate hikes than low-exposure occupations. Why? High-exposure occupations (analysts, accountants, administrators) are typically white-collar roles in sectors that are arguably less interest-rate-sensitive than construction or real estate (which tend to be lower-exposure). The paper provides no theoretical argument for why beta_1 should be negative. Without this mechanism, the significant beta_1 may simply reflect a pre-existing differential trend that happens to begin around the rate hike. The occupation-specific trend specification (column 3) is telling: beta_1 loses significance when trends are included, which is consistent with this concern.

**3. The paper dismisses the AI effect too quickly given its own robustness results.**
The claim that "the ChatGPT coefficient is essentially zero" is not well-supported by the robustness table. Beta_2 is negative in 7 of 8 specifications, significantly so at the 5% level in 2 (all-apps, terciles) and at the 10% level in a third (excluding IT/tech). The all-apps specification yields beta_2 = -0.091 (p = 0.018), which is economically and statistically meaningful. Rather than declaring a null, the paper should honestly characterise its results as showing a dominant monetary policy effect with suggestive but imprecise evidence of an additional AI-related decline. The current framing overstates the strength of the null finding.

**4. Job postings are a noisy and potentially misleading outcome.**
Postings reflect both labour demand and posting behaviour. Firms may reduce the number of ads posted per vacancy (consolidation), shift to other platforms (Indeed, LinkedIn), or change duplicate-posting practices -- all without changing actual hiring. The authors should discuss this measurement issue and, ideally, show that the vacancy count per ad (which is in their data) tells the same story. The vacancy-weighted robustness check partially addresses this but is buried in the appendix without discussion.

**5. The event study does not convincingly establish parallel pre-trends.**
The event study figure (Figure A3) shows substantial volatility in the pre-period, with coefficients of -0.21 in August 2020 and +0.26 in December 2020 -- swings of nearly 50 log points. Several pre-period coefficients are individually significant. The authors state "parallel pre-trends" but the figure shows noisy, not flat, pre-treatment differences. A formal pre-trend test (e.g., joint F-test on pre-period coefficients) is needed. With this much noise, the power to detect a post-ChatGPT structural break is questionable.

**6. Statistical power with 369 occupations split into quartiles.**
Each quartile contains roughly 92 occupations. The median cell has only 43 ads per month. Given the enormous variance (SD = 438 vs. mean = 167), many cells are near zero or very thin. Can the authors show that their results are not driven by a handful of large occupations? A leave-one-out sensitivity analysis or a histogram of cell sizes by quartile would be informative.

## 5. Minor Comments

- The abstract says "all data and code are publicly available" but the GitHub URL is a placeholder. This must be resolved before publication.
- The paper references an "Online Appendix" in two places (OMXSPI comparison, event study), but no appendix is included in the submission. Please provide it for review.
- The DAIOE index is the authors' own measure (Lodefalk et al. 2024). A robustness check with an independent exposure measure (e.g., Eloundou et al. 2024 or Felten et al. 2023 crosswalked to SSYK) would strengthen the results and address self-citation concerns.
- The sample starts in January 2020, right before the pandemic. The pandemic itself massively affected posting patterns, and the "indexed to 100 at February 2020" normalisation may create a misleading visual impression if Q4 and Q1 occupations were differentially affected by COVID. The exclusion-of-pandemic robustness check helps but does not fully resolve this because the treatment period itself is defined relative to a pandemic-distorted baseline.
- Clustering at the occupation level with 369 clusters is reasonable, but the authors should report wild cluster bootstrap p-values given the quartile treatment assignment, which creates only 4 effective treatment groups for the cross-sectional variation.
- The paper is exactly at the Economics Letters word limit. Some space could be recovered by cutting the Kauhanen references to a single sentence and using the freed words to address the mechanism question raised in Concern 2.
