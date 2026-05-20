# ProMarket Draft v2 — Revised in Magnus's voice
## Status: SKELETON — requires author rewrite before submission (ProMarket AI policy)
## American English throughout (ProMarket is US-based)
## Word count: ~1,260 words

---

## TITLE OPTIONS
1. Generative AI Is Changing Who Gets Hired, Not How Many
2. The Stock Market Divergence Is Not an AI Story. The Hiring Freeze Is.
3. AI Has a Quiet Labor Market Effect. It Is Showing Up in Entry-Level Hiring.

---

## ARTICLE

Since late 2022, stock prices have risen sharply while job openings have fallen almost as steeply. The divergence has become one of the most cited charts in discussions of artificial intelligence and the labor market, and for many observers the interpretation is straightforward: generative AI is substituting for workers, depressing labor demand while lifting corporate profits. Our research, using full-population register data for Sweden and 4.6 million job advertisements, finds that this reading of the aggregate picture is not supported by the evidence. The underlying concern, however, is well founded. The effect of generative AI operates below the surface of standard statistics, through a selective recomposition of who gets hired.

**The aggregate decline is a monetary policy story**

The key is timing. Sweden's central bank, the Riksbank, raised interest rates for the first time in April 2022, seven months before ChatGPT launched in November 2022. If monetary tightening is the primary driver of the posting decline, it should begin with the rate hike. If generative AI is the driver, it should not appear before late 2022 and should be concentrated in high-exposure occupations.

Using the full population of advertisements published on Platsbanken, Sweden's public employment service vacancy portal, matched to an occupation-level measure of generative-AI exposure (the DAIOE index, Engberg et al. 2024), we find that postings across all AI-exposure groups declined together from April 2022 onwards. The post-rate-hike interaction is estimated at −0.127 (p < 0.01); the post-ChatGPT interaction at −0.062 and not statistically distinguishable from zero (p = 0.11). AI-exposed occupations are also uncorrelated with interest-rate sensitivity, consistent with the two channels being distinct. [Thompson 2025] has argued similarly for the United States: the posting decline tracks the Federal Reserve's tightening cycle, not the ChatGPT launch.

**Within employers, the age gradient is large and accelerating**

The aggregate picture conceals a different pattern at the level of individual employers. Using monthly employer-declaration data (the AGI register) covering the full Swedish working population, linked to four-digit occupation codes and age from the LISA longitudinal register, we estimate an employer-level difference-in-differences design (following [Brynjolfsson et al. 2025]). Employer-by-quartile and employer-by-month fixed effects absorb time-invariant compositional differences and all firm-level time-varying shocks, so identification comes from within-employer recomposition across AI-exposure quartiles over time.

Employment of workers aged 22 to 25 in the top quartile of AI-exposed occupations falls 5.5 percent below less-exposed occupations within the same employers by the first half of 2025 (95 percent confidence interval: −5.8 to −4.9 percent). This is our headline event-study estimate. Poisson pseudo-maximum-likelihood estimates, which better accommodate the zero-heavy distribution of employment counts, yield a decline of 16 percent across four consistent specifications (confidence interval: −18 to −14 percent). The two estimates are consistent: the gap in magnitude reflects the scale difference between the log-linear and Poisson specifications, not a genuine difference in economic magnitude. Workers aged 31 to 49 are essentially unaffected. Workers over 50 show a small positive gain, though this finding is sensitive to the choice of AI exposure measure.

The adjustment loads almost entirely on the hiring margin. The hires coefficient for ages 22 to 25 is −0.0051 (p < 0.001); the separations coefficient is one-quarter as large (−0.0012, p < 0.01). Sweden's employment-protection legislation, based on a last-in-first-out principle, constrains dismissal of incumbent workers. High-AI-exposure employers are therefore not laying off junior staff. Rather, they are not replacing departing workers and not taking on new cohorts. [Hosseini and Lichtinger 2025], analyzing résumé and job-posting data for roughly 65 million workers across more than 280,000 U.S. firms, find an approximately 9 percent junior employment decline in generative-AI-adopting firms driven by a comparable hiring slowdown. The hiring-freeze pattern appears consistent across institutional settings.

**Young women face a larger adjustment**

The most under-examined finding concerns young women. Women aged 22 to 25 in high-AI-exposure occupations have experienced a difference-in-differences employment decline of −0.016 (p < 0.001), compared with −0.007 (p < 0.01) for men of the same age. The gender gap is itself statistically significant (triple-difference p < 0.01). Roughly two-fifths of this disparity is accounted for by occupational composition: women in Sweden are concentrated in administrative, payroll, and customer-service roles near the top of the AI-exposure distribution. Payroll administrators are 82 percent female and sit at the 96th percentile of the DAIOE index; customer service agents and receptionists are 66 percent female.

The remaining three-fifths of the gender gap is within-occupation: young women are losing ground relative to young men in the same AI-exposed jobs. The mechanism is not identified in our data, but the within-occupation component implies that occupational sorting alone does not account for the disparity.

This employment finding is consistent with simulation evidence for Sweden from [Gardberg, Heyman, Olsson and Tåg (IFN 2025)], who show that pre-AI patterns of gender-based occupational sorting can widen the post-AI gender wage gap. Our data suggest that employment consequences are already materializing ahead of the wage effects. The International Labour Organization (2026) has estimated that female-dominated occupations globally face roughly twice the generative-AI exposure risk of male-dominated ones. In Sweden, that elevated risk is now reflected in employer-level hiring decisions.

**Why outcomes differ across countries**

Three studies, three countries, three outcomes. [Brynjolfsson, Chandar and Chen (Stanford Digital Economy Lab 2025)] find a 16 percent Poisson-estimated decline in employment for young U.S. workers in AI-exposed occupations. Our Swedish Poisson estimates cluster in the same range. [Kauhanen and Rouvinen (ETLA 2026)], however, find no effect in Finland using a comparable research design on Finnish population data. We test directly whether this divergence is methodological: reweighting our Swedish sample to match Finland's industry-occupation composition and applying Finland's preferred exposure measure leaves our estimated decline at approximately 15 percent, nearly unchanged. The divergence is not an artifact of study design.

What explains it remains uncertain. Candidate factors include differences in the intensity of AI adoption, in the occupational structure of youth employment, and in the vocational-training pathways that channel young Finns into less-exposed fields. The cross-country heterogeneity is itself substantive: institutional context appears to shape the distributional consequences of the same underlying technology.

**A skill-formation concern**

Job counts alone understate what is at stake. Entry-level positions in AI-exposed occupations are not merely employment. They are the mechanism through which workers accumulate applied skills, professional networks, and the tacit knowledge that constitutes mid-career capability. An employer that stops taking on new cohorts today is not only reducing near-term costs. It is also failing to develop the experienced workers it will need a decade from now.

[Brynjolfsson et al. 2025] describe young workers as the "canaries in the coal mine," early signals of a broader adjustment. If the entry-level hiring freeze persists, what follows is not merely a transient employment dip but a generational skill-formation gap. That gap will appear in career trajectories and aggregate productivity well before it registers in headline unemployment figures.

**Implications**

Three practical implications follow. First, monitoring: aggregate unemployment and job-posting indices will not detect this adjustment early. Employer-level data at monthly frequency, tracking hiring composition by age and occupation, is what makes it visible. Countries with employer-declaration registers of the kind Sweden maintains hold a significant informational advantage. Second, education and training: if entry-level positions in AI-exposed fields contract, universities and vocational programs face a mandate to restructure applied learning so as to provide the experiential component that employment has historically supplied. Third, coverage gaps: recent graduates who cannot find work and have not yet accrued eligibility for unemployment-insurance benefits are invisible to the social insurance system. Monitoring whether this population is growing disproportionately in AI-exposed fields warrants attention.

**Disclosure.** Funded by the Torsten Söderberg Foundation (grants E46/21, ET3/23) and WASP-HS (grant 805). Employment data from Statistics Sweden via the MONA platform under ethical approvals detailed in the working paper. No conflicts of interest.

---

## REFERENCES FOR HYPERLINKS
[Thompson 2025] → https://www.derekthompson.org/p/is-this-the-new-scariest-chart-in (verify)
[Brynjolfsson et al. 2025] → https://digitaleconomy.stanford.edu/wp-content/uploads/2025/11/CanariesintheCoalMine_Nov25.pdf (verify)
[Hosseini and Lichtinger 2025] → https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5425555 (verify)
[Gardberg et al. IFN 2025] → https://www.ifn.se/en/publications/working-papers/2025/1534/ (verify)
[Kauhanen and Rouvinen 2026] → https://www.etla.fi/en/publications/working-papers-en/ai-has-not-impacted-the-youth-labor-market-in-finland/ (verify)
[Engberg et al. DAIOE] → https://docs.iza.org/dp16717.pdf
[ILO gender AI risk 2026] → https://www.ilo.org/resource/news/new-ilo-data-confirm-women-face-higher-workplace-risks-generative-ai-men (verified live, published 5 March 2026)
Working paper → https://www.ai-econlab.com/papers
