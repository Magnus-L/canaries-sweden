# Pick-up prompt for Claude Code

Copy-paste this when resuming work on the "Two Economies?" paper.

---

## Quick resume

```
I'm resuming work on the Economics Letters paper "Two Economies? Stock Markets, Job Postings, and AI Exposure in Sweden" in ~/Documents/Research/papers/2026/canaries-sweden/.

Read SESSION_SUMMARY.md for full context. The paper is complete at 1,995/2,000 words, both PDFs compiled, 12 commits on main.

Key files: paper/main.tex, paper/appendix.tex, src/*.py, mona_package/.

What I need to do next: [describe task here]
```

---

## Task-specific prompts

### Incorporate MONA results (from Lydia)
```
Lydia sent back MONA results: mona_canaries_regression.csv and figA8_mona_canaries.png. Please update appendix.tex section A5 with the AGI-based results (replacing or supplementing the SCB annual results), copy the figure to figures/, and recompile both PDFs.
```

### Create GitHub repo + cover letter
```
Two tasks: (1) Create a public GitHub repo for the replication package — all src/ scripts, data/raw/ non-JSONL files, figures, tables, paper. Update the "[GitHub repository URL]" placeholder in main.tex. (2) Draft a cover letter for Economics Letters emphasising: postings (not employment) as outcome, Swedish open data, timing identification (Riksbanken before ChatGPT), complements Finnish studies.
```

### Final pre-submission check
```
Final check before Economics Letters submission. Please: (1) verify word count ≤ 2,000, (2) verify abstract ≤ 100 words, (3) check all figures/tables render in PDF, (4) verify GitHub URL is live, (5) verify AI disclosure statement present, (6) recompile both PDFs, (7) list any remaining issues.
```
