# Manual Spot-Check Instructions for Erik Engberg

**Paper:** Two Economies? Stock Markets, Job Postings, and AI Exposure in Sweden
**Purpose:** Independent verification of data pipeline (Platsbanken processing + telework crosswalk)
**Estimated time:** 1–2 hours

---

## What you need

All files are in the project repo: `canaries-sweden/`

| File | Location |
|------|----------|
| Raw Platsbanken JSONL (2023) | `data/raw/2023.jsonl.zip` |
| Our processed counts | `data/processed/postings_ssyk4_monthly.csv` |
| Telework mapping (our output) | `tables/telework_ssyk_mapping.csv` |
| BLS SOC→ISCO crosswalk | `data/raw/isco_soc_crosswalk2.xls` (sheet: "2010 SOC to ISCO-08", data starts row 7) |
| SCB SSYK→ISCO crosswalk | `data/raw/ssyk2012_isco08.xlsx` (sheet: "Nyckel", data starts row 5) |
| Dingel-Neiman telework scores | `data/raw/dingel_neiman_telework.csv` |
| Processing script (reference) | `src/02_process_platsbanken.py` |
| Crosswalk script (reference) | `src/11_remote_work_robustness.py` |

You can use Python, R, Stata, Excel, or even `zcat`+`grep` — the point is to verify independently, so use whatever you prefer.

---

## Task 1: Verify 5 SSYK codes against raw JSONL (June 2023)

Pick the JSONL file `data/raw/2023.jsonl.zip`. Each line is one job ad in JSON. The SSYK 4-digit code is in:

```
occupation_group[0].legacy_ams_taxonomy_id
```

(The field `occupation_group` is a list of dicts; take the first element.)

Count how many ads have `publication_date` starting with `"2023-06"` for each of these 5 SSYK codes:

| SSYK | Description | AI quartile | Our count (June 2023) |
|------|------------|-------------|----------------------|
| 2512 | Software developers | Q4 (highest) | 2,644 ads |
| 4222 | Contact centre agents | Q4 (highest) | 3,124 ads |
| 9412 | Kitchen helpers | Q1 (lowest) | 922 ads |
| 3311 | Securities/finance dealers | Q4 (highest) | 3 ads |
| 7231 | Motor vehicle mechanics | Q1 (lowest) | 1,285 ads |

**What to check:**
- Do your raw counts match our processed counts? Exact match expected.
- Note any ads where the SSYK code is present but in a different JSON structure than expected (e.g., `occupation_group` as a dict instead of a list).

**Quick approach (command line):**
```bash
cd data/raw
python3 -c "
import zipfile, json
from collections import Counter

counts = Counter()
with zipfile.ZipFile('2023.jsonl.zip') as z:
    with z.open(z.namelist()[0]) as f:
        for line in f:
            ad = json.loads(line)
            pub = ad.get('publication_date', '')
            if not pub.startswith('2023-06'):
                continue
            occ = ad.get('occupation_group')
            if isinstance(occ, list) and len(occ) > 0:
                code = occ[0].get('legacy_ams_taxonomy_id')
            elif isinstance(occ, dict):
                code = occ.get('legacy_ams_taxonomy_id')
            else:
                continue
            if code in ('2512','4222','9412','3311','7231'):
                counts[code] += 1

for code in ('2512','4222','9412','3311','7231'):
    print(f'SSYK {code}: {counts.get(code, 0)} ads')
"
```

---

## Task 2: Trace 3 crosswalk chains by hand (SOC → ISCO → SSYK)

The teleworkability analysis uses a three-step crosswalk:

```
Dingel-Neiman O*NET-SOC → SOC 2010 → ISCO-08 → SSYK 2012
```

For each occupation below, trace the chain manually through the Excel files and verify the final teleworkability score matches `tables/telework_ssyk_mapping.csv`.

### Chain A: SSYK 2512 (Software developers) → expected teleworkable = 1.0

1. **SSYK→ISCO:** Open `data/raw/ssyk2012_isco08.xlsx`, sheet "Nyckel". Find SSYK 2512. Column A = SSYK code, Column C = corresponding ISCO-08 code(s). Record the ISCO code(s).
2. **ISCO→SOC:** Open `data/raw/isco_soc_crosswalk2.xls`, sheet "2010 SOC to ISCO-08". Find rows where column D (ISCO-08 Code) matches the ISCO code(s) from step 1. Record the SOC 2010 codes (column A).
3. **SOC→Telework:** Open `data/raw/dingel_neiman_telework.csv`. The `onetsoccode` column has O*NET-SOC codes (XX-XXXX.XX). Truncate to 6-digit SOC (XX-XXXX) to match. For each matching SOC, record the `teleworkable` value (0 or 1).
4. **Aggregate:** Average the teleworkable scores across all SOC codes found. This should equal 1.0.

### Chain B: SSYK 7231 (Motor vehicle mechanics) → expected teleworkable = 0.0

Same steps. Should yield 0.0.

### Chain C: SSYK 1112 (Senior government officials) → expected teleworkable = 0.5

Same steps. This is a mixed case — should yield 0.5, meaning some source SOC codes are teleworkable and some are not.

**What to check:**
- Does the final averaged score match our output?
- Are there any ISCO codes in the chain that map to unexpected SOC codes?
- Are there any many-to-many mapping issues that could distort the average?

---

## Reporting

Just send Magnus a short note with:
1. For Task 1: your 5 counts vs ours (match / discrepancy)
2. For Task 2: for each chain, the intermediate codes you found and whether the final score matches
3. Any anomalies or concerns

No formal write-up needed — an email or message with the numbers is fine.
