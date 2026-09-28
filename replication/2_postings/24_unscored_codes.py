#!/usr/bin/env python3
"""
24_unscored_codes.py: the occupation codes that advertisements carry but the
DAIOE index does not score, counted by kind, and their share of
advertisements.

WHAT IT COUNTS
Advertisements published between January 2020 and June 2026 carry 400
four-digit SSYK 2012 codes; the index scores 369 of them. The 31 it does not
score fall into three kinds, which the online appendix and the response
letter state:
  - military occupations (major group 0), which O*NET does not cover;
  - managerial occupations (major group 1). Most of these are codes that end
    in 0 (e.g. 1210), which advertisements use for a manager group as a
    whole, whereas the index scores the group's two levels separately (1211
    and 1212). The script counts how many unscored managerial codes are of
    that kind: the code ends in 0 and the index scores both xxx1 and xxx2;
  - single codes in other major groups.
The share of advertisements is the unscored codes' advertisements over all
advertisements in the same window, so it answers how much of the posting
data the 369-occupation sample leaves out.

The unscored set is derived here from the counts and the index, and checked
against the list that 02 writes for 2020 to 2025, so that the two windows
are shown to lose the same codes.

INPUTS   data/processed/postings_ssyk4_monthly.csv (from 1_data_public/02);
         data/processed/daioe_quartiles.csv (from 1_data_public/04);
         output/results/occupation_reconciliation_lists.txt (from 02)
OUTPUTS  output/results/unscored_codes.csv (statistic, value);
         output/results/unscored_codes_list.csv (one row per code)
SERVES   Online Appendix II.7 (the 3, 26 of which 25, and 2 codes; 3.7 per
         cent of advertisements); Section 1 of the paper (the index scores
         96 per cent of advertisements); the response letter, Point A2
RUNTIME  seconds
"""

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

WINDOW = ("2020-01", config.POSTINGS_REGRESSION_END)   # January 2020 to June 2026


def main():
    print("Occupation codes the index does not score")
    post = pd.read_csv(config.PROCESSED / "postings_ssyk4_monthly.csv",
                       dtype={"ssyk4": str})
    post["ssyk4"] = post["ssyk4"].str.zfill(4)
    post = post[(post["year_month"] >= WINDOW[0])
                & (post["year_month"] <= WINDOW[1])]
    scored = set(pd.read_csv(config.PROCESSED / "daioe_quartiles.csv",
                             dtype={"ssyk4": str})["ssyk4"].str.zfill(4))

    codes = set(post["ssyk4"])
    unscored = sorted(codes - scored)

    # The same codes must be lost on 02's 2020-2025 window; if not, the text's
    # single list of 31 would describe only one of the two windows.
    listed = (config.RESULTS / "occupation_reconciliation_lists.txt").read_text()
    m = re.search(r"lost_at_DAIOE_match \(\d+\): (.*)", listed)
    listed_codes = sorted(c.strip() for c in m.group(1).split(","))
    assert unscored == listed_codes, (
        f"the unscored codes differ between windows: {unscored} vs {listed_codes}")

    def kind(c: str) -> str:
        if c[0] == "0":
            return "military"
        if c[0] == "1":
            # a manager group recorded as a whole, whose two levels the
            # index scores separately
            if c.endswith("0") and {c[:3] + "1", c[:3] + "2"} <= scored:
                return "managerial, levels not split"
            return "managerial, other"
        return "other"

    lst = pd.DataFrame({"ssyk4": unscored})
    lst["kind"] = lst["ssyk4"].map(kind)
    ads = post.groupby("ssyk4")["n_ads"].sum()
    lst["n_ads"] = lst["ssyk4"].map(ads).astype(int)
    lst.to_csv(config.RESULTS / "unscored_codes_list.csv", index=False)

    n = lst["kind"].value_counts()
    total_ads = int(post["n_ads"].sum())
    share = 100 * lst["n_ads"].sum() / total_ads
    stats = {
        "codes_in_advertisements": len(codes),
        "codes_scored": len(codes & scored),
        "codes_unscored": len(unscored),
        "unscored_military": int(n.get("military", 0)),
        "unscored_managerial": int(n.get("managerial, levels not split", 0)
                                   + n.get("managerial, other", 0)),
        "unscored_managerial_levels_not_split": int(n.get("managerial, levels not split", 0)),
        "unscored_other": int(n.get("other", 0)),
        "unscored_nonmilitary": len(unscored) - int(n.get("military", 0)),
        "advertisements_total": total_ads,
        "advertisements_unscored": int(lst["n_ads"].sum()),
        "advertisements_unscored_pct": round(share, 4),
        "advertisements_scored_pct": round(100 - share, 4),
    }
    out = pd.DataFrame(list(stats.items()), columns=["statistic", "value"])
    out.to_csv(config.RESULTS / "unscored_codes.csv", index=False)
    for k, v in stats.items():
        print(f"  {k:40s} {v}")
    print("  other codes:", ", ".join(lst.loc[lst["kind"] == "other", "ssyk4"]))
    print("  managerial, other:",
          ", ".join(lst.loc[lst["kind"] == "managerial, other", "ssyk4"]))


if __name__ == "__main__":
    main()
