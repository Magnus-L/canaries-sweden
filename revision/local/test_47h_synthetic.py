#!/usr/bin/env python3
"""
test_47h_synthetic.py -- run 47h end to end, locally, with no MONA.

Injects synthetic frames at the pull-function boundary (the SQL is the only
thing that cannot run here), uses the REAL key, DAIOE and Eloundou inputs
from revision/upload, the REAL scoring, assignment, balancing and R + fixest
estimation. Nothing estimated here means anything: the data are generated.
The assertions are about mechanics: it runs, every file appears, the fast
balanced panel equals mona_common's, the degraded paths degrade instead of
crashing, caches are reused, and the gate halts when it should.

    CANARIES_DRYRUN=1 python3 revision/local/test_47h_synthetic.py
"""

import importlib.util
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["CANARIES_DRYRUN"] = "1"
HERE = Path(__file__).resolve().parent
MONA = HERE.parent / "mona"
UPLOAD = HERE.parent / "upload"
TMP = Path(tempfile.mkdtemp(prefix="canaries47h_"))
SHARE = TMP / "input"
SHARE.mkdir()
for f in ("daioe_quartiles.dta", "eloundou_ssyk4.dta",
          "utb_grupp2_sun2020_niva3_inr4_nyckel.dta"):
    shutil.copy(UPLOAD / f, SHARE / f)
os.environ["CANARIES_SHARE"] = str(SHARE)

sys.path.insert(0, str(MONA))
import mona_common as mc  # noqa: E402

spec = importlib.util.spec_from_file_location("s47h", MONA / "47h_edu_horserace.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

RNG = np.random.default_rng(20260918)
N_EMP = 300
MONTHS = {y: [f"{y}-{m:02d}" for m in range(1, 13)] for y in range(2019, 2024)}
AGES = ["22-25", "26-30", "31-34", "35-40", "41-49", "50+"]
KEY = mod.load_key()
DAIOE = pd.read_stata(SHARE / "daioe_quartiles.dta")
SSYK = DAIOE["ssyk4"].astype(str).str.zfill(4).to_numpy()
SSYK_HI = DAIOE.loc[DAIOE["high_exposure"] == 1, "ssyk4"].astype(str).str.zfill(4).to_numpy()
SSYK_LO = DAIOE.loc[DAIOE["high_exposure"] == 0, "ssyk4"].astype(str).str.zfill(4).to_numpy()

# universe of education cells: 40 tertiary, 20 gymnasial, from the real key
ter = KEY[KEY["niva"].str[:1].isin(["4", "5", "6"])].sample(40, random_state=1)
gym = KEY[KEY["niva"].str[:1] == "3"].sample(20, random_state=2)
CELLS = pd.concat([ter, gym]).reset_index(drop=True)
CELLS["tilt"] = np.where(CELLS["niva"].str[:1].isin(["4", "5", "6"]),
                         RNG.uniform(0.2, 0.9, len(CELLS)), RNG.uniform(0.0, 0.4, len(CELLS)))


def synth_weights(year, conn):
    rows = []
    for _, c in CELLS.iterrows():
        codes = np.concatenate([RNG.choice(SSYK_HI, 3), RNG.choice(SSYK_LO, 3)])
        for code in codes:
            hi = code in set(SSYK_HI)
            base = 600 if (hi == (c["tilt"] > 0.5)) else 150
            for fresh in (0, 1):
                for band in mod.EXP_BANDS:
                    for young in (0, 1):
                        scale = 1.0 if band in ("0-2", "3-5") else 0.7
                        n = int(RNG.poisson(base * scale * (0.6 if fresh else 0.4)
                                            * (0.7 if young else 0.3))) + 1
                        rows.append((c["niva"], c["inr"], code, fresh, band, young, n))
    return pd.DataFrame(rows, columns=["niva", "inr", "ssyk4", "fresh", "expband", "young", "n"])


def synth_year(year, conn, enrol_ok):
    rows = []
    stale_p = {2021: 0.45, 2022: 0.25}
    for emp in range(1000, 1000 + N_EMP):
        cell_ids = RNG.choice(len(CELLS), 8, replace=False)
        for ym in MONTHS[year]:
            for cid in cell_ids:
                c = CELLS.iloc[cid]
                for age in RNG.choice(AGES, 3, replace=False):
                    n = int(RNG.poisson(3)) + 1
                    expb_t = RNG.choice(["0-2", "3-5"]) if age in ("22-25", "26-30") \
                        else RNG.choice(["6-10", "11-20", "21+", "na"])
                    rec = dict(employer_id=emp, year_month=ym, niva_t=c["niva"],
                               inr_t=c["inr"], expb_t=expb_t, age_group=age, n_emp=n)
                    for T in (2021, 2022):
                        p = stale_p[T] * (1.0 if age == "22-25" else 0.35 if age == "26-30" else 0.02)
                        if year > T and RNG.uniform() < p and c["niva"][:1] in "456":
                            g = gym.iloc[RNG.integers(len(gym))]
                            rec[f"niva_{T%100}"], rec[f"inr_{T%100}"] = g["niva"], g["inr"]
                            rec[f"expb_{T%100}"] = RNG.choice(["3-5", "6-10"])
                            rec[f"enr_{T%100}"] = c["inr"] if (enrol_ok and RNG.uniform() < 0.6) else None
                        else:
                            rec[f"niva_{T%100}"], rec[f"inr_{T%100}"] = c["niva"], c["inr"]
                            rec[f"expb_{T%100}"] = expb_t
                            rec[f"enr_{T%100}"] = None
                    # the legacy (47b) cascade columns the gate compares
                    # against. Deliberately DIFFERENT from the corrected ones
                    # for a share of rows, so a test that silently aliased the
                    # two as-of arms would fail rather than pass.
                    if RNG.uniform() < 0.3:
                        g = gym.iloc[RNG.integers(len(gym))]
                        rec["niva_21g"], rec["inr_21g"] = g["niva"], g["inr"]
                    else:
                        rec["niva_21g"] = rec["niva_21"]
                        rec["inr_21g"] = rec["inr_21"]
                    rows.append(rec)
    df = pd.DataFrame(rows)
    # a few null employers must be dropped, not crash
    df.loc[df.sample(3, random_state=3).index, "employer_id"] = np.nan
    return mod.compact(df[mod.YEAR_COLS + ["n_emp"]])


_STDOUT = sys.stdout


def wire(tmp: Path, enrol=True, pull_year=None):
    # each main() installs a Tee over sys.stdout; unwind before the next run
    # or every later line is echoed once per earlier run
    sys.stdout = _STDOUT
    out = tmp / "output_47h"
    cache = tmp / "cache"
    out.mkdir(exist_ok=True)
    cache.mkdir(exist_ok=True)
    mod.OUT, mod.CACHE = out, cache
    mc.connect = lambda: object()
    mod.probe_enrolment = lambda conn: enrol
    mod.pull_weights = synth_weights
    mod.pull_year = pull_year or synth_year
    mod.GATE_WARN, mod.GATE_HALT = float("inf"), float("inf")   # synthetic data cannot reproduce 47b
    return out, cache


def test_fast_balance_equals_mona_common():
    sub = pd.DataFrame({
        "employer_id": RNG.integers(1, 40, 600),
        "exposure_quartile": RNG.integers(1, 5, 600),
        "year_month": RNG.choice([f"2020-{m:02d}" for m in range(1, 13)], 600),
        "n_emp": RNG.integers(1, 9, 600)})
    sub = sub.groupby(["employer_id", "exposure_quartile", "year_month"])["n_emp"].sum().reset_index()
    months = sorted(sub["year_month"].unique())
    a = mc.balance_panel(sub, months)
    b = mod.fast_balance(sub, months)
    key = ["employer_id", "exposure_quartile", "year_month"]
    a = a.sort_values(key).reset_index(drop=True)[key + ["n_emp"]]
    b = b.sort_values(key).reset_index(drop=True)[key + ["n_emp"]]
    a["exposure_quartile"] = a["exposure_quartile"].astype(int)
    b["exposure_quartile"] = b["exposure_quartile"].astype(int)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)
    assert (b["n_emp"] == 0).any(), "zero-fill missing"
    print("PASS fast_balance == mona_common.balance_panel "
          f"({len(b):,} rows, {int((b['n_emp'] == 0).sum())} zero-filled)")


def test_happy_path():
    tmp = TMP / "happy"; tmp.mkdir()
    out, cache = wire(tmp, enrol=True)
    t0 = time.time()
    mod.main()
    est = pd.read_csv(out / "horserace_estimates.csv")
    n_designs = len(mod.DESIGNS)
    tier_b_designs = est.loc[est["tier"] == "B", "design"].nunique()
    expected = (3 + (n_designs * 4 - 2)      # the gate now runs three arms
                + tier_b_designs * 2 * 2 * 2
                + 2 * 3 * 2)
    assert len(est) == expected, (len(est), expected)
    assert est["gamma2"].notna().all(), est[est["gamma2"].isna()]
    assert (est["status"] == "ok").all(), est["status"].value_counts()
    for name in mod.DESIGNS:
        assert (out / f"score_{name}.csv").exists(), name
        sc = pd.read_csv(out / f"score_{name}.csv")
        assert (sc["n_workers"] >= 5).all()
        assert set(sc["quartile"]) <= {1, 2, 3, 4}
    for f in ("47h_summary.txt", "anchoring_rates.csv", "score_diagnostics.csv",
              "47h_log.txt", "gate_decomposition.csv"):
        assert (out / f).exists(), f
    gd = pd.read_csv(out / "gate_decomposition.csv")
    assert set(gd["arm"]) == {"true", "asof", "asof_legacy"}, gd["arm"].tolist()
    assert gd.loc[gd.arm == "asof_legacy", "gamma2"].notna().all()
    assert (gd.loc[gd.arm == "asof_legacy", "gamma2"].iloc[0]
            != gd.loc[gd.arm == "asof", "gamma2"].iloc[0]), \
        "the legacy arm must not be an alias of the corrected one"
    summ = (out / "47h_summary.txt").read_text()
    assert "ARTEFACT BY DESIGN" in summ and "Tier C" in summ and "NOTE" not in summ
    ar = pd.read_csv(out / "anchoring_rates.csv")
    assert (ar["anchored_share"] > 0).any(), "enrolment anchoring never fired"
    # 3 weight pulls + 5 year pulls + (designs x arms x truncations x years)
    # + the gate's legacy pass: OL_daioe, one arm, T=2021 only, 5 years
    assert len(list(cache.glob("edu_hr_*.parquet"))) == 3 + 5 + n_designs * 2 * 2 * 5 + 5, \
        len(list(cache.glob("edu_hr_*.parquet")))
    # wiring guard: the as-of pieces must differ from the true pieces at BOTH
    # truncations, and the two truncations from each other, in years after T
    for name in ("OL_daioe", "enrol"):
        for T in (2021, 2022):
            tr = pd.read_parquet(cache / f"edu_hr_coll_{name}_true_T{T}_2023.parquet")
            af = pd.read_parquet(cache / f"edu_hr_coll_{name}_asof_T{T}_2023.parquet")
            key = ["employer_id", "year_month", "exposure_quartile", "age_group"]
            m = tr.merge(af, on=key, how="outer", suffixes=("_t", "_a")).fillna(0)
            assert (m["n_emp_t"] != m["n_emp_a"]).any(), (name, T, "as-of == true")
        a21 = pd.read_parquet(cache / f"edu_hr_coll_{name}_asof_T2021_2023.parquet")
        a22 = pd.read_parquet(cache / f"edu_hr_coll_{name}_asof_T2022_2023.parquet")
        assert not a21.equals(a22), (name, "T2021 as-of == T2022 as-of")
    print(f"PASS happy path: {len(est)} fits, all status ok, {time.time()-t0:.0f}s")
    return tmp


def test_cache_reuse(tmp):
    def boom(*a, **k):
        raise AssertionError("pull called although the cache exists")
    wire(tmp, enrol=True, pull_year=boom)
    mod.pull_weights = boom
    mod.main()
    print("PASS cache reuse: second run made no pull")


def test_enrolment_degrades():
    tmp = TMP / "noenrol"; tmp.mkdir()
    out, _ = wire(tmp, enrol=False)
    mod.main()
    summ = (out / "47h_summary.txt").read_text()
    assert "NOTE: enrolment register probe failed" in summ
    assert not (out / "anchoring_rates.csv").exists()
    print("PASS degraded path: no enrolment -> designs run unanchored, said so")


def test_resume(tmp):
    """
    47h reached 95 GB against a 100 GB per-job cap on 19 Sep with about
    seventy fits behind it. A memory kill would have discarded every one,
    because the script deleted its results file at the start of each run.
    It now reuses them, guarded by a code tag so two versions of the
    script can never be mixed into one table.
    """
    out, _ = wire(tmp, enrol=True)
    est = out / "horserace_estimates.csv"
    before = pd.read_csv(est)
    assert "code_tag" in before.columns, "results carry no code tag"
    n_before = len(before)

    real = mod.estimate
    mod.estimate = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("refitted a cell already on disk"))
    try:
        mod.main()
    finally:
        mod.estimate = real
    after = pd.read_csv(est)
    assert len(after) == n_before, (n_before, len(after))
    print(f"PASS a restart reuses all {n_before} fits and refits nothing")

    tampered = before.copy()
    tampered["code_tag"] = "deadbeef1234"
    tampered.to_csv(est, index=False)
    mod.main()
    fresh = pd.read_csv(est)
    assert (fresh["code_tag"] != "deadbeef1234").all(), \
        f"rows from another code version were reused: {fresh['code_tag'].unique()[:3]}"
    print("PASS rows written under a different code tag are discarded")


def test_gate_halts():
    """
    The TRUE arm still halts. If 47h cannot reproduce 47b where the two
    should agree almost exactly, the pull is in doubt and nothing
    downstream is worth computing.
    """
    tmp = TMP / "gate"; tmp.mkdir()
    out, _ = wire(tmp, enrol=True)
    mod.GATE_WARN, mod.GATE_HALT = 0.005, 0.05
    # break the TRUE arm specifically. The previous version of this test
    # returned 0.0 for every arm and therefore halted on the LEGACY arm,
    # which is the behaviour that changed on 20 Sep; it would have passed
    # while testing nothing.
    mod.estimate = lambda coll, age, tag: dict(
        gamma2=(-0.9 if "_true_" in tag else mod.GATE_47B["asof"]),
        se=0.01, p=0.5, n_obs=1, status="ok", elapsed_s=0.0)
    try:
        mod.main()
    except SystemExit as ex:
        assert "TRUE arm" in str(ex), str(ex)
        print("PASS the gate still halts when the TRUE arm does not reproduce 47b")
        return
    raise AssertionError("gate did not halt on a bad true arm")


def test_gate_warns_on_legacy():
    """
    The LEGACY arm only warns. On 19 Sep it came back identical to the
    corrected arm, so 47b's cascade is not what separates the scripts, and
    refusing to estimate the other seven designs over an unexplained
    discrepancy with a superseded script buys a reconciliation rather than
    a result. The discrepancy must still reach the summary.
    """
    tmp = TMP / "gatewarn"; tmp.mkdir()
    out, _ = wire(tmp, enrol=True)
    mod.GATE_WARN, mod.GATE_HALT = 0.005, 0.05
    # true arm agrees with 47b; legacy arm does not
    def fake(coll, age, tag):
        g = mod.GATE_47B["true"] if "_true_" in tag else -0.9
        return dict(gamma2=g, se=0.01, p=0.5, n_obs=1, status="ok",
                    elapsed_s=0.0)
    mod.estimate = fake
    mod.main()
    summ = (out / "47h_summary.txt").read_text()
    assert "UNRECONCILED" in summ, "the discrepancy never reached the summary"
    est = pd.read_csv(out / "horserace_estimates.csv")
    assert len(est) > 3, f"only {len(est)} fits: the run did not proceed past the gate"
    print(f"PASS the legacy arm warns, records UNRECONCILED, and the run "
          f"proceeds ({len(est)} fits)")


if __name__ == "__main__":
    test_fast_balance_equals_mona_common()
    happy = test_happy_path()
    test_cache_reuse(happy)
    test_enrolment_degrades()
    test_resume(happy)
    test_gate_halts()
    test_gate_warns_on_legacy()
    print(f"\nALL PASS  (tmp: {TMP})")
