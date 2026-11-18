"""Model calculations: identification checks, HAC, BH, interaction recovery and its covariance-based SE."""
import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from mfd import regression as R


def _sim(n=600, seed=1, b_oil=0.5, b_int=1.5, b_debt=-0.05, center=0.30):
    rng = np.random.default_rng(seed)
    months = pd.period_range("1970-01", periods=n, freq="M")
    oil = rng.normal(0, 0.09, n)
    mkt = rng.normal(0.007, 0.04, n)
    debt = np.clip(0.3 + 0.1 * np.sin(np.arange(n) / 15) + rng.normal(0, 0.03, n), 0.05, 0.7)
    y = 0.002 + 0.9 * mkt + (b_oil + b_int * (debt - center)) * oil + b_debt * (debt - center) + rng.normal(0, 0.03, n)
    return pd.DataFrame({"ret": y, "market": mkt, "oil": oil, "debt_to_assets": debt}, index=months)


def test_constant_feature_rejected():
    df = _sim(100)
    df["k"] = 1.0
    with pytest.raises(R.ModelSpecError, match="constant"):
        R.fit_ols(df, "ret", ["market", "k"], "X")


def test_rank_deficient_rejected():
    df = _sim(100)
    df["m2"] = 2 * df["market"]
    with pytest.raises(R.ModelSpecError, match="rank deficient"):
        R.fit_ols(df, "ret", ["market", "m2"], "X")


def test_insufficient_history_rejected():
    df = _sim(12)  # 12 months, 4 parameters → 8 residual df < 10
    with pytest.raises(R.ModelSpecError, match="insufficient history"):
        R.fit_ols(df, "ret", ["market", "oil", "debt_to_assets"], "X")


def test_missing_rows_dropped_and_counted():
    df = _sim(120)
    df.iloc[:10, 1] = np.nan
    r = R.fit_ols(df, "ret", ["market", "oil"], "X")
    assert r.nobs == 110 and r.dropped_rows == 10


def test_hac_lag_rules():
    assert R.newey_west_lags(100) == 4
    assert R.resolve_lags(180, "fixed", 3) == 3
    assert R.resolve_lags(8, "fixed", 12) == 2  # capped at n/4


def test_hac_matches_statsmodels_directly():
    import statsmodels.api as sm
    df = _sim(200)
    r = R.fit_ols(df, "ret", ["market", "oil"], "X", "fixed", 3)
    m = sm.OLS(df["ret"], sm.add_constant(df[["market", "oil"]])).fit(cov_type="HAC", cov_kwds={"maxlags": 3, "use_correction": True}, use_t=True)
    assert np.allclose(r.bse.values, m.bse.values)


def test_bh_q_values():
    df = _sim(200)
    fits = {"A": R.fit_ols(df, "ret", ["market", "oil"], "A"), "B": R.fit_ols(_sim(200, seed=5), "ret", ["market", "oil"], "B")}
    t = R.effects_table(fits)
    nc = t[t.term != "const"]
    assert np.allclose(nc["q_value"].values, multipletests(nc["p_value"].values, method="fdr_bh")[1])
    assert t[t.term == "const"]["q_value"].isna().all()


def test_known_interaction_recovered_with_covariance_based_uncertainty():
    df = _sim(800)
    r = R.fit_interaction(df, "ret", "oil", "debt_to_assets", ["market"], "SIM")
    c = r.centers["debt_to_assets"]
    # interaction coefficient recovered (within 3 SE) and CI covers the truth
    assert abs(r.params["oil_x_debt_to_assets"] - 1.5) < 3 * r.bse["oil_x_debt_to_assets"]
    assert r.conf_int.loc["oil_x_debt_to_assets", "lo"] < 1.5 < r.conf_int.loc["oil_x_debt_to_assets", "hi"]
    grid = np.array([0.15, 0.30, 0.45])
    cs = R.conditional_slope(r, "oil", "debt_to_assets", grid)
    truth = 0.5 + 1.5 * (grid - 0.30)
    assert np.all(np.abs(cs["slope"] - truth) < 3 * cs["se"])
    # Exact check: re-centering debt at the grid value makes β_oil equal to the conditional slope,
    # so its HAC SE must equal the covariance-based SE (including Cov(β_oil, β_int)).
    for g, se in zip(grid, cs["se"]):
        d2, _ = R.add_interaction(df, "oil", "debt_to_assets", center=g)
        r2 = R.fit_ols(d2, "ret", ["market", "oil", "debt_to_assets_c", "oil_x_debt_to_assets"], "SIM")
        assert r2.params["oil"] == pytest.approx(r.params["oil"] + r.params["oil_x_debt_to_assets"] * (g - c), rel=1e-8)
        assert r2.bse["oil"] == pytest.approx(se, rel=1e-8)
    # and ignoring the covariance term gives a different (wrong) SE away from the center
    V = r.cov
    naive = np.sqrt(V.loc["oil", "oil"] + (0.45 - c) ** 2 * V.loc["oil_x_debt_to_assets", "oil_x_debt_to_assets"])
    assert not np.isclose(naive, cs["se"].iloc[2], rtol=1e-3)


def test_constant_moderator_reports_identification_limit():
    df = _sim(200)
    df["debt_to_assets"] = 0.3
    with pytest.raises(R.ModelSpecError, match="constant over the estimation window"):
        R.fit_interaction(df, "ret", "oil", "debt_to_assets", ["market"], "X")


def test_scenario_uses_log_conversion_and_flags_extrapolation():
    df = _sim(300)
    r = R.fit_interaction(df, "ret", "oil", "debt_to_assets", ["market"], "X")
    import math
    a = math.log(1.10)
    out = R.scenario_mean(r, {"oil": a, "market": 0.0, "debt_to_assets_c": 0.0, "oil_x_debt_to_assets": 0.0})
    assert out["estimate"] == pytest.approx(r.params["const"] + r.params["oil"] * a)
    far = R.scenario_mean(r, {"oil": 5.0})
    assert far["extrapolation"]


def test_rolling_uses_only_window_observations():
    df = _sim(150)
    roll, _ = R.rolling_fits(df, "ret", ["market", "oil"], window=60, min_obs=60)
    last = roll[(roll.term == "oil")].iloc[-1]
    direct = R.fit_ols(df.iloc[-60:], "ret", ["market", "oil"], "")
    assert last["estimate"] == pytest.approx(direct.params["oil"])
    assert last["n"] == 60
    assert roll["overlap_with_previous"].iloc[0] == pytest.approx(59 / 60)
