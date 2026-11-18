"""Portfolio accounting: feasibility, weights, annualization, drift, execution timing, costs, delisting."""
import numpy as np
import pandas as pd
import pytest

from conftest import make_prices
from mfd import portfolio as P
from mfd import prep
from mfd.calendar import first_session_after

SYMS = ["A", "B", "C", "D"]


@pytest.fixture(scope="module")
def setup():
    p = make_prices(SYMS + ["M"], "2014-01-01", "2018-12-31", seed=3)
    out = prep.build_panel(p, None, None, SYMS, "M")
    mw = out["monthly_returns"].pivot(index="month", columns="symbol", values="ret")
    return p, out, mw


def test_infeasible_cap_is_reported(setup):
    _, _, mw = setup
    with pytest.raises(P.PortfolioInfeasible, match="impossible"):
        P.min_vol_weights(mw[SYMS].iloc[-36:], max_weight=0.2)


def test_weights_sum_to_one_within_bounds(setup):
    _, _, mw = setup
    w, S, info = P.min_vol_weights(mw[SYMS].iloc[-36:], max_weight=0.3)
    assert w.sum() == pytest.approx(1.0, abs=1e-8)
    assert (w >= -1e-10).all() and (w <= 0.3 + 1e-6).all()


def test_monthly_annualization_frequency_12(setup):
    from sklearn.covariance import ledoit_wolf
    _, _, mw = setup
    R = mw[SYMS].iloc[-36:]
    _, S, _ = P.min_vol_weights(R, max_weight=0.5)
    expected = ledoit_wolf(R.to_numpy())[0] * 12
    assert np.allclose(S.to_numpy(), expected, rtol=1e-6)


def test_execution_timing_drift_and_costs(setup):
    p, out, mw = setup
    res, wh, _ = P.backtest(out["tri"], mw, out["month_calendar"], SYMS, "M", lookback=24, max_weight=0.5,
                            fee_bps=5, slippage_bps=10, rebalance_every=1, charge_initial=True)
    sessions = out["tri"].index
    mc = out["month_calendar"].set_index("month")
    eq = res[res.strategy == "equal_weight"].sort_values("execution").reset_index(drop=True)
    # execution = first session after the decision month's last session
    for _, r in eq.head(5).iterrows():
        assert r["execution"] == first_session_after(mc.loc[r["decision_month"], "last_session"], sessions)
    # first rebalance buys from cash: traded = 1, cost = 15 bp
    assert eq.loc[0, "traded_notional"] == pytest.approx(1.0)
    assert eq.loc[0, "cost"] == pytest.approx(0.0015)
    # drift: pre-trade weights at rebalance k+1 = target_k·(1+r_i)/(1+r_p) over execution→next execution
    tri = out["tri"]
    k = 3
    r0 = eq.loc[k]
    rr = pd.Series({s: tri[s][r0["next_execution"]] / tri[s][r0["execution"]] - 1 for s in SYMS})
    w = pd.Series(0.25, index=SYMS)
    gross = float((w * rr).sum())
    assert r0["gross_return"] == pytest.approx(gross)
    drift = w * (1 + rr) / (1 + gross)
    nxt = wh[(wh.strategy == "equal_weight") & (wh.execution == eq.loc[k + 1, "execution"])].set_index("symbol")
    assert np.allclose(nxt.loc[SYMS, "pre_trade_weight"].to_numpy(), drift.to_numpy())
    traded = float((0.25 - drift).abs().sum())
    assert eq.loc[k + 1, "traded_notional"] == pytest.approx(traded)
    assert eq.loc[k + 1, "half_turnover"] == pytest.approx(traded / 2)
    assert eq.loc[k + 1, "cost"] == pytest.approx(traded * 0.0015)
    assert eq.loc[k + 1, "net_return"] == pytest.approx((1 - traded * 0.0015) * (1 + eq.loc[k + 1, "gross_return"]) - 1)
    # zero-cost sensitivity: gross equals net when costs are zero
    res0, _, _ = P.backtest(out["tri"], mw, out["month_calendar"], SYMS, "M", 24, 0.5, 0, 0, 1, True)
    assert np.allclose(res0["gross_return"], res0["net_return"])
    # all strategies share the same execution dates
    assert res.groupby("strategy")["execution"].apply(tuple).nunique() == 1


def test_no_rebalance_means_pure_drift(setup):
    _, out, mw = setup
    res, wh, _ = P.backtest(out["tri"], mw, out["month_calendar"], SYMS, "M", 24, 0.5, 5, 10, rebalance_every=12)
    eq = res[res.strategy == "equal_weight"].sort_values("execution")
    assert (eq["rebalanced"].iloc[1:12] == False).all()  # noqa: E712
    assert (eq["traded_notional"].iloc[1:12] == 0).all()


def test_delisted_holding_becomes_cash():
    p = make_prices(SYMS + ["M"], "2014-01-01", "2018-12-31", seed=4)
    p = p[~((p.symbol == "D") & (p.session_date > "2017-06-14"))]
    out = prep.build_panel(p, None, None, SYMS, "M")
    mw = out["monthly_returns"].pivot(index="month", columns="symbol", values="ret")
    res, wh, _ = P.backtest(out["tri"], mw, out["month_calendar"], SYMS, "M", 24, 0.5, 5, 10, 1)
    eq = res[res.strategy == "equal_weight"]
    hit = eq[eq["notes"].str.contains("D stops trading")]
    assert len(hit) == 1
    later = wh[(wh.strategy == "equal_weight") & (wh.execution > hit["execution"].iloc[0])]
    assert "D" not in set(later["symbol"])


def test_performance_summary_annualization():
    res = pd.DataFrame({"strategy": "x", "decision_month": pd.period_range("2020-01", periods=24, freq="M"),
                        "execution": pd.date_range("2020-02-03", periods=24, freq="MS"),
                        "next_execution": pd.date_range("2020-03-02", periods=24, freq="MS"),
                        "gross_return": 0.01, "net_return": 0.01, "traded_notional": 0.1, "cost": 0.0})
    s = P.performance_summary(res).iloc[0]
    assert s["net_ann_return"] == pytest.approx(1.01 ** 12 - 1)
    assert s["avg_annual_traded_notional"] == pytest.approx(1.2)
