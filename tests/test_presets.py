"""US tech preset: NYSE calendar, profitability feature, control factor, per-symbol start, interaction recovery."""
import numpy as np
import pandas as pd
import pytest

from conftest import make_prices
from mfd import prep, regression as R
from mfd.calendar import exchange_sessions
from mfd.config import PROFILES


@pytest.fixture(scope="module")
def us():
    from mfd.providers.synthetic import generate
    d = generate("us_tech")
    p = prep.build_panel(d["prices"], d["macro"], d["fundamentals"], list(d["truth"]["symbol"]), "SYN-NDX",
                         exchange="NYSE", control=d["control"])
    return d, p


def test_nyse_holidays():
    s = exchange_sessions("2025-01-01", "2025-12-31", "NYSE")
    for d in ["2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26", "2025-06-19", "2025-07-04",
              "2025-09-01", "2025-11-27", "2025-12-25"]:
        assert pd.Timestamp(d) not in s
    assert pd.Timestamp("2025-10-13") in s  # Columbus Day: NYSE open (TSX closed for Thanksgiving)
    assert pd.Timestamp("2025-10-13") not in exchange_sessions("2025-10-01", "2025-10-31", "TSX")


def test_profiles_are_consistent():
    for p in PROFILES.values():
        assert p.default_benchmark in p.benchmarks
        assert p.interaction[0] in p.default_factors
        syms = [m["symbol"] for m in p.universe]
        assert len(syms) == len(set(syms))
        if p.control_symbol:
            assert p.control_symbol in syms


def test_control_factor_and_profitability_columns(us):
    d, p = us
    pan = p["panel"]
    ctrl = p["monthly_returns"]
    ctrl = ctrl[ctrl.symbol == "SYN-CTRL"].set_index("month")["ret"]
    g = pan[pan.symbol == "SYN-IP"].set_index("month")
    m = pd.Period("2022-06", "M")
    assert g.loc[m, "control"] == pytest.approx(ctrl[m])
    assert g.loc[m, "lag_control"] == pytest.approx(ctrl[m])
    assert g["profitability"].notna().sum() > 100
    # profitability = net income / assets of the latest report known before the month
    assert g.loc[m, "profitability"] == pytest.approx(g.loc[m - 1, "lag_profitability"])


def test_us_interaction_recovered(us):
    d, p = us
    pan = p["panel"]
    truth = d["truth"].set_index("symbol")
    for sym in ["SYN-QNT", "SYN-DC"]:
        df = pan[pan.symbol == sym].set_index("month")
        r = R.fit_interaction(df, "ret", "y10", "profitability", ["market", "control", "vix", "usd"], sym)
        t = truth.loc[sym, "y10_x_profitability"]
        lo, hi = r.conf_int.loc["y10_x_profitability", "lo"], r.conf_int.loc["y10_x_profitability", "hi"]
        assert lo < t < hi
        assert r.params["y10_x_profitability"] > 0


def test_analysis_start_drops_predecessor_history():
    p = make_prices(["A", "B", "M"], "2016-01-01", "2019-12-31")
    out = prep.build_panel(p, None, None, ["A", "B"], "M", analysis_start={"A": "2018-06-01"})
    a = out["panel"][(out["panel"].symbol == "A")].dropna(subset=["ret"])
    assert a["month"].min() >= pd.Period("2018-07", "M")
    b = out["panel"][(out["panel"].symbol == "B")].dropna(subset=["ret"])
    assert b["month"].min() < pd.Period("2017-01", "M")
