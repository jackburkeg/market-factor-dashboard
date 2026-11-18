"""Time alignment: return shifts, month boundaries, incomplete months, release-time joins,
dividend/split treatment, macro availability."""
import numpy as np
import pandas as pd
import pytest

from conftest import make_prices, ts
from mfd import prep
from mfd.calendar import latest_completed_month, decision_cutoff, tsx_sessions
from mfd.providers.market import apply_split_adjustment


def test_month_return_uses_last_session_of_each_month():
    p = make_prices(["A"], "2017-01-01", "2017-04-30")
    mcal = prep.month_calendar(p, pd.Period("2017-01", "M"), pd.Period("2017-04", "M"))
    tri = prep.total_return_index(p)
    mret, exc = prep.monthly_returns(tri, mcal, pd.Period("2017-04", "M"))
    s = p.set_index("session_date")["adjusted_close"]
    feb_end = s[s.index.to_period("M") == pd.Period("2017-02", "M")].iloc[-1]
    jan_end = s[s.index.to_period("M") == pd.Period("2017-01", "M")].iloc[-1]
    got = mret.set_index("month").loc[pd.Period("2017-02", "M"), "ret"]
    assert got == pytest.approx(feb_end / jan_end - 1, rel=1e-12)
    # first month has no prior month-end → no return
    assert pd.Period("2017-01", "M") not in set(mret["month"])


def test_target_is_next_month_and_lags_are_current(synth_panel):
    pan = synth_panel["panel"]
    g = pan[pan.symbol == "SYN-EPA"].set_index("month")
    m = pd.Period("2020-05", "M")
    assert g.loc[m, "target_next"] == pytest.approx(g.loc[m + 1, "ret"])
    assert g.loc[m, "lag_own"] == pytest.approx(g.loc[m, "ret"])
    assert g.loc[m, "target_month"] == m + 1
    # descriptive fundamentals for month t equal point-in-time values at cutoff t-1
    assert g.loc[m, "debt_to_assets"] == pytest.approx(g.loc[m - 1, "lag_debt_to_assets"])


def test_incomplete_month_excluded():
    # data ending mid-month: that month must be dropped from monthly returns
    p = make_prices(["A", "B"], "2017-01-01", "2017-05-16")
    out = prep.build_panel(p, None, None, ["A"], "B")
    assert out["end_month"] == pd.Period("2017-04", "M")
    assert any("dropped" in n for n in out["notes"])
    assert out["panel"]["month"].max() == pd.Period("2017-04", "M")


def test_latest_completed_month_respects_cutoff():
    m = pd.Period("2026-10", "M")
    cut = decision_cutoff(m)
    assert latest_completed_month(cut - pd.Timedelta(minutes=1)) == m - 1
    assert latest_completed_month(cut + pd.Timedelta(minutes=1)) == m


def test_partial_month_for_delisted_symbol_is_excluded(synth_panel):
    exc = synth_panel["excluded"]
    row = exc[(exc.symbol == "SYN-GAS") & (exc.month == "2025-06")]
    assert len(row) == 1 and "partial month" in row["reason"].iloc[0]
    pan = synth_panel["panel"]
    assert pan[(pan.symbol == "SYN-GAS") & (pan.month >= pd.Period("2025-06", "M"))]["ret"].isna().all()


def test_tsx_holidays():
    s = tsx_sessions("2024-01-01", "2024-12-31")
    for d in ["2024-01-01", "2024-02-19", "2024-03-29", "2024-05-20", "2024-07-01", "2024-08-05", "2024-09-02",
              "2024-10-14", "2024-12-25", "2024-12-26"]:
        assert pd.Timestamp(d) not in s
    assert pd.Timestamp("2024-11-11") in s  # Remembrance Day: TSX open


# ------------------------------------------------------------------ fundamentals release-time joins
def _fund(rows):
    df = pd.DataFrame(rows, columns=["symbol", "fiscal_period_end", "available_at", "metric", "value", "revision"])
    df["fiscal_period_end"] = pd.to_datetime(df["fiscal_period_end"])
    df["available_at"] = df["available_at"].apply(ts)
    df["period_type"] = "Q"
    return df


def _mcal(months):
    ms = pd.period_range(months[0], months[1], freq="M")
    s = tsx_sessions(ms[0].start_time, ms[-1].end_time)
    last = [s[(s >= m.start_time) & (s <= m.end_time)].max() for m in ms]
    m = pd.DataFrame({"month": ms, "last_session": last})
    m["decision_at"] = (m["last_session"] + pd.Timedelta(hours=17)).dt.tz_localize("America/Toronto")
    return m


def test_fundamentals_enter_only_after_available_at():
    f = _fund([
        ("A", "2019-12-31", "2020-02-14 08:00", "interest_bearing_debt", 30.0, "1"),
        ("A", "2019-12-31", "2020-02-14 08:00", "total_assets", 100.0, "1"),
    ])
    out = prep.fundamentals_point_in_time(f, ["A"], _mcal(("2020-01", "2020-03"))).set_index("month")
    assert np.isnan(out.loc[pd.Period("2020-01", "M"), "debt_to_assets"])
    assert out.loc[pd.Period("2020-02", "M"), "debt_to_assets"] == pytest.approx(0.30)


def test_same_day_after_cutoff_is_not_used():
    # released at 18:00 on the last session (after the 17:00 cutoff) → not usable that month
    f = _fund([
        ("A", "2019-12-31", "2020-01-31 18:00", "interest_bearing_debt", 30.0, "1"),
        ("A", "2019-12-31", "2020-01-31 18:00", "total_assets", 100.0, "1"),
    ])
    out = prep.fundamentals_point_in_time(f, ["A"], _mcal(("2020-01", "2020-02"))).set_index("month")
    assert np.isnan(out.loc[pd.Period("2020-01", "M"), "debt_to_assets"])
    assert out.loc[pd.Period("2020-02", "M"), "debt_to_assets"] == pytest.approx(0.30)


def test_restatement_does_not_overwrite_earlier_vintage():
    f = _fund([
        ("A", "2019-12-31", "2020-02-14 08:00", "interest_bearing_debt", 30.0, "1"),
        ("A", "2019-12-31", "2020-02-14 08:00", "total_assets", 100.0, "1"),
        ("A", "2019-12-31", "2020-06-10 08:00", "interest_bearing_debt", 40.0, "2"),  # restated later
    ])
    out = prep.fundamentals_point_in_time(f, ["A"], _mcal(("2020-02", "2020-07"))).set_index("month")
    assert out.loc[pd.Period("2020-05", "M"), "debt_to_assets"] == pytest.approx(0.30)
    assert out.loc[pd.Period("2020-06", "M"), "debt_to_assets"] == pytest.approx(0.40)


def test_duplicate_reports_are_deterministic():
    rows = [("A", "2019-12-31", "2020-02-14 08:00", "interest_bearing_debt", 30.0, "1"),
            ("A", "2019-12-31", "2020-02-14 08:00", "interest_bearing_debt", 30.0, "1"),
            ("A", "2019-12-31", "2020-02-14 08:00", "total_assets", 100.0, "1")]
    a = prep.fundamentals_point_in_time(_fund(rows), ["A"], _mcal(("2020-02", "2020-03")))
    b = prep.fundamentals_point_in_time(_fund(rows[::-1]), ["A"], _mcal(("2020-02", "2020-03")))
    assert a["debt_to_assets"].tolist() == b["debt_to_assets"].tolist()


def test_earnings_growth_small_denominator_and_transition():
    f = _fund([
        ("A", "2019-03-31", "2019-05-10 08:00", "net_income", -0.01, "1"),
        ("A", "2019-03-31", "2019-05-10 08:00", "total_assets", 100.0, "1"),
        ("A", "2020-03-31", "2020-05-10 08:00", "net_income", 2.0, "1"),
        ("A", "2020-03-31", "2020-05-10 08:00", "total_assets", 100.0, "1"),
    ])
    out = prep.fundamentals_point_in_time(f, ["A"], _mcal(("2020-05", "2020-05")), small_denominator_frac=0.002)
    r = out.iloc[0]
    assert np.isnan(r["earnings_growth"])                # |−0.01| < 0.2% of 100
    assert r["earnings_transition"] == "loss→profit"
    assert r["earnings_change_scaled"] == pytest.approx(2.01 / 100)


# ------------------------------------------------------------------ dividends & splits
def test_dividends_counted_once(synth):
    p = synth["prices"]
    a = p[p.symbol == "SYN-EPA"]
    tri_adj = prep.total_return_index(a, "adjusted_close")["SYN-EPA"]
    tri_cpd = prep.total_return_index(a, "close_plus_dividends")["SYN-EPA"]
    ra, rc = tri_adj.pct_change().dropna(), tri_cpd.pct_change().dropna()
    assert np.allclose(ra.values, rc.values, atol=1e-10)
    assert (a["dividend"] > 0).sum() > 40
    # adjusted convention ignores the dividend column entirely: doubling it changes nothing
    b = a.assign(dividend=a["dividend"] * 2)
    assert np.allclose(prep.total_return_index(b, "adjusted_close")["SYN-EPA"].values, tri_adj.values)


def test_split_adjustment_of_unadjusted_closes():
    d = pd.DataFrame({"symbol": "A", "session_date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]),
                      "adjusted_close": np.nan, "close": [100.0, 102.0, 51.5, 52.0], "dividend": [0, 1.0, 0, 0],
                      "split_ratio": [1, 1, 2.0, 1], "close_basis": "unadjusted"})
    adj = apply_split_adjustment(d)
    assert adj["close"].tolist() == pytest.approx([50.0, 51.0, 51.5, 52.0])
    assert adj["dividend"].tolist() == pytest.approx([0, 0.5, 0, 0])
    tri = prep.total_return_index(adj, "close_plus_dividends")["A"]
    r = tri.pct_change().dropna().to_numpy()
    assert r == pytest.approx([(51 + 0.5) / 50 - 1, 51.5 / 51 - 1, 52 / 51.5 - 1])


def test_close_plus_dividends_rejects_unadjusted():
    d = pd.DataFrame({"symbol": "A", "session_date": pd.to_datetime(["2020-01-02", "2020-01-03"]), "adjusted_close": np.nan,
                      "close": [1.0, 1.1], "dividend": 0.0, "split_ratio": 1.0, "close_basis": "unadjusted"})
    with pytest.raises(ValueError):
        prep.total_return_index(d, "close_plus_dividends")


# ------------------------------------------------------------------ macro availability
def test_macro_point_in_time_respects_publication_lag(synth, synth_panel):
    mpit = synth_panel["macro_pit"]
    oil = mpit[mpit.factor == "oil"].set_index("month")
    m = pd.Period("2021-03", "M")
    r = oil.loc[m]
    assert r["source_available_at"] <= r["decision_at"]
    # with a 7-day lag the month-end observation is NOT yet known at the cutoff
    assert r["observation_date"] < pd.Timestamp("2021-03-31") - pd.Timedelta(days=5)
    # and the descriptive (observation-date) change differs from the point-in-time change
    desc = synth_panel["macro_desc"].loc[m, "oil"]
    assert not np.isclose(desc, r["value"])


def test_oil_log_change_requires_positive_prices():
    obs = pd.DataFrame({"factor": "oil", "observation_date": pd.to_datetime(["2020-03-31", "2020-04-30", "2020-05-29"]),
                        "available_at": [ts("2020-04-07"), ts("2020-05-07"), ts("2020-06-05")],
                        "value": [20.0, -5.0, 35.0], "unit": "", "currency": "USD", "source": "t", "vintage": "t"})
    d = prep.macro_descriptive(obs, pd.period_range("2020-04", "2020-05", freq="M"))
    assert d["oil"].isna().all()
