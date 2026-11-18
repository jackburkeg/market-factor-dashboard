"""Deterministic SYNTHETIC dataset that exercises every feature of the app.

All symbols start with "SYN-" and every result computed from them is labelled
SYNTHETIC in the UI and exports. The data-generating process is known, so the app's
estimates can be checked against the true coefficients (shown in the Stock drivers view).

Design
* Monthly factors 2009-12 … 2026-10: market return, WTI log change, policy-rate steps (pp),
  USD/CAD log change. Daily paths are bridged so month-end values are exact.
* Six stocks with different exposures. E&P names have an oil × debt interaction:
      oil_sensitivity(debt) = b_oil + b_int · (debt − 0.30)
  where debt is the debt/assets ratio *known before the month* (latest report available at the
  prior month's cutoff), i.e. exactly what the app's point-in-time join produces.
* Quarterly fundamentals released 35–55 days after period end; some periods are restated
  ~200 days later (revision "2") so vintage handling can be checked.
* SYN-GAS is "delisted" mid-June 2025 (partial final month); SYN-NEW starts in 2018 (short history).
* Quarterly dividends; adjusted_close and close+dividend conventions give identical returns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..calendar import tsx_sessions
from ..config import MARKET_TZ, DECISION_CUTOFF_HOUR

SEED = 20261117
START, END = pd.Period("2009-12", "M"), pd.Period("2026-10", "M")
LABEL = "SYNTHETIC"

TRUTH = {
    #            alpha   market  oil    rate   fx     debt   earn   b_int  noise  start      end
    "SYN-EPA":  (0.002, 0.9,    0.55, -0.010, 0.10, -0.05, 0.010, 1.50, 0.055, None, None),
    "SYN-EPB":  (0.001, 1.0,    0.65, -0.015, 0.05, -0.08, 0.010, 1.20, 0.060, None, None),
    "SYN-INT":  (0.002, 0.8,    0.35, -0.010, 0.00, -0.03, 0.005, 0.40, 0.045, None, None),
    "SYN-GAS":  (0.000, 0.9,    0.25, -0.020, 0.00, -0.06, 0.010, 0.80, 0.070, None, "2025-06-12"),
    "SYN-PIPE": (0.003, 0.6,    0.10, -0.040, -0.05, -0.02, 0.000, 0.00, 0.035, None, None),
    "SYN-NEW":  (0.001, 1.1,    0.50, -0.010, 0.00, -0.05, 0.010, 1.00, 0.065, "2018-01", None),
}
SUBSECTOR = {"SYN-EPA": "Synthetic E&P", "SYN-EPB": "Synthetic E&P", "SYN-INT": "Synthetic integrated",
             "SYN-GAS": "Synthetic gas E&P (delisted 2025-06)", "SYN-PIPE": "Synthetic pipeline",
             "SYN-NEW": "Synthetic E&P (short history)"}
BENCH = ["SYN-MKT", "SYN-ENERGY"]
DEBT_CENTER = 0.30


def _bridge(month_log_changes: pd.Series, sessions: pd.DatetimeIndex, rng, daily_sd: float, start_log: float):
    """Daily log path whose month-end values match the monthly log changes exactly."""
    vals, dates, level = [], [], start_log
    for m, total in month_log_changes.items():
        d = sessions[(sessions >= m.start_time) & (sessions <= m.end_time)]
        z = rng.normal(0, daily_sd, len(d))
        inc = z - z.mean() + total / len(d)
        path = level + np.cumsum(inc)
        vals.extend(path), dates.extend(d)
        level = path[-1]
    return pd.Series(vals, index=pd.DatetimeIndex(dates))


def generate():
    rng = np.random.default_rng(SEED)
    months = pd.period_range(START, END, freq="M")
    sessions = tsx_sessions(START.start_time, END.end_time)
    n = len(months)

    # ---- factors (month 0 is the base month; its changes are unused)
    cov = np.array([[0.04 ** 2, 0.3 * 0.04 * 0.09, -0.3 * 0.04 * 0.02],
                    [0.3 * 0.04 * 0.09, 0.09 ** 2, -0.4 * 0.09 * 0.02],
                    [-0.3 * 0.04 * 0.02, -0.4 * 0.09 * 0.02, 0.02 ** 2]])
    z = rng.multivariate_normal([0.007, 0.0, 0.0], cov, n)
    mkt, oil, fx = z[:, 0], z[:, 1], z[:, 2]
    rate_level = np.empty(n)
    r = 1.0
    for i in range(n):
        if rng.random() < 0.15:
            r = float(np.clip(r + rng.choice([-0.25, 0.25, 0.5, -0.5], p=[0.35, 0.35, 0.15, 0.15]), 0.25, 5.0))
        rate_level[i] = r
    rate_chg = np.r_[0.0, np.diff(rate_level)]
    fac = pd.DataFrame({"market": mkt, "oil": oil, "rate": rate_chg, "fx": fx}, index=months)

    # ---- macro daily observations
    macro_rows = []
    oil_path = np.exp(_bridge(fac["oil"].iloc[1:], sessions, rng, 0.02, np.log(75.0)))
    fx_path = np.exp(_bridge(fac["fx"].iloc[1:], sessions, rng, 0.004, np.log(1.05)))
    # policy rate: daily step series, changes take effect on a session mid-month
    rate_daily = []
    for i, m in enumerate(months[1:], start=1):
        d = sessions[(sessions >= m.start_time) & (sessions <= m.end_time)]
        change_day = d[min(9, len(d) - 1)]
        for day in d:
            rate_daily.append((day, rate_level[i] if day >= change_day else rate_level[i - 1]))
    rate_s = pd.Series(dict(rate_daily))
    for f, s, unit, cur, lag in (("oil", oil_path, "USD per barrel", "USD", 7),
                                 ("fx", fx_path, "CAD per 1 USD", None, 1),
                                 ("rate", rate_s, "percent", "CAD", 1)):
        av = (s.index + pd.Timedelta(days=lag) + pd.Timedelta(hours=17)).tz_localize(MARKET_TZ)
        macro_rows.append(pd.DataFrame({"factor": f, "observation_date": s.index, "available_at": av,
                                        "value": s.values, "unit": unit, "currency": cur,
                                        "source": f"{LABEL}:{f}", "vintage": f"synthetic; lag +{lag}d"}))
    macro = pd.concat(macro_rows, ignore_index=True)

    # ---- fundamentals (quarterly)
    qends = pd.period_range("2009Q1", "2026Q3", freq="Q")
    fund_rows = []
    for k, sym in enumerate(TRUTH):
        debt = 0.30 + 0.12 * np.sin(np.arange(len(qends)) / (4.0 + k)) + rng.normal(0, 0.03, len(qends)).cumsum() * 0.3
        debt = np.clip(debt, 0.05, 0.65)
        assets = 40e9 * (1.01 ** np.arange(len(qends))) * (1 + 0.2 * k)
        ni = assets * (0.015 + 0.010 * np.sin(np.arange(len(qends)) / 3.0 + k)) + rng.normal(0, 0.003, len(qends)) * assets
        if sym == "SYN-INT":
            ni[20] = assets[20] * 0.0001  # near-zero quarter → small-denominator case a year later
        for j, q in enumerate(qends):
            pe = q.end_time.normalize()
            av = (pe + pd.Timedelta(days=int(rng.integers(35, 56))) + pd.Timedelta(hours=7)).tz_localize(MARKET_TZ)
            for metric, val in (("interest_bearing_debt", debt[j] * assets[j]), ("total_assets", assets[j]),
                                ("lease_liabilities", 0.03 * assets[j]), ("net_income", ni[j])):
                fund_rows.append(dict(symbol=sym, fiscal_period_end=pe, period_type="Q", available_at=av,
                                      metric=metric, value=val, currency="CAD", unit="CAD", source=LABEL,
                                      revision="1", availability_quality="synthetic timestamp"))
            if j % 11 == 5:  # restatement ~200 days later: debt revised up 10%
                av2 = av + pd.Timedelta(days=200)
                fund_rows.append(dict(symbol=sym, fiscal_period_end=pe, period_type="Q", available_at=av2,
                                      metric="interest_bearing_debt", value=debt[j] * assets[j] * 1.10,
                                      currency="CAD", unit="CAD", source=LABEL, revision="2",
                                      availability_quality="synthetic timestamp (restatement)"))
    fund = pd.DataFrame(fund_rows)

    # ---- point-in-time firm features exactly as the app computes them
    from ..prep import fundamentals_point_in_time
    mcal = pd.DataFrame({"month": months})
    mcal["last_session"] = [sessions[(sessions >= m.start_time) & (sessions <= m.end_time)].max() for m in months]
    mcal["decision_at"] = (mcal["last_session"] + pd.Timedelta(hours=DECISION_CUTOFF_HOUR)).dt.tz_localize(MARKET_TZ)
    firm = fundamentals_point_in_time(fund, list(TRUTH), mcal)

    # ---- stock monthly returns
    price_rows = []
    truth_rows = []
    for sym, (a, bm, bo, br, bf, bd, be, bint, sd, start, end) in TRUTH.items():
        f = firm[firm["symbol"] == sym].set_index("month")
        debt_known = f["debt_to_assets"].shift(1).reindex(months)          # known before month t
        earn_known = f["earnings_growth"].shift(1).reindex(months).fillna(0).clip(-3, 3)
        dc = (debt_known - DEBT_CENTER).fillna(0)
        eps = rng.normal(0, sd, n)
        ret = (a + bm * fac["market"] + (bo + bint * dc) * fac["oil"] + br * fac["rate"] + bf * fac["fx"]
               + bd * dc + be * earn_known + eps)
        logr = np.log1p(ret.iloc[1:])
        path = np.exp(_bridge(logr, sessions, rng, 0.012, np.log(30.0)))
        # dividends: quarterly ~0.9% yield, ex-date = 15th session of Mar/Jun/Sep/Dec
        tri = path
        ex_days = set()
        for m in months[1:]:
            if m.month in (3, 6, 9, 12):
                d = tri.index[(tri.index >= m.start_time) & (tri.index <= m.end_time)]
                ex_days.add(d[min(14, len(d) - 1)])
        g = (tri / tri.shift(1)).fillna(1.0).to_numpy()
        is_ex = tri.index.isin(list(ex_days))
        factor = np.where(is_ex, g - 0.009, g)           # c_i = c_{i-1}·g_i − 0.009·c_{i-1} on ex-days
        close = pd.Series(tri.iloc[0] * np.cumprod(factor), index=tri.index)
        div = pd.Series(np.where(is_ex, 0.009 * close.shift(1).fillna(0).to_numpy(), 0.0), index=tri.index)
        # adjusted close in Yahoo style: scaled so the last value equals the last close
        adj = tri / tri.iloc[-1] * close.iloc[-1]
        df = pd.DataFrame({"symbol": sym, "session_date": tri.index, "adjusted_close": adj.values,
                           "close": close.values, "dividend": div.values, "split_ratio": 1.0,
                           "close_basis": "split_adjusted", "currency": "CAD", "exchange": "SYNTHETIC",
                           "source": LABEL, "retrieved_at": "deterministic (seed %d)" % SEED})
        if start:
            df = df[df["session_date"] >= pd.Period(start, "M").start_time]
        if end:
            df = df[df["session_date"] <= pd.Timestamp(end)]
        price_rows.append(df)
        truth_rows.append(dict(symbol=sym, intercept=a, market=bm, oil=bo, rate=br, fx=bf, debt_to_assets=bd,
                               earnings_growth=be, oil_x_debt=bint, noise_sd=sd, debt_center=DEBT_CENTER))
    # benchmarks
    for b, extra in (("SYN-MKT", 0.0), ("SYN-ENERGY", 0.5)):
        noise = np.zeros(n) if extra == 0 else rng.normal(0, 0.02, n)
        rb = fac["market"] + extra * fac["oil"] + noise  # SYN-MKT return equals the latent market factor
        path = np.exp(_bridge(np.log1p(rb.iloc[1:]), sessions, rng, 0.008, np.log(25.0)))
        price_rows.append(pd.DataFrame({"symbol": b, "session_date": path.index, "adjusted_close": path.values,
                                        "close": path.values, "dividend": 0.0, "split_ratio": 1.0,
                                        "close_basis": "split_adjusted", "currency": "CAD", "exchange": "SYNTHETIC",
                                        "source": LABEL, "retrieved_at": "deterministic (seed %d)" % SEED}))
    prices = pd.concat(price_rows, ignore_index=True)
    meta = pd.DataFrame([dict(symbol=s, name=f"{s} ({LABEL})", exchange="SYNTHETIC", currency="CAD",
                              subsector=SUBSECTOR.get(s, "Synthetic benchmark"),
                              status="delisted" if TRUTH.get(s, (None,) * 11)[10] else "listed",
                              notes=LABEL) for s in list(TRUTH) + BENCH])
    return dict(prices=prices, macro=macro, fundamentals=fund, metadata=meta,
                truth=pd.DataFrame(truth_rows), factors=fac)
