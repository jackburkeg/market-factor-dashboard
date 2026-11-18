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


def generate(profile: str = "canada_energy"):
    if profile == "us_tech":
        return generate_us()
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
                               earnings_growth=be, oil_x_debt=bint, oil_x_debt_to_assets=bint, noise_sd=sd,
                               debt_center=DEBT_CENTER, center=DEBT_CENTER))
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


# =========================================================================== US tech (synthetic)
US_START, US_END = pd.Period("2016-12", "M"), pd.Period("2026-10", "M")
PROF_CENTER = 0.0
US_TRUTH = {
    #               alpha   market control  y10    vix    usd   prof   b_int  noise  start
    "SYN-CONN1": (0.010, 1.10, 0.45, -0.06, -0.02, -0.30, 0.20, 6.0, 0.075, "2023-03"),
    "SYN-CONN2": (0.008, 1.20, 0.40, -0.05, -0.03, -0.40, 0.20, 6.0, 0.070, "2021-01"),
    "SYN-CLOUD": (0.012, 1.00, 0.60, -0.08, -0.02, -0.20, 0.30, 6.0, 0.090, "2023-11"),
    "SYN-IP":    (0.005, 1.30, 0.30, -0.05, -0.02, -0.50, 0.10, 4.0, 0.060, "2017-01"),
    "SYN-QNT":   (0.004, 1.40, 0.20, -0.10, -0.05, -0.20, 0.30, 6.0, 0.095, "2017-01"),
    "SYN-DC":    (0.006, 1.20, 0.50, -0.07, -0.03, -0.30, 0.25, 6.0, 0.080, "2017-01"),
}
US_SUBSECTOR = {"SYN-CONN1": "Synthetic AI connectivity (short history)", "SYN-CONN2": "Synthetic AI connectivity",
                "SYN-CLOUD": "Synthetic AI cloud (short history)", "SYN-IP": "Synthetic chip IP",
                "SYN-QNT": "Synthetic quantum (loss-making)", "SYN-DC": "Synthetic AI data centres"}
US_BENCH = ["SYN-NDX", "SYN-SEMI"]
US_CONTROL = "SYN-CTRL"


def _price_frame(sym, path, start=None):
    df = pd.DataFrame({"symbol": sym, "session_date": path.index, "adjusted_close": path.values, "close": path.values,
                       "dividend": 0.0, "split_ratio": 1.0, "close_basis": "split_adjusted", "currency": "USD",
                       "exchange": "SYNTHETIC", "source": LABEL, "retrieved_at": "deterministic (seed %d)" % SEED})
    if start:
        df = df[df["session_date"] >= pd.Period(start, "M").start_time]
    return df


def generate_us():
    """Synthetic US tech dataset with a known 10-year-yield × profitability interaction:
        yield_sensitivity(profitability) = b_y10 + b_int · (ROA − 0)
    where ROA is the latest quarterly net income / total assets known before the month."""
    from ..calendar import exchange_sessions
    rng = np.random.default_rng(SEED + 1)
    months = pd.period_range(US_START, US_END, freq="M")
    sessions = exchange_sessions(US_START.start_time, US_END.end_time, "NYSE")
    n = len(months)
    sd = np.array([0.055, 0.20, 0.018])
    corr = np.array([[1, -0.7, -0.2], [-0.7, 1, 0.1], [-0.2, 0.1, 1]])
    z = rng.multivariate_normal([0.012, 0.0, 0.0], corr * np.outer(sd, sd), n)
    mkt, vix, usd = z[:, 0], z[:, 1], z[:, 2]
    y10_level = np.empty(n)
    lvl = 2.4
    for i in range(n):  # mean-reverting around 3%, so the yield keeps moving (no sticky bounds)
        lvl = lvl + 0.06 * (3.0 - lvl) + rng.normal(0, 0.22)
        y10_level[i] = abs(lvl - 0.3) + 0.3 if lvl < 0.3 else lvl
        lvl = y10_level[i]
    y10_chg = np.r_[0.0, np.diff(y10_level)]
    ffr_level = np.empty(n)
    r = 1.0
    for i in range(n):
        if rng.random() < 0.12:
            r = float(np.clip(r + rng.choice([-0.25, 0.25, 0.5, -0.5, 0.75], p=[0.3, 0.35, 0.15, 0.1, 0.1]), 0.05, 5.5))
        ffr_level[i] = r
    ffr_chg = np.r_[0.0, np.diff(ffr_level)]
    control = 0.02 + 1.6 * mkt + rng.normal(0, 0.09, n)
    fac = pd.DataFrame({"market": mkt, "y10": y10_chg, "ffr": ffr_chg, "vix": vix, "usd": usd, "control": control},
                       index=months)

    # ---- macro daily observations
    def level_bridge(levels, daily_sd):
        vals, dates = [], []
        for i, m in enumerate(months[1:], start=1):
            d = sessions[(sessions >= m.start_time) & (sessions <= m.end_time)]
            zz = rng.normal(0, daily_sd, len(d))
            inc = zz - zz.mean() + (levels[i] - levels[i - 1]) / len(d)
            path = levels[i - 1] + np.cumsum(inc)
            vals.extend(path), dates.extend(d)
        return pd.Series(vals, index=pd.DatetimeIndex(dates))
    y10_s = level_bridge(y10_level, 0.03)
    vix_s = np.exp(_bridge(fac["vix"].iloc[1:], sessions, rng, 0.05, np.log(18.0)))
    usd_s = np.exp(_bridge(fac["usd"].iloc[1:], sessions, rng, 0.003, np.log(115.0)))
    ffr_rows = []
    for i, m in enumerate(months[1:], start=1):
        d = sessions[(sessions >= m.start_time) & (sessions <= m.end_time)]
        ch = d[min(12, len(d) - 1)]
        ffr_rows += [(day, ffr_level[i] if day >= ch else ffr_level[i - 1]) for day in d]
    ffr_s = pd.Series(dict(ffr_rows))
    macro = []
    for f, s_, unit, lag in (("y10", y10_s, "percent", 1), ("ffr", ffr_s, "percent", 1),
                             ("vix", vix_s, "index points", 1), ("usd", usd_s, "index", 7)):
        av = (s_.index + pd.Timedelta(days=lag) + pd.Timedelta(hours=17)).tz_localize(MARKET_TZ)
        macro.append(pd.DataFrame({"factor": f, "observation_date": s_.index, "available_at": av, "value": s_.values,
                                   "unit": unit, "currency": "USD", "source": f"{LABEL}:{f}",
                                   "vintage": f"synthetic; lag +{lag}d"}))
    macro = pd.concat(macro, ignore_index=True)

    # ---- quarterly fundamentals: ROA paths (loss-making names improve over time)
    qends = pd.period_range("2016Q1", "2026Q3", freq="Q")
    nq = len(qends)
    fund_rows = []
    for k, sym in enumerate(US_TRUTH):
        start_roa = [-0.02, 0.0, -0.05, 0.02, -0.08, -0.04][k]
        drift = [0.0012, 0.0010, 0.0020, 0.0002, 0.0008, 0.0012][k]
        roa = start_roa + drift * np.arange(nq) + 0.02 * np.sin(np.arange(nq) / 3.0 + k) + rng.normal(0, 0.004, nq)
        assets = 2e9 * (1.06 ** np.arange(nq)) * (1 + 0.5 * k)
        debt = np.clip(0.15 + 0.05 * np.sin(np.arange(nq) / 5 + k) + rng.normal(0, 0.01, nq), 0.0, 0.6)
        for j, q in enumerate(qends):
            pe = q.end_time.normalize()
            av = (pe + pd.Timedelta(days=int(rng.integers(25, 46))) + pd.Timedelta(hours=16, minutes=5)).tz_localize(MARKET_TZ)
            for metric, val in (("net_income", roa[j] * assets[j]), ("total_assets", assets[j]),
                                ("interest_bearing_debt", debt[j] * assets[j]), ("lease_liabilities", 0.02 * assets[j])):
                fund_rows.append(dict(symbol=sym, fiscal_period_end=pe, period_type="Q", available_at=av, metric=metric,
                                      value=val, currency="USD", unit="USD", source=LABEL, revision="1",
                                      availability_quality="synthetic timestamp"))
    fund = pd.DataFrame(fund_rows)

    from ..prep import fundamentals_point_in_time
    mcal = pd.DataFrame({"month": months})
    mcal["last_session"] = [sessions[(sessions >= m.start_time) & (sessions <= m.end_time)].max() for m in months]
    mcal["decision_at"] = (mcal["last_session"] + pd.Timedelta(hours=DECISION_CUTOFF_HOUR)).dt.tz_localize(MARKET_TZ)
    firm = fundamentals_point_in_time(fund, list(US_TRUTH), mcal)

    prices, truth = [], []
    for sym, (a, bm, bc, by, bv, bu, bp, bint, sdv, start) in US_TRUTH.items():
        f = firm[firm["symbol"] == sym].set_index("month")
        prof = f["profitability"].shift(1).reindex(months)
        pc = (prof - PROF_CENTER).fillna(0)
        eps = rng.normal(0, sdv, n)
        ret = (a + bm * fac["market"] + bc * fac["control"] + (by + bint * pc) * fac["y10"] + bv * fac["vix"]
               + bu * fac["usd"] + bp * pc + eps).clip(lower=-0.9)
        path = np.exp(_bridge(np.log1p(ret.iloc[1:]), sessions, rng, 0.025, np.log(20.0)))
        prices.append(_price_frame(sym, path, start))
        truth.append(dict(symbol=sym, intercept=a, market=bm, control=bc, y10=by, vix=bv, usd=bu, profitability=bp,
                          y10_x_profitability=bint, noise_sd=sdv, center=PROF_CENTER))
    for b, extra, sd_ in (("SYN-NDX", 0.0, 0.0), ("SYN-SEMI", 0.3, 0.03)):
        rb = fac["market"] + extra * (fac["control"] - fac["control"].mean()) + (rng.normal(0, sd_, n) if sd_ else 0.0)
        prices.append(_price_frame(b, np.exp(_bridge(np.log1p(rb.iloc[1:]), sessions, rng, 0.012, np.log(100.0)))))
    prices.append(_price_frame(US_CONTROL, np.exp(_bridge(np.log1p(fac["control"].iloc[1:]), sessions, rng, 0.03,
                                                         np.log(50.0)))))
    prices = pd.concat(prices, ignore_index=True)
    meta = pd.DataFrame([dict(symbol=s, name=f"{s} ({LABEL})", exchange="SYNTHETIC", currency="USD",
                              subsector=US_SUBSECTOR.get(s, "Synthetic benchmark" if s in US_BENCH else
                                                         "Synthetic AI-leader control"),
                              status="listed", notes=LABEL) for s in list(US_TRUTH) + US_BENCH + [US_CONTROL]])
    return dict(prices=prices, macro=macro, fundamentals=fund, metadata=meta, truth=pd.DataFrame(truth), factors=fac,
                control=US_CONTROL)
