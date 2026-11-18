"""Time alignment: total returns, completed months, point-in-time macro and fundamentals.

Conventions (shown in the UI):
* Month t's simple total return = TRI(last session of t) / TRI(last session of t-1) − 1.
  Both month-ends must be observed; a gap month makes both adjacent returns missing.
* Decision cutoff for month t = 17:00 America/Toronto on t's last observed session.
* Descriptive (explanatory) rows use *contemporaneous* month-t macro changes computed by
  observation date, and firm fundamentals known at the cutoff of month t−1.
* Forecast rows at decision month t use only values whose availability timestamp is ≤ the
  cutoff of month t (as-of joins), and target the month t+1 return.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .calendar import tsx_sessions, latest_completed_month
from .config import FACTOR_SPECS, MARKET_TZ, DECISION_CUTOFF_HOUR

FIRM_COLS = ["debt_to_assets", "earnings_growth", "earnings_change_scaled"]


# --------------------------------------------------------------------------- prices
def total_return_index(prices: pd.DataFrame, convention: str = "adjusted_close") -> pd.DataFrame:
    """Wide daily TRI (index=session_date, columns=symbol).

    adjusted_close        : TRI = adjusted close. Dividend column is ignored, so dividends are
                            counted exactly once (inside the adjustment).
    close_plus_dividends  : TRI_t = TRI_{t-1} · (close_t + dividend_t) / close_{t-1}, with close and
                            dividend on the same split-adjusted basis.
    """
    out = {}
    for sym, g in prices.sort_values("session_date").groupby("symbol"):
        g = g.drop_duplicates("session_date", keep="last").set_index("session_date")
        if convention == "adjusted_close":
            s = pd.to_numeric(g["adjusted_close"], errors="coerce")
            if s.isna().all():
                raise ValueError(f"{sym}: no adjusted_close values; choose the close_plus_dividends convention")
        elif convention == "close_plus_dividends":
            if (g["close_basis"] != "split_adjusted").any():
                raise ValueError(f"{sym}: closes must be split-adjusted before using close_plus_dividends")
            c = pd.to_numeric(g["close"], errors="coerce")
            d = pd.to_numeric(g["dividend"], errors="coerce").fillna(0.0)
            gross = (c + d) / c.shift(1)
            s = gross.fillna(1.0).cumprod() * c.dropna().iloc[0]
            s[c.isna()] = np.nan
        else:
            raise ValueError(f"unknown return convention {convention}")
        out[sym] = s
    return pd.DataFrame(out).sort_index()


def observed_sessions(prices: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(pd.to_datetime(prices["session_date"]).unique()))


def month_calendar(prices: pd.DataFrame, start_month: pd.Period, end_month: pd.Period) -> pd.DataFrame:
    """Per month: last session (data-driven where data exists, TSX calendar otherwise) and cutoff."""
    obs = observed_sessions(prices) if len(prices) else pd.DatetimeIndex([])
    cal = tsx_sessions(start_month.start_time, end_month.end_time)
    rows = []
    for m in pd.period_range(start_month, end_month, freq="M"):
        o = obs[(obs >= m.start_time) & (obs <= m.end_time)]
        c = cal[(cal >= m.start_time) & (cal <= m.end_time)]
        last = o.max() if len(o) else (c.max() if len(c) else pd.NaT)
        rows.append(dict(month=m, last_session=last, calendar_last_session=c.max() if len(c) else pd.NaT,
                         first_session=(o.min() if len(o) else (c.min() if len(c) else pd.NaT))))
    mc = pd.DataFrame(rows)
    mc["decision_at"] = (mc["last_session"] + pd.Timedelta(hours=DECISION_CUTOFF_HOUR)).dt.tz_localize(MARKET_TZ)
    return mc


def monthly_returns(tri: pd.DataFrame, mcal: pd.DataFrame, end_month: pd.Period):
    """Long monthly returns + a table of excluded rows with reasons."""
    rows, excluded = [], []
    mc = mcal.set_index("month")
    for sym in tri.columns:
        s = tri[sym].dropna()
        if s.empty:
            excluded.append(dict(symbol=sym, month=None, reason="no price observations"))
            continue
        per = s.index.to_period("M")
        last_by_m = s.groupby(per).tail(1)
        last_by_m.index = last_by_m.index.to_period("M")
        last_date = s.groupby(per).apply(lambda x: x.index.max())
        prev_val = None
        prev_m = None
        for m, v in last_by_m.items():
            if m > end_month:
                excluded.append(dict(symbol=sym, month=str(m), reason="incomplete month (after latest completed month)"))
                continue
            expected = mc.loc[m, "last_session"] if m in mc.index else pd.NaT
            partial = pd.notna(expected) and last_date[m] < expected
            if prev_m is not None and m == prev_m + 1 and not partial:
                rows.append(dict(symbol=sym, month=m, ret=v / prev_val - 1.0, month_end_session=last_date[m]))
            elif partial:
                excluded.append(dict(symbol=sym, month=str(m),
                                     reason=f"partial month: last observation {last_date[m].date()} before "
                                            f"month's last session {pd.Timestamp(expected).date()} "
                                            "(delisting, halt or missing data)"))
            elif prev_m is not None and m != prev_m + 1:
                excluded.append(dict(symbol=sym, month=str(m), reason=f"gap: no observation in {prev_m + 1}"))
            if not partial:
                prev_val, prev_m = v, m
            else:
                prev_val, prev_m = None, None
    return pd.DataFrame(rows), pd.DataFrame(excluded, columns=["symbol", "month", "reason"])


# --------------------------------------------------------------------------- macro
def _transform(level_now, level_prev, how: str):
    if how == "log_change":
        ok = (level_now > 0) & (level_prev > 0)
        return np.where(ok, np.log(np.where(ok, level_now, 1.0) / np.where(ok, level_prev, 1.0)), np.nan)
    if how in ("diff_pp", "diff"):
        return level_now - level_prev
    if how == "level":
        return level_now
    raise ValueError(f"unknown transformation {how}")


def factor_transformations(macro: pd.DataFrame) -> dict:
    out = {}
    for f in macro["factor"].unique():
        if f in FACTOR_SPECS:
            out[f] = FACTOR_SPECS[f].transformation
        else:
            t = macro.loc[macro["factor"] == f, "transformation"].dropna() if "transformation" in macro else pd.Series([], dtype=object)
            out[f] = t.iloc[0] if len(t) else "log_change"
    return out


def macro_descriptive(macro: pd.DataFrame, months: pd.PeriodIndex) -> pd.DataFrame:
    """Month-t change by *observation date* (latest vintage), for explanatory models."""
    trans = factor_transformations(macro)
    out = pd.DataFrame(index=months)
    for f, g in macro.groupby("factor"):
        g = g.sort_values(["observation_date", "available_at"]).drop_duplicates("observation_date", keep="last")
        s = g.set_index("observation_date")["value"]
        lvl = s.groupby(s.index.to_period("M")).last()
        lvl = lvl.reindex(pd.period_range(min(lvl.index.min(), months.min() - 1), months.max(), freq="M"))
        chg = _transform(lvl.values, lvl.shift(1).values, trans[f])
        out[f] = pd.Series(chg, index=lvl.index).reindex(months)
        out[f + "__level"] = lvl.reindex(months)
    return out


def macro_point_in_time(macro: pd.DataFrame, mcal: pd.DataFrame) -> pd.DataFrame:
    """At each decision cutoff: level of the latest observation *available* by then, and the
    change versus the previous cutoff's level. Returns long derived-feature rows."""
    trans = factor_transformations(macro)
    rows = []
    cut = mcal[["month", "decision_at"]].dropna()
    for f, g in macro.groupby("factor"):
        g = g.dropna(subset=["available_at", "value"]).sort_values("available_at")
        av = g["available_at"].dt.tz_convert(MARKET_TZ)
        quality = "point-in-time vintage" if g["vintage"].astype(str).str.startswith("ALFRED").all() else (
            "approximate (documented publication lag; latest revised values)")
        prev_level = None
        for _, r in cut.iterrows():
            known = g[av <= r["decision_at"]]
            if known.empty:
                level, obs_date, src_av = np.nan, pd.NaT, pd.NaT
            else:
                latest_obs = known["observation_date"].max()
                kk = known[known["observation_date"] == latest_obs].iloc[-1]  # latest vintage of it
                level, obs_date, src_av = kk["value"], latest_obs, kk["available_at"]
            chg = (_transform(np.array([level]), np.array([prev_level]), trans[f])[0]
                   if prev_level is not None and pd.notna(level) and pd.notna(prev_level) else np.nan)
            rows.append(dict(factor=f, month=r["month"], decision_at=r["decision_at"], level=level,
                             value=chg, observation_date=obs_date, source_available_at=src_av,
                             transformation=trans[f], availability_quality=quality))
            prev_level = level
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- fundamentals
def _features_from_state(state: dict, ptypes: dict, include_leases: bool, small_denominator_frac: float) -> dict:
    """Compute firm features from {(metric, period_end): (value, available_at)} known at one time."""
    rec = {}
    by_metric: dict = {}
    for (metric, pe), (v, av) in state.items():
        if pd.notna(v):
            by_metric.setdefault(metric, {})[pe] = (v, av)
    need = ["interest_bearing_debt", "total_assets"] + (["lease_liabilities"] if include_leases else [])
    if all(m in by_metric for m in need):
        common = set.intersection(*(set(by_metric[m]) for m in need))
        if common:
            pe = max(common)
            debt = by_metric["interest_bearing_debt"][pe][0] + (by_metric["lease_liabilities"][pe][0] if include_leases else 0.0)
            assets = by_metric["total_assets"][pe][0]
            if assets > 0:
                rec.update(debt_to_assets=debt / assets, debt_period_end=pe,
                           debt_available_at=max(by_metric[m][pe][1] for m in need))
    ni = by_metric.get("net_income")
    if ni:
        pe = max(ni)
        target = pe - pd.DateOffset(years=1)
        cands = [q for q in ni if ptypes.get(q) == ptypes.get(pe) and abs((q - target).days) <= 15]
        if cands:
            q = max(cands)
            cur, prior = ni[pe][0], ni[q][0]
            assets = by_metric.get("total_assets", {}).get(pe, (np.nan, None))[0]
            thresh = small_denominator_frac * assets if pd.notna(assets) and assets > 0 else 0.0
            rec.update(earn_period_end=pe, earn_available_at=ni[pe][1], net_income=cur, net_income_prior=prior,
                       earnings_transition=("loss→profit" if prior < 0 <= cur else "profit→loss" if prior >= 0 > cur
                                            else "loss→loss" if prior < 0 and cur < 0 else "profit→profit"))
            if prior != 0 and abs(prior) > thresh:
                rec["earnings_growth"] = (cur - prior) / abs(prior)
            else:
                rec["earnings_flag"] = "small prior-year denominator; growth left missing"
            if pd.notna(assets) and assets > 0:
                rec["earnings_change_scaled"] = (cur - prior) / assets
    return rec


def fundamentals_point_in_time(fund: pd.DataFrame, symbols, mcal: pd.DataFrame, include_leases: bool = False,
                               small_denominator_frac: float = 0.002, max_age_days: int = 400) -> pd.DataFrame:
    """Firm features at each decision cutoff, built only from reports available by then.

    Reports are replayed in availability order. For each (metric, fiscal period) the state holds the
    latest revision *available so far*, so a later restatement never overwrites an earlier vintage
    for an earlier cutoff. Duplicate reports with the same timestamp are resolved by `revision`
    (lexicographic, last wins) and then by file order, deterministically.
    """
    cols = ["symbol", "month", "decision_at"] + FIRM_COLS + ["earnings_transition", "debt_age_days", "earn_age_days"]
    if fund is None or fund.empty:
        return pd.DataFrame(columns=cols)
    fund = fund.copy()
    ns = f"datetime64[ns, {MARKET_TZ}]"
    fund["available_at"] = fund["available_at"].dt.tz_convert(MARKET_TZ).astype(ns)
    fund["_order"] = np.arange(len(fund))
    fund = fund.sort_values(["available_at", "revision", "_order"])
    cut = mcal[["month", "decision_at"]].dropna().sort_values("decision_at").reset_index(drop=True)
    cut["decision_at"] = cut["decision_at"].astype(ns)
    out = []
    for sym in symbols:
        f = fund[fund["symbol"] == sym]
        if f.empty:
            out.append(cut.assign(symbol=sym))
            continue
        ptypes = f.groupby("fiscal_period_end")["period_type"].last().to_dict()
        state, events = {}, []
        for av, g in f.groupby("available_at", sort=True):
            for r in g.itertuples(index=False):
                state[(r.metric, r.fiscal_period_end)] = (r.value, r.available_at)
            events.append(dict(event_at=av, **_features_from_state(state, ptypes, include_leases, small_denominator_frac)))
        ev = pd.DataFrame(events).sort_values("event_at")
        ev["event_at"] = ev["event_at"].astype(ns)
        m = pd.merge_asof(cut, ev, left_on="decision_at", right_on="event_at", direction="backward")
        m["symbol"] = sym
        for feat, av, agecol in (("debt_to_assets", "debt_available_at", "debt_age_days"),
                                 ("earnings_growth", "earn_available_at", "earn_age_days")):
            if av in m:
                m[agecol] = (m["decision_at"] - pd.to_datetime(m[av])).dt.days
                stale = m[agecol] > max_age_days
                if feat == "debt_to_assets":
                    m.loc[stale, "debt_to_assets"] = np.nan
                    m.loc[stale, "debt_flag"] = f"stale (> {max_age_days} days)"
                else:
                    for c in ("earnings_growth", "earnings_change_scaled"):
                        if c in m:
                            m.loc[stale, c] = np.nan
        out.append(m)
    res = pd.concat(out, ignore_index=True)
    for c in cols:
        if c not in res:
            res[c] = np.nan
    return res


# --------------------------------------------------------------------------- panel
def build_panel(prices, macro, fund, symbols, benchmark, end_month=None, start_month=None,
                convention="adjusted_close", include_leases=False, small_denominator_frac=0.002):
    """Assemble descriptive and forecast rows for all symbols. Returns a dict of tables."""
    asof_end = latest_completed_month()
    end_month = min(end_month, asof_end) if end_month is not None else asof_end
    data_last = pd.Period(pd.to_datetime(prices["session_date"]).max(), freq="M")
    end_month = min(end_month, data_last)
    data_first = pd.Period(pd.to_datetime(prices["session_date"]).min(), freq="M")
    start_month = max(start_month, data_first) if start_month is not None else data_first
    mcal = month_calendar(prices, start_month, end_month)
    notes = []
    last = mcal.iloc[-1]
    if pd.notna(last["calendar_last_session"]) and last["last_session"] < last["calendar_last_session"]:
        notes.append(f"{end_month} dropped: data end {pd.Timestamp(last['last_session']).date()} precedes the month's "
                     f"last TSX session {pd.Timestamp(last['calendar_last_session']).date()} (incomplete month).")
        end_month = end_month - 1
        mcal = mcal.iloc[:-1]
    tri =total_return_index(prices[prices["symbol"].isin(list(symbols) + [benchmark])], convention)
    tri = tri[tri.index <= mcal["last_session"].max()]
    mret, excluded = monthly_returns(tri, mcal, end_month)
    if mret.empty:
        raise ValueError("no complete monthly returns could be computed from the price data")
    months = pd.period_range(start_month, end_month, freq="M")
    bench = mret[mret["symbol"] == benchmark].set_index("month")["ret"].rename("market")
    stocks = mret[mret["symbol"].isin(symbols)]
    mdesc = macro_descriptive(macro, months) if macro is not None and len(macro) else pd.DataFrame(index=months)
    mpit = macro_point_in_time(macro, mcal) if macro is not None and len(macro) else pd.DataFrame()
    firm = fundamentals_point_in_time(fund, symbols, mcal, include_leases, small_denominator_frac)

    factor_names = [c for c in mdesc.columns if not c.endswith("__level")]
    panel_rows = []
    for sym in symbols:
        s = stocks[stocks["symbol"] == sym].set_index("month")["ret"].reindex(months)
        df = pd.DataFrame(index=months)
        df["symbol"] = sym
        df["ret"] = s
        df["market"] = bench.reindex(months)
        for f in factor_names:
            df[f] = mdesc[f]
        fs = firm[firm["symbol"] == sym].set_index("month").reindex(months) if len(firm) else pd.DataFrame(index=months)
        # descriptive: known before month t  (= point-in-time at cutoff t-1)
        for c in FIRM_COLS + ["earnings_transition", "debt_age_days", "earn_age_days"]:
            df[c] = fs[c].shift(1) if c in fs else np.nan
        # forecast features at decision month t
        df["decision_at"] = mcal.set_index("month")["decision_at"].reindex(months)
        df["lag_own"] = df["ret"]
        df["lag_market"] = df["market"]
        if len(mpit):
            for f in mpit["factor"].unique():
                df["lag_" + f] = mpit[mpit["factor"] == f].set_index("month")["value"].reindex(months)
        for c in FIRM_COLS:
            df["lag_" + c] = fs[c] if c in fs else np.nan
        srcs = []
        if len(mpit):
            srcs.append(mpit.groupby("month")["source_available_at"].max().reindex(months))
        for av in ("debt_available_at", "earn_available_at"):
            if av in fs:
                srcs.append(pd.to_datetime(fs[av]))
        if srcs:
            df["feature_source_available_at"] = pd.concat(srcs, axis=1).max(axis=1)
        df["target_next"] = df["ret"].shift(-1)
        df["target_month"] = df.index + 1
        panel_rows.append(df.reset_index(names="month"))
    panel = pd.concat(panel_rows, ignore_index=True)

    # derived-feature contract table (long), for display and export
    feats = []
    if len(mpit):
        feats.append(mpit.assign(symbol="*", feature=lambda d: "pit_" + d["factor"])[
            ["symbol", "decision_at", "feature", "value", "source_available_at", "transformation", "availability_quality"]])
    if len(firm):
        for c, av in (("debt_to_assets", "debt_available_at"), ("earnings_growth", "earn_available_at"),
                      ("earnings_change_scaled", "earn_available_at")):
            if c in firm and av in firm:
                t = firm[["symbol", "decision_at", c, av]].dropna(subset=[c]).rename(columns={c: "value", av: "source_available_at"})
                t["feature"], t["transformation"] = c, "as-of latest available report (carried forward)"
                t["availability_quality"] = "supplied availability timestamp"
                feats.append(t[["symbol", "decision_at", "feature", "value", "source_available_at", "transformation", "availability_quality"]])
    features_long = pd.concat(feats, ignore_index=True) if feats else pd.DataFrame()

    return dict(panel=panel, monthly_returns=mret, excluded=excluded, month_calendar=mcal, tri=tri,
                macro_desc=mdesc, macro_pit=mpit, firm=firm, features_long=features_long,
                factor_names=factor_names, start_month=start_month, end_month=end_month, notes=notes)


def coverage_table(prices, macro, fund, meta: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for sym, g in prices.groupby("symbol"):
        rows.append(dict(series=sym, kind="price", source=g["source"].iloc[0], first=g["session_date"].min().date(),
                         last=g["session_date"].max().date(), observations=len(g),
                         missing_values=int(g["adjusted_close"].isna().sum() if g["adjusted_close"].notna().any() else g["close"].isna().sum()),
                         currency=g["currency"].iloc[0]))
    if macro is not None and len(macro):
        for f, g in macro.groupby("factor"):
            rows.append(dict(series=f, kind="macro", source=g["source"].iloc[0], first=g["observation_date"].min().date(),
                             last=g["observation_date"].max().date(), observations=len(g),
                             missing_values=int(g["value"].isna().sum()), currency=g["currency"].iloc[0]))
    if fund is not None and len(fund):
        for sym, g in fund.groupby("symbol"):
            rows.append(dict(series=f"{sym} fundamentals", kind="fundamentals", source=g["source"].iloc[0],
                             first=g["fiscal_period_end"].min().date(), last=g["fiscal_period_end"].max().date(),
                             observations=g["fiscal_period_end"].nunique(), missing_values=int(g["value"].isna().sum()),
                             currency=g["currency"].iloc[0]))
    return pd.DataFrame(rows)
