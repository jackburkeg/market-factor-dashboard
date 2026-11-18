"""Optional Financial Modeling Prep adapter for historical statements.

Unverified for Canadian coverage in this build (no key available while building).
FMP documents that for some international symbols the filing-date field holds the
report date, and its statements reflect the provider's current (possibly restated)
values. Rows from this adapter are therefore labelled `availability_quality` =
"provider filing date (unverified)" or "suspect: filing date = period end", and the
app warns before using them in predictive tests.

Normalized fundamentals table (long):
    symbol, fiscal_period_end, period_type, available_at, metric, value, currency,
    unit, source, revision, availability_quality
"""
from __future__ import annotations

import os

import pandas as pd

from ..config import MARKET_TZ
from .base import ProviderError, cached, get_json

FUND_COLUMNS = ["symbol", "fiscal_period_end", "period_type", "available_at", "metric", "value", "currency",
                "unit", "source", "revision", "availability_quality"]
BASE = "https://financialmodelingprep.com/stable"


def fetch_fmp(symbols: list[str], period: str = "quarter", refresh: bool = False):
    key = os.environ.get("FMP_API_KEY", "").strip()
    if not key:
        raise ProviderError("FMP", "FMP_API_KEY not set", "add it to .env, or use the fundamentals CSV import")
    frames, problems = [], []
    for sym in symbols:
        try:
            def _f(endpoint):
                js = get_json("FMP", f"{BASE}/{endpoint}", params=dict(symbol=sym, period=period, limit=120, apikey=key))
                if not isinstance(js, list) or not js:
                    raise ProviderError("FMP", f"{sym}: no {endpoint} rows (coverage or plan limit)")
                return pd.DataFrame(js).astype(str)
            bs = cached("fmp_bs", dict(symbol=sym, period=period), lambda: _f("balance-sheet-statement"), refresh).df
            inc = cached("fmp_is", dict(symbol=sym, period=period), lambda: _f("income-statement"), refresh).df
            frames.append(_normalize(sym, bs, inc, period))
        except ProviderError as e:
            problems.append(str(e))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=FUND_COLUMNS)
    return out, problems


def _num(df, col):
    return pd.to_numeric(df[col], errors="coerce") if col in df else pd.Series(float("nan"), index=df.index)


def _normalize(sym, bs, inc, period):
    rows = []
    for kind, df in (("bs", bs), ("is", inc)):
        fdate = df.get("filingDate", df.get("fillingDate"))
        acc = pd.to_datetime(df.get("acceptedDate"), errors="coerce")
        fil = pd.to_datetime(fdate, errors="coerce")
        pend = pd.to_datetime(df["date"], errors="coerce")
        avail = acc.where(acc.notna(), fil + pd.Timedelta(hours=23, minutes=59))
        avail = avail.dt.tz_localize(MARKET_TZ, nonexistent="shift_forward", ambiguous="NaT")
        quality = pd.Series("provider filing date (unverified)", index=df.index)
        quality[(fil.dt.normalize() == pend.dt.normalize())] = "suspect: filing date = period end"
        cur = df.get("reportedCurrency", pd.Series(None, index=df.index))
        if kind == "bs":
            metrics = {
                "interest_bearing_debt": _num(df, "shortTermDebt").fillna(0) + _num(df, "longTermDebt"),
                "lease_liabilities": _num(df, "capitalLeaseObligations"),
                "total_assets": _num(df, "totalAssets"),
            }
        else:
            metrics = {"net_income": _num(df, "netIncome")}
        for m, v in metrics.items():
            rows.append(pd.DataFrame({
                "symbol": sym, "fiscal_period_end": pend, "period_type": "Q" if period == "quarter" else "FY",
                "available_at": avail, "metric": m, "value": v, "currency": cur, "unit": "currency units",
                "source": "FMP", "revision": "provider current values", "availability_quality": quality,
            }))
    return pd.concat(rows, ignore_index=True)[FUND_COLUMNS]
