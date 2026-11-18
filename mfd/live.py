"""On-demand latest-market snapshot and return attribution.

Nothing here refreshes on a timer: quotes are fetched only when the user presses
"Refresh", and the result is cached (15 minutes by default in the app).

Attribution: a stock's return over a period is split into
    Σ_k β_k · (factor k's move over the same period)   +   the remainder (stock-specific)
using the per-stock betas from the descriptive regressions. Betas are monthly estimates,
so applying them to a partial month is an approximation and is labelled as such.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .providers.base import ProviderError

# Yahoo proxies for the live factor moves
LIVE_FACTOR_SYMBOLS = {
    "oil": ("CL=F", "WTI crude front-month future (proxy for the WTI spot series)", "log"),
    "fx": ("CAD=X", "USD/CAD (CAD per USD)", "log"),
}


def fetch_snapshot(symbols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Latest daily closes over ~2 months for symbols + factor proxies (one request per symbol)."""
    try:
        import yfinance as yf
    except ImportError:
        raise ProviderError("yfinance", "package not installed") from None
    rows, problems = [], []
    for sym in list(symbols) + [v[0] for v in LIVE_FACTOR_SYMBOLS.values()]:
        try:
            h = yf.Ticker(sym).history(period="3mo", interval="1d", auto_adjust=True, raise_errors=True)
            if h is None or h.empty:
                raise ValueError("no rows")
            idx = h.index.tz_convert("America/Toronto").tz_localize(None) if h.index.tz is not None else h.index
            s = pd.Series(h["Close"].to_numpy(), index=idx.normalize()).dropna()
            rows.append(dict(symbol=sym, series=s))
        except Exception as e:  # yfinance raises many types
            problems.append(f"[yfinance] {sym}: {type(e).__name__}: {str(e)[:150]}")
    return pd.DataFrame(rows), problems


def period_moves(series_by_symbol: dict, start_level_date: pd.Timestamp) -> pd.DataFrame:
    """Move from the last close on/before `start_level_date` to the latest close."""
    out = []
    for sym, s in series_by_symbol.items():
        base = s[s.index <= start_level_date]
        if base.empty or s.empty:
            continue
        b, last = float(base.iloc[-1]), float(s.iloc[-1])
        prev = float(s.iloc[-2]) if len(s) > 1 else np.nan
        out.append(dict(symbol=sym, last_close=last, as_of=s.index[-1].date(), day_change=last / prev - 1 if prev else np.nan,
                        period_return=last / b - 1, period_log=math.log(last / b) if last > 0 and b > 0 else np.nan))
    return pd.DataFrame(out)


def attribute(stock_returns: pd.Series, betas: pd.DataFrame, factor_moves: dict) -> pd.DataFrame:
    """Split each stock's return into factor contributions and a stock-specific remainder.

    stock_returns: index symbol → simple return over the period
    betas: index symbol, columns factor names (raw-unit coefficients)
    factor_moves: factor → move in the same units the model uses (market: simple return;
                  oil, fx: log change; rate: change in percentage points)
    """
    rows = []
    for sym, r in stock_returns.items():
        if sym not in betas.index or pd.isna(r):
            continue
        rec = {"symbol": sym, "actual": float(r)}
        explained = 0.0
        for f, mv in factor_moves.items():
            if f in betas.columns and pd.notna(mv) and pd.notna(betas.loc[sym, f]):
                c = float(betas.loc[sym, f]) * float(mv)
                rec[f] = c
                explained += c
        rec["explained_by_factors"] = explained
        rec["stock_specific"] = float(r) - explained
        rows.append(rec)
    return pd.DataFrame(rows)
