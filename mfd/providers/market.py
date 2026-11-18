"""Stock / ETF price providers.

Normalized daily price table (one row per symbol-session):
    symbol, session_date, adjusted_close, close, dividend, split_ratio, close_basis,
    currency, exchange, source, retrieved_at

`adjusted_close`  split- and dividend-adjusted close (a total-return index up to scale)
`close`           split-adjusted (close_basis="split_adjusted") or unadjusted
                  (close_basis="unadjusted") close
`dividend`        cash dividend on its ex-date, on the same split basis as `close`
`split_ratio`     new shares per old share on the split's ex-date (1.0 otherwise)
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from .base import ProviderError, Snapshot, cached

PRICE_COLUMNS = ["symbol", "session_date", "adjusted_close", "close", "dividend", "split_ratio",
                 "close_basis", "currency", "exchange", "source", "retrieved_at"]


class PriceProvider(ABC):
    name = "abstract"

    @abstractmethod
    def fetch(self, symbols: list[str], start: str, end: str, refresh: bool = False):
        """Return (prices DataFrame, metadata DataFrame, list of per-symbol problems)."""


class YFinanceProvider(PriceProvider):
    """yfinance adapter for *personal research*.

    yfinance describes itself as intended for research and educational use and points
    users to Yahoo's terms; check suitability before any public/commercial deployment.
    Adjustment behaviour is set explicitly: auto_adjust=False so that both the
    split-adjusted Close and the split+dividend-adjusted 'Adj Close' are returned,
    together with Dividends and Stock Splits (actions=True).
    """
    name = "yfinance"

    def _one(self, sym: str, start: str, end: str) -> tuple[pd.DataFrame, dict]:
        try:
            import yfinance as yf
        except ImportError:
            raise ProviderError("yfinance", "package not installed", "pip install yfinance") from None
        t = yf.Ticker(sym)
        try:
            h = t.history(start=start, end=(pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat(),
                          interval="1d", auto_adjust=False, actions=True, raise_errors=True)
        except Exception as e:  # yfinance raises a variety of types
            raise ProviderError("yfinance", f"{sym}: {type(e).__name__}: {str(e)[:200]}",
                                "symbol may be delisted, renamed, or Yahoo unreachable") from None
        if h is None or h.empty:
            raise ProviderError("yfinance", f"{sym}: no rows returned",
                                "symbol may be delisted or renamed; try CSV import for its history")
        try:
            md = t.get_history_metadata() or {}
        except Exception:
            md = {}
        idx = h.index
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert("America/Toronto").tz_localize(None)
        df = pd.DataFrame({
            "symbol": sym,
            "session_date": idx.normalize(),
            "adjusted_close": h["Adj Close"].to_numpy() if "Adj Close" in h else float("nan"),
            "close": h["Close"].to_numpy(),
            "dividend": h.get("Dividends", pd.Series(0.0, index=h.index)).to_numpy(),
            "split_ratio": h.get("Stock Splits", pd.Series(0.0, index=h.index)).replace(0, 1.0).to_numpy(),
            "close_basis": "split_adjusted",
            "currency": md.get("currency"),
            "exchange": md.get("fullExchangeName") or md.get("exchangeName"),
        })
        first_trade = md.get("firstTradeDate")
        meta = dict(symbol=sym, provider_name=md.get("longName") or md.get("shortName"),
                    exchange=df["exchange"].iloc[0], currency=df["currency"].iloc[0],
                    instrument_type=md.get("instrumentType"), timezone=md.get("exchangeTimezoneName"),
                    first_trade_date=pd.to_datetime(first_trade, unit="s", errors="coerce") if isinstance(first_trade, (int, float)) else first_trade)
        return df, meta

    def fetch(self, symbols, start, end, refresh=False):
        frames, metas, problems = [], [], []
        for sym in symbols:
            params = {"symbol": sym, "start": start, "end": end, "auto_adjust": False, "actions": True}
            try:
                holder = {}

                def _fetch():
                    d, m = self._one(sym, start, end)
                    holder["meta"] = m
                    return d
                snap: Snapshot = cached("yfinance", params, _fetch, refresh=refresh)
                d = snap.df.copy()
                d["source"] = "yfinance"
                d["retrieved_at"] = snap.meta["retrieved_at"]
                frames.append(d)
                m = holder.get("meta") or {"symbol": sym, "exchange": d["exchange"].iloc[0],
                                           "currency": d["currency"].iloc[0]}
                m.update(retrieved_at=snap.meta["retrieved_at"], cache_status=snap.meta["cache_status"])
                metas.append(m)
            except ProviderError as e:
                problems.append(str(e))
        prices = pd.concat(frames, ignore_index=True)[PRICE_COLUMNS] if frames else pd.DataFrame(columns=PRICE_COLUMNS)
        return prices, pd.DataFrame(metas), problems


def apply_split_adjustment(df: pd.DataFrame) -> pd.DataFrame:
    """Convert unadjusted close/dividends to a split-adjusted basis (latest share count).

    For a split of ratio r on ex-date e, prices and per-share dividends before e are
    divided by r. Rows already marked split_adjusted are returned unchanged.
    """
    out = []
    for sym, g in df.sort_values(["symbol", "session_date"]).groupby("symbol", sort=False):
        g = g.copy()
        if (g["close_basis"] == "unadjusted").any():
            r = g["split_ratio"].fillna(1.0).replace(0, 1.0).astype(float)
            # cumulative product of splits strictly AFTER each row
            future = r[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
            g["close"] = g["close"] / future
            g["dividend"] = g["dividend"].fillna(0.0) / future
            g["close_basis"] = "split_adjusted"
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else df
