"""CSV import for prices, macro observations, fundamentals and universe metadata.

CSV import is the provider-independent route: it works offline and supports the full
debt and earnings workflow. Templates live in ./templates.
"""
from __future__ import annotations

import io

import pandas as pd

from ..config import MARKET_TZ
from .fundamentals import FUND_COLUMNS
from .macro import MACRO_COLUMNS
from .market import PRICE_COLUMNS, apply_split_adjustment

REQUIRED = {
    "prices": ["symbol", "session_date"],
    "macro": ["factor", "observation_date", "available_at", "value"],
    "fundamentals": ["symbol", "fiscal_period_end", "available_at", "metric", "value"],
    "metadata": ["symbol"],
}


class CSVSchemaError(ValueError):
    pass


def _read(src) -> pd.DataFrame:
    if isinstance(src, (bytes, bytearray)):
        src = io.BytesIO(src)
    df = pd.read_csv(src)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _tz(series: pd.Series, notes: list, label: str) -> pd.Series:
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        return series.dt.tz_convert(MARKET_TZ)
    txt = series.astype(str).str.strip()
    has_offset = txt.str.contains(r"(?:[+-]\d\d:?\d\d|Z)$", regex=True)
    out = pd.Series(pd.NaT, index=series.index, dtype=f"datetime64[ns, {MARKET_TZ}]")
    if has_offset.any():
        out[has_offset] = pd.to_datetime(txt[has_offset], errors="coerce", utc=True, format="mixed").dt.tz_convert(MARKET_TZ)
    naive = ~has_offset
    if naive.any():
        notes.append(f"{label}: {int(naive.sum())} timestamps had no time zone; interpreted as {MARKET_TZ}.")
        out[naive] = pd.to_datetime(txt[naive], errors="coerce", format="mixed").dt.tz_localize(
            MARKET_TZ, nonexistent="shift_forward", ambiguous="NaT")
    return out


def load_prices(src, source_label="CSV"):
    df, notes = _read(src), []
    _check(df, "prices")
    if "adjusted_close" not in df and "close" not in df:
        raise CSVSchemaError("prices CSV needs adjusted_close, or close (+ dividend, split_ratio)")
    for c, default in (("adjusted_close", float("nan")), ("close", float("nan")), ("dividend", 0.0),
                       ("split_ratio", 1.0), ("close_basis", "split_adjusted"), ("currency", None),
                       ("exchange", None), ("source", source_label)):
        if c not in df:
            df[c] = default
    df["session_date"] = pd.to_datetime(df["session_date"]).dt.normalize()
    df["dividend"] = pd.to_numeric(df["dividend"], errors="coerce").fillna(0.0)
    df["split_ratio"] = pd.to_numeric(df["split_ratio"], errors="coerce").fillna(1.0).replace(0, 1.0)
    df["retrieved_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    dup = df.duplicated(["symbol", "session_date"], keep="last")
    if dup.any():
        notes.append(f"prices: {int(dup.sum())} duplicate symbol/date rows; kept the last occurrence.")
        df = df[~dup]
    df = apply_split_adjustment(df)
    return df[PRICE_COLUMNS], notes


def load_macro(src, source_label="CSV"):
    df, notes = _read(src), []
    _check(df, "macro")
    df["observation_date"] = pd.to_datetime(df["observation_date"]).dt.normalize()
    df["available_at"] = _tz(df["available_at"], notes, "macro.available_at")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    for c, d in (("unit", None), ("currency", None), ("source", source_label), ("vintage", "CSV supplied"),
                 ("transformation", None)):
        if c not in df:
            df[c] = d
    return df[MACRO_COLUMNS + ["transformation"]], notes


def load_fundamentals(src, source_label="CSV"):
    df, notes = _read(src), []
    _check(df, "fundamentals")
    df["fiscal_period_end"] = pd.to_datetime(df["fiscal_period_end"]).dt.normalize()
    df["available_at"] = _tz(df["available_at"], notes, "fundamentals.available_at")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    for c, d in (("period_type", "Q"), ("currency", None), ("unit", "currency units"), ("source", source_label),
                 ("revision", "1"), ("availability_quality", "CSV supplied timestamp")):
        if c not in df:
            df[c] = d
    df["revision"] = df["revision"].astype(str)
    bad = df["available_at"].isna().sum()
    if bad:
        notes.append(f"fundamentals: {int(bad)} rows with unparseable available_at were dropped.")
        df = df.dropna(subset=["available_at"])
    early = (df["available_at"].dt.tz_localize(None) < df["fiscal_period_end"]).sum()
    if early:
        notes.append(f"fundamentals: {int(early)} rows are 'available' before their fiscal period ended — check timestamps.")
    return df[FUND_COLUMNS], notes


def load_metadata(src):
    df = _read(src)
    _check(df, "metadata")
    return df


def _check(df, kind):
    missing = [c for c in REQUIRED[kind] if c not in df.columns]
    if missing:
        raise CSVSchemaError(f"{kind} CSV is missing required column(s): {', '.join(missing)}. "
                             f"See templates/{kind}_template.csv")
    if df.empty:
        raise CSVSchemaError(f"{kind} CSV has no rows")
