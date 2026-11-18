"""Macro / commodity providers: FRED (WTI) and Bank of Canada Valet (policy rate, USD/CAD).

Normalized macro observation table:
    factor, observation_date, available_at, value, unit, currency, source, vintage
`available_at` is tz-aware. `vintage` is either "ALFRED realtime_start" (true point-in-time
vintage) or "latest; lag rule +Nd" (current revised values with a documented
publication-lag approximation).
"""
from __future__ import annotations

import io
import os

import pandas as pd

from ..config import FACTOR_SPECS, MARKET_TZ
from .base import ProviderError, cached, get_json, get_text

MACRO_COLUMNS = ["factor", "observation_date", "available_at", "value", "unit", "currency", "source", "vintage"]


def lag_rule_available_at(obs_dates: pd.Series, lag_days: int) -> pd.Series:
    """observation date + lag days, at 17:00 Toronto (conservative end-of-day)."""
    d = pd.to_datetime(obs_dates).dt.normalize() + pd.Timedelta(days=lag_days) + pd.Timedelta(hours=17)
    return d.dt.tz_localize(MARKET_TZ)


def fetch_fred(factor: str, start: str, refresh: bool = False):
    spec = FACTOR_SPECS[factor]
    key = os.environ.get("FRED_API_KEY", "").strip()
    if key:
        def _f():
            js = get_json("FRED", "https://api.stlouisfed.org/fred/series/observations",
                          params=dict(series_id=spec.series_id, api_key=key, file_type="json",
                                      observation_start=start, realtime_start="1776-07-04",
                                      realtime_end="9999-12-31"))
            obs = pd.DataFrame(js.get("observations", []))
            if obs.empty:
                raise ProviderError("FRED", f"{spec.series_id}: no observations returned")
            return obs[["date", "value", "realtime_start"]]
        snap = cached("fred_alfred", dict(series=spec.series_id, start=start), _f, refresh=refresh)
        obs = snap.df.copy()
        obs["observation_date"] = pd.to_datetime(obs["date"])
        obs["value"] = pd.to_numeric(obs["value"].replace(".", None), errors="coerce")
        rs = pd.to_datetime(obs["realtime_start"])
        avail = (rs + pd.Timedelta(hours=23, minutes=59)).dt.tz_localize("America/New_York").dt.tz_convert(MARKET_TZ)
        # Earliest ALFRED vintage dates are an archive artefact for old observations: when the
        # first vintage is > 60 days after the observation, fall back to the lag rule and say so.
        artefact = (rs - obs["observation_date"]).dt.days > 60
        first_vintage = obs.groupby("observation_date")["realtime_start"].transform("min") == obs["realtime_start"]
        lagged = lag_rule_available_at(obs["observation_date"], spec.publication_lag_days)
        use_lag = artefact & first_vintage
        obs["available_at"] = avail.where(~use_lag, lagged)
        obs["vintage"] = ("ALFRED " + obs["realtime_start"].astype(str)).where(
            ~use_lag, f"pre-archive; lag rule +{spec.publication_lag_days}d")
        obs = obs.dropna(subset=["value"])
    else:
        def _f():
            txt = get_text("FRED", "https://fred.stlouisfed.org/graph/fredgraph.csv",
                           params=dict(id=spec.series_id, cosd=start))
            d = pd.read_csv(io.StringIO(txt))
            if d.shape[1] < 2:
                raise ProviderError("FRED", f"{spec.series_id}: unexpected CSV layout")
            d.columns = ["date", "value"]
            return d
        snap = cached("fred_csv", dict(series=spec.series_id, start=start), _f, refresh=refresh)
        obs = snap.df.copy()
        obs["observation_date"] = pd.to_datetime(obs["date"])
        obs["value"] = pd.to_numeric(obs["value"], errors="coerce")
        obs = obs.dropna(subset=["value"])
        obs["available_at"] = lag_rule_available_at(obs["observation_date"], spec.publication_lag_days)
        obs["vintage"] = f"latest; lag rule +{spec.publication_lag_days}d (approximate)"
    obs["factor"] = factor
    obs["unit"] = spec.level_unit
    obs["currency"] = spec.currency
    obs["source"] = f"FRED:{spec.series_id}"
    return obs[MACRO_COLUMNS].reset_index(drop=True), snap.meta


def fetch_boc(factor: str, start: str, refresh: bool = False):
    spec = FACTOR_SPECS[factor]

    def _f():
        js = get_json("Bank of Canada Valet",
                      f"https://www.bankofcanada.ca/valet/observations/{spec.series_id}/json",
                      params=dict(start_date=start))
        rows = [(o.get("d"), (o.get(spec.series_id) or {}).get("v")) for o in js.get("observations", [])]
        if not rows:
            raise ProviderError("Bank of Canada Valet", f"{spec.series_id}: no observations returned",
                                "check the series id at bankofcanada.ca/valet/series")
        return pd.DataFrame(rows, columns=["date", "value"])
    snap = cached("boc_valet", dict(series=spec.series_id, start=start), _f, refresh=refresh)
    obs = snap.df.copy()
    obs["observation_date"] = pd.to_datetime(obs["date"])
    obs["value"] = pd.to_numeric(obs["value"], errors="coerce")
    obs = obs.dropna(subset=["value"])
    obs["available_at"] = lag_rule_available_at(obs["observation_date"], spec.publication_lag_days)
    obs["vintage"] = f"latest; lag rule +{spec.publication_lag_days}d (approximate)"
    obs["factor"] = factor
    obs["unit"] = spec.level_unit
    obs["currency"] = spec.currency
    obs["source"] = f"BoC Valet:{spec.series_id}"
    return obs[MACRO_COLUMNS].reset_index(drop=True), snap.meta


def fetch_macro(factors: list[str], start: str, refresh: bool = False):
    frames, metas, problems = [], [], []
    for f in factors:
        if f not in FACTOR_SPECS:
            continue
        try:
            fn = fetch_fred if FACTOR_SPECS[f].source == "fred" else fetch_boc
            d, m = fn(f, start, refresh)
            frames.append(d)
            metas.append({"factor": f, **m})
        except ProviderError as e:
            problems.append(str(e))
    macro = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=MACRO_COLUMNS)
    return macro, metas, problems
