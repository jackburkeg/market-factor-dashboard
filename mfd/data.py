"""Dataset assembly for the three data modes: synthetic, live providers, CSV import."""
from __future__ import annotations

import pandas as pd

from .config import FACTOR_SPECS, PROFILES, DEFAULT_PROFILE
from .providers import csv_import, synthetic
from .providers.base import ProviderError
from .providers.macro import fetch_macro
from .providers.market import YFinanceProvider


def load_synthetic(profile: str = DEFAULT_PROFILE):
    d = synthetic.generate(profile)
    pm = {"prices": {"source": "SYNTHETIC", "seed": synthetic.SEED}, "macro": {"source": "SYNTHETIC"},
          "fundamentals": {"source": "SYNTHETIC"}}
    return dict(prices=d["prices"], macro=d["macro"], fundamentals=d["fundamentals"], metadata=d["metadata"],
                truth=d["truth"], provider_meta=pm, problems=[], notes=[], synthetic=True,
                control=d.get("control"))


def load_live(symbols, benchmark, start, end, factors, refresh=False, fundamentals_csv=None, use_fmp=False,
              profile: str = DEFAULT_PROFILE):
    problems, notes, pm = [], [], {}
    ctrl = PROFILES[profile].control_symbol
    extra = [benchmark] + ([ctrl] if ctrl and ctrl not in symbols else [])
    prices, meta, p = YFinanceProvider().fetch(list(symbols) + extra, start, end, refresh)
    problems += p
    pm["prices"] = {"provider": "yfinance", "auto_adjust": False, "actions": True,
                    "symbols": meta.to_dict("records") if len(meta) else []}
    macro, mmeta, p = fetch_macro([f for f in factors if f in FACTOR_SPECS], start, refresh)
    problems += p
    pm["macro"] = mmeta
    fund = None
    if fundamentals_csv is not None:
        fund, n = csv_import.load_fundamentals(fundamentals_csv)
        notes += n
        pm["fundamentals"] = {"source": "CSV upload"}
    elif use_fmp:
        from .providers.fundamentals import fetch_fmp
        try:
            fund, p = fetch_fmp(list(symbols))
            problems += p
            pm["fundamentals"] = {"source": "FMP", "availability": "provider filing dates, unverified"}
            notes.append("FMP statements reflect current (possibly restated) values with provider filing dates; "
                         "treat predictive results using them as approximate until release times are verified.")
        except ProviderError as e:
            problems.append(str(e))
    md = metadata_table(meta, profile=profile)
    return dict(prices=prices, macro=macro, fundamentals=fund, metadata=md, truth=None, provider_meta=pm,
                problems=problems, notes=notes, synthetic=False, control=ctrl)


def load_csv(prices_file, macro_file=None, fundamentals_file=None, metadata_file=None, profile: str = DEFAULT_PROFILE):
    notes = []
    prices, n = csv_import.load_prices(prices_file)
    notes += n
    macro = None
    if macro_file is not None:
        macro, n = csv_import.load_macro(macro_file)
        notes += n
    fund = None
    if fundamentals_file is not None:
        fund, n = csv_import.load_fundamentals(fundamentals_file)
        notes += n
    md = (csv_import.load_metadata(metadata_file) if metadata_file is not None
          else metadata_table(None, prices, profile=profile))
    pm = {"prices": {"source": "CSV"}, "macro": {"source": "CSV" if macro is not None else None},
          "fundamentals": {"source": "CSV" if fund is not None else None}}
    return dict(prices=prices, macro=macro, fundamentals=fund, metadata=md, truth=None, provider_meta=pm,
                problems=[], notes=notes, synthetic=bool(prices["symbol"].astype(str).str.startswith("SYN-").all()),
                control=("SYN-CTRL" if "SYN-CTRL" in set(prices["symbol"]) else PROFILES[profile].control_symbol))


def metadata_table(provider_meta: pd.DataFrame | None, prices: pd.DataFrame | None = None,
                   profile: str = DEFAULT_PROFILE) -> pd.DataFrame:
    base = pd.DataFrame(list(PROFILES[profile].universe))
    if provider_meta is not None and len(provider_meta):
        pmeta = provider_meta.rename(columns={"exchange": "provider_exchange", "currency": "provider_currency"})
        base = base.merge(pmeta, on="symbol", how="outer")
        base["currency_check"] = [
            ("ok" if (pd.isna(a) or pd.isna(b) or a == b) else f"MISMATCH: expected {a}, provider {b}")
            for a, b in zip(base.get("currency"), base.get("provider_currency"))]
    elif prices is not None:
        extra = prices.groupby("symbol").agg(currency=("currency", "first")).reset_index()
        base = base.merge(extra, on="symbol", how="outer", suffixes=("", "_csv"))
    return base
