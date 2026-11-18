"""CSV round trip, provider failure handling, provenance export."""
import io
import json

import numpy as np
import pandas as pd
import pytest

from mfd import provenance
from mfd.data import load_csv
from mfd.prep import build_panel
from mfd.providers import base
from mfd.providers.csv_import import CSVSchemaError, load_prices


def _csv(df):
    return df.to_csv(index=False).encode()


def test_csv_round_trip_reproduces_panel(synth, synth_panel):
    d = load_csv(_csv(synth["prices"]), _csv(synth["macro"]), _csv(synth["fundamentals"]))
    assert d["synthetic"]  # SYN- symbols keep their label through CSV
    p2 = build_panel(d["prices"], d["macro"], d["fundamentals"], list(synth["truth"]["symbol"]), "SYN-MKT")
    a, b = synth_panel["panel"], p2["panel"]
    cols = ["ret", "market", "oil", "rate", "fx", "debt_to_assets", "lag_oil", "lag_debt_to_assets", "target_next"]
    assert np.allclose(a[cols].to_numpy(float), b[cols].to_numpy(float), equal_nan=True)


def test_csv_schema_errors_are_actionable():
    with pytest.raises(CSVSchemaError, match="missing required column"):
        load_prices(b"ticker,date\nA,2020-01-01\n")


def test_naive_timestamps_are_flagged(synth):
    from mfd.providers.csv_import import load_fundamentals
    f = synth["fundamentals"].head(8).copy()
    f["available_at"] = f["available_at"].dt.tz_localize(None).astype(str)
    out, notes = load_fundamentals(_csv(f))
    assert str(out["available_at"].dt.tz) == "America/Toronto"
    assert any("no time zone" in n for n in notes)


def test_provider_failure_is_source_specific(monkeypatch, tmp_path):
    import requests
    monkeypatch.setattr(base, "CACHE_DIR", tmp_path)

    def boom(*a, **k):
        raise requests.ConnectionError("blocked")
    monkeypatch.setattr(requests.Session, "get", boom)
    from mfd.providers.macro import fetch_macro
    macro, metas, problems = fetch_macro(["oil", "rate"], "2020-01-01")
    assert macro.empty
    assert any("[FRED]" in p for p in problems) and any("Bank of Canada" in p for p in problems)
    assert all("CSV" in p for p in problems)


def test_cache_round_trip(monkeypatch, tmp_path):
    monkeypatch.setattr(base, "CACHE_DIR", tmp_path)
    calls = []

    def fetch():
        calls.append(1)
        return pd.DataFrame({"a": [1, 2]})
    s1 = base.cached("test", {"x": 1}, fetch)
    s2 = base.cached("test", {"x": 1}, fetch)
    assert len(calls) == 1 and s2.meta["cache_status"] == "cached snapshot"
    base.cached("test", {"x": 1}, fetch, refresh=True)
    assert len(calls) == 2


def test_run_config_has_provenance_and_no_secrets(synth):
    settings = {"data_mode": "synthetic", "return_convention": "adjusted_close", "fmp_api_key": "SHOULD_NOT_APPEAR"}
    cfg = provenance.run_config(settings, synth, {"prices": {"source": "SYNTHETIC", "apikey": "x"}})
    blob = json.dumps(cfg)
    assert "SHOULD_NOT_APPEAR" not in blob and '"apikey": "x"' not in blob
    assert cfg["dataset_fingerprints"]["prices"] and cfg["config_fingerprint"]
    assert cfg["package_versions"]["statsmodels"]
    z = provenance.bundle(cfg, {"input_prices": synth["prices"].head(50)})
    cfg2, files = provenance.read_bundle(z)
    assert cfg2["config_fingerprint"] == cfg["config_fingerprint"] and "input_prices" in files


def test_fingerprint_changes_with_data(synth):
    a = provenance.frame_fingerprint(synth["prices"].head(100))
    b = provenance.frame_fingerprint(synth["prices"].head(100).assign(adjusted_close=lambda d: d.adjusted_close * 1.0001))
    assert a != b
