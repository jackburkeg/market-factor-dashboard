"""Shared provider plumbing: errors, bounded-retry HTTP, Parquet snapshot cache."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

CACHE_DIR = Path(os.environ.get("MFD_CACHE_DIR", Path(__file__).resolve().parents[2] / "data_cache"))
HTTP_TIMEOUT = 20
USER_AGENT = "MarketFactorDashboard/0.1 (personal research)"


class ProviderError(RuntimeError):
    """A source-specific failure with a user-actionable message."""

    def __init__(self, source: str, message: str, hint: str = ""):
        self.source, self.message, self.hint = source, message, hint
        super().__init__(f"[{source}] {message}" + (f" — {hint}" if hint else ""))


def http_session(retries: int = 3, backoff: float = 1.0) -> requests.Session:
    s = requests.Session()
    retry = Retry(total=retries, connect=retries, read=retries, backoff_factor=backoff,
                  status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",),
                  raise_on_status=False)
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.headers["User-Agent"] = USER_AGENT
    return s


def get_json(source: str, url: str, params: dict | None = None, redact: tuple = ("apikey", "api_key")) -> object:
    safe_params = {k: ("***" if k in redact else v) for k, v in (params or {}).items()}
    try:
        r = http_session().get(url, params=params, timeout=HTTP_TIMEOUT)
    except requests.RequestException as e:
        raise ProviderError(source, f"network error contacting {url}: {type(e).__name__}",
                            "check your connection, or switch Data mode to CSV import") from None
    if r.status_code != 200:
        raise ProviderError(source, f"HTTP {r.status_code} from {url} (params {safe_params})",
                            "the series id may be wrong or the service unavailable; CSV import still works")
    try:
        return r.json()
    except ValueError:
        raise ProviderError(source, f"non-JSON response from {url}") from None


def get_text(source: str, url: str, params: dict | None = None) -> str:
    try:
        r = http_session().get(url, params=params, timeout=HTTP_TIMEOUT)
    except requests.RequestException as e:
        raise ProviderError(source, f"network error contacting {url}: {type(e).__name__}",
                            "check your connection, or switch Data mode to CSV import") from None
    if r.status_code != 200:
        raise ProviderError(source, f"HTTP {r.status_code} from {url}")
    return r.text


@dataclass
class Snapshot:
    """A cached dataset plus provenance."""
    df: pd.DataFrame
    meta: dict = field(default_factory=dict)


def _key(source: str, params: dict) -> str:
    blob = json.dumps({"source": source, **params}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def cached(source: str, params: dict, fetch: Callable[[], pd.DataFrame], refresh: bool = False,
           extra_meta: dict | None = None) -> Snapshot:
    """Return a Parquet snapshot for (source, params), fetching if absent or refresh=True.

    The cache never stores API keys: `params` must already exclude them.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    k = _key(source, params)
    pq, mj = CACHE_DIR / f"{source}_{k}.parquet", CACHE_DIR / f"{source}_{k}.json"
    if pq.exists() and mj.exists() and not refresh:
        meta = json.loads(mj.read_text())
        meta["cache_status"] = "cached snapshot"
        return Snapshot(pd.read_parquet(pq), meta)
    df = fetch()
    retrieved = pd.Timestamp.now(tz="UTC").isoformat()
    meta = {"source": source, "params": params, "retrieved_at": retrieved, "rows": int(len(df)),
            "snapshot_file": pq.name, **(extra_meta or {})}
    df.to_parquet(pq, index=False)
    mj.write_text(json.dumps(meta, indent=2, default=str))
    meta["cache_status"] = "fresh download"
    return Snapshot(df, meta)


def list_snapshots() -> pd.DataFrame:
    if not CACHE_DIR.exists():
        return pd.DataFrame(columns=["file", "source", "retrieved_at", "rows"])
    rows = []
    for mj in sorted(CACHE_DIR.glob("*.json")):
        try:
            m = json.loads(mj.read_text())
            rows.append(dict(file=m.get("snapshot_file"), source=m.get("source"),
                             retrieved_at=m.get("retrieved_at"), rows=m.get("rows"),
                             params=json.dumps(m.get("params"), default=str)))
        except Exception:  # corrupt metadata is shown, not fatal
            rows.append(dict(file=mj.name, source="?", retrieved_at=None, rows=None, params="unreadable"))
    return pd.DataFrame(rows)
