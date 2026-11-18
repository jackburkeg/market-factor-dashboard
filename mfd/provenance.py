"""Run configuration, dataset fingerprints and reproducible export bundles."""
from __future__ import annotations

import hashlib
import io
import json
import platform
import zipfile
from importlib import metadata

import pandas as pd

PACKAGES = ["pandas", "numpy", "scipy", "statsmodels", "scikit-learn", "pyportfolioopt", "cvxpy", "streamlit",
            "plotly", "pyarrow", "requests", "yfinance"]
SECRET_HINTS = ("key", "token", "secret", "password")


def package_versions() -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def frame_fingerprint(df: pd.DataFrame | None) -> str | None:
    if df is None:
        return None
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_datetime64_any_dtype(d[c]) or isinstance(d[c].dtype, pd.PeriodDtype):
            d[c] = d[c].astype(str)
    d = d.reindex(sorted(d.columns), axis=1)
    h = pd.util.hash_pandas_object(d.astype(str), index=False).to_numpy()
    return hashlib.sha256(h.tobytes()).hexdigest()


def _scrub(obj):
    if isinstance(obj, dict):
        return {k: ("***" if any(s in k.lower() for s in SECRET_HINTS) else _scrub(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub(v) for v in obj]
    return obj


def run_config(settings: dict, data: dict, provider_meta: dict, extra: dict | None = None) -> dict:
    fps = {k: frame_fingerprint(data.get(k)) for k in ("prices", "macro", "fundamentals")}
    cfg = {
        "app": "Market Factor Dashboard",
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "data_mode": settings.get("data_mode"),
        "synthetic": settings.get("data_mode") == "synthetic",
        "settings": settings,
        "sources": provider_meta,
        "transformations": {
            "returns": f"monthly simple total return from {settings.get('return_convention')}; completed months only",
            "oil": "log(month-end WTI / prior month-end), positive prices only",
            "rate": "month-end level difference, percentage points",
            "fx": "log change of CAD per USD",
            "debt_to_assets": "interest-bearing debt / total assets" + (" incl. leases" if settings.get("include_leases_in_debt") else " excl. leases"),
            "earnings_growth": f"YoY net-income change / |prior|; missing if |prior| ≤ {settings.get('small_denominator_frac')}·assets",
            "decision_cutoff": "17:00 America/Toronto on the month's last TSX session; as-of joins on available_at",
        },
        "package_versions": package_versions(),
        "dataset_fingerprints": fps,
        **(extra or {}),
    }
    blob = json.dumps({"settings": settings, "fps": fps}, sort_keys=True, default=str)
    cfg["config_fingerprint"] = hashlib.sha256(blob.encode()).hexdigest()
    return _scrub(json.loads(json.dumps(cfg, default=str)))


def bundle(cfg: dict, tables: dict) -> bytes:
    """Zip of run_config.json plus normalized input/output CSVs; reloadable via CSV import."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("run_config.json", json.dumps(cfg, indent=2, default=str))
        for name, df in tables.items():
            if df is not None and len(df):
                z.writestr(f"{name}.csv", df.to_csv(index=False))
    return buf.getvalue()


def read_bundle(data: bytes) -> tuple[dict, dict]:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        cfg = json.loads(z.read("run_config.json"))
        files = {n[:-4]: z.read(n) for n in z.namelist() if n.endswith(".csv")}
    return cfg, files
