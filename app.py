"""Market Factor Dashboard — Streamlit entry point.

Run:  streamlit run app.py
"""
from __future__ import annotations

import json
import math
import uuid

import numpy as np
import pandas as pd
import streamlit as st


def _fresh_mfd_modules():
    """When hosted apps pull a new commit, Streamlit reruns app.py but can keep older copies of the
    mfd package in memory. If any mfd file changed on disk, drop the cached modules so they re-import."""
    import sys
    from pathlib import Path
    root = Path(__file__).with_name("mfd")
    stamp = tuple(sorted((str(p), p.stat().st_mtime_ns) for p in root.rglob("*.py")))
    pkg = sys.modules.get("mfd")
    if pkg is not None and getattr(pkg, "_source_stamp", None) != stamp:
        for name in [m for m in sys.modules if m == "mfd" or m.startswith("mfd.")]:
            del sys.modules[name]
    import mfd
    mfd._source_stamp = stamp


_fresh_mfd_modules()

from mfd import charts as C  # noqa: E402
from mfd import forecast as F
from mfd import live as LV
from mfd import portfolio as P
from mfd import regression as R
from mfd.config import (APP_TITLE, AnalysisSettings, FACTOR_SPECS, FIRM_FEATURES, PROFILES, DEFAULT_PROFILE)
from mfd.data import load_csv, load_live, load_synthetic
from mfd.prep import build_panel, coverage_table
from mfd.providers.base import ProviderError, list_snapshots
from mfd.providers.csv_import import CSVSchemaError
from mfd.provenance import bundle, read_bundle, run_config

def _load_dotenv(path=".env"):
    """Minimal .env reader (KEY=VALUE lines); existing environment variables win."""
    import os
    from pathlib import Path
    f = Path(__file__).with_name(path)
    if f.exists():
        for line in f.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                if v.strip():
                    os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()
st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="📈")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown, .stDataFrame, button, input, label { font-family: 'Inter', -apple-system, sans-serif; }
.block-container { padding-top: 1.6rem; }
.syn-banner { background:#fff4e5; border:1px solid #eda100; color:#5a3b00; padding:.5rem .8rem; border-radius:6px;
              font-weight:600; margin-bottom:.6rem; }
.small-note { color:#52514e; font-size:0.86rem; }
</style>""", unsafe_allow_html=True)

DEFAULTS = AnalysisSettings()
FACTOR_LABELS = {"market": "Benchmark total return", **{k: v.label for k, v in FACTOR_SPECS.items()},
                 **{k: v["label"] for k, v in FIRM_FEATURES.items()}}
FACTOR_UNITS = {"market": "simple return", **{k: v.unit for k, v in FACTOR_SPECS.items()},
                **{k: v["unit"] for k, v in FIRM_FEATURES.items()}}


def pct(x, d=2):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.{d}f}%"


def wkw(key, name, value):
    """Pass a widget default only when Session State does not already hold its value."""
    return {} if key in st.session_state else {name: value}


def fix(key, options, default, multi=False):
    """Drop stale Session State values that are not valid options any more (e.g. after switching data)."""
    if key not in st.session_state:
        return
    v = st.session_state[key]
    if multi:
        keep = [x for x in (v or []) if x in options]
        st.session_state[key] = keep if keep else [x for x in default if x in options]
    elif v not in options:
        st.session_state[key] = default if default in options else (options[0] if options else None)


def sel(container, label, options, default, key, **kw):
    options = list(options)
    fix(key, options, default)
    idx = options.index(default) if default in options else 0
    return container.selectbox(label, options, key=key, **wkw(key, "index", idx), **kw)


def msel(container, label, options, default, key, **kw):
    options = list(options)
    fix(key, options, default, multi=True)
    return container.multiselect(label, options, key=key, **wkw(key, "default", [d for d in default if d in options]), **kw)


# =========================================================================== data loading (cached)
@st.cache_data(show_spinner="Generating deterministic synthetic dataset…")
def _synthetic(preset):
    return load_synthetic(preset)


@st.cache_data(show_spinner="Downloading / reading cached provider data…")
def _live(symbols, benchmark, start, end, factors, refresh_token, fund_bytes, use_fmp, preset):
    return load_live(list(symbols), benchmark, start, end, list(factors), refresh=refresh_token > 0,
                     fundamentals_csv=fund_bytes, use_fmp=use_fmp, profile=preset)


@st.cache_data(show_spinner="Reading CSV files…")
def _csv(prices_b, macro_b, fund_b, meta_b, preset):
    return load_csv(prices_b, macro_b, fund_b, meta_b, profile=preset)


@st.cache_data(show_spinner="Aligning returns, macro and fundamentals on the decision calendar…")
def _panel(_data, data_key, symbols, benchmark, start_month, end_month, convention, include_leases, small_frac,
           exchange, control, analysis_start):
    return build_panel(_data["prices"], _data["macro"], _data["fundamentals"], list(symbols), benchmark,
                       end_month=end_month, start_month=start_month, convention=convention,
                       include_leases=include_leases, small_denominator_frac=small_frac, exchange=exchange,
                       control=control, analysis_start=dict(analysis_start) if analysis_start else None)


@st.cache_data(show_spinner="Fitting per-stock regressions…")
def _fits(_panel_df, key, symbols, xs, start, end, lag_rule, lags):
    fits, errors = {}, {}
    for s in symbols:
        df = _panel_df[_panel_df["symbol"] == s].set_index("month")
        df = df[(df.index >= start) & (df.index <= end)]
        try:
            fits[s] = R.fit_ols(df, "ret", list(xs), s, lag_rule, lags)
        except R.ModelSpecError as e:
            errors[s] = str(e)
    return fits, errors


@st.cache_data(show_spinner="Running walk-forward forecasts (training-window-only preprocessing and tuning)…")
def _forecasts(_panel_df, key, symbols, feats, min_train, window, trailing, alphas, cv, impute, interaction,
               retune, pi_level, run_id):
    recs, skipped = [], []
    for s in symbols:
        d = _panel_df[_panel_df["symbol"] == s].set_index("month")
        fc, sk = F.walk_forward(d, list(feats), s, min_train, window, trailing, alphas, cv, impute, interaction,
                                retune, run_id)
        if len(fc):
            recs.append(fc)
        skipped += sk
    if not recs:
        return pd.DataFrame(), skipped
    fc = pd.concat(recs, ignore_index=True)
    return F.add_empirical_intervals(fc, pi_level), skipped


@st.cache_data(show_spinner="Rolling refits…")
def _rolling(_panel_df, key, symbols, xs, window, min_obs, lag_rule, lags, interaction, slope_at, start, end):
    out, skipped = [], []
    for s in symbols:
        d = _panel_df[_panel_df["symbol"] == s].set_index("month")
        d = d[(d.index >= start) & (d.index <= end)]
        r, sk = R.rolling_fits(d, "ret", list(xs), window, min_obs, lag_rule, lags, interaction, slope_at, s)
        if len(r):
            out.append(r)
        skipped += sk
    return (pd.concat(out, ignore_index=True) if out else pd.DataFrame()), skipped


@st.cache_data(show_spinner="Backtesting rebalanced portfolios…")
def _backtest(_p, key, symbols, benchmark, lookback, cap, fee, slip, every, charge_initial):
    mw = _p["monthly_returns"].pivot(index="month", columns="symbol", values="ret")
    return P.backtest(_p["tri"], mw, _p["month_calendar"], list(symbols), benchmark, lookback, cap, fee, slip,
                      every, charge_initial)


# =========================================================================== bundle restore
def _restore_from_bundle(cfg: dict):
    s = cfg.get("settings", {})
    pr = s.get("preset", DEFAULT_PROFILE)
    st.session_state["_pending_preset"] = pr
    mapping = {"tickers": "w_tickers", "benchmark": "w_benchmark", "factors": "w_factors",
               "firm_features": "w_firm", "hac_lag_rule": "w_hac_rule", "hac_lags": "w_hac_lags",
               "rolling_window": "w_roll", "return_convention": "w_conv", "include_leases_in_debt": "w_leases",
               "small_denominator_frac": "w_smallden", "start": "w_start", "end": "w_end"}
    for k, w in mapping.items():
        if s.get(k) not in (None, ""):
            st.session_state[f"{w}__{pr}"] = s[k]
    st.session_state["restored_config"] = cfg


# =========================================================================== sidebar
sb = st.sidebar
sb.title(APP_TITLE)
sb.markdown("**1 · Choose market and data**")
if "_pending_preset" in st.session_state:
    st.session_state["w_preset"] = st.session_state.pop("_pending_preset")
preset = sb.selectbox("Market preset", list(PROFILES), key="w_preset", format_func=lambda k: PROFILES[k].label,
                      help="Sets the default stocks, benchmarks, factors, trading calendar, currency and the "
                           "interaction question. Everything stays editable.")
PF = PROFILES[preset]


def K(name: str) -> str:
    """Widget keys are per preset, so switching presets never shows stale selections."""
    return f"{name}__{preset}"

mode = sb.radio("Data mode", ["Synthetic (labelled)", "Live providers", "CSV import"], key="w_mode",
                help="Synthetic exercises every feature with a known data-generating process. Live uses yfinance "
                     "and FRED (plus Bank of Canada Valet for the Canada preset), for personal research use. "
                     "CSV import works offline.")
data, load_error = None, None
if "refresh_token" not in st.session_state:
    st.session_state.refresh_token = 0

if mode.startswith("Synthetic"):
    data = _synthetic(preset)
elif mode == "Live providers":
    default_live = [m["symbol"] for m in PF.universe if not m.get("optional") and not m.get("control")]
    optional = [m["symbol"] for m in PF.universe if m.get("optional")]
    tick_text = sb.text_input("Tickers (comma-separated)", ", ".join(default_live), key=K("w_live_tickers"),
                              help=("Optional (very short history): " + ", ".join(optional)) if optional else None)
    live_syms = tuple(t.strip().upper() for t in tick_text.split(",") if t.strip())
    bench_live = sb.text_input("Benchmark ETF", PF.default_benchmark, key=K("w_live_bench"),
                               help="; ".join(f"{k}: {v}" for k, v in PF.benchmarks.items())).strip().upper()
    years = sb.number_input("Requested history (years)", 3, 30, DEFAULTS.history_years, key="w_years")
    fund_up = sb.file_uploader("Fundamentals CSV (optional, recommended)", type="csv", key="w_live_fund")
    import os
    use_fmp = sb.checkbox("Also try Financial Modeling Prep (needs FMP_API_KEY)", False,
                          disabled=not os.environ.get("FMP_API_KEY"), key="w_fmp")
    c1, c2 = sb.columns(2)
    if c1.button("Refresh data", help="Bypass the Parquet cache and download again"):
        st.session_state.refresh_token += 1
        st.cache_data.clear()
    end_d = pd.Timestamp.today().normalize()
    start_d = end_d - pd.DateOffset(years=int(years))
    try:
        data = _live(live_syms, bench_live, start_d.date().isoformat(), end_d.date().isoformat(),
                     PF.macro_factors, st.session_state.refresh_token,
                     fund_up.getvalue() if fund_up else None, use_fmp, preset)
        if data["prices"].empty:
            load_error = ("No price data could be retrieved. Source-specific errors:\n\n" +
                          "\n".join(f"- {p}" for p in data["problems"]) +
                          "\n\nSwitch Data mode to **CSV import** (templates in `templates/`) to continue offline.")
            data = None
    except (ProviderError, CSVSchemaError) as e:
        load_error = str(e)
else:
    bundle_up = sb.file_uploader("Run bundle (.zip from Data & exports)", type="zip", key="w_bundle")
    if bundle_up is not None and st.session_state.get("_bundle_name") != bundle_up.name + str(bundle_up.size):
        cfg, files = read_bundle(bundle_up.getvalue())
        st.session_state["_bundle_files"] = files
        st.session_state["_bundle_name"] = bundle_up.name + str(bundle_up.size)
        _restore_from_bundle(cfg)
        st.rerun()
    files = st.session_state.get("_bundle_files", {})
    pu = sb.file_uploader("Prices CSV (required)", type="csv", key="w_prices")
    mu = sb.file_uploader("Macro observations CSV", type="csv", key="w_macro")
    fu = sb.file_uploader("Fundamentals CSV", type="csv", key="w_fund")
    du = sb.file_uploader("Universe metadata CSV", type="csv", key="w_meta")
    pb = pu.getvalue() if pu else files.get("input_prices")
    mb = mu.getvalue() if mu else files.get("input_macro")
    fb = fu.getvalue() if fu else files.get("input_fundamentals")
    db = du.getvalue() if du else files.get("input_metadata")
    if pb is None:
        load_error = "Upload a prices CSV (see templates/prices_template.csv) or a run bundle to begin."
    else:
        try:
            data = _csv(pb, mb, fb, db, preset)
        except (CSVSchemaError, ValueError) as e:
            load_error = f"CSV import failed: {e}"

st.title(APP_TITLE)
st.caption("Research on how fundamentals, commodity prices, interest rates and market conditions relate to stock "
           "returns. Results describe the selected universe only; nothing here is investment advice.")

if data is None:
    st.error(load_error or "No data loaded.")
    st.info("**Ways forward:** choose *Synthetic (labelled)* to explore every feature, or *CSV import* with the "
            "templates in the `templates/` folder.")
    st.stop()

SYN = bool(data.get("synthetic"))
if SYN:
    st.markdown('<div class="syn-banner">SYNTHETIC DATA — every number on this page comes from a simulated dataset '
                'with a known data-generating process. It is for exercising and checking the methods, not for '
                'conclusions about real companies.</div>', unsafe_allow_html=True)
for p in data.get("problems", []):
    st.warning(f"Data problem: {p}")

prices = data["prices"]
all_syms = sorted(prices["symbol"].unique())
meta = data["metadata"]
SYN_BENCH = ("SYN-MKT", "SYN-ENERGY", "SYN-NDX", "SYN-SEMI")
CONTROL = data.get("control") if data.get("control") in all_syms else None
bench_candidates = [s for s in all_syms if s in PF.benchmarks or s in SYN_BENCH] or all_syms
default_bench = (PF.synthetic_benchmark if SYN and PF.synthetic_benchmark in all_syms else
                 (PF.default_benchmark if PF.default_benchmark in all_syms else bench_candidates[0]))
sb.markdown("**2 · Choose stocks and dates**")
benchmark = sel(sb, "Benchmark (market factor)", all_syms, default_bench, K("w_benchmark"),
                         help="Its monthly total return is the market factor. " +
                              "; ".join(f"{k} = {v}" for k, v in PF.benchmarks.items()))
stock_opts = [s for s in all_syms if s not in PF.benchmarks and s not in SYN_BENCH and s != benchmark and s != CONTROL]
tickers = msel(sb, "Watchlist", stock_opts, stock_opts, K("w_tickers"))
if not tickers:
    st.warning("Select at least one stock in the sidebar watchlist.")
    st.stop()

pmin = pd.Period(prices["session_date"].min(), "M")
pmax = pd.Period(prices["session_date"].max(), "M")
month_opts = [str(m) for m in pd.period_range(pmin, pmax, freq="M")]
start_s = sel(sb, "Start month", month_opts, str(max(pmin, pmax - 12 * DEFAULTS.history_years)), K("w_start"))
end_s = sel(sb, "End month (latest completed month by default)", month_opts, month_opts[-1], K("w_end"))
start_m, end_m = pd.Period(start_s, "M"), pd.Period(end_s, "M")

macro_factors = sorted(data["macro"]["factor"].unique()) if data.get("macro") is not None and len(data["macro"]) else []
has_fund = data.get("fundamentals") is not None and len(data["fundamentals"]) > 0
factor_opts = ["market"] + (["control"] if CONTROL else []) + macro_factors
if CONTROL:
    FACTOR_LABELS["control"] = f"{CONTROL} return (control)" if SYN else PF.control_label
    FACTOR_UNITS["control"] = "simple return"
sb.markdown("**3 · Choose what to test**")
factors = msel(sb, "Market & macro factors", factor_opts, list(PF.default_factors), K("w_factors"),
               format_func=lambda k: FACTOR_LABELS.get(k, k))
firm_opts = list(FIRM_FEATURES) if has_fund else []
firm = msel(sb, "Company fundamentals", firm_opts, list(PF.default_firm), K("w_firm"),
                      format_func=lambda k: FIRM_FEATURES[k]["label"],
                      disabled=not has_fund,
                      help=None if has_fund else "Needs dated fundamentals: upload a fundamentals CSV.")
with sb.expander("Advanced settings", expanded=False):
    hac_rule = st.radio("HAC lag rule", ["fixed", "newey_west"], key=K("w_hac_rule"),
                        format_func=lambda x: "Fixed lags" if x == "fixed" else "Newey–West plug-in",
                        help="Heteroscedasticity- and autocorrelation-consistent (Newey–West) standard errors.")
    hac_lags = st.number_input("Fixed HAC lags (months)", 0, 24, key=K("w_hac_lags"), **wkw(K("w_hac_lags"), "value", DEFAULTS.hac_lags))
    roll_window = st.number_input("Rolling window (months)", 12, 180, step=6, key=K("w_roll"), **wkw(K("w_roll"), "value", PF.rolling_window))
    min_obs = st.number_input("Minimum observations per window", 12, 180, key=K("w_minobs"), **wkw(K("w_minobs"), "value", PF.min_obs_rolling))
    convention = st.selectbox("Total-return convention", ["adjusted_close", "close_plus_dividends"], key=K("w_conv"),
                              help="adjusted_close: provider split+dividend-adjusted closes (dividends counted once, "
                                   "inside the adjustment). close_plus_dividends: (close + dividend)/previous close "
                                   "with split-adjusted closes.")
    include_leases = st.checkbox("Include lease liabilities in debt", key=K("w_leases"), **wkw(K("w_leases"), "value", DEFAULTS.include_leases_in_debt))
    small_frac = st.number_input("Small-denominator threshold (share of assets)", 0.0, 0.05,
                                 step=0.001, **wkw(K("w_smallden"), "value", DEFAULTS.small_denominator_frac), format="%.3f", key=K("w_smallden"),
                                 help="Earnings growth is left missing when |prior-year net income| is below this "
                                      "fraction of total assets.")
with sb.expander("Data status & cache", expanded=False):
    st.write(f"Mode: **{mode}**")
    st.write(f"Prices: {prices['session_date'].min().date()} → {prices['session_date'].max().date()}")
    snaps = list_snapshots()
    if len(snaps):
        st.dataframe(snaps[["source", "retrieved_at", "rows"]], hide_index=True, width="stretch")
    else:
        st.caption("No provider snapshots cached yet.")
    if mode == "Live providers":
        st.caption("Use **Refresh data** above to bypass the cache.")

settings = DEFAULTS.to_dict()
settings.update(tickers=list(tickers), benchmark=benchmark, start=str(start_m), end=str(end_m), factors=list(factors),
                firm_features=list(firm), hac_lag_rule=hac_rule, hac_lags=int(hac_lags), rolling_window=int(roll_window),
                min_obs_rolling=int(min_obs), return_convention=convention, include_leases_in_debt=include_leases,
                small_denominator_frac=float(small_frac), data_mode="synthetic" if SYN else mode, preset=preset,
                control_symbol=CONTROL, exchange=PF.exchange, currency=PF.currency)

# =========================================================================== panel
data_key = json.dumps(data.get("provider_meta", {}), default=str) + str(len(prices)) + str(prices["session_date"].max())
try:
    a_start = () if SYN else tuple((m["symbol"], m["analysis_start"]) for m in PF.universe if m.get("analysis_start"))
    pan = _panel(data, data_key + preset, tuple(tickers), benchmark, start_m - 1, end_m, convention, include_leases,
                 small_frac, PF.exchange, CONTROL, a_start)
except Exception as e:
    st.error(f"Could not align the data: {e}")
    st.stop()
panel = pan["panel"]
pkey = data_key + preset + str((CONTROL, tuple(tickers), benchmark, str(start_m), str(end_m), convention, include_leases, small_frac))
eff_end = min(end_m, pan["end_month"])
colors = C.color_map(tickers)
X = list(factors) + list(firm)
if "results" not in st.session_state:
    st.session_state.results = {}
RES = st.session_state.results

(t_home, t_today, t_over, t_drv, t_int, t_roll, t_fc, t_pf, t_exp) = st.tabs(
    ["Start here", "Latest month", "Data", "What moves each stock", PF.interaction_label, "Stability over time",
     "Can it predict?", "Portfolio", "Export"])

FITS, FIT_ERRS = (_fits(panel, pkey, tuple(tickers), tuple(X), start_m, eff_end, hac_rule, int(hac_lags))
                  if X else ({}, {}))
BETAS = pd.DataFrame({s: r.params for s, r in FITS.items()}).T if FITS else pd.DataFrame()
n_months = len(pd.period_range(start_m, eff_end, freq="M"))


def _sig(q):
    return "clear" if pd.notna(q) and q < 0.05 else "not clearly different from zero"


# =========================================================================== Start here
with t_home:
    st.subheader("What this dashboard does")
    drivers = ("**the market, oil prices, interest rates and the Canadian dollar**" if preset == "canada_energy" else
               "**the market, NVIDIA (as an AI-cycle control), Treasury yields, market volatility and the US dollar**")
    st.markdown(f"It asks how the stocks in your watchlist have moved with {drivers}, and whether **company "
                "fundamentals** (debt, profitability, earnings) change those relationships. It then tests whether any "
                "of this **predicts next month's returns** better than simple guesses, and explores a low-risk "
                f"**portfolio**. Everything is calculated from historical monthly data. Preset: **{PF.label}**.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data source", "Demo" if SYN else mode.split()[0], help="Synthetic demo data" if SYN else mode)
    c2.metric("Stocks", len(tickers))
    c3.metric("Months analysed", n_months, help=f"{start_m} to {eff_end}")
    c4.metric("Latest prices", prices["session_date"].max().strftime("%b %d, %Y"))

    st.subheader(f"Key findings, {start_m} to {eff_end}")
    if not FITS:
        st.info("Select at least one factor in the sidebar to see findings.")
    else:
        eff_all = R.effects_table(FITS)
        lines = []
        HEADLINE_TEXT = {"oil": "when WTI rose 10% in a month", "y10": "when the 10-year Treasury yield rose 25 bp in a month"}
        hf = PF.headline_factor
        if hf in X and (eff_all.term == hf).any():
            o = eff_all[eff_all.term == hf].sort_values("effect", ascending=False)
            top, bot = o.iloc[0], o.iloc[-1]
            lines.append(f"**{FACTOR_LABELS.get(hf, hf).split(' (')[0]}:** {HEADLINE_TEXT.get(hf, R.EFFECT_SCALES[hf][1])}, "
                         f"**{top.symbol}** had the most positive typical response ({top.effect * 100:+.1f} percentage "
                         f"points, range {top.effect_ci_low * 100:+.1f} to {top.effect_ci_high * 100:+.1f}) and "
                         f"**{bot.symbol}** the most negative ({bot.effect * 100:+.1f} pp, range "
                         f"{bot.effect_ci_low * 100:+.1f} to {bot.effect_ci_high * 100:+.1f}). "
                         + (f"Both links were {_sig(top.q_value)}." if _sig(top.q_value) == _sig(bot.q_value) else
                            f"The link was {_sig(top.q_value)} for {top.symbol} and {_sig(bot.q_value)} for {bot.symbol}."))
        if "market" in X:
            m = eff_all[eff_all.term == "market"].sort_values("estimate")
            lines.append(f"**Market:** betas ranged from {m.iloc[0].estimate:.2f} ({m.iloc[0].symbol}, steadiest relative "
                         f"to the benchmark) to {m.iloc[-1].estimate:.2f} ({m.iloc[-1].symbol}). A beta of 1 means the "
                         "stock tended to move one-for-one with the market"
                         + (", holding the control stock fixed." if "control" in X else "."))
        if "control" in X and (eff_all.term == "control").any():
            g = eff_all[eff_all.term == "control"].sort_values("estimate")
            clear = g[g.q_value < 0.05]["symbol"].tolist()
            lines.append(f"**{CONTROL} control:** beyond the general market, a 1 pp higher {CONTROL} return was associated "
                         f"with {g.estimate.min():+.2f} to {g.estimate.max():+.2f} pp for these stocks; clear in "
                         f"{len(clear)} of {len(g)}" + (f" ({', '.join(clear)})." if clear else ".")
                         + " Note that the benchmark and the control move together, so their separate estimates are less precise.")
        for f in [f for f in X if f in FACTOR_SPECS and f != hf]:
            g = eff_all[eff_all.term == f]
            if len(g):
                clear = g[g.q_value < 0.05]["symbol"].tolist()
                lines.append(f"**{FACTOR_LABELS.get(f, f).split(' (')[0]}:** a clear association in {len(clear)} of "
                             f"{len(g)} stocks" + (f" ({', '.join(clear)})." if clear else "."))
        ia, ib = PF.interaction
        if ia in X and ib in X and ib in panel.columns:
            hits_pos, hits_neg, n_ok = [], [], 0
            for s_ in tickers:
                d = panel[panel.symbol == s_].set_index("month")
                d = d[(d.index >= start_m) & (d.index <= eff_end)]
                try:
                    ri = R.fit_interaction(d, "ret", ia, ib, [x for x in X if x not in (ia, ib)], s_, hac_rule, int(hac_lags))
                    n_ok += 1
                    term = f"{ia}_x_{ib}"
                    if ri.pvalues[term] < 0.05:
                        (hits_pos if ri.params[term] > 0 else hits_neg).append(s_)
                except R.ModelSpecError:
                    pass
            lines.append(f"**{PF.interaction_label}:** asks {PF.interaction_question}. A clear dependence appeared in "
                         f"{len(hits_pos) + len(hits_neg)} of {n_ok} stocks"
                         + (f": sensitivity rose with higher {FACTOR_LABELS.get(ib, ib).lower()} for {', '.join(hits_pos)}"
                            if hits_pos else "")
                         + (f"; it fell for {', '.join(hits_neg)}" if hits_neg else "") + ". "
                         + f"(Raw p < 0.05, not adjusted for multiple tests; see *{PF.interaction_label}*.)")
        st.markdown("\n\n".join(f"- {l}" for l in lines))
        for s, e in FIT_ERRS.items():
            st.warning(e)
        st.caption("“Clear” = Benjamini–Hochberg q < 0.05. These are historical associations, not causes. "
                   "Ranges are 95% confidence intervals.")
    home_forecast = st.empty()
    home_portfolio = st.empty()

    st.subheader("How to use it")
    st.markdown(f"""
1. **Pick your data and stocks** in the left sidebar. *Synthetic* is a built-in demo with known answers. *Live* downloads real prices. *CSV import* uses your own files.
2. **Latest month**: what explained last month's moves for each stock, plus this month so far in Live mode.
3. **What moves each stock**: how strongly each stock has moved with each factor, with uncertainty ranges.
4. **{PF.interaction_label}**: {PF.interaction_question}.
5. **Stability over time**: whether those relationships held steady or shifted.
6. **Can it predict?**: an honest test of next-month forecasts against simple baselines.
7. **Portfolio**: a minimum-risk mix of the stocks compared with equal weights, including trading costs.
8. **Export**: download every table, or a bundle that reproduces this exact analysis.
""")
    with st.expander("Glossary (plain-language definitions)"):
        st.markdown("""
- **Beta / sensitivity**: how much a stock's monthly return tended to move per unit move in a factor.
- **Percentage point (pp)**: the difference between two percentages (5% → 7% is +2 pp).
- **Basis point (bp)**: one hundredth of a percentage point (25 bp = 0.25 pp).
- **Log change**: a way of measuring percentage moves that adds up neatly over time. A +10% move is log(1.10) ≈ 0.095.
- **USD/CAD**: Canadian dollars per US dollar. When it rises, the Canadian dollar has weakened.
- **Confidence interval (95%)**: the range of values consistent with the data. If it excludes zero, the association is unlikely to be noise.
- **HAC standard errors**: uncertainty estimates that allow for volatile periods and month-to-month carry-over in the data.
- **q-value**: a p-value adjusted for testing many coefficients at once (false discovery rate).
- **Interaction**: when one factor's effect depends on another (e.g. oil matters more for indebted companies).
- **Out-of-sample R²**: how much better a forecast did than simply predicting the historical average. Below 0 means worse.
- **Ledoit–Wolf shrinkage**: a standard way to stabilize a covariance estimate from limited data.
- **Drawdown**: the fall from a previous peak value.
- **Total return**: price change plus dividends.
""")

# =========================================================================== Latest month / today
with t_today:
    st.subheader(f"What explained {eff_end}'s moves")
    comps = [f for f in ["market", "control"] + list(FACTOR_SPECS) if f in X]
    if BETAS.empty or not comps:
        st.info("Needs at least one market/macro factor and a successful fit (see *What moves each stock*).")
    else:
        last = panel[panel.month == eff_end].set_index("symbol")
        moves = {f: last[f].dropna().iloc[0] if f in last and last[f].notna().any() else np.nan for f in comps}
        att = LV.attribute(last["ret"], BETAS, moves)
        if att.empty:
            st.info(f"No complete returns for {eff_end}.")
        else:
            mv = []
            for f, v in moves.items():
                if pd.isna(v):
                    continue
                if f == "market":
                    mv.append(f"benchmark {pct(v, 1)}")
                elif f == "control":
                    mv.append(f"{CONTROL} {pct(v, 1)}")
                elif FACTOR_SPECS[f].transformation == "log_change":
                    mv.append(f"{FACTOR_LABELS.get(f, f).split(' (')[0]} {pct(math.exp(v) - 1, 1)}")
                else:
                    mv.append(f"{FACTOR_LABELS.get(f, f).split(' (')[0]} {v * 100:+.0f} bp")
            st.markdown("**Factor moves that month:** " + " · ".join(mv))
            st.plotly_chart(C.attribution_bar(att, comps, f"{eff_end}: return split into factor contributions", SYN),
                            width="stretch")
            biggest = att.reindex(att["stock_specific"].abs().sort_values(ascending=False).index).iloc[0]
            st.markdown(f"Each bar adds the factor contributions (beta × that month's factor move) and the remaining "
                        f"stock-specific part; the diamond is the actual return. The largest stock-specific move was "
                        f"**{biggest.symbol}** ({biggest.stock_specific * 100:+.1f} pp), i.e. news particular to that "
                        "company or factors not in the model.")
            st.dataframe((att.set_index("symbol") * 100).round(2).rename(columns={
                "actual": "actual return", **{k: v for k, v in C.COMPONENT_LABELS.items()},
                "explained_by_factors": "explained by factors"}).reset_index(), hide_index=True, width="stretch")
            st.caption("Values in percentage points. Betas are estimated over the selected window, so the split is "
                       "descriptive, not causal.")

    st.subheader("This month so far")
    if mode != "Live providers":
        st.info("Switch the sidebar **Data mode** to *Live providers* to see this month's moves for the real "
                "watchlist. This uses one download per stock, only when you press refresh.")
    elif BETAS.empty:
        st.info("Needs a successful model fit first.")
    else:
        @st.cache_data(ttl=900, show_spinner="Fetching latest closes…")
        def _snapshot(syms, preset_):
            from mfd.live import fetch_snapshot
            return fetch_snapshot(list(syms), preset_)
        if st.button("Refresh latest quotes", help="Downloads once and caches for 15 minutes. Nothing refreshes automatically."):
            st.session_state[K("live_snapshot")] = _snapshot(tuple(tickers) + (benchmark,) + ((CONTROL,) if CONTROL else ()), preset)
            st.session_state.live_snapshot_at = pd.Timestamp.now(tz="America/Toronto")
        snap = st.session_state.get(K("live_snapshot"))
        if snap is None:
            st.caption("Press **Refresh latest quotes** to load this month's moves.")
        else:
            from mfd.live import LIVE_PROXIES, attribute, period_moves
            snap_df, probs = snap
            for p in probs:
                st.warning(p)
            series = {r.symbol: r.series for r in snap_df.itertuples()} if len(snap_df) else {}
            last_month_end = pd.Timestamp(pan["month_calendar"].dropna(subset=["last_session"])["last_session"].max())
            pm = period_moves(series, last_month_end)
            if pm.empty:
                st.error("No quotes could be loaded.")
            else:
                pm = pm.set_index("symbol")
                st.caption(f"Fetched {st.session_state.live_snapshot_at:%Y-%m-%d %H:%M} Toronto time. Month-to-date = "
                           f"latest close vs the {last_month_end.date()} close. Free Yahoo quotes can be delayed.")
                tbl = pm[["last_close", "as_of", "day_change", "period_return"]].rename(
                    columns={"day_change": "last day", "period_return": "month to date"})
                st.dataframe(tbl.style.format({"last_close": "{:.2f}", "last day": "{:+.2%}", "month to date": "{:+.2%}"}),
                             width="stretch")
                fm = {"market": pm["period_return"].get(benchmark, np.nan)}
                if CONTROL:
                    fm["control"] = pm["period_return"].get(CONTROL, np.nan)
                for f, (ysym, _, kind) in LIVE_PROXIES[preset].items():
                    fm[f] = pm["period_log" if kind == "log" else "period_diff"].get(ysym, np.nan)
                fm = {k: v for k, v in fm.items() if k in comps}
                att_live = attribute(pm["period_return"].reindex(tickers).dropna(), BETAS, fm)
                if len(att_live):
                    st.plotly_chart(C.attribution_bar(att_live, [c for c in comps if c in fm],
                                                      "Month to date: return split into factor contributions"), width="stretch")
                    st.caption("Live proxies: " + "; ".join(f"{f} uses {v[0]} ({v[1]})" for f, v in LIVE_PROXIES[preset].items())
                               + ". Policy rates are not moved intra-month. Applying monthly betas to a partial month is "
                               "an approximation.")


# =========================================================================== Overview
with t_over:
    st.subheader("Universe")
    cols = [c for c in ["symbol", "name", "subsector", "exchange", "currency", "provider_exchange", "provider_currency",
                        "currency_check", "status", "notes", "first_trade_date"] if c in meta]
    st.dataframe(meta[meta["symbol"].isin(list(tickers) + [benchmark])][cols], hide_index=True, width="stretch")
    st.markdown('<p class="small-note">Survivorship: the watchlist is chosen today. Companies that failed or were '
                'acquired earlier are absent unless you add them, so results are research on <b>this selected '
                'universe</b>, not on Canadian energy as it existed historically. Delisted names are kept where data '
                'exists and their final partial month is excluded.</p>', unsafe_allow_html=True)
    for n in data.get("notes", []) + pan.get("notes", []):
        st.info(n)
    st.markdown(f"**Analysis window:** {start_m} → {eff_end} (latest completed month with data: {pan['end_month']}). "
                f"Decision cutoff each month: 17:00 Eastern time on its last {PF.exchange} session. Currency: {PF.currency}.")
    c1, c2 = st.columns([3, 2])
    with c1:
        mr = pan["monthly_returns"]
        mr = mr[(mr["month"] >= start_m) & (mr["month"] <= eff_end)]
        st.plotly_chart(C.cumulative_returns(mr, {**colors, benchmark: C.NEUTRAL}, SYN), width="stretch")
    with c2:
        st.markdown("**Coverage by series**")
        st.dataframe(coverage_table(prices, data.get("macro"), data.get("fundamentals")), hide_index=True, width="stretch")
    if macro_factors:
        md = pan["macro_desc"]
        md = md[(md.index >= start_m) & (md.index <= eff_end)]
        st.plotly_chart(C.factor_history(md, FACTOR_LABELS, FACTOR_UNITS, SYN), width="stretch")
    else:
        st.info("No macro series loaded; only the market factor is available.")
    st.subheader("Observations and missing data")
    win = panel[(panel["month"] >= start_m) & (panel["month"] <= eff_end)]
    miss = win.groupby("symbol")[["ret"] + [c for c in X if c in win]].apply(lambda g: g.notna().sum()).reset_index()
    miss.insert(1, "months_in_window", len(pd.period_range(start_m, eff_end, freq="M")))
    complete = win.dropna(subset=["ret"] + [c for c in X if c in win]).groupby("symbol").size().rename("complete_rows_for_model")
    miss = miss.merge(complete, on="symbol", how="left").fillna({"complete_rows_for_model": 0})
    st.caption("Non-missing months per variable in the selected window (descriptive rows).")
    st.dataframe(miss, hide_index=True, width="stretch")
    with st.expander(f"Excluded rows ({len(pan['excluded'])})"):
        st.dataframe(pan["excluded"], hide_index=True, width="stretch")
    with st.expander("Derived features with availability timestamps (point-in-time contract)"):
        fl = pan["features_long"]
        st.dataframe(fl.tail(500) if len(fl) else fl, hide_index=True, width="stretch")
    with st.expander("Definitions and units"):
        for k in X:
            if k in FACTOR_SPECS:
                st.markdown(f"**{FACTOR_SPECS[k].label}** — {FACTOR_SPECS[k].description}")
            elif k in FIRM_FEATURES:
                st.markdown(f"**{FIRM_FEATURES[k]['label']}** — {FIRM_FEATURES[k]['definition']}")
            elif k == "control":
                st.markdown(f"**{FACTOR_LABELS['control']}** — monthly total return of {CONTROL}. Included as a control "
                            "factor: its coefficient measures co-movement with this AI leader beyond the benchmark.")
            elif k == "market":
                st.markdown(f"**Benchmark total return** — monthly simple total return of {benchmark}, used as a "
                            "contemporaneous control in descriptive models and lagged in forecasts.")

# =========================================================================== Stock drivers
with t_drv:
    st.subheader("Historical sensitivity (descriptive, full selected window)")
    st.markdown('<p class="small-note">One OLS regression per stock: monthly total return on the selected factors. '
                'Macro factors are contemporaneous (same month); fundamentals are the latest values known before the '
                'month began. Estimates describe historical co-movement — “associated with”, not “caused by”.</p>',
                unsafe_allow_html=True)
    if not X:
        st.warning("Select at least one factor in the sidebar.")
    else:
        fits, errs = FITS, FIT_ERRS
        for s, e in errs.items():
            st.error(e)
        if fits:
            family = st.radio("Multiple-testing family (Benjamini–Hochberg)",
                              ["all displayed coefficients", "within each stock"], horizontal=True, key="w_family")
            if family == "within each stock":
                eff = pd.concat([R.effects_table({s: r}, f"non-intercept coefficients of {s}") for s, r in fits.items()],
                                ignore_index=True)
            else:
                eff = R.effects_table(fits, f"all non-intercept coefficients across {len(fits)} stocks")
            RES["coefficients"] = eff
            view = st.radio("Display", ["Effect of a typical move (pp)", "Raw coefficient"], horizontal=True, key="w_coefview")
            truth = data.get("truth") if SYN and view == "Raw coefficient" else None
            st.plotly_chart(C.coefficient_plot(eff, colors, view.startswith("Effect"), SYN, truth), width="stretch")
            if SYN and view.startswith("Effect"):
                st.caption("Switch to *Raw coefficient* to overlay the synthetic data's true coefficients (×).")
            show = eff.copy()
            show["estimate"] = show["estimate"].round(4)
            fmt_cols = {"effect": "effect (pp)", "effect_ci_low": "effect CI low", "effect_ci_high": "effect CI high"}
            for c in fmt_cols:
                show[c] = show[c] * 100
            show = show.rename(columns=fmt_cols)
            st.dataframe(show[["symbol", "term", "estimate", "std_error", "ci_low", "ci_high", "p_value", "q_value",
                               "effect_scenario", "effect (pp)", "effect CI low", "effect CI high", "n", "window",
                               "hac_lags", "test_family"]],
                         hide_index=True, width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="%.4f") for c in
                                        ["std_error", "ci_low", "ci_high", "p_value", "q_value"]} |
                         {c: st.column_config.NumberColumn(format="%.2f") for c in
                          ["effect (pp)", "effect CI low", "effect CI high"]})
            st.caption("q-values: Benjamini–Hochberg false-discovery-rate adjustment over the stated family. "
                       "Intercepts are excluded from the family. Confidence intervals use HAC standard errors with "
                       "the t distribution (residual df).")
            sym = sel(st, "Stock detail", list(fits), list(fits)[0], K("w_detail"))
            r = fits[sym]
            fit_cols = st.columns(5)
            fit_cols[0].metric("Months used", r.nobs, help=f"{r.start} → {r.end}; {r.dropped_rows} rows dropped for missing values")
            fit_cols[1].metric("Parameters", len(r.params), help="Including intercept")
            fit_cols[2].metric("Obs / parameter", f"{r.nobs / len(r.params):.1f}")
            fit_cols[3].metric("Adjusted R²", f"{r.rsquared_adj:.3f}", help="Share of return variance described, "
                               "penalized for the number of factors. In-sample description, not predictive skill.")
            fit_cols[4].metric("HAC lags", r.hac_lags, help=r.diagnostics.get("hac_rule"))
            # plain-language interpretation
            lines = []
            for term in r.xs:
                scale, lab = R.EFFECT_SCALES.get(term, (None, ""))
                if scale:
                    lo, hi = r.conf_int.loc[term, "lo"] * scale * 100, r.conf_int.loc[term, "hi"] * scale * 100
                    lines.append(f"- **{FACTOR_LABELS.get(term, term)}**: {lab} was associated with a "
                                 f"{r.params[term] * scale * 100:+.2f} pp monthly return (95% CI {lo:+.2f} to {hi:+.2f}), "
                                 "holding the other factors fixed.")
            st.markdown(f"**{sym}, {r.start}–{r.end}, n={r.nobs}:**\n" + "\n".join(lines))
            for w in r.warnings:
                st.warning(w)
            d = r.diagnostics
            dc1, dc2 = st.columns([2, 3])
            with dc1:
                diag = pd.DataFrame([
                    ("Design rank", f"{d.get('rank')} of {len(r.params)}"),
                    ("Condition number (standardized)", f"{d.get('condition_number', float('nan')):.1f}"),
                    ("Durbin–Watson", f"{d.get('durbin_watson'):.2f}"),
                    (f"Ljung–Box p (lag {d.get('ljung_box_lag')})", f"{d.get('ljung_box_p'):.3f}"),
                    ("Jarque–Bera p (normality)", f"{d.get('jarque_bera_p'):.3f}"),
                    ("Breusch–Pagan p (heteroscedasticity)", f"{d.get('breusch_pagan_p'):.3f}"),
                ], columns=["diagnostic", "value"])
                st.dataframe(diag, hide_index=True, width="stretch")
                if d.get("vif"):
                    st.dataframe(pd.Series(d["vif"], name="VIF").round(2).reset_index().rename(columns={"index": "feature"}),
                                 hide_index=True, width="stretch")
            with dc2:
                if isinstance(d.get("correlation"), pd.DataFrame):
                    st.plotly_chart(C.correlation_heatmap(d["correlation"]), width="stretch")
            st.plotly_chart(C.fitted_vs_actual(r, SYN), width="stretch")
            std = st.checkbox("Show standardized coefficients (β · SD(x) / SD(y), same estimation sample)", key="w_std")
            if std:
                st.dataframe(pd.DataFrame({s: R.standardized_coefs(f) for s, f in fits.items()}).round(3), width="stretch")

# =========================================================================== Interactions
with t_int:
    st.subheader("Interaction: does one factor's sensitivity depend on another?")
    st.markdown('<p class="small-note">Model: return = controls + β<sub>a</sub>·a + β<sub>b</sub>·(b − b̄) + '
                'β<sub>int</sub>·a·(b − b̄). With b centered at its window mean b̄, β<sub>a</sub> is the sensitivity '
                'to a at average b, and the sensitivity at any b is β<sub>a</sub> + β<sub>int</sub>·(b − b̄). Its '
                'uncertainty uses the full coefficient covariance, including Cov(β<sub>a</sub>, β<sub>int</sub>).</p>',
                unsafe_allow_html=True)
    a_opts = [f for f in factor_opts if f in panel]
    b_opts = [f for f in firm_opts if f in panel] + [f for f in a_opts]
    if not a_opts or not b_opts:
        st.info("Interactions need at least one factor and one moderator (e.g. debt/assets from a fundamentals CSV).")
    else:
        c1, c2, c3 = st.columns(3)
        a = sel(c1, "Factor (a)", a_opts, PF.interaction[0] if PF.interaction[0] in a_opts else a_opts[0], K("w_ia"),
                format_func=lambda k: FACTOR_LABELS.get(k, k))
        bo = [x for x in b_opts if x != a]
        b = sel(c2, "Moderator (b)", bo, PF.interaction[1] if PF.interaction[1] in bo else bo[0], K("w_ib"),
                format_func=lambda k: FACTOR_LABELS.get(k, k))
        isym = sel(c3, "Stock", tickers, tickers[0], K("w_isym"))
        ctrl_opts = [x for x in X if x not in (a, b)]
        controls = msel(st, "Controls", ctrl_opts, ctrl_opts, K("w_ictrl"), format_func=lambda k: FACTOR_LABELS.get(k, k))
        rows, ires = [], {}
        for s in tickers:
            d = panel[panel["symbol"] == s].set_index("month")
            d = d[(d.index >= start_m) & (d.index <= eff_end)]
            try:
                ires[s] = R.fit_interaction(d, "ret", a, b, controls, s, hac_rule, int(hac_lags))
            except R.ModelSpecError as e:
                ires[s] = str(e)
        ok = {s: r for s, r in ires.items() if not isinstance(r, str)}
        inter = f"{a}_x_{b}"
        if ok:
            it = R.effects_table(ok, f"all coefficients in {len(ok)} interaction models")
            RES["interaction_coefficients"] = it
            summ = it[it["term"].isin([a, b + "_c", inter])][["symbol", "term", "estimate", "ci_low", "ci_high",
                                                                 "p_value", "q_value", "n", "window"]]
            st.markdown("**Main effects and interaction across stocks**")
            st.dataframe(summ, hide_index=True, width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="%.4f") for c in
                                        ["estimate", "ci_low", "ci_high", "p_value", "q_value"]})
        if isinstance(ires.get(isym), str):
            st.error(ires[isym])
        elif isym in ok:
            r = ok[isym]
            for w in r.warnings:
                st.warning(w)
            obs_b = r.data[b + "_c"] + r.centers[b]
            grid = np.linspace(obs_b.min() - 0.1 * (obs_b.max() - obs_b.min()),
                               obs_b.max() + 0.1 * (obs_b.max() - obs_b.min()), 60)
            cs = R.conditional_slope(r, a, b, grid)
            scale, slab = R.EFFECT_SCALES.get(a, (1.0, "+1 unit"))
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f"β {a} at mean {b}", f"{r.params[a]:.3f}", help=f"mean {b} = {r.centers[b]:.4f} (this window)")
            m2.metric(f"β interaction", f"{r.params[inter]:.3f}",
                      help=f"95% CI [{r.conf_int.loc[inter, 'lo']:.3f}, {r.conf_int.loc[inter, 'hi']:.3f}]")
            m3.metric("p (interaction)", f"{r.pvalues[inter]:.4f}")
            m4.metric("Months", r.nobs, help=f"{r.start} → {r.end}")
            truth_line = None
            tr_ = data.get("truth") if SYN else None
            if tr_ is not None and f"{a}_x_{b}" in tr_.columns and isym in set(tr_["symbol"]):
                t = tr_.set_index("symbol").loc[isym]
                truth_line = (t[a], t[f"{a}_x_{b}"], t["center"])
            st.plotly_chart(C.interaction_curve(cs, b, FACTOR_LABELS.get(a, a), scale, slab, SYN, truth_line), width="stretch")
            st.caption(f"Hollow markers: grid points with fewer than max(5, 5% of months) observations within ±0.5 SD of "
                       f"{b}, or outside the observed range ({obs_b.min():.3f}–{obs_b.max():.3f}). Treat those parts "
                       "of the curve as extrapolation.")
            jd = r.data.assign(**{b: obs_b})
            st.plotly_chart(C.joint_distribution(jd, a, b, SYN), width="stretch")
            st.markdown("**Scenario (fitted conditional mean within the observed range)**")
            sc1, sc2 = st.columns(2)
            if FACTOR_SPECS.get(a) and FACTOR_SPECS[a].transformation == "log_change":
                pct_move = sc1.number_input(f"Assumed {a} move this month (%)", -50.0, 100.0, 10.0, step=1.0, key=K("w_scen_a"))
                a_val = math.log(1 + pct_move / 100)
                sc1.caption(f"Log-change model: a {pct_move:+.1f}% move enters as log(1{pct_move / 100:+.3f}) = {a_val:.4f}.")
            else:
                a_val = sc1.number_input(f"Assumed {a} value", value=float(scale or 0.01), format="%.4f", key=K("w_scen_a2"))
            b_val = sc2.number_input(f"Assumed {b}", value=float(round(r.centers[b], 3)), format="%.3f", key=K("w_scen_b"))
            vals = {c: float(r.data[c].mean()) for c in controls}
            vals.update({a: a_val, b + "_c": b_val - r.centers[b], inter: a_val * (b_val - r.centers[b])})
            sm = R.scenario_mean(r, vals)
            raw_ext = []
            if not (obs_b.min() <= b_val <= obs_b.max()):
                raw_ext.append(f"{b}={b_val:.3f} outside observed [{obs_b.min():.3f}, {obs_b.max():.3f}]")
            if not (r.data[a].min() <= a_val <= r.data[a].max()):
                raw_ext.append(f"{a}={a_val:.4f} outside observed [{r.data[a].min():.4f}, {r.data[a].max():.4f}]")
            st.markdown(f"Fitted mean monthly return for **{isym}**: **{sm['estimate'] * 100:+.2f}%** "
                        f"(95% CI for the conditional mean {sm['lo'] * 100:+.2f}% to {sm['hi'] * 100:+.2f}%), with "
                        f"controls held at their window means. Estimation period {r.start}–{r.end}. "
                        "This is a conditional mean under assumed factor values, not a forecast or prediction interval.")
            for e in raw_ext:
                st.warning("Extrapolation: " + e)

# =========================================================================== Rolling
with t_roll:
    st.subheader("Rolling relationships — coefficient stability analysis")
    st.markdown(f'<p class="small-note">At each window end the model is refit using only the trailing {roll_window} '
                f'months. Consecutive windows share {roll_window - 1} of {roll_window} months '
                f'({(roll_window - 1) / roll_window:.0%} overlap), so neighbouring estimates are strongly dependent; '
                'look for persistent shifts, not month-to-month wiggles.</p>', unsafe_allow_html=True)
    if not X:
        st.warning("Select factors in the sidebar.")
    else:
        rc1, rc2, rc3 = st.columns(3)
        rsyms = msel(rc1, "Stocks", tickers, tickers[: min(4, len(tickers))], K("w_rsyms"))
        use_int = rc2.checkbox("Add interaction (factor × moderator)", key="w_rint",
                               disabled=not any(f in X for f in firm_opts))
        standardized = rc3.radio("Scale", ["Raw units", "Standardized (shared full-period SDs)"], key="w_rscale") != "Raw units"
        interaction, slope_at = None, None
        if use_int:
            ic1, ic2, ic3 = st.columns(3)
            ra_opts, rb_opts = [f for f in X if f in factor_opts], [f for f in X if f in firm_opts]
            ra = sel(ic1, "Factor", ra_opts, PF.interaction[0] if PF.interaction[0] in ra_opts else ra_opts[0], K("w_ra"))
            rb = sel(ic2, "Moderator", rb_opts, PF.interaction[1] if PF.interaction[1] in rb_opts else rb_opts[0], K("w_rb"))
            slope_at = ic3.number_input(f"Report {ra} sensitivity at fixed {rb} =", format="%.3f", key=K("w_rslope"),
                                        **wkw(K("w_rslope"), "value", 0.30 if rb == "debt_to_assets" else 0.0))
            interaction = (ra, rb)
        roll, rsk = _rolling(panel, pkey, tuple(rsyms), tuple(X), int(roll_window), int(min_obs), hac_rule,
                             int(hac_lags), interaction, slope_at, start_m, eff_end)
        if roll.empty:
            st.warning(f"No window has ≥ {min_obs} complete months. Shorten the window or the minimum, or widen dates.")
        else:
            RES["rolling"] = roll
            terms_all = list(dict.fromkeys(roll["term"]))
            terms = msel(st, "Terms", terms_all, [t for t in terms_all if t.startswith("slope_")] or terms_all[:3], K("w_rterms"))
            if standardized:
                full = panel[(panel["month"] >= start_m) & (panel["month"] <= eff_end)]
                sds = full.groupby("symbol")[[c for c in X if c in full] + ["ret"]].std()
                roll = roll.copy()
                roll["std_scale"] = [
                    (sds.loc[s, t] / sds.loc[s, "ret"]) if t in sds.columns else np.nan
                    for s, t in zip(roll["symbol"], roll["term"])]
                st.caption("Standardized with each stock's SD of x and y over the whole selected period, so a change "
                           "in a window's own volatility does not silently change the plotted quantity.")
                terms = [t for t in terms if roll.loc[roll["term"] == t, "std_scale"].notna().any()]
            if terms:
                st.plotly_chart(C.rolling_plot(roll, terms, colors, standardized, SYN), width="stretch")
            st.plotly_chart(C.rolling_n_plot(roll, colors), width="stretch")
            with st.expander("Rolling estimates table"):
                st.dataframe(roll, hide_index=True, width="stretch")
        if rsk:
            with st.expander(f"Windows skipped ({len(rsk)})"):
                st.write("\n".join(rsk[:200]))

# =========================================================================== Forecast validation
with t_fc:
    st.subheader("Forecast validation — out-of-sample, chronological")
    st.markdown('<p class="small-note">Separate from the descriptive views. Each forecast of month t+1 uses only '
                'information available at month t’s decision cutoff (lagged returns, macro values released by then, '
                'fundamentals released by then). Models are refit at every month on an expanding (or trailing) '
                'training history; scaling, imputation, interaction centering and ridge tuning are fitted on that '
                'history only. Baselines: the training-history mean return and a zero-return forecast.</p>',
                unsafe_allow_html=True)
    lag_opts = [c for c in panel.columns if c.startswith("lag_") and panel[c].notna().any()]
    lag_labels = {"lag_own": "own return, month t", "lag_market": "benchmark return, month t",
                  **{f"lag_{k}": f"{v.label} — value known at cutoff t" for k, v in FACTOR_SPECS.items()},
                  **{f"lag_{k}": f"{v['label']} — latest report released by cutoff t" for k, v in FIRM_FEATURES.items()}}
    fdef = [c for c in ["lag_own", "lag_market", "lag_control"] + [f"lag_{f}" for f in PF.macro_factors]
            + [f"lag_{f}" for f in PF.default_firm] if c in lag_opts]
    lag_labels["lag_control"] = f"{CONTROL} return, month t" if CONTROL else "control"
    fc1, fc2 = st.columns([3, 2])
    feats = msel(fc1, "Predictors (all known at the decision cutoff)", lag_opts, fdef, K("w_ffeats"),
                 format_func=lambda c: lag_labels.get(c, c))
    la, lb = f"lag_{PF.interaction[0]}", f"lag_{PF.interaction[1]}"
    fint = fc2.checkbox(f"Add {la} × {lb} (centered on training mean)", key=K("w_fint"),
                        disabled=not ({la, lb} <= set(feats)))
    with st.expander("Training and evaluation settings", expanded=False):
        e1, e2, e3, e4 = st.columns(4)
        fwin = e1.radio("Training window", ["expanding", "trailing"], key="w_fwin")
        ftrail = e1.number_input("Trailing length (months)", 24, 240, DEFAULTS.forecast_trailing, key=K("w_ftrail"))
        fmin = e2.number_input("Initial training months", 12, 180, key=K("w_fmin"), **wkw(K("w_fmin"), "value", PF.forecast_min_train))
        fimp = e2.selectbox("Missing predictors", ["drop", "train_median"], key="w_fimp",
                            format_func=lambda x: "drop row (default)" if x == "drop" else "impute training-window median")
        alph = e3.text_input("Ridge alphas", ", ".join(str(a) for a in DEFAULTS.ridge_alphas), key="w_falpha")
        fcv = e3.number_input("Time-ordered CV folds", 2, 10, DEFAULTS.ridge_cv_splits, key="w_fcv")
        fret = e4.number_input("Retune ridge every N months", 1, 24, 1, key="w_fret")
        fhold = e4.number_input("Final holdout (months)", 0, 60, DEFAULTS.holdout_months, key="w_fhold")
        fpi = e1.slider("Empirical prediction-interval level", 0.5, 0.95, 0.8, 0.05, key="w_fpi")
    try:
        alphas = tuple(float(x) for x in alph.split(",") if x.strip())
    except ValueError:
        alphas = tuple(DEFAULTS.ridge_alphas)
        st.warning("Could not parse ridge alphas; using defaults.")
    if not feats:
        st.warning("Select at least one predictor.")
    else:
        fpanel = panel[(panel["month"] >= start_m - 1) & (panel["month"] <= eff_end)]
        run_id = uuid.uuid5(uuid.NAMESPACE_URL, pkey + str((feats, fint, fwin, ftrail, fmin, fimp, alphas, fcv, fret))).hex[:12]
        fc, fsk = _forecasts(fpanel, pkey, tuple(tickers), tuple(feats), int(fmin), fwin, int(ftrail), alphas, int(fcv),
                             fimp, (la, lb) if fint else None, int(fret), float(fpi), run_id)
        if fc.empty:
            st.error(f"No forecasts could be made: need ≥ {fmin} complete training months before the first decision. "
                     "Reduce the initial training window, drop sparse predictors, or extend the date range.")
        else:
            fc = F.split_holdout(fc, int(fhold))
            RES["forecasts"] = fc
            reveal = st.checkbox(f"Reveal final holdout ({fhold} months: {fc.loc[fc.segment == 'holdout', 'target_month'].min()}"
                                 f"–{fc['target_month'].max()})", key="w_reveal",
                                 help="Choose features and settings on the validation period; open the holdout once.")
            seg = fc if reveal else fc[fc["segment"] == "validation"]
            st.caption(f"Evaluation targets: {seg['target_month'].min()} → {seg['target_month'].max()} "
                       f"({'validation + holdout' if reveal else 'validation only'}). Metrics use exactly matching "
                       "(stock, month) pairs across all four models.")
            met = F.metrics(seg)
            if len(met):
                mm = met.set_index("model")["oos_r2_vs_hist_mean"]
                best = mm.drop(["hist_mean", "zero"], errors="ignore")
                if len(best):
                    bname = best.idxmax()
                    verdict = ("did slightly better than" if best.max() > 0.01 else
                               "did about the same as" if best.max() > -0.01 else "did worse than")
                    home_forecast.markdown(
                        f"- **Prediction test:** the best model ({F.MODEL_LABELS[bname]}) {verdict} simply predicting the "
                        f"historical average return (out-of-sample R² {best.max():+.3f} over {int(met['n_forecasts'].iloc[0])} "
                        "stock-months). Monthly stock returns are mostly unpredictable, so values near zero are normal.")
                met["model"] = met["model"].map(F.MODEL_LABELS)
            st.markdown("**Pooled across stocks**")
            st.dataframe(met, hide_index=True, width="stretch",
                         column_config={"MAE": st.column_config.NumberColumn(format="%.4f"),
                                        "RMSE": st.column_config.NumberColumn(format="%.4f"),
                                        "directional_accuracy": st.column_config.NumberColumn(format="%.3f"),
                                        "bias": st.column_config.NumberColumn(format="%.4f"),
                                        "oos_r2_vs_hist_mean": st.column_config.NumberColumn("OOS R² vs hist. mean", format="%.4f")})
            st.caption("OOS R² = 1 − Σ(model error²) / Σ(historical-mean error²) on the same dates. Negative values "
                       "mean the model forecast worse than simply using the training-history mean; they are kept. "
                       "Directional accuracy excludes zero forecasts, so it is blank for the zero model.")
            with st.expander("Per-stock metrics"):
                ms = F.metrics(seg, by_symbol=True)
                st.dataframe(ms, hide_index=True, width="stretch")
            models = msel(st, "Models to plot", F.MODELS, ["ols", "ridge", "hist_mean"], "w_fmodels",
                          format_func=lambda m: F.MODEL_LABELS[m])
            st.plotly_chart(C.cumulative_sse_gap(seg, models, synthetic=SYN), width="stretch")
            g1, g2 = st.columns(2)
            fsym = sel(g1, "Stock", tickers, tickers[0], K("w_fsym"))
            pim = sel(g2, "Prediction interval for", ["none"] + models, "none", "w_fpim")
            st.plotly_chart(C.forecast_history(seg, fsym, models, SYN, None if pim == "none" else pim), width="stretch")
            h1, h2 = st.columns(2)
            with h1:
                st.plotly_chart(C.actual_vs_pred(seg, models[0] if models else "ols", colors, SYN), width="stretch")
            with h2:
                st.plotly_chart(C.error_distribution(seg, models, SYN), width="stretch")
            cov = seg.dropna(subset=["pi_covered"]).groupby("model")["pi_covered"].agg(["mean", "count"]).reset_index()
            cov.columns = ["model", "empirical_coverage", "n_intervals"]
            cov["nominal"] = fpi
            st.markdown("**Prediction-interval calibration**")
            st.dataframe(cov, hide_index=True, width="stretch")
            st.caption("Method: for a forecast made at month t, the interval is the prediction plus the "
                       f"{(1 - fpi) / 2:.0%} and {(1 + fpi) / 2:.0%} quantiles of that model's own earlier out-of-sample "
                       "errors already realized by t (≥ 24 required). Coverage above is measured chronologically on "
                       "later forecasts. These are *prediction* intervals for a new return; the coefficient intervals "
                       "elsewhere are for conditional means and are much narrower.")
            if "ridge" in fc["model"].unique():
                al = fc[fc.model == "ridge"]["settings"].apply(lambda s: s.get("alpha"))
                st.caption(f"Ridge α chosen by time-ordered CV inside each training window: "
                           f"median {al.median():g}, range {al.min():g}–{al.max():g}.")
            with st.expander("Forecast records (every forecast with its training range and settings)"):
                st.dataframe(seg.drop(columns=["settings"]).head(2000), hide_index=True, width="stretch")
            if fsk:
                with st.expander(f"Skipped forecasts ({len(fsk)})"):
                    st.write("\n".join(fsk[:300]))

# =========================================================================== Portfolio
with t_pf:
    st.subheader("Portfolio exploration (PyPortfolioOpt)")
    st.markdown('<p class="small-note">Long-only minimum-volatility weights from a Ledoit–Wolf shrunk covariance of '
                'monthly total returns (annualized with frequency = 12), compared with equal weights in the same '
                'eligible universe. Risk numbers are estimates from these fitted inputs.</p>', unsafe_allow_html=True)
    pc1, pc2, pc3 = st.columns(3)
    cap = pc1.slider("Max weight per stock", 0.05, 1.0, DEFAULTS.max_weight, 0.05, key="w_cap")
    look = pc2.number_input("Covariance lookback (months)", 12, 180, key=K("w_look"), **wkw(K("w_look"), "value", PF.cov_lookback))
    mw = pan["monthly_returns"].pivot(index="month", columns="symbol", values="ret")
    win = mw.loc[(mw.index > eff_end - int(look)) & (mw.index <= eff_end), [s for s in tickers if s in mw]]
    dropped = [s for s in win.columns if win[s].isna().any()]
    if dropped:
        pc3.warning("Excluded (incomplete lookback): " + ", ".join(dropped))
    try:
        w_mv, S, info = P.min_vol_weights(win, cap)
        w_eq = P.equal_weights(w_mv.index)
        W = pd.DataFrame({"min-volatility": w_mv, "equal weight": w_eq})
        RES["weights"] = W.reset_index(names="symbol")
        st.plotly_chart(C.weights_bar(W, SYN), width="stretch")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Estimated annual volatility (min-vol)", pct(info["est_annual_vol"], 1))
        k2.metric("Estimated annual volatility (equal)", pct(float(np.sqrt(w_eq @ S.to_numpy() @ w_eq)), 1))
        k3.metric("Ledoit–Wolf shrinkage intensity", f"{info['shrinkage']:.2f}")
        conc = P.concentration(w_mv)
        k4.metric("Effective number of holdings", f"{conc['effective_n']:.1f}", help=f"1/HHI; HHI={conc['hhi']:.3f}")
        st.caption(f"Window {win.index.min()} → {win.index.max()} ({info['n_months']} months). A single weight vector is "
                   "an allocation calculation from historical inputs, not a recommendation.")
        if X:
            pf, _ = _fits(panel, pkey + "pf" + str(look), tuple(w_mv.index), tuple(f for f in X if f in factor_opts),
                          win.index.min(), win.index.max(), hac_rule, int(hac_lags))
            if pf:
                betas = pd.DataFrame({s: r.params.drop("const") for s, r in pf.items()}).T
                ex = pd.DataFrame({"min-volatility": P.holdings_exposure(w_mv, betas),
                                   "equal weight": P.holdings_exposure(w_eq, betas)})
                st.markdown("**Factor exposure implied by holdings** (weighted per-stock betas over the same window)")
                st.dataframe(ex.round(3), width="stretch")
    except P.PortfolioInfeasible as e:
        st.error(f"Optimizer: {e}")
    st.markdown("---")
    st.markdown("**Historical evaluation with rebalancing**")
    b1, b2, b3, b4, b5 = st.columns(5)
    every = b1.selectbox("Rebalance", [1, 3, 12], format_func=lambda k: {1: "monthly", 3: "quarterly", 12: "annually"}[k],
                         key="w_every")
    fee = b2.number_input("Fees (bps per traded $)", 0.0, 100.0, DEFAULTS.fee_bps, key="w_fee")
    slip = b3.number_input("Slippage (bps per traded $)", 0.0, 200.0, DEFAULTS.slippage_bps, key="w_slip")
    ci = b4.checkbox("Charge cost on initial purchase", True, key="w_ci")
    rf_mode = b5.selectbox("Risk-free (for Sharpe)", ["none", "constant", f"{PF.rf_label} (proxy)"], key=K("w_rf"))
    rf_const = (st.number_input(f"Constant {PF.currency} risk-free rate (annual %)", 0.0, 10.0, 2.0, key="w_rfc")
                if rf_mode == "constant" else None)
    try:
        res, wh, bmsg = _backtest(pan, pkey + str(start_m), tuple(tickers), benchmark, int(look), float(cap), float(fee),
                                  float(slip), int(every), bool(ci))
        res = res[(res["decision_month"] >= start_m) & (res["decision_month"] <= eff_end)]
        rf, rf_label = None, "none"
        if rf_mode == "constant":
            rf = pd.Series((1 + rf_const / 100) ** (1 / 12) - 1, index=pd.period_range(start_m, eff_end, freq="M"))
            rf_label = f"constant {rf_const:.2f}% p.a. ({PF.currency}), compounded monthly"
        elif rf_mode.endswith("(proxy)") and len(pan["macro_pit"]) and PF.rf_factor in pan["macro_pit"]["factor"].unique():
            lv = pan["macro_pit"][pan["macro_pit"]["factor"] == PF.rf_factor].set_index("month")["level"]
            rf = lv / 100 / 12
            rf_label = (f"{PF.rf_label} level at the decision cutoff ÷ 12 (approximate {PF.currency} cash return; "
                        "not a T-bill series)")
        summ = P.performance_summary(res, rf, rf_label)
        _s = summ.set_index("strategy")
        if {"min_vol", "equal_weight"} <= set(_s.index):
            home_portfolio.markdown(
                f"- **Portfolio:** the minimum-volatility mix had annual volatility of {pct(_s.loc['min_vol', 'net_ann_vol'], 1)} "
                f"vs {pct(_s.loc['equal_weight', 'net_ann_vol'], 1)} for equal weights, with net annual returns of "
                f"{pct(_s.loc['min_vol', 'net_ann_return'], 1)} vs {pct(_s.loc['equal_weight', 'net_ann_return'], 1)} "
                f"({_s.loc['min_vol', 'first_execution']} to {_s.loc['min_vol', 'last_period_end']}, after costs).")
        RES["backtest"] = res
        RES["weights_history"] = wh
        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(C.growth_plot(res, "net", SYN), width="stretch")
        with g2:
            st.plotly_chart(C.drawdown_plot(res, SYN), width="stretch")
        st.dataframe(summ, hide_index=True, width="stretch")
        st.caption(f"Execution: close of the first {PF.exchange} session after each month-end cutoff; returns run execution-to-"
                   "execution from the total-return index; weights drift between rebalances. Traded notional = "
                   "Σ|target − drifted weight|; each traded dollar pays fees + slippage once. Half-turnover = traded "
                   "notional ÷ 2 (equal to one-sided turnover when the portfolio stays fully invested). Gross results "
                   "are the zero-cost sensitivity. Sharpe is shown only with an explicit CAD risk-free assumption.")
        with st.expander("Rebalance log and weights"):
            st.dataframe(res, hide_index=True, width="stretch")
            st.dataframe(wh, hide_index=True, width="stretch")
        if bmsg:
            st.warning(f"{len(bmsg)} scheduled rebalance(s) were skipped, mostly early on when too few stocks had enough "
                       f"history for the {look}-month lookback (first: {bmsg[0].split(':')[0]}, last: "
                       f"{bmsg[-1].split(':')[0]}). The evaluation starts at the first feasible rebalance.")
            with st.expander("Skipped rebalances"):
                st.write("\n".join(bmsg))
    except P.PortfolioInfeasible as e:
        st.error(f"Backtest: {e}")

# =========================================================================== Data & exports
with t_exp:
    st.subheader("Exports and reproducibility")
    cfg = run_config(settings, data, data.get("provider_meta", {}),
                     extra={"analysis_window": {"start": str(start_m), "end": str(eff_end)},
                            "synthetic_label": "SYNTHETIC DATA" if SYN else None,
                            "provider_assumptions": {
                                "yfinance": "personal research use; adjustment set explicitly (auto_adjust=False, actions=True)",
                                "FRED": "ALFRED vintages if FRED_API_KEY set, else latest values + 7-day lag rule",
                                "BoC Valet": "latest values + 1-day lag rule",
                                "fundamentals": "CSV-supplied available_at; FMP filing dates unverified"},
                            "excluded_rows": int(len(pan["excluded"]))})
    st.download_button("Run configuration (JSON)", json.dumps(cfg, indent=2), "run_config.json", "application/json")
    tables = {"input_prices": prices, "input_macro": data.get("macro"), "input_fundamentals": data.get("fundamentals"),
              "input_metadata": meta, "monthly_panel": panel.astype({"month": str, "target_month": str}),
              "derived_features": pan["features_long"], **{k: v for k, v in RES.items() if isinstance(v, pd.DataFrame)}}
    cols = st.columns(4)
    for i, (name, df) in enumerate(tables.items()):
        if df is not None and len(df):
            out = df.copy()
            if "settings" in out:
                out["settings"] = out["settings"].apply(lambda s: json.dumps(s, default=str))
            cols[i % 4].download_button(f"{name}.csv", out.to_csv(index=False), f"{name}.csv", "text/csv", key=f"dl_{name}")
    st.download_button("Reproducible bundle (.zip: inputs + outputs + config)",
                       bundle(cfg, {k: (v.assign(settings=v["settings"].apply(lambda s: json.dumps(s, default=str)))
                                        if v is not None and "settings" in v else v) for k, v in tables.items()}),
                       "market_factor_run.zip", "application/zip")
    st.caption("To reproduce: choose *CSV import*, upload the bundle. Inputs and settings are restored from it; outputs "
               "computed from them should match the exported CSVs (same package versions recorded in the JSON).")
    st.markdown(f"**Config fingerprint:** `{cfg['config_fingerprint'][:16]}` · "
                f"**Prices fingerprint:** `{(cfg['dataset_fingerprints']['prices'] or '')[:16]}`")
    with st.expander("Run configuration preview"):
        st.json(cfg, expanded=False)
    with st.expander("Optional integrations"):
        st.markdown("- **QuantConnect / LEAN** — `integrations/lean/` holds an example algorithm that reads exported "
                    "weights. Requires the LEAN CLI and Docker, and a QuantConnect account for cloud data. Canadian "
                    "(TSX) equity and corporate-action coverage in LEAN has not been verified for this build.\n"
                    "- **MSCI Barra** — not implemented. It needs a licensed export or API; add an adapter only "
                    "against documented, licensed files.\n"
                    "- **Financial Modeling Prep** — optional fundamentals adapter (needs `FMP_API_KEY`). Canadian "
                    "coverage and release timestamps are unverified, so its rows are labelled accordingly.")
