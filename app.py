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

from mfd import charts as C
from mfd import forecast as F
from mfd import portfolio as P
from mfd import regression as R
from mfd.config import (APP_TITLE, AnalysisSettings, BENCHMARKS, DEFAULT_TICKERS, FACTOR_SPECS, FIRM_FEATURES,
                        DEFAULT_BENCHMARK)
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
def _synthetic():
    return load_synthetic()


@st.cache_data(show_spinner="Downloading / reading cached provider data…")
def _live(symbols, benchmark, start, end, factors, refresh_token, fund_bytes, use_fmp):
    return load_live(list(symbols), benchmark, start, end, list(factors), refresh=refresh_token > 0,
                     fundamentals_csv=fund_bytes, use_fmp=use_fmp)


@st.cache_data(show_spinner="Reading CSV files…")
def _csv(prices_b, macro_b, fund_b, meta_b):
    return load_csv(prices_b, macro_b, fund_b, meta_b)


@st.cache_data(show_spinner="Aligning returns, macro and fundamentals on the decision calendar…")
def _panel(_data, data_key, symbols, benchmark, start_month, end_month, convention, include_leases, small_frac):
    return build_panel(_data["prices"], _data["macro"], _data["fundamentals"], list(symbols), benchmark,
                       end_month=end_month, start_month=start_month, convention=convention,
                       include_leases=include_leases, small_denominator_frac=small_frac)


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
    mapping = {"tickers": "w_tickers", "benchmark": "w_benchmark", "factors": "w_factors",
               "firm_features": "w_firm", "hac_lag_rule": "w_hac_rule", "hac_lags": "w_hac_lags",
               "rolling_window": "w_roll", "return_convention": "w_conv", "include_leases_in_debt": "w_leases",
               "small_denominator_frac": "w_smallden"}
    for k, w in mapping.items():
        if k in s:
            st.session_state[w] = s[k]
    for k in ("start", "end"):
        if s.get(k):
            st.session_state["w_" + k] = s[k]
    st.session_state["restored_config"] = cfg


# =========================================================================== sidebar
sb = st.sidebar
sb.title(APP_TITLE)
mode = sb.radio("Data mode", ["Synthetic (labelled)", "Live providers", "CSV import"], key="w_mode",
                help="Synthetic exercises every feature with a known data-generating process. Live uses yfinance, "
                     "FRED and Bank of Canada Valet (personal research use). CSV import works offline.")
data, load_error = None, None
if "refresh_token" not in st.session_state:
    st.session_state.refresh_token = 0

if mode.startswith("Synthetic"):
    data = _synthetic()
elif mode == "Live providers":
    tick_text = sb.text_input("Tickers (comma-separated)", ", ".join(DEFAULT_TICKERS), key="w_live_tickers")
    live_syms = tuple(t.strip().upper() for t in tick_text.split(",") if t.strip())
    bench_live = sb.text_input("Benchmark ETF", DEFAULT_BENCHMARK, key="w_live_bench",
                               help="; ".join(f"{k}: {v}" for k, v in BENCHMARKS.items())).strip().upper()
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
                     ("oil", "rate", "fx"), st.session_state.refresh_token,
                     fund_up.getvalue() if fund_up else None, use_fmp)
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
            data = _csv(pb, mb, fb, db)
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
bench_candidates = [s for s in all_syms if s in BENCHMARKS or s.startswith("SYN-MKT") or s.startswith("SYN-ENERGY")
                    or s.startswith("X")] or all_syms
default_bench = ("SYN-MKT" if SYN else (DEFAULT_BENCHMARK if DEFAULT_BENCHMARK in all_syms else bench_candidates[0]))
benchmark = sel(sb, "Benchmark (market factor)", all_syms, default_bench, "w_benchmark",
                         help="Its monthly total return is the market control. XIC.TO = broad Canadian market; "
                              "XEG.TO = Canadian energy.")
stock_opts = [s for s in all_syms if s not in BENCHMARKS and not s.startswith(("SYN-MKT", "SYN-ENERGY")) and s != benchmark]
tickers = msel(sb, "Watchlist", stock_opts, stock_opts, "w_tickers")
if not tickers:
    st.warning("Select at least one stock in the sidebar watchlist.")
    st.stop()

pmin = pd.Period(prices["session_date"].min(), "M")
pmax = pd.Period(prices["session_date"].max(), "M")
month_opts = [str(m) for m in pd.period_range(pmin, pmax, freq="M")]
start_s = sel(sb, "Start month", month_opts, str(max(pmin, pmax - 12 * DEFAULTS.history_years)), "w_start")
end_s = sel(sb, "End month (latest completed month by default)", month_opts, month_opts[-1], "w_end")
start_m, end_m = pd.Period(start_s, "M"), pd.Period(end_s, "M")

macro_factors = sorted(data["macro"]["factor"].unique()) if data.get("macro") is not None and len(data["macro"]) else []
has_fund = data.get("fundamentals") is not None and len(data["fundamentals"]) > 0
factor_opts = ["market"] + macro_factors
factors = msel(sb, "Market & macro factors", factor_opts, DEFAULTS.factors, "w_factors",
               format_func=lambda k: FACTOR_LABELS.get(k, k))
firm_opts = list(FIRM_FEATURES) if has_fund else []
firm = msel(sb, "Company fundamentals", firm_opts, DEFAULTS.firm_features, "w_firm",
                      format_func=lambda k: FIRM_FEATURES[k]["label"],
                      disabled=not has_fund,
                      help=None if has_fund else "Needs dated fundamentals: upload a fundamentals CSV.")
with sb.expander("Estimation settings", expanded=False):
    hac_rule = st.radio("HAC lag rule", ["fixed", "newey_west"], key="w_hac_rule",
                        format_func=lambda x: "Fixed lags" if x == "fixed" else "Newey–West plug-in",
                        help="Heteroscedasticity- and autocorrelation-consistent (Newey–West) standard errors.")
    hac_lags = st.number_input("Fixed HAC lags (months)", 0, 24, key="w_hac_lags", **wkw("w_hac_lags", "value", DEFAULTS.hac_lags))
    roll_window = st.number_input("Rolling window (months)", 24, 180, step=6, key="w_roll", **wkw("w_roll", "value", DEFAULTS.rolling_window))
    min_obs = st.number_input("Minimum observations per window", 18, 180, DEFAULTS.min_obs_rolling, key="w_minobs")
    convention = st.selectbox("Total-return convention", ["adjusted_close", "close_plus_dividends"], key="w_conv",
                              help="adjusted_close: provider split+dividend-adjusted closes (dividends counted once, "
                                   "inside the adjustment). close_plus_dividends: (close + dividend)/previous close "
                                   "with split-adjusted closes.")
    include_leases = st.checkbox("Include lease liabilities in debt", key="w_leases", **wkw("w_leases", "value", DEFAULTS.include_leases_in_debt))
    small_frac = st.number_input("Small-denominator threshold (share of assets)", 0.0, 0.05,
                                 step=0.001, **wkw("w_smallden", "value", DEFAULTS.small_denominator_frac), format="%.3f", key="w_smallden",
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
                small_denominator_frac=float(small_frac), data_mode="synthetic" if SYN else mode)

# =========================================================================== panel
data_key = json.dumps(data.get("provider_meta", {}), default=str) + str(len(prices)) + str(prices["session_date"].max())
try:
    pan = _panel(data, data_key, tuple(tickers), benchmark, start_m - 1, end_m, convention, include_leases, small_frac)
except Exception as e:
    st.error(f"Could not align the data: {e}")
    st.stop()
panel = pan["panel"]
pkey = data_key + str((tuple(tickers), benchmark, str(start_m), str(end_m), convention, include_leases, small_frac))
eff_end = min(end_m, pan["end_month"])
colors = C.color_map(tickers)
X = list(factors) + list(firm)
if "results" not in st.session_state:
    st.session_state.results = {}
RES = st.session_state.results

tabs = st.tabs(["Overview", "Stock drivers", "Interactions", "Rolling relationships", "Forecast validation",
                "Portfolio", "Data & exports"])

# =========================================================================== Overview
with tabs[0]:
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
                "Decision cutoff each month: 17:00 Toronto on its last TSX session.")
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
            elif k == "market":
                st.markdown(f"**Benchmark total return** — monthly simple total return of {benchmark}, used as a "
                            "contemporaneous control in descriptive models and lagged in forecasts.")

# =========================================================================== Stock drivers
with tabs[1]:
    st.subheader("Historical sensitivity (descriptive, full selected window)")
    st.markdown('<p class="small-note">One OLS regression per stock: monthly total return on the selected factors. '
                'Macro factors are contemporaneous (same month); fundamentals are the latest values known before the '
                'month began. Estimates describe historical co-movement — “associated with”, not “caused by”.</p>',
                unsafe_allow_html=True)
    if not X:
        st.warning("Select at least one factor in the sidebar.")
    else:
        fits, errs = _fits(panel, pkey, tuple(tickers), tuple(X), start_m, eff_end, hac_rule, int(hac_lags))
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
            sym = sel(st, "Stock detail", list(fits), list(fits)[0], "w_detail")
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
with tabs[2]:
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
        a = sel(c1, "Factor (a)", a_opts, "oil" if "oil" in a_opts else a_opts[0], "w_ia",
                format_func=lambda k: FACTOR_LABELS.get(k, k))
        bo = [x for x in b_opts if x != a]
        b = sel(c2, "Moderator (b)", bo, "debt_to_assets" if "debt_to_assets" in bo else bo[0], "w_ib",
                format_func=lambda k: FACTOR_LABELS.get(k, k))
        isym = sel(c3, "Stock", tickers, tickers[0], "w_isym")
        ctrl_opts = [x for x in X if x not in (a, b)]
        controls = msel(st, "Controls", ctrl_opts, ctrl_opts, "w_ictrl", format_func=lambda k: FACTOR_LABELS.get(k, k))
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
            if SYN and data.get("truth") is not None and a == "oil" and b == "debt_to_assets":
                t = data["truth"].set_index("symbol").loc[isym]
                truth_line = (t["oil"], t["oil_x_debt"], t["debt_center"])
            st.plotly_chart(C.interaction_curve(cs, b, FACTOR_LABELS.get(a, a), scale, slab, SYN, truth_line), width="stretch")
            st.caption(f"Hollow markers: grid points with fewer than max(5, 5% of months) observations within ±0.5 SD of "
                       f"{b}, or outside the observed range ({obs_b.min():.3f}–{obs_b.max():.3f}). Treat those parts "
                       "of the curve as extrapolation.")
            jd = r.data.assign(**{b: obs_b})
            st.plotly_chart(C.joint_distribution(jd, a, b, SYN), width="stretch")
            st.markdown("**Scenario (fitted conditional mean within the observed range)**")
            sc1, sc2 = st.columns(2)
            if FACTOR_SPECS.get(a) and FACTOR_SPECS[a].transformation == "log_change":
                pct_move = sc1.number_input(f"Assumed {a} move this month (%)", -50.0, 100.0, 10.0, step=1.0, key="w_scen_a")
                a_val = math.log(1 + pct_move / 100)
                sc1.caption(f"Log-change model: a {pct_move:+.1f}% move enters as log(1{pct_move / 100:+.3f}) = {a_val:.4f}.")
            else:
                a_val = sc1.number_input(f"Assumed {a} value", value=float(scale or 0.01), format="%.4f", key="w_scen_a2")
            b_val = sc2.number_input(f"Assumed {b}", value=float(round(r.centers[b], 3)), format="%.3f", key="w_scen_b")
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
with tabs[3]:
    st.subheader("Rolling relationships — coefficient stability analysis")
    st.markdown(f'<p class="small-note">At each window end the model is refit using only the trailing {roll_window} '
                f'months. Consecutive windows share {roll_window - 1} of {roll_window} months '
                f'({(roll_window - 1) / roll_window:.0%} overlap), so neighbouring estimates are strongly dependent; '
                'look for persistent shifts, not month-to-month wiggles.</p>', unsafe_allow_html=True)
    if not X:
        st.warning("Select factors in the sidebar.")
    else:
        rc1, rc2, rc3 = st.columns(3)
        rsyms = msel(rc1, "Stocks", tickers, tickers[: min(4, len(tickers))], "w_rsyms")
        use_int = rc2.checkbox("Add interaction (factor × moderator)", key="w_rint",
                               disabled=not any(f in X for f in firm_opts))
        standardized = rc3.radio("Scale", ["Raw units", "Standardized (shared full-period SDs)"], key="w_rscale") != "Raw units"
        interaction, slope_at = None, None
        if use_int:
            ic1, ic2, ic3 = st.columns(3)
            ra_opts, rb_opts = [f for f in X if f in factor_opts], [f for f in X if f in firm_opts]
            ra = sel(ic1, "Factor", ra_opts, "oil" if "oil" in ra_opts else ra_opts[0], "w_ra")
            rb = sel(ic2, "Moderator", rb_opts, "debt_to_assets" if "debt_to_assets" in rb_opts else rb_opts[0], "w_rb")
            slope_at = ic3.number_input(f"Report {ra} sensitivity at fixed {rb} =", value=0.30, format="%.3f", key="w_rslope")
            interaction = (ra, rb)
        roll, rsk = _rolling(panel, pkey, tuple(rsyms), tuple(X), int(roll_window), int(min_obs), hac_rule,
                             int(hac_lags), interaction, slope_at, start_m, eff_end)
        if roll.empty:
            st.warning(f"No window has ≥ {min_obs} complete months. Shorten the window or the minimum, or widen dates.")
        else:
            RES["rolling"] = roll
            terms_all = list(dict.fromkeys(roll["term"]))
            terms = msel(st, "Terms", terms_all, [t for t in terms_all if t.startswith("slope_")] or terms_all[:3], "w_rterms")
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
with tabs[4]:
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
    fdef = [c for c in ["lag_own", "lag_market", "lag_oil", "lag_rate", "lag_fx", "lag_debt_to_assets",
                        "lag_earnings_growth"] if c in lag_opts]
    fc1, fc2 = st.columns([3, 2])
    feats = msel(fc1, "Predictors (all known at the decision cutoff)", lag_opts, fdef, "w_ffeats",
                 format_func=lambda c: lag_labels.get(c, c))
    fint = fc2.checkbox("Add lag_oil × lag_debt_to_assets (centered on training mean)", key="w_fint",
                        disabled=not ({"lag_oil", "lag_debt_to_assets"} <= set(feats)))
    with st.expander("Training and evaluation settings", expanded=False):
        e1, e2, e3, e4 = st.columns(4)
        fwin = e1.radio("Training window", ["expanding", "trailing"], key="w_fwin")
        ftrail = e1.number_input("Trailing length (months)", 24, 240, DEFAULTS.forecast_trailing, key="w_ftrail")
        fmin = e2.number_input("Initial training months", 24, 180, DEFAULTS.forecast_min_train, key="w_fmin")
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
                             fimp, ("lag_oil", "lag_debt_to_assets") if fint else None, int(fret), float(fpi), run_id)
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
            fsym = sel(g1, "Stock", tickers, tickers[0], "w_fsym")
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
with tabs[5]:
    st.subheader("Portfolio exploration (PyPortfolioOpt)")
    st.markdown('<p class="small-note">Long-only minimum-volatility weights from a Ledoit–Wolf shrunk covariance of '
                'monthly total returns (annualized with frequency = 12), compared with equal weights in the same '
                'eligible universe. Risk numbers are estimates from these fitted inputs.</p>', unsafe_allow_html=True)
    pc1, pc2, pc3 = st.columns(3)
    cap = pc1.slider("Max weight per stock", 0.05, 1.0, DEFAULTS.max_weight, 0.05, key="w_cap")
    look = pc2.number_input("Covariance lookback (months)", 24, 180, DEFAULTS.cov_lookback, key="w_look")
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
    rf_mode = b5.selectbox("Risk-free (for Sharpe)", ["none", "constant", "BoC policy rate (proxy)"], key="w_rf")
    rf_const = st.number_input("Constant CAD risk-free rate (annual %)", 0.0, 10.0, 2.0, key="w_rfc") if rf_mode == "constant" else None
    try:
        res, wh, bmsg = _backtest(pan, pkey + str(start_m), tuple(tickers), benchmark, int(look), float(cap), float(fee),
                                  float(slip), int(every), bool(ci))
        res = res[(res["decision_month"] >= start_m) & (res["decision_month"] <= eff_end)]
        rf, rf_label = None, "none"
        if rf_mode == "constant":
            rf = pd.Series((1 + rf_const / 100) ** (1 / 12) - 1, index=pd.period_range(start_m, eff_end, freq="M"))
            rf_label = f"constant {rf_const:.2f}% p.a. (CAD), compounded monthly"
        elif rf_mode.startswith("BoC") and len(pan["macro_pit"]) and "rate" in pan["macro_pit"]["factor"].unique():
            lv = pan["macro_pit"][pan["macro_pit"]["factor"] == "rate"].set_index("month")["level"]
            rf = lv / 100 / 12
            rf_label = "BoC policy-rate level at the decision cutoff ÷ 12 (approximate CAD cash return; not a T-bill series)"
        summ = P.performance_summary(res, rf, rf_label)
        RES["backtest"] = res
        RES["weights_history"] = wh
        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(C.growth_plot(res, "net", SYN), width="stretch")
        with g2:
            st.plotly_chart(C.drawdown_plot(res, SYN), width="stretch")
        st.dataframe(summ, hide_index=True, width="stretch")
        st.caption("Execution: close of the first TSX session after each month-end cutoff; returns run execution-to-"
                   "execution from the total-return index; weights drift between rebalances. Traded notional = "
                   "Σ|target − drifted weight|; each traded dollar pays fees + slippage once. Half-turnover = traded "
                   "notional ÷ 2 (equal to one-sided turnover when the portfolio stays fully invested). Gross results "
                   "are the zero-cost sensitivity. Sharpe is shown only with an explicit CAD risk-free assumption.")
        with st.expander("Rebalance log and weights"):
            st.dataframe(res, hide_index=True, width="stretch")
            st.dataframe(wh, hide_index=True, width="stretch")
        for m in bmsg[:10]:
            st.warning(m)
    except P.PortfolioInfeasible as e:
        st.error(f"Backtest: {e}")

# =========================================================================== Data & exports
with tabs[6]:
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
