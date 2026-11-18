# Market Factor Dashboard

A local Streamlit app for studying how company fundamentals, oil prices, interest rates, the USD/CAD rate and the broad market relate to stock returns. The starter universe is Canadian energy. Tickers and factors can be edited.

It covers five research tasks: per-stock factor sensitivities, interactions (e.g. oil × debt), rolling-window stability, out-of-sample forecast validation against simple baselines, and uncertainty plus historical prediction errors throughout. It also includes a small PyPortfolioOpt portfolio module.

> Research tool only. Results describe the selected universe and period. Estimates are associations, not causal effects, and nothing here is investment advice.

## Setup (macOS)

```bash
cd market-factor-dashboard
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock        # exact versions tested (see note below)
#   or, if a pin has no wheel for your Python/macOS:
pip install -e ".[dev]"
cp .env.example .env                    # optional API keys
streamlit run app.py
```

The app opens at http://localhost:8501. Run the tests with `pytest`.

The tested environment was Python 3.13 with pandas 3.0.5, numpy 2.5.3, scipy 1.18.1, statsmodels 0.15.0, scikit-learn 1.9.1, PyPortfolioOpt 1.6.0 (cvxpy 1.9.3), Streamlit 1.67.0, Plotly 7.1.0, pyarrow 26.0.1 and yfinance 1.8.0. `requirements.lock` pins the full dependency tree. That tree was resolved on a Linux build host, so if one transitive pin has no macOS wheel, use the `pip install -e .` route. It uses the compatible ranges in `pyproject.toml`.

## Data modes (sidebar)

| Mode | What it does |
|---|---|
| **Synthetic (labelled)** | A deterministic simulated dataset (seed 20261117) with known coefficients, including an oil × debt interaction, quarterly restatements, a mid-month delisting (SYN-GAS) and a short-history stock (SYN-NEW). Every page and export is labelled SYNTHETIC. In *Raw coefficient* view the true values are overlaid as × marks. |
| **Live providers** | Prices come from yfinance with `auto_adjust=False, actions=True`, giving split-adjusted Close, Adj Close, dividends and splits. WTI comes from FRED `DCOILWTICO`. The policy rate (`V39079`) and USD/CAD (`FXUSDCAD`) come from Bank of Canada Valet. Downloads are cached as Parquet snapshots in `data_cache/`; **Refresh data** bypasses the cache. Failures are reported per source. |
| **CSV import** | Works offline. Upload prices, macro, fundamentals and metadata CSVs (templates in `templates/`), or upload a **run bundle** exported from the app to restore its inputs and settings. |

**Fundamentals (debt, earnings)** come from your own dated CSV, which is the core route. Each row needs `available_at`, the time the number became public, such as a press-release timestamp. There is also an optional Financial Modeling Prep adapter (`FMP_API_KEY`). Its Canadian coverage and its filing dates have **not** been verified. FMP documents that some international filing dates are report dates, so its rows are labelled "unverified". If no fundamentals are loaded, the price and macro analyses still run and the firm-feature controls are disabled with an explanation.

**Optional keys (`.env`):** with `FRED_API_KEY` set, WTI uses ALFRED vintages (true point-in-time values). Without it, the app uses current values plus a documented 7-day publication lag. Keys are never written to caches or exports.

## Time conventions

- **Returns:** monthly simple total returns. The default uses the adjusted close, so dividends are counted once, inside the adjustment. The alternative is (close + dividend) / previous close on split-adjusted prices. Months must be complete: the current month, and any month whose data ends early (for example at a delisting), are excluded and listed.
- **Decision cutoff:** 17:00 America/Toronto on each month's last TSX session. The TSX holiday rules are built in, and observed sessions in the data take precedence.
- **Descriptive models** use same-month macro changes (by observation date) and the fundamentals known *before* the month began.
- **Forecast models** use only values whose `available_at` is at or before the cutoff, via as-of joins. Released fundamentals carry forward with their age shown, up to 400 days. A later restatement never overwrites the earlier vintage when training on earlier dates.
- **Macro availability rules** when no vintage exists: WTI +7 days (EIA publishes weekly), BoC series +1 day. These are labelled "approximate".

## Methods by view

| View | Method |
|---|---|
| Overview | Universe metadata with subsector, coverage per series, missing-data counts, excluded rows, and the derived-feature table with availability timestamps. |
| Stock drivers | Per-stock OLS with HAC (Newey–West) SEs using fixed lags (default 3) or the plug-in rule, with t-based 95% CIs. Benjamini–Hochberg q-values over a stated family. Effects are shown for interpretable moves (+10% WTI = log 1.10; +25 bp rate; +5% USD/CAD; +0.10 debt/assets). Diagnostics include rank, standardized condition number, VIF, correlations, Durbin–Watson, Ljung–Box, Jarque–Bera and Breusch–Pagan. Unidentified designs (constant columns, rank deficiency, too few months) are refused with a message. |
| Interactions | Moderator centered within the window. The conditional slope β_a + β_int·(b − b̄) gets its CI from the full covariance matrix. Sparse or out-of-range regions are marked, the joint distribution is shown, and a scenario calculator gives conditional means with extrapolation warnings. |
| Rolling | Trailing-window refits (default 60 months) with CIs, observation counts, overlap shown, and raw or shared-full-period standardized scales. Optionally tracks the interaction slope at a fixed debt level. |
| Forecast validation | Expanding (or trailing) walk-forward. Preprocessing (median imputation, scaling, interaction centering) and ridge α tuning (sklearn `TimeSeriesSplit`) happen inside each training window. Compared against the training-mean and zero forecasts on matched dates using MAE, RMSE, directional accuracy, bias and OOS R² vs the historical mean (negative values kept). A final holdout stays hidden until revealed. Empirical prediction intervals are built from each model's past errors and their coverage is reported. |
| Portfolio | Ledoit–Wolf covariance with `frequency=12` feeds a long-only minimum-volatility portfolio with a weight cap (default 25%) and a feasibility check, compared with equal weight. The backtest rebalances on then-available data and executes at the first session after the cutoff, using execution-to-execution returns with drift. Costs are traded notional × (fees + slippage), and half-turnover = traded/2. Results are shown gross and net, against buy-and-hold of the benchmark ETF. Sharpe appears only with an explicit CAD risk-free assumption. Delisted holdings become cash at their last close. |
| Data & exports | CSVs for every table, a `run_config.json` (settings, sources, retrieval times, transformations, package versions, fingerprints) and a reproducible `.zip` bundle that reloads via CSV import. |

## Starter universe (checked November 2026)

| Symbol | Company | Subsector | Note |
|---|---|---|---|
| CNQ.TO | Canadian Natural Resources | Oil sands / E&P | |
| SU.TO | Suncor Energy | Integrated | |
| CVE.TO | Cenovus Energy | Integrated | acquired MEG Energy in 2025 |
| TOU.TO | Tourmaline Oil | Gas-weighted E&P | |
| ARX.TO | ARC Resources | Gas/condensate E&P | **Acquired by Shell, delisted from TSX at close on 2026-09-04.** Its history ends there; the deal consideration is not modelled. |
| ENB.TO | Enbridge | Pipelines | |
| XIC.TO / XEG.TO | iShares broad-market / energy ETFs | Benchmarks | |

All are TSX listings quoted in CAD. At run time the app compares the provider's exchange and currency with these expectations and flags any mismatch. **Survivorship:** the list was chosen today, so results apply to this selected universe, not to the historical sector.

## Verification status of this build

- **Exercised:** all five core analyses plus the portfolio module on the synthetic dataset, through the running app in a headless browser. The run-bundle export → CSV import → identical results round trip. The live-provider failure path: no network was available in the build environment, and the app showed a source-specific error per symbol and pointed to CSV import. `pytest`: 47 tests covering return shifts, month boundaries, incomplete months, release-time joins and restatements, dividend/split handling, missing and constant features, recovery of a known interaction with an exact check of the covariance-based slope SE, look-ahead invariance, matched-date metrics with negative OOS R², and portfolio feasibility, weights, monthly annualization, drift, execution timing, costs and delisting.
- **Not exercised here:** live downloads from yfinance, FRED, BoC Valet and FMP, because the build host had no access to those sites. Run Live mode once on your Mac to confirm. QuantConnect/LEAN (`integrations/lean/`) is an untested example, and TSX coverage depends on your subscription. MSCI Barra is not implemented because it needs a licensed export.
- **Approximations:** macro availability without vintages uses lag rules; FMP filing dates are unverified; the backtest uses close-to-close execution with a flat per-dollar cost; cash earns zero.

## Project layout

```
app.py                     Streamlit entry point (all views)
mfd/config.py              defaults, universe metadata, factor definitions
mfd/calendar.py            TSX sessions, decision cutoffs, completed months
mfd/providers/             yfinance, FRED, BoC Valet, FMP, CSV import, synthetic data, Parquet cache
mfd/prep.py                total returns, monthly alignment, point-in-time macro & fundamentals
mfd/regression.py          OLS+HAC, diagnostics, BH, interactions, scenarios, rolling fits
mfd/forecast.py            walk-forward evaluation, metrics, empirical intervals
mfd/portfolio.py           PyPortfolioOpt weights, rebalancing backtest, performance summary
mfd/charts.py              Plotly figures
mfd/provenance.py          run config, fingerprints, export bundles
templates/                 CSV templates
tests/                     pytest suite
integrations/lean/         optional LEAN replay example
```

Natural next steps are a pooled panel model with firm and time clustering, natural gas as a factor, and richer factor sets (e.g. currency-matched Fama–French).
