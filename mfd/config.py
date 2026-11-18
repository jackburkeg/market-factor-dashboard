"""Defaults, universe metadata and factor specifications.

Everything here is a *starting* product choice; the Streamlit sidebar exposes
the editable pieces. Nothing in this module performs I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

APP_TITLE = "Market Factor Dashboard"
REPORTING_CURRENCY = "CAD"
MARKET_TZ = "America/Toronto"
# Decision cutoff: 17:00 Toronto time on the last TSX session of a month.
# The TSX closes at 16:00; the extra hour allows closing prints to settle.
DECISION_CUTOFF_HOUR = 17

# ---------------------------------------------------------------------------
# Universe metadata. Identity facts were checked in November 2026; the app
# re-verifies exchange, quote currency and first/last trade dates from the
# provider at run time and shows any mismatch.
# ---------------------------------------------------------------------------
UNIVERSE_METADATA = [
    dict(symbol="CNQ.TO", name="Canadian Natural Resources", exchange="TSX", currency="CAD",
         subsector="Oil sands / E&P", status="listed", notes=""),
    dict(symbol="SU.TO", name="Suncor Energy", exchange="TSX", currency="CAD",
         subsector="Integrated (oil sands + refining)", status="listed", notes=""),
    dict(symbol="CVE.TO", name="Cenovus Energy", exchange="TSX", currency="CAD",
         subsector="Integrated (oil sands + refining)", status="listed",
         notes="Acquired MEG Energy in 2025."),
    dict(symbol="TOU.TO", name="Tourmaline Oil", exchange="TSX", currency="CAD",
         subsector="Natural-gas-weighted E&P", status="listed", notes=""),
    dict(symbol="ARX.TO", name="ARC Resources", exchange="TSX", currency="CAD",
         subsector="Natural-gas / condensate E&P (Montney)", status="delisted",
         notes="Acquired by Shell plc (closed 2026-09-02); delisted from TSX at close 2026-09-04. "
               "Price history ends there; deal consideration (cash + Shell shares) is not modelled."),
    dict(symbol="ENB.TO", name="Enbridge", exchange="TSX", currency="CAD",
         subsector="Pipelines / midstream", status="listed", notes=""),
]
DEFAULT_TICKERS = [m["symbol"] for m in UNIVERSE_METADATA]

BENCHMARKS = {
    "XIC.TO": "iShares Core S&P/TSX Capped Composite ETF (broad Canadian market)",
    "XEG.TO": "iShares S&P/TSX Capped Energy ETF (Canadian energy)",
}
DEFAULT_BENCHMARK = "XIC.TO"


@dataclass(frozen=True)
class FactorSpec:
    key: str                  # internal column name of the transformed factor
    label: str                # plain-language label
    source: str               # provider key
    series_id: str
    unit: str                 # unit of the *transformed* factor
    level_unit: str           # unit of the raw level
    currency: Optional[str]
    transformation: str       # "log_change" | "diff_pp"
    publication_lag_days: int  # documented availability rule when no vintage exists
    description: str


FACTOR_SPECS = {
    "oil": FactorSpec(
        key="oil", label="WTI crude oil (log change)", source="fred", series_id="DCOILWTICO",
        unit="log change (≈ % change / 100)", level_unit="USD per barrel", currency="USD",
        transformation="log_change", publication_lag_days=7,
        description="WTI Cushing spot price from the U.S. EIA via FRED. Monthly change is "
                    "log(month-end price / prior month-end price), only where both prices are positive "
                    "(the April 2020 negative print is excluded). Availability rule without vintages: "
                    "EIA publishes spot prices weekly, so an observation is treated as usable 7 days later.",
    ),
    "rate": FactorSpec(
        key="rate", label="BoC policy rate (change, pp)", source="boc", series_id="V39079",
        unit="percentage points", level_unit="percent", currency="CAD",
        transformation="diff_pp", publication_lag_days=1,
        description="Bank of Canada target for the overnight rate (Valet series V39079). Month-end level; "
                    "the factor is the month-over-month change in percentage points (0.25 pp = 25 bp). "
                    "Decisions are announced publicly; treated as usable the next day.",
    ),
    "fx": FactorSpec(
        key="fx", label="USD/CAD (log change, CAD per USD)", source="boc", series_id="FXUSDCAD",
        unit="log change (≈ % change / 100)", level_unit="CAD per 1 USD", currency=None,
        transformation="log_change", publication_lag_days=1,
        description="Bank of Canada daily USD/CAD rate, quoted as CAD per 1 USD. A positive change "
                    "means the U.S. dollar strengthened (the Canadian dollar weakened). "
                    "Published late afternoon; treated as usable the next day.",
    ),
}

FIRM_FEATURES = {
    "debt_to_assets": dict(
        label="Debt / total assets",
        unit="ratio (0.30 = 30%)",
        definition="Interest-bearing debt (short-term borrowings + current portion of long-term debt + "
                   "long-term debt) divided by total assets, from the most recent report whose "
                   "availability timestamp is before the decision cutoff. Lease liabilities are "
                   "excluded by default (toggle in settings); the same policy applies to every company.",
    ),
    "earnings_growth": dict(
        label="Net-income growth (YoY)",
        unit="ratio (0.25 = +25%)",
        definition="(Net income for the latest reported period − net income for the same period one year "
                   "earlier) ÷ |prior-year net income|, using only values available at the cutoff. "
                   "Undefined (left missing) when |prior-year net income| is below the small-denominator "
                   "threshold (default 0.2% of total assets, suited to quarterly reports). Loss→profit transitions are labelled.",
    ),
    "earnings_change_scaled": dict(
        label="Net-income change ÷ total assets",
        unit="ratio of assets",
        definition="Absolute year-over-year change in net income divided by total assets. Defined even when "
                   "prior earnings are near zero, so it is the recommended view around loss/profit transitions.",
    ),
}


@dataclass
class AnalysisSettings:
    tickers: list = field(default_factory=lambda: list(DEFAULT_TICKERS))
    benchmark: str = DEFAULT_BENCHMARK
    start: str = ""  # ISO date; blank → 15 years before end
    end: str = ""    # ISO date; blank → latest completed month
    history_years: int = 15
    factors: list = field(default_factory=lambda: ["market", "oil", "rate", "fx"])
    firm_features: list = field(default_factory=lambda: ["debt_to_assets", "earnings_growth"])
    hac_lag_rule: str = "fixed"   # "fixed" | "newey_west"
    hac_lags: int = 3
    rolling_window: int = 60
    min_obs_rolling: int = 36
    forecast_min_train: int = 60
    forecast_window: str = "expanding"  # or "trailing"
    forecast_trailing: int = 60
    ridge_alphas: list = field(default_factory=lambda: [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0])
    ridge_cv_splits: int = 5
    holdout_months: int = 24
    max_weight: float = 0.25
    cov_lookback: int = 60
    fee_bps: float = 5.0
    slippage_bps: float = 10.0
    include_leases_in_debt: bool = False
    small_denominator_frac: float = 0.002
    return_convention: str = "adjusted_close"  # or "close_plus_dividends"
    data_mode: str = "synthetic"

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Additional factors (all FRED, no key needed) used by the US tech preset.
# ---------------------------------------------------------------------------
FACTOR_SPECS.update({
    "y10": FactorSpec(
        key="y10", label="10-year Treasury yield (change, pp)", source="fred", series_id="DGS10",
        unit="percentage points", level_unit="percent", currency="USD", transformation="diff_pp",
        publication_lag_days=1,
        description="U.S. 10-year Treasury constant-maturity yield (FRED DGS10). Month-end level; the factor is the "
                    "monthly change in percentage points (0.25 pp = 25 bp). Long-duration growth stocks are often "
                    "sensitive to it. Treated as usable the next day."),
    "ffr": FactorSpec(
        key="ffr", label="Effective fed funds rate (change, pp)", source="fred", series_id="DFF",
        unit="percentage points", level_unit="percent", currency="USD", transformation="diff_pp",
        publication_lag_days=1,
        description="Effective federal funds rate (FRED DFF), the U.S. policy rate in practice. Month-end level; "
                    "factor is the monthly change in percentage points. Published the next business day."),
    "vix": FactorSpec(
        key="vix", label="VIX (log change)", source="fred", series_id="VIXCLS",
        unit="log change", level_unit="index points", currency=None, transformation="log_change",
        publication_lag_days=1,
        description="CBOE Volatility Index close (FRED VIXCLS), the market's expected 30-day S&P 500 volatility "
                    "(a 'fear gauge'). Factor is the monthly log change; +20% is log(1.20)."),
    "usd": FactorSpec(
        key="usd", label="Broad US dollar index (log change)", source="fred", series_id="DTWEXBGS",
        unit="log change", level_unit="index (Jan 2006 = 100)", currency=None, transformation="log_change",
        publication_lag_days=7,
        description="Federal Reserve nominal broad U.S. dollar index (FRED DTWEXBGS). A rise means a stronger dollar, "
                    "which can weigh on companies with large foreign sales. Published weekly (H.10), so treated as "
                    "usable 7 days after the observation date."),
})

FIRM_FEATURES["profitability"] = dict(
    label="Profitability (return on assets)",
    unit="ratio per quarter (0.02 = 2%)",
    definition="Net income for the latest reported period ÷ total assets at that period end (return on assets), "
               "from the most recent report available before the decision cutoff. Negative for loss-making companies.",
)


@dataclass(frozen=True)
class Profile:
    key: str
    label: str
    exchange: str            # calendar: "TSX" | "NYSE"
    currency: str
    universe: tuple          # metadata dicts
    benchmarks: dict
    default_benchmark: str
    macro_factors: tuple     # factor keys fetched from providers
    default_factors: tuple
    default_firm: tuple
    interaction: tuple       # (factor a, moderator b)
    interaction_label: str
    interaction_question: str
    headline_factor: str
    control_symbol: str | None = None
    control_label: str = ""
    rf_factor: str | None = None       # factor whose level approximates a cash rate (for Sharpe)
    rf_label: str = ""
    rolling_window: int = 60
    min_obs_rolling: int = 36
    forecast_min_train: int = 60
    cov_lookback: int = 60
    synthetic_benchmark: str = "SYN-MKT"


US_TECH_UNIVERSE = (
    dict(symbol="ALAB", name="Astera Labs", exchange="NASDAQ", currency="USD", subsector="AI connectivity chips",
         status="listed", notes="IPO March 2024: short history."),
    dict(symbol="CRDO", name="Credo Technology", exchange="NASDAQ", currency="USD",
         subsector="AI connectivity (active electrical cables, SerDes)", status="listed", notes="IPO January 2022."),
    dict(symbol="NBIS", name="Nebius Group", exchange="NASDAQ", currency="USD", subsector="AI cloud (GPU infrastructure)",
         status="listed", analysis_start="2024-11-01",
         notes="Former Yandex N.V.; trading resumed Oct 2024 after a suspension. Earlier history belongs to a different "
               "business and is excluded (analysis starts Nov 2024)."),
    dict(symbol="ARM", name="Arm Holdings (ADS)", exchange="NASDAQ", currency="USD", subsector="Chip architecture / IP",
         status="listed", notes="IPO September 2023."),
    dict(symbol="IONQ", name="IonQ", exchange="NYSE", currency="USD", subsector="Quantum computing (loss-making)",
         status="listed", notes="Listed via SPAC October 2021."),
    dict(symbol="APLD", name="Applied Digital", exchange="NASDAQ", currency="USD",
         subsector="AI data centres (loss-making)", status="listed", notes=""),
    dict(symbol="CRWV", name="CoreWeave", exchange="NASDAQ", currency="USD", subsector="AI cloud",
         status="listed", optional=True, notes="IPO March 2025: too little monthly history for most analyses."),
    dict(symbol="CBRS", name="Cerebras Systems", exchange="NASDAQ", currency="USD", subsector="AI chips (wafer-scale)",
         status="listed", optional=True, notes="IPO May 2026: too little monthly history for most analyses."),
    dict(symbol="NVDA", name="NVIDIA", exchange="NASDAQ", currency="USD", subsector="Control: mature AI leader",
         status="listed", control=True, notes="Used as a control factor, not as a target stock."),
)

PROFILES = {
    "canada_energy": Profile(
        key="canada_energy", label="Canada – Energy", exchange="TSX", currency="CAD",
        universe=tuple(UNIVERSE_METADATA), benchmarks=BENCHMARKS, default_benchmark=DEFAULT_BENCHMARK,
        macro_factors=("oil", "rate", "fx"), default_factors=("market", "oil", "rate", "fx"),
        default_firm=("debt_to_assets", "earnings_growth"), interaction=("oil", "debt_to_assets"),
        interaction_label="Oil × debt",
        interaction_question="whether oil sensitivity depends on how indebted the company is",
        headline_factor="oil", rf_factor="rate", rf_label="BoC policy rate", synthetic_benchmark="SYN-MKT"),
    "us_tech": Profile(
        key="us_tech", label="US – Tech (early-stage AI)", exchange="NYSE", currency="USD",
        universe=US_TECH_UNIVERSE,
        benchmarks={"QQQ": "Invesco QQQ (Nasdaq-100)", "SPY": "SPDR S&P 500 ETF",
                    "SMH": "VanEck Semiconductor ETF"},
        default_benchmark="QQQ",
        macro_factors=("y10", "ffr", "vix", "usd"), default_factors=("market", "control", "y10", "vix"),
        default_firm=("profitability",), interaction=("y10", "profitability"),
        interaction_label="Rates × profitability",
        interaction_question="whether sensitivity to the 10-year yield depends on how profitable the company is",
        headline_factor="y10", control_symbol="NVDA", control_label="NVIDIA return (AI-leader control)",
        rf_factor="ffr", rf_label="effective fed funds rate",
        rolling_window=24, min_obs_rolling=18, forecast_min_train=24, cov_lookback=24,
        synthetic_benchmark="SYN-NDX"),
}
DEFAULT_PROFILE = "canada_energy"
