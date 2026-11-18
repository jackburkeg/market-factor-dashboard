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
