# region imports
from AlgorithmImports import *  # noqa: F401,F403  (provided by QuantConnect LEAN)
import csv
# endregion

# Optional LEAN example: replays the dashboard's exported rebalance weights
# (weights_history.csv from "Data & exports") as an event-driven backtest so that
# fills, fees, slippage and brokerage rules come from LEAN's execution models.
#
# NOT VERIFIED in this build: LEAN was not installed, and TSX equity/corporate-action
# coverage depends on your QuantConnect data subscription. Check that each symbol
# resolves (Market.TSX may not be available on every plan) before comparing engines.


class DashboardWeightsReplay(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2015, 2, 1)
        self.set_end_date(2026, 10, 31)
        self.set_cash("CAD", 100_000)  # cash currency must match your account / data
        self.strategy = self.get_parameter("strategy") or "min_vol"
        self.schedule_by_date = {}
        # weights_history.csv: strategy,decision_month,execution,symbol,target_weight,pre_trade_weight
        with open(self.get_parameter("weights_csv") or "weights_history.csv") as f:
            for row in csv.DictReader(f):
                if row["strategy"] != self.strategy:
                    continue
                d = row["execution"][:10]
                self.schedule_by_date.setdefault(d, {})[row["symbol"]] = float(row["target_weight"])
        self.symbols = {}
        for tickers in self.schedule_by_date.values():
            for t in tickers:
                if t not in self.symbols:
                    base = t.replace(".TO", "")
                    # Adjust market/security type to what your data subscription supports.
                    self.symbols[t] = self.add_equity(base, Resolution.DAILY).symbol
        self.schedule.on(self.date_rules.every_day(), self.time_rules.at(15, 55), self.rebalance)

    def rebalance(self):
        key = self.time.strftime("%Y-%m-%d")
        if key not in self.schedule_by_date:
            return
        targets = [PortfolioTarget(self.symbols[t], w) for t, w in self.schedule_by_date[key].items()]
        self.set_holdings(targets, liquidate_existing_holdings=True)
