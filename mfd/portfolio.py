"""Portfolio exploration with PyPortfolioOpt and a transparent rebalancing backtest.

Conventions (displayed in the UI)
* Inputs are monthly simple total returns; covariance is Ledoit–Wolf shrinkage annualized
  with frequency=12. Long-only, per-stock cap (default 25%), fully invested.
* Decision at month-end cutoff t uses only returns of months ≤ t. Execution happens at the
  close of the first TSX session after the cutoff. Holding-period returns run from one
  execution close to the next (execution-to-execution), from the total-return index.
* Weights drift with realized returns between rebalances. Traded notional at a rebalance =
  Σ|target − pre-trade weight| over stocks (fraction of portfolio value). Cost = traded notional ×
  (fee + slippage) bps, charged once per traded dollar (one-way). Half-turnover = traded/2.
* A position whose price history stops (delisting) is converted to cash at its last available
  close; deal consideration is not modelled. Cash earns zero.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .calendar import first_session_after


class PortfolioInfeasible(ValueError):
    pass


def check_feasible(n_assets: int, max_weight: float):
    if n_assets == 0:
        raise PortfolioInfeasible("no eligible stocks with complete return history in the lookback window")
    if n_assets * max_weight < 1 - 1e-9:
        need = int(np.ceil(1 / max_weight))
        raise PortfolioInfeasible(
            f"{n_assets} eligible stock(s) × {max_weight:.0%} cap = {n_assets * max_weight:.0%} < 100%: a fully invested "
            f"long-only portfolio is impossible. Raise the cap to ≥ {1 / n_assets:.0%} or include ≥ {need} stocks.")


def min_vol_weights(rets: pd.DataFrame, max_weight: float = 0.25):
    """Long-only minimum-volatility weights from Ledoit–Wolf shrunk covariance (monthly → annual)."""
    from pypfopt import EfficientFrontier
    from pypfopt.risk_models import CovarianceShrinkage

    rets = rets.dropna(axis=1, how="any")
    check_feasible(rets.shape[1], max_weight)
    if len(rets) < max(24, rets.shape[1] + 2):
        raise PortfolioInfeasible(f"only {len(rets)} months of complete returns; need ≥ {max(24, rets.shape[1] + 2)}")
    cs = CovarianceShrinkage(rets, returns_data=True, frequency=12)
    S = cs.ledoit_wolf()
    ef = EfficientFrontier(None, S, weight_bounds=(0, max_weight))
    try:
        ef.min_volatility()
    except Exception as e:
        raise PortfolioInfeasible(f"optimizer failed: {type(e).__name__}: {e}") from None
    w = pd.Series(ef.weights, index=rets.columns).clip(lower=0)
    w = w / w.sum()
    vol = float(np.sqrt(w @ S.to_numpy() @ w))
    return w, S, dict(shrinkage=float(cs.delta), est_annual_vol=vol, n_months=len(rets))


def equal_weights(cols) -> pd.Series:
    return pd.Series(1.0 / len(cols), index=list(cols))


def concentration(w: pd.Series) -> dict:
    w = w[w > 0]
    hhi = float((w ** 2).sum())
    return dict(max_weight=float(w.max()), hhi=hhi, effective_n=1 / hhi if hhi else np.nan, n_holdings=int(len(w)))


def holdings_exposure(w: pd.Series, betas: pd.DataFrame) -> pd.Series:
    """Weighted-average factor betas (betas: index symbol, columns factor)."""
    b = betas.reindex(w.index)
    return (b.mul(w, axis=0)).sum(min_count=1)


def _tri_at(tri: pd.Series, date: pd.Timestamp):
    s = tri.dropna()
    s = s[s.index <= date]
    return (s.iloc[-1], s.index[-1]) if len(s) else (np.nan, None)


def backtest(tri: pd.DataFrame, mret_wide: pd.DataFrame, mcal: pd.DataFrame, symbols, benchmark: str,
             lookback: int = 60, max_weight: float = 0.25, fee_bps: float = 5.0, slippage_bps: float = 10.0,
             rebalance_every: int = 1, charge_initial: bool = True, min_months: int | None = None):
    """Run min-vol and equal-weight strategies plus buy-and-hold benchmark on common execution dates."""
    cost_rate = (fee_bps + slippage_bps) / 1e4
    min_months = min_months or lookback
    sessions = tri.index
    mc = mcal.dropna(subset=["last_session"]).reset_index(drop=True)
    months = list(mc["month"])
    plans, msgs = [], []
    for i, m in enumerate(months[:-1]):
        exe = first_session_after(mc.loc[i, "last_session"], sessions)
        nxt = first_session_after(mc.loc[i + 1, "last_session"], sessions) if i + 1 < len(months) else None
        if exe is None or nxt is None:
            continue
        plans.append(dict(decision_month=m, execution=exe, next_execution=nxt))
    if not plans:
        raise PortfolioInfeasible("not enough sessions after the decision cutoffs to run a backtest")
    strategies = {"min_vol": {}, "equal_weight": {}}
    state = {k: dict(w=None, value=1.0, started=False) for k in strategies}
    rows, weights_hist = [], []
    first_exec = None
    k_reb = 0
    for p in plans:
        m, exe, nxt = p["decision_month"], p["execution"], p["next_execution"]
        win = mret_wide.loc[(mret_wide.index > m - lookback) & (mret_wide.index <= m), list(symbols)]
        tradable = [s for s in symbols if s in tri and pd.notna(tri[s].get(exe, np.nan))]
        complete = win[tradable].dropna(axis=1, how="any") if len(win) >= min_months else win.iloc[0:0]
        eligible = list(complete.columns) if len(win) >= min_months else []
        do_reb = (k_reb % rebalance_every == 0)
        targets = {}
        if eligible and do_reb:
            try:
                wmv, _, info = min_vol_weights(complete, max_weight)
                targets["min_vol"] = wmv
                targets["equal_weight"] = equal_weights(eligible)
            except Exception as e:  # PortfolioInfeasible
                msgs.append(f"{m}: rebalance skipped — {e}")
        if not any(st["started"] for st in state.values()) and not targets:
            continue
        if first_exec is None:
            first_exec = exe
        k_reb += 1
        # realized execution-to-execution returns
        r = {}
        flags = []
        for s in set(symbols):
            if s not in tri:
                continue
            v0, d0 = _tri_at(tri[s], exe)
            v1, d1 = _tri_at(tri[s], nxt)
            if d0 is None or d0 != exe:
                r[s] = np.nan
                continue
            r[s] = v1 / v0 - 1
            if d1 is not None and d1 < nxt:
                flags.append(f"{s} stops trading {d1.date()} (converted to cash at last close)")
        for name, st in state.items():
            w_pre = st["w"] if st["w"] is not None else pd.Series(dtype=float)
            if name in targets:
                tgt = targets[name]
                idx = tgt.index.union(w_pre.index)
                pre = w_pre.reindex(idx).fillna(0.0)
                # positions that became cash (delisted) are not "sold" again
                traded = float((tgt.reindex(idx).fillna(0.0) - pre).abs().sum())
                if not st["started"] and not charge_initial:
                    traded_charged = 0.0
                else:
                    traded_charged = traded
                cost = traded_charged * cost_rate
                w = tgt
                st["started"] = True
            else:
                if not st["started"]:
                    continue
                traded, traded_charged, cost = 0.0, 0.0, 0.0
                w = w_pre
            rr = pd.Series({s: r.get(s, np.nan) for s in w.index})
            gross = float((w * rr.fillna(0.0)).sum())  # missing → held as cash (0%)
            net = (1 - cost) * (1 + gross) - 1
            st["value"] *= (1 + net)
            grown = w * (1 + rr.fillna(0.0))
            # delisted names become cash: remove from risky weights after the interval
            stopped = [s for s in w.index if _tri_at(tri[s], nxt)[1] is not None and _tri_at(tri[s], nxt)[1] < nxt]
            denom = (1 + gross)
            st["w"] = (grown / denom).drop(index=stopped, errors="ignore")
            rows.append(dict(strategy=name, decision_month=m, execution=exe, next_execution=nxt,
                             gross_return=gross, net_return=net, traded_notional=traded,
                             traded_charged=traded_charged, half_turnover=traded / 2, cost=cost,
                             value=st["value"], rebalanced=name in targets, n_holdings=int((w > 0).sum()),
                             notes="; ".join(flags)))
            if name in targets:
                weights_hist.append(pd.DataFrame({"strategy": name, "decision_month": m, "execution": exe,
                                                  "symbol": w.index, "target_weight": w.values,
                                                  "pre_trade_weight": w_pre.reindex(w.index).fillna(0.0).values}))
    res = pd.DataFrame(rows)
    if res.empty:
        raise PortfolioInfeasible("no feasible rebalance in the selected period: " + (msgs[-1] if msgs else ""))
    # buy-and-hold ETF on the same execution dates
    dates = sorted(res["execution"].unique())
    nexts = res.drop_duplicates("execution").set_index("execution")["next_execution"]
    bvals, v = [], 1.0
    for j, d in enumerate(dates):
        v0, d0 = _tri_at(tri[benchmark], d)
        v1, _ = _tri_at(tri[benchmark], nexts[d])
        g = v1 / v0 - 1 if d0 is not None else np.nan
        c = cost_rate if (j == 0 and charge_initial) else 0.0
        n = (1 - c) * (1 + g) - 1
        v *= 1 + n
        bvals.append(dict(strategy=f"buy_hold_{benchmark}", decision_month=res.loc[res["execution"] == d, "decision_month"].iloc[0],
                          execution=d, next_execution=nexts[d], gross_return=g, net_return=n,
                          traded_notional=1.0 if j == 0 else 0.0, traded_charged=1.0 if (j == 0 and charge_initial) else 0.0,
                          half_turnover=0.5 if j == 0 else 0.0, cost=c, value=v, rebalanced=j == 0, n_holdings=1, notes=""))
    res = pd.concat([res, pd.DataFrame(bvals)], ignore_index=True)
    # restrict all strategies to common execution dates
    common = set.intersection(*[set(g["execution"]) for _, g in res.groupby("strategy")])
    res = res[res["execution"].isin(common)].sort_values(["strategy", "execution"]).reset_index(drop=True)
    wh = pd.concat(weights_hist, ignore_index=True) if weights_hist else pd.DataFrame()
    return res, wh, msgs


def performance_summary(res: pd.DataFrame, rf_monthly: pd.Series | None = None, rf_label: str = "none") -> pd.DataFrame:
    rows = []
    for name, g in res.groupby("strategy"):
        g = g.sort_values("execution")
        out = dict(strategy=name, first_execution=g["execution"].min().date(), last_period_end=g["next_execution"].max().date(),
                   periods=len(g))
        for kind in ("gross", "net"):
            r = g[f"{kind}_return"].astype(float)
            curve = (1 + r).cumprod()
            n = len(r)
            out[f"{kind}_growth_of_1"] = float(curve.iloc[-1])
            out[f"{kind}_ann_return"] = float(curve.iloc[-1] ** (12 / n) - 1) if n else np.nan
            out[f"{kind}_ann_vol"] = float(r.std(ddof=1) * np.sqrt(12)) if n > 1 else np.nan
            out[f"{kind}_max_drawdown"] = float((curve / curve.cummax() - 1).min())
        years = len(g) / 12
        out["avg_annual_traded_notional"] = float(g["traded_notional"].sum() / years) if years else np.nan
        out["avg_annual_half_turnover"] = out["avg_annual_traded_notional"] / 2
        out["total_cost_charged"] = float(g["cost"].sum())
        out["cost_drag_ann"] = out["gross_ann_return"] - out["net_ann_return"]
        if rf_monthly is not None:
            rf = rf_monthly.reindex(g["decision_month"]).to_numpy()
            ex = g["net_return"].to_numpy() - rf
            out["ann_excess_return"] = float(np.nanmean(ex) * 12)
            out["sharpe_net"] = float(np.nanmean(ex) / np.nanstd(ex, ddof=1) * np.sqrt(12))
            out["risk_free_assumption"] = rf_label
        rows.append(out)
    return pd.DataFrame(rows)
