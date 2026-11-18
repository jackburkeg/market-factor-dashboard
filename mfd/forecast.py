"""Chronological (walk-forward) forecast evaluation of next-month returns.

At each decision month t (cutoff = 17:00 Toronto on t's last session):
* training rows are decision months s whose target month s+1 has been realized by t
  (s + 1 ≤ t), optionally restricted to a trailing window;
* all preprocessing (median imputation, standardization, interaction centering) and ridge
  tuning (time-ordered folds, sklearn TimeSeriesSplit) are fitted on those rows only;
* the prediction for month t+1 uses the features known at the cutoff of month t.
Because nothing after t enters the computation, changing later observations cannot change
earlier forecasts (tested in tests/test_forecast.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

MODELS = ["ols", "ridge", "hist_mean", "zero"]
MODEL_LABELS = {"ols": "OLS", "ridge": "Ridge", "hist_mean": "Historical mean (baseline)", "zero": "Zero return"}


def _prep_fold(Xtr: pd.DataFrame, Xte: pd.DataFrame, impute: str, interaction):
    """Fit preprocessing on training rows only; apply to test rows."""
    Xtr, Xte = Xtr.copy(), Xte.copy()
    info = {}
    if impute == "train_median":
        med = Xtr.median()
        info["medians"] = med.to_dict()
        Xtr, Xte = Xtr.fillna(med), Xte.fillna(med)
    if interaction:
        a, b = interaction
        c = float(Xtr[b].mean())
        info["center_" + b] = c
        Xtr[f"{a}_x_{b}"] = Xtr[a] * (Xtr[b] - c)
        Xte[f"{a}_x_{b}"] = Xte[a] * (Xte[b] - c)
    return Xtr, Xte, info


def _tune_ridge(X: np.ndarray, y: np.ndarray, alphas, n_splits: int, gap: int = 0):
    n_splits = max(2, min(n_splits, len(y) // 12))
    tss = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    scores = {a: [] for a in alphas}
    for tr, va in tss.split(X):
        sc = StandardScaler().fit(X[tr])
        Xt, Xv = sc.transform(X[tr]), sc.transform(X[va])
        # Closed form identical to sklearn Ridge(fit_intercept=True) on standardized X:
        # intercept = mean(y_train), β = (X'X + αI)⁻¹ X'(y − ȳ). Columns of Xt are already centered.
        ym = y[tr].mean()
        XtX, Xty = Xt.T @ Xt, Xt.T @ (y[tr] - ym)
        I = np.eye(Xt.shape[1])
        for a in alphas:
            beta = np.linalg.solve(XtX + a * I, Xty)
            scores[a].append(np.mean((Xv @ beta + ym - y[va]) ** 2))
    mse = {a: float(np.mean(v)) for a, v in scores.items()}
    best = min(mse, key=mse.get)
    return best, mse


def walk_forward(panel_sym: pd.DataFrame, features: list, symbol: str, min_train: int = 60,
                 window: str = "expanding", trailing: int = 60, alphas=(0.1, 1, 10, 100, 1000),
                 cv_splits: int = 5, impute: str = "drop", interaction=None, retune_every: int = 1,
                 run_id: str = "") -> tuple[pd.DataFrame, list]:
    """Return (forecast records, skipped messages) for one stock.

    panel_sym: rows indexed by decision month with columns features + target_next, decision_at.
    """
    df = panel_sym.sort_index()
    months = df.index
    recs, skipped = [], []
    best_alpha, last_tune = None, None
    for t in months:
        row = df.loc[[t]]
        if pd.isna(row["target_next"].iloc[0]):
            continue
        tr = df[(df.index + 1 <= t) & df["target_next"].notna()]
        if window == "trailing":
            tr = tr[tr.index > t - trailing - 1]
        if impute == "drop":
            tr = tr.dropna(subset=features)
        if len(tr) < min_train:
            continue
        Xr = row[features]
        if impute == "drop" and Xr.isna().any(axis=1).iloc[0]:
            skipped.append(f"{symbol} {t}: feature(s) missing at cutoff: "
                           + ", ".join(Xr.columns[Xr.isna().iloc[0]]))
            continue
        Xtr, Xte, info = _prep_fold(tr[features], Xr, impute, interaction)
        const = [c for c in Xtr.columns if Xtr[c].std(ddof=0) < 1e-12]
        if const:
            Xtr, Xte = Xtr.drop(columns=const), Xte.drop(columns=const)
            info["dropped_constant"] = const
        ytr = tr["target_next"].to_numpy(float)
        y_true = float(row["target_next"].iloc[0])
        base = dict(run_id=run_id, symbol=symbol, decision_month=t, target_month=t + 1,
                    decision_at=row["decision_at"].iloc[0], training_start=tr.index.min(), training_end=tr.index.max(),
                    n_train=len(tr), features="|".join(Xtr.columns), actual=y_true,
                    feature_source_available_at=row.get("feature_source_available_at", pd.Series([pd.NaT])).iloc[0])
        hist = float(ytr.mean())
        preds = {"hist_mean": (hist, {}), "zero": (0.0, {})}
        if Xtr.shape[1] > 0:
            sc = StandardScaler().fit(Xtr.to_numpy(float))
            Xs, Xts = sc.transform(Xtr.to_numpy(float)), sc.transform(Xte.to_numpy(float))
            ols = LinearRegression().fit(Xs, ytr)
            preds["ols"] = (float(ols.predict(Xts)[0]), {})
            if best_alpha is None or last_tune is None or (t - last_tune).n >= retune_every:
                best_alpha, mse = _tune_ridge(Xtr.to_numpy(float), ytr, list(alphas), cv_splits)
                last_tune = t
            rid = Ridge(alpha=best_alpha).fit(Xs, ytr)
            preds["ridge"] = (float(rid.predict(Xts)[0]), {"alpha": best_alpha})
        for m, (p, extra) in preds.items():
            recs.append({**base, "model": m, "prediction": p, "error": y_true - p,
                         "settings": {**extra, **info, "window": window, "impute": impute}})
    out = pd.DataFrame(recs)
    return out, skipped


def add_empirical_intervals(fc: pd.DataFrame, level: float = 0.80, min_errors: int = 24) -> pd.DataFrame:
    """Empirical prediction intervals from each model's *past* out-of-sample errors.

    For a forecast made at decision month t, only errors whose target month ≤ t (already realized)
    are used. Interval = prediction + [q_(1-level)/2, q_(1+level)/2] of those errors.
    """
    out = []
    lo_q, hi_q = (1 - level) / 2, (1 + level) / 2
    for (sym, m), g in fc.groupby(["symbol", "model"]):
        g = g.sort_values("decision_month").copy()
        los, his = [], []
        for _, r in g.iterrows():
            past = g[g["target_month"] <= r["decision_month"]]["error"]
            if len(past) >= min_errors:
                los.append(r["prediction"] + past.quantile(lo_q))
                his.append(r["prediction"] + past.quantile(hi_q))
            else:
                los.append(np.nan), his.append(np.nan)
        g["pi_low"], g["pi_high"] = los, his
        g["pi_covered"] = np.where(g["pi_low"].notna(), (g["actual"] >= g["pi_low"]) & (g["actual"] <= g["pi_high"]), np.nan)
        out.append(g)
    res = pd.concat(out, ignore_index=True) if out else fc
    res["pi_level"] = level
    return res


def matched(fc: pd.DataFrame, models=None) -> pd.DataFrame:
    """Keep only (symbol, target_month) pairs where every requested model has a forecast."""
    models = models or sorted(fc["model"].unique())
    f = fc[fc["model"].isin(models)]
    cnt = f.groupby(["symbol", "target_month"])["model"].nunique()
    keep = cnt[cnt == len(models)].index
    return f.set_index(["symbol", "target_month"]).loc[keep].reset_index()


def metrics(fc: pd.DataFrame, by_symbol: bool = False, baseline: str = "hist_mean") -> pd.DataFrame:
    """MAE, RMSE, directional accuracy, bias and OOS R² vs the historical-mean baseline,
    computed on exactly matching evaluation dates. Negative OOS R² is preserved."""
    f = matched(fc)
    if f.empty:
        return pd.DataFrame()
    keys = ["symbol", "model"] if by_symbol else ["model"]
    base = f[f["model"] == baseline].set_index(["symbol", "target_month"])["error"]
    rows = []
    for k, g in f.groupby(keys):
        k = k if isinstance(k, tuple) else (k,)
        e = g["error"].to_numpy()
        nz = g["actual"] != 0
        direction = (np.sign(g["prediction"]) == np.sign(g["actual"]))[nz & (g["prediction"] != 0)]
        b = base.reindex(pd.MultiIndex.from_frame(g[["symbol", "target_month"]])).to_numpy()
        sse_m, sse_b = float(np.sum(e ** 2)), float(np.sum(b ** 2))
        rows.append(dict(zip(keys, k), n_forecasts=len(g),
                         first_target=g["target_month"].min(), last_target=g["target_month"].max(),
                         MAE=float(np.mean(np.abs(e))), RMSE=float(np.sqrt(np.mean(e ** 2))),
                         directional_accuracy=float(direction.mean()) if len(direction) else np.nan,
                         bias=float(np.mean(-e)),  # mean(prediction − actual)
                         oos_r2_vs_hist_mean=1 - sse_m / sse_b if sse_b > 0 else np.nan))
    return pd.DataFrame(rows)


def split_holdout(fc: pd.DataFrame, holdout_months: int):
    """Tag the last `holdout_months` target months as the final holdout."""
    if fc.empty or holdout_months <= 0:
        return fc.assign(segment="validation")
    last = fc["target_month"].max()
    cut = last - holdout_months
    return fc.assign(segment=np.where(fc["target_month"] > cut, "holdout", "validation"))
