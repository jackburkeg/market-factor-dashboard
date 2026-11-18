"""Descriptive per-stock regressions: OLS with HAC (Newey–West) standard errors, diagnostics,
Benjamini–Hochberg q-values, interaction slopes with full-covariance uncertainty, scenarios,
and rolling-window refits.

All inference uses the t distribution with the residual degrees of freedom (use_t=True),
which is slightly more conservative than the normal approximation in small monthly samples.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera

# Scenario size used to translate a coefficient into a readable effect.
EFFECT_SCALES = {
    "market": (0.01, "+1 pp benchmark return"),
    "oil": (math.log(1.10), "+10% WTI (log 1.10 = 0.0953)"),
    "rate": (0.25, "+25 bp policy-rate change"),
    "fx": (math.log(1.05), "+5% USD/CAD (log 1.05)"),
    "debt_to_assets": (0.10, "+0.10 debt/assets (10 pp of assets)"),
    "earnings_growth": (0.10, "+10 pp YoY net-income growth"),
    "earnings_change_scaled": (0.01, "+1% of assets net-income change"),
    "oil_x_debt": (None, "see conditional slope"),
}
MIN_RESID_DF = 10


class ModelSpecError(ValueError):
    """The model is not identified or the sample is insufficient; message is user-facing."""


@dataclass
class FitResult:
    symbol: str
    y: str
    xs: list
    params: pd.Series
    bse: pd.Series
    tvalues: pd.Series
    pvalues: pd.Series
    conf_int: pd.DataFrame
    cov: pd.DataFrame
    nobs: int
    df_resid: float
    rsquared: float
    rsquared_adj: float
    hac_lags: int
    start: object
    end: object
    resid: pd.Series
    fitted: pd.Series
    data: pd.DataFrame
    dropped_rows: int
    diagnostics: dict = field(default_factory=dict)
    centers: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


def newey_west_lags(n: int) -> int:
    """Newey–West (1994) plug-in rule: floor(4·(n/100)^(2/9))."""
    return int(math.floor(4 * (n / 100.0) ** (2.0 / 9.0)))


def resolve_lags(n: int, rule: str, fixed: int) -> int:
    lags = newey_west_lags(n) if rule == "newey_west" else int(fixed)
    return max(0, min(lags, max(0, n // 4)))


def identification_report(X: pd.DataFrame, n_params: int):
    """Return (errors, warnings, info) about a design matrix without an intercept column."""
    errors, warns, info = [], [], {}
    n = len(X)
    info["n"], info["k"] = n, n_params
    info["obs_per_param"] = n / n_params if n_params else float("nan")
    const = [c for c in X.columns if X[c].std(ddof=0) < 1e-12]
    if const:
        errors.append("constant (zero-variance) column(s): " + ", ".join(const) +
                      ". Remove them or widen the window; a constant feature cannot be separated from the intercept.")
    near = [c for c in X.columns if c not in const and X[c].nunique() <= 2 and n > 10]
    if near:
        warns.append("very few distinct values in: " + ", ".join(near))
    if n - n_params < MIN_RESID_DF:
        errors.append(f"insufficient history: {n} usable months for {n_params} parameters "
                      f"(need ≥ {n_params + MIN_RESID_DF}). Use fewer features or a longer window.")
    Xc = sm.add_constant(X, has_constant="add").to_numpy(float)
    if n > 0:
        rank = np.linalg.matrix_rank(Xc)
        info["rank"] = int(rank)
        if rank < Xc.shape[1]:
            errors.append(f"design is rank deficient (rank {rank} < {Xc.shape[1]} columns): some features are exact "
                          "linear combinations of others. Drop one of the collinear features.")
    if not errors and X.shape[1] > 0:
        Z = (X - X.mean()) / X.std(ddof=0)
        info["condition_number"] = float(np.linalg.cond(sm.add_constant(Z).to_numpy(float)))
        if X.shape[1] > 1:
            Zc = sm.add_constant(Z).to_numpy(float)
            info["vif"] = {c: float(variance_inflation_factor(Zc, i + 1)) for i, c in enumerate(X.columns)}
            hi = [c for c, v in info["vif"].items() if v > 10]
            if hi:
                warns.append("high collinearity (VIF > 10): " + ", ".join(hi) +
                             ". Their individual coefficients are imprecise; consider dropping one.")
        info["correlation"] = X.corr()
    return errors, warns, info


def fit_ols(df: pd.DataFrame, y: str, xs: list, symbol: str = "", lag_rule: str = "fixed", lags: int = 3,
            centers: dict | None = None) -> FitResult:
    """Fit y ~ 1 + xs with HAC covariance. `df` must be indexed by month (Period)."""
    cols = [y] + list(xs)
    missing_cols = [c for c in cols if c not in df]
    if missing_cols:
        raise ModelSpecError(f"{symbol}: feature(s) not available in this dataset: {', '.join(missing_cols)}")
    d = df[cols].apply(pd.to_numeric, errors="coerce")
    full = len(d)
    d = d.dropna()
    if d.empty:
        raise ModelSpecError(f"{symbol}: no months with all of {', '.join(cols)} observed")
    X = d[list(xs)]
    errors, warns, info = identification_report(X, len(xs) + 1)
    if errors:
        raise ModelSpecError(f"{symbol}: " + " ".join(errors))
    L = resolve_lags(len(d), lag_rule, lags)
    model = sm.OLS(d[y], sm.add_constant(X, has_constant="add"))
    res = model.fit(cov_type="HAC", cov_kwds={"maxlags": L, "use_correction": True}, use_t=True)
    ci = res.conf_int(0.05)
    ci.columns = ["lo", "hi"]
    resid = res.resid
    diag = dict(info)
    diag["hac_lags"] = L
    diag["hac_rule"] = ("Newey–West plug-in floor(4·(n/100)^(2/9))" if lag_rule == "newey_west" else "fixed")
    diag["durbin_watson"] = float(durbin_watson(resid))
    lb_lag = max(L, 6) if len(resid) > 12 else max(1, len(resid) // 3)
    try:
        lb = acorr_ljungbox(resid, lags=[lb_lag])
        diag["ljung_box_p"] = float(lb["lb_pvalue"].iloc[0])
        diag["ljung_box_lag"] = lb_lag
    except Exception:
        diag["ljung_box_p"] = float("nan")
    jb = jarque_bera(resid)
    diag["jarque_bera_p"] = float(jb[1])
    try:
        diag["breusch_pagan_p"] = float(het_breuschpagan(resid, model.exog)[1])
    except Exception:
        diag["breusch_pagan_p"] = float("nan")
    if diag.get("ljung_box_p", 1) < 0.05:
        warns.append(f"residual autocorrelation detected (Ljung–Box p={diag['ljung_box_p']:.3f} at lag {lb_lag}); "
                     f"consider more HAC lags than {L}.")
    return FitResult(symbol=symbol, y=y, xs=list(xs), params=res.params, bse=res.bse, tvalues=res.tvalues,
                     pvalues=res.pvalues, conf_int=ci, cov=res.cov_params(), nobs=int(res.nobs),
                     df_resid=float(res.df_resid), rsquared=float(res.rsquared), rsquared_adj=float(res.rsquared_adj),
                     hac_lags=L, start=d.index.min(), end=d.index.max(), resid=resid, fitted=res.fittedvalues,
                     data=d, dropped_rows=full - len(d), diagnostics=diag, centers=centers or {}, warnings=warns)


def effects_table(fits: dict, family: str = "all displayed non-intercept coefficients") -> pd.DataFrame:
    """Long table of coefficients across stocks with BH q-values over the displayed family."""
    rows = []
    for sym, r in fits.items():
        for name in r.params.index:
            scale, scale_label = EFFECT_SCALES.get(name, (None, ""))
            rows.append(dict(symbol=sym, term=name, estimate=r.params[name], std_error=r.bse[name],
                             ci_low=r.conf_int.loc[name, "lo"], ci_high=r.conf_int.loc[name, "hi"],
                             t=r.tvalues[name], p_value=r.pvalues[name], n=r.nobs,
                             window=f"{r.start}–{r.end}", hac_lags=r.hac_lags,
                             effect_scenario=scale_label,
                             effect=(r.params[name] * scale if scale else np.nan),
                             effect_ci_low=(r.conf_int.loc[name, "lo"] * scale if scale else np.nan),
                             effect_ci_high=(r.conf_int.loc[name, "hi"] * scale if scale else np.nan)))
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    mask = t["term"] != "const"
    t["q_value"] = np.nan
    if mask.sum():
        t.loc[mask, "q_value"] = multipletests(t.loc[mask, "p_value"].to_numpy(), method="fdr_bh")[1]
    t["test_family"] = np.where(mask, f"BH over {int(mask.sum())} tests: {family}", "intercept (not in family)")
    return t


def standardized_coefs(r: FitResult, scale_df: pd.DataFrame | None = None) -> pd.Series:
    """β·sd(x)/sd(y). `scale_df` supplies the reference sample for the SDs (default: fit sample)."""
    ref = scale_df if scale_df is not None else r.data
    sy = ref[r.y].std()
    return pd.Series({x: r.params[x] * ref[x].std() / sy for x in r.xs})


# --------------------------------------------------------------------------- interactions
def add_interaction(df: pd.DataFrame, a: str, b: str, center: float | None = None):
    """Return (df with centered b and a×b_c columns, center constant)."""
    d = df.copy()
    c = float(d[b].mean()) if center is None else float(center)
    d[b + "_c"] = d[b] - c
    d[f"{a}_x_{b}"] = d[a] * d[b + "_c"]
    return d, c


def fit_interaction(df, y, a, b, controls, symbol="", lag_rule="fixed", lags=3, window_mask=None):
    """Fit y ~ controls + a + b_c + a×b_c with b centered within the estimation sample."""
    base = df[[y, a, b] + list(controls)].apply(pd.to_numeric, errors="coerce").dropna()
    if window_mask is not None:
        base = base[window_mask.reindex(base.index).fillna(False)]
    if base.empty:
        raise ModelSpecError(f"{symbol}: no months with {y}, {a}, {b} and controls all observed")
    sd_b = base[b].std(ddof=0)
    mean_b = base[b].mean()
    if sd_b < 1e-9:
        raise ModelSpecError(f"{symbol}: {b} is constant over the estimation window ({mean_b:.4g}), so its "
                             f"interaction with {a} is not identified for this single stock. Use a longer window "
                             f"or a panel of stocks (planned extension).")
    d, c = add_interaction(base, a, b)
    xs = list(controls) + [a, b + "_c", f"{a}_x_{b}"]
    r = fit_ols(d, y, xs, symbol, lag_rule, lags, centers={b: c})
    cv = sd_b / abs(mean_b) if mean_b else np.inf
    if cv < 0.05:
        r.warnings.append(f"{b} is nearly constant (coefficient of variation {cv:.1%}); the interaction is weakly "
                          "identified and its uncertainty will be large.")
    return r


def conditional_slope(r: FitResult, a: str, b: str, grid: np.ndarray, level: float = 0.95) -> pd.DataFrame:
    """slope_a(b) = β_a + β_int·(b − c), with Var = V_aa + (b−c)²V_ii + 2(b−c)V_ai (full covariance)."""
    inter = f"{a}_x_{b}"
    c = r.centers[b]
    ba, bi = r.params[a], r.params[inter]
    V = r.cov
    dv = np.asarray(grid, float) - c
    var = V.loc[a, a] + dv ** 2 * V.loc[inter, inter] + 2 * dv * V.loc[a, inter]
    se = np.sqrt(np.maximum(var, 0))
    tq = stats.t.ppf(0.5 + level / 2, r.df_resid)
    slope = ba + bi * dv
    obs = r.data[b] if b in r.data else (r.data[b + "_c"] + c)
    h = 0.5 * obs.std(ddof=0) if obs.std(ddof=0) > 0 else 1e-6
    support = np.array([int(((obs - g).abs() <= h).sum()) for g in grid])
    sparse_thresh = max(5, int(0.05 * len(obs)))
    return pd.DataFrame({b: grid, "slope": slope, "se": se, "lo": slope - tq * se, "hi": slope + tq * se,
                         "support_n": support, "sparse": support < sparse_thresh,
                         "outside_observed": (grid < obs.min()) | (grid > obs.max())})


def scenario_mean(r: FitResult, values: dict, level: float = 0.95) -> dict:
    """Fitted conditional mean at user-set factor values, with a CI from x'Vx (not a prediction interval)."""
    x = pd.Series(0.0, index=r.params.index)
    x["const"] = 1.0
    for k, v in values.items():
        if k in x.index:
            x[k] = v
    est = float(x @ r.params)
    se = float(np.sqrt(max(x @ r.cov.to_numpy() @ x, 0)))
    tq = stats.t.ppf(0.5 + level / 2, r.df_resid)
    extrap = []
    for k, v in values.items():
        if k in r.data and k != r.y:
            lo, hi = r.data[k].min(), r.data[k].max()
            if v < lo or v > hi:
                extrap.append(f"{k}={v:.4g} outside observed [{lo:.4g}, {hi:.4g}]")
    return dict(estimate=est, se=se, lo=est - tq * se, hi=est + tq * se, extrapolation=extrap,
                x=x.to_dict())


# --------------------------------------------------------------------------- rolling
def rolling_fits(df, y, xs, window=60, min_obs=36, lag_rule="fixed", lags=3, interaction=None,
                 slope_at=None, symbol=""):
    """Refit at every window end using only observations inside the trailing window.

    interaction: optional (a, b); b is centered within each window and the reported
    `slope_at_b*` = β_a + β_int·(b* − c_window) keeps a fixed meaning across windows.
    Returns (long DataFrame, list of skipped-window messages).
    """
    months = df.index.sort_values()
    rows, skipped = [], []
    for i in range(len(months)):
        end = months[i]
        start = end - (window - 1)
        w = df[(df.index >= start) & (df.index <= end)]
        if len(w.dropna(subset=[y])) < min_obs:
            continue
        try:
            if interaction:
                a, b = interaction
                r = fit_interaction(w, y, a, b, [x for x in xs if x not in (a, b)], symbol, lag_rule, lags)
            else:
                r = fit_ols(w, y, xs, symbol, lag_rule, lags)
        except ModelSpecError as e:
            skipped.append(f"{end}: {e}")
            continue
        if r.nobs < min_obs:
            continue
        for term in r.params.index:
            if term == "const":
                continue
            rows.append(dict(symbol=symbol, window_end=end, window_start=start, term=term, estimate=r.params[term],
                             lo=r.conf_int.loc[term, "lo"], hi=r.conf_int.loc[term, "hi"], n=r.nobs,
                             adj_r2=r.rsquared_adj, sd_x=r.data[term].std(), sd_y=r.data[y].std()))
        if interaction and slope_at is not None:
            a, b = interaction
            cs = conditional_slope(r, a, b, np.array([slope_at]))
            rows.append(dict(symbol=symbol, window_end=end, window_start=start, term=f"slope_{a}_at_{b}={slope_at:.3g}",
                             estimate=cs["slope"].iloc[0], lo=cs["lo"].iloc[0], hi=cs["hi"].iloc[0], n=r.nobs,
                             adj_r2=r.rsquared_adj, sd_x=np.nan, sd_y=r.data[y].std()))
    out = pd.DataFrame(rows)
    if len(out):
        out["overlap_with_previous"] = (window - 1) / window
    return out, skipped
