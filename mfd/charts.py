"""Plotly figures. Colors follow a validated categorical order (fixed per entity, never cycled);
one y-axis per chart; thin marks; recessive grid; hover on every chart."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
NEUTRAL = "#8a8983"
FONT = "Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif"


def color_map(entities) -> dict:
    """Stable entity→color mapping in fixed slot order (a 9th+ entity reuses neutral gray)."""
    return {e: (SERIES[i] if i < len(SERIES) else NEUTRAL) for i, e in enumerate(entities)}


def _style(fig: go.Figure, title: str = "", height: int = 380, legend: bool = True, synthetic: bool = False):
    if synthetic:
        title = f"{title}  ·  SYNTHETIC DATA" if title else "SYNTHETIC DATA"
    top = 100 if legend else 56
    fig.update_layout(title=dict(text=title, font=dict(size=15, color=TEXT), x=0, xanchor="left",
                                 yref="container", y=1, yanchor="top", pad=dict(t=12)),
                      font=dict(family=FONT, size=12, color=TEXT2), paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                      height=height + (top - 56), margin=dict(l=60, r=20, t=top, b=50), showlegend=legend,
                      legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0, title=None),
                      hoverlabel=dict(font_family=FONT, bgcolor="white"))
    fig.update_xaxes(showgrid=False, linecolor=GRID, ticks="outside", tickcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, gridwidth=1, zeroline=True, zerolinecolor="#c9c8c1", zerolinewidth=1,
                     linecolor=GRID)
    return fig


def _to_ts(idx):
    return idx.to_timestamp(how="end") if isinstance(idx, pd.PeriodIndex) else idx


# --------------------------------------------------------------------------- overview
def cumulative_returns(mret: pd.DataFrame, colors: dict, synthetic=False):
    w = mret.pivot(index="month", columns="symbol", values="ret").sort_index()
    fig = go.Figure()
    for s in w.columns:
        r = w[s]
        first = r.first_valid_index()
        curve = (1 + r.loc[first:].fillna(0)).cumprod()
        curve[r.loc[first:].isna()] = np.nan
        fig.add_trace(go.Scatter(x=_to_ts(curve.index), y=curve.values, name=s, mode="lines",
                                 line=dict(width=2, color=colors.get(s, NEUTRAL)),
                                 hovertemplate=f"{s}<br>%{{x|%Y-%m}}<br>growth of 1: %{{y:.2f}}<extra></extra>"))
    fig.update_yaxes(type="log", title="Growth of 1 (log scale, total return)")
    return _style(fig, "Cumulative monthly total return", 400, synthetic=synthetic)


def factor_history(mdesc: pd.DataFrame, labels: dict, units: dict, synthetic=False):
    cols = [c for c in mdesc.columns if not c.endswith("__level")]
    fig = make_subplots(rows=len(cols), cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=[f"{labels.get(c, c)} — {units.get(c, '')}" for c in cols])
    for i, c in enumerate(cols, start=1):
        s = mdesc[c]
        fig.add_trace(go.Bar(x=_to_ts(s.index), y=s.values, marker_color=SERIES[0], name=c,
                             hovertemplate=f"{c}<br>%{{x|%Y-%m}}: %{{y:.4f}}<extra></extra>"), row=i, col=1)
    fig = _style(fig, "Monthly factor changes (by observation date)", 150 * max(len(cols), 1) + 80, legend=False,
                 synthetic=synthetic)
    fig.update_annotations(font=dict(size=12, color=TEXT2))
    return fig


# --------------------------------------------------------------------------- drivers
def coefficient_plot(eff: pd.DataFrame, colors: dict, use_effect=True, synthetic=False, truth: pd.DataFrame | None = None):
    e = eff[eff["term"] != "const"].copy()
    terms = list(dict.fromkeys(e["term"]))
    fig = make_subplots(rows=1, cols=len(terms), shared_yaxes=True, horizontal_spacing=0.03,
                        subplot_titles=[t for t in terms])
    for j, t in enumerate(terms, start=1):
        g = e[e["term"] == t]
        val = "effect" if use_effect and g["effect"].notna().all() else "estimate"
        lo, hi = ("effect_ci_low", "effect_ci_high") if val == "effect" else ("ci_low", "ci_high")
        scale = 100 if val == "effect" else 1
        x = g[val] * scale
        fig.add_trace(go.Scatter(
            x=x, y=g["symbol"], mode="markers", marker=dict(size=9, color=[colors.get(s, NEUTRAL) for s in g["symbol"]],
                                                            line=dict(color=SURFACE, width=2)),
            error_x=dict(type="data", symmetric=False, array=(g[hi] * scale - x), arrayminus=(x - g[lo] * scale),
                         color=TEXT2, thickness=1.5, width=0),
            customdata=np.c_[g["p_value"], g["q_value"], g["n"], g[lo] * scale, g[hi] * scale],
            hovertemplate=("%{y}<br>" + ("effect (pp): " if val == "effect" else "coef: ") +
                           "%{x:.3f}<br>95% CI [%{customdata[3]:.3f}, %{customdata[4]:.3f}]<br>"
                           "p=%{customdata[0]:.3f}, q=%{customdata[1]:.3f}, n=%{customdata[2]}<extra></extra>"),
            showlegend=False), row=1, col=j)
        if truth is not None and val == "estimate":
            tt = truth.set_index("symbol")
            tcol = {"debt_to_assets_c": "debt_to_assets", "profitability_c": "profitability"}.get(t, t)
            if tcol in tt:
                tv = tt[tcol].reindex(g["symbol"])
                fig.add_trace(go.Scatter(x=tv.values, y=g["symbol"], mode="markers", name="true value (synthetic)",
                                         marker=dict(symbol="x-thin", size=10, line=dict(width=2, color=TEXT)),
                                         showlegend=(j == 1),
                                         hovertemplate="true coefficient %{x:.3f}<extra></extra>"), row=1, col=j)
        fig.add_vline(x=0, line=dict(color="#c9c8c1", width=1), row=1, col=j)
        sub = e.loc[e["term"] == t, "effect_scenario"].iloc[0] if val == "effect" else "coefficient (raw units)"
        fig.update_xaxes(title_text=sub, title_font=dict(size=10), row=1, col=j)
    fig = _style(fig, "Estimated associations with monthly return (95% HAC confidence intervals)",
                 120 + 34 * e["symbol"].nunique() + 80, legend=truth is not None, synthetic=synthetic)
    fig.update_annotations(font=dict(size=12, color=TEXT))
    return fig


def correlation_heatmap(corr: pd.DataFrame, title="Feature correlations (estimation sample)"):
    fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
                               colorscale=[[0, "#2a78d6"], [0.5, "#f0efec"], [1, "#e34948"]],
                               text=np.round(corr.values, 2), texttemplate="%{text}",
                               hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>"))
    return _style(fig, title, 340, legend=False)


def fitted_vs_actual(r, synthetic=False):
    idx = _to_ts(r.data.index)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=r.data[r.y] * 100, name="actual", mode="lines", line=dict(width=2, color=SERIES[0]),
                             hovertemplate="%{x|%Y-%m} actual %{y:.2f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=idx, y=r.fitted * 100, name="fitted (in-sample)", mode="lines",
                             line=dict(width=2, color=SERIES[1]),
                             hovertemplate="%{x|%Y-%m} fitted %{y:.2f}%<extra></extra>"))
    fig.update_yaxes(title="monthly return (%)")
    return _style(fig, f"{r.symbol}: actual vs in-sample fitted (descriptive, not a forecast)", 320, synthetic=synthetic)


# --------------------------------------------------------------------------- interactions
def interaction_curve(cs: pd.DataFrame, b: str, a_label: str, scale: float, scale_label: str, synthetic=False,
                      truth_line: tuple | None = None):
    x = cs[b]
    y, lo, hi = cs["slope"] * scale * 100, cs["lo"] * scale * 100, cs["hi"] * scale * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[hi, lo[::-1]], fill="toself", fillcolor="rgba(42,120,214,0.15)",
                             line=dict(width=0), hoverinfo="skip", name="95% CI (full covariance)"))
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(width=2, color=SERIES[0]), name="conditional effect",
                             customdata=np.c_[lo, hi, cs["support_n"]],
                             hovertemplate=(f"{b}=%{{x:.3f}}<br>effect of {scale_label}: %{{y:.2f}} pp<br>"
                                            "95% CI [%{customdata[0]:.2f}, %{customdata[1]:.2f}]<br>"
                                            "months within ±0.5 SD: %{customdata[2]}<extra></extra>")))
    sp = cs[cs["sparse"] | cs["outside_observed"]]
    if len(sp):
        fig.add_trace(go.Scatter(x=sp[b], y=sp["slope"] * scale * 100, mode="markers", name="sparse / outside data",
                                 marker=dict(size=7, color="white", line=dict(color=NEUTRAL, width=1.5)),
                                 hovertemplate=f"{b}=%{{x:.3f}}: few nearby observations<extra></extra>"))
    if truth_line is not None:
        tb, ti, tc = truth_line
        fig.add_trace(go.Scatter(x=x, y=(tb + ti * (x - tc)) * scale * 100, mode="lines", name="true (synthetic)",
                                 line=dict(width=1.5, color=TEXT, dash="dot")))
    fig.update_xaxes(title=f"{b} (known before the month)")
    fig.update_yaxes(title=f"return effect of {scale_label} (pp)")
    return _style(fig, f"Conditional {a_label} sensitivity across observed {b}", 400, synthetic=synthetic)


def joint_distribution(df: pd.DataFrame, a: str, b: str, synthetic=False):
    fig = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4], subplot_titles=[f"{a} vs {b}", f"distribution of {b}"])
    fig.add_trace(go.Scatter(x=df[b], y=df[a], mode="markers", marker=dict(size=8, color=SERIES[0], opacity=0.6,
                                                                            line=dict(color=SURFACE, width=1)),
                             text=[str(m) for m in df.index],
                             hovertemplate=f"%{{text}}<br>{b}=%{{x:.3f}}<br>{a}=%{{y:.4f}}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Histogram(x=df[b], nbinsx=25, marker=dict(color=SERIES[0], line=dict(color=SURFACE, width=2)),
                               hovertemplate=f"{b} %{{x}}: %{{y}} months<extra></extra>"), row=1, col=2)
    fig.update_xaxes(title_text=b, row=1, col=1)
    fig.update_yaxes(title_text=a, row=1, col=1)
    fig.update_xaxes(title_text=b, row=1, col=2)
    return _style(fig, "Observed joint support of the interacting factors", 330, legend=False, synthetic=synthetic)


# --------------------------------------------------------------------------- rolling
def rolling_plot(roll: pd.DataFrame, terms, colors: dict, standardized=False, synthetic=False):
    fig = make_subplots(rows=len(terms), cols=1, shared_xaxes=True, vertical_spacing=0.07, subplot_titles=list(terms))
    for i, t in enumerate(terms, start=1):
        g = roll[roll["term"] == t]
        for s, gs in g.groupby("symbol"):
            gs = gs.sort_values("window_end")
            k = gs["std_scale"] if standardized and "std_scale" in gs else 1.0
            x = _to_ts(pd.PeriodIndex(gs["window_end"]))
            col = colors.get(s, NEUTRAL)
            rgb = tuple(int(col[j:j + 2], 16) for j in (1, 3, 5))
            fig.add_trace(go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[gs["hi"] * k, (gs["lo"] * k)[::-1]], fill="toself",
                                     fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.12)", line=dict(width=0),
                                     hoverinfo="skip", showlegend=False, legendgroup=s), row=i, col=1)
            fig.add_trace(go.Scatter(x=x, y=gs["estimate"] * k, mode="lines", line=dict(width=2, color=col), name=s,
                                     legendgroup=s, showlegend=(i == 1),
                                     customdata=np.c_[gs["lo"] * k, gs["hi"] * k, gs["n"], gs["window_start"].astype(str)],
                                     hovertemplate=(f"{s}<br>window %{{customdata[3]}} → %{{x|%Y-%m}}<br>"
                                                    "estimate %{y:.3f} [%{customdata[0]:.3f}, %{customdata[1]:.3f}]"
                                                    "<br>n=%{customdata[2]}<extra></extra>")), row=i, col=1)
    ttl = "Rolling-window coefficient stability" + (" (standardized on a shared full-period scale)" if standardized else " (raw units)")
    fig = _style(fig, ttl, 230 * len(terms) + 80, synthetic=synthetic)
    fig.update_annotations(font=dict(size=12, color=TEXT))
    return fig


def rolling_n_plot(roll: pd.DataFrame, colors: dict):
    g = roll.drop_duplicates(["symbol", "window_end"])
    fig = go.Figure()
    for s, gs in g.groupby("symbol"):
        gs = gs.sort_values("window_end")
        fig.add_trace(go.Scatter(x=_to_ts(pd.PeriodIndex(gs["window_end"])), y=gs["n"], mode="lines", name=s,
                                 line=dict(width=2, color=colors.get(s, NEUTRAL)),
                                 hovertemplate=f"{s} %{{x|%Y-%m}}: n=%{{y}}<extra></extra>"))
    fig.update_yaxes(title="observations in window", rangemode="tozero")
    return _style(fig, "Observations per window", 260)


# --------------------------------------------------------------------------- forecasts
MODEL_COLORS = {"ols": SERIES[0], "ridge": SERIES[1], "hist_mean": SERIES[2], "zero": NEUTRAL}


def forecast_history(fc: pd.DataFrame, symbol: str, models, synthetic=False, show_pi=None):
    g = fc[fc["symbol"] == symbol]
    fig = go.Figure()
    act = g.drop_duplicates("target_month").sort_values("target_month")
    x = _to_ts(pd.PeriodIndex(act["target_month"]))
    if show_pi is not None:
        p = g[g["model"] == show_pi].sort_values("target_month")
        px = _to_ts(pd.PeriodIndex(p["target_month"]))
        ok = p["pi_low"].notna()
        fig.add_trace(go.Scatter(x=np.r_[px[ok], px[ok][::-1]], y=np.r_[p["pi_high"][ok] * 100, (p["pi_low"][ok] * 100)[::-1]],
                                 fill="toself", fillcolor="rgba(42,120,214,0.12)", line=dict(width=0), hoverinfo="skip",
                                 name=f"{int(p['pi_level'].iloc[0] * 100)}% empirical prediction interval ({show_pi})"))
    fig.add_trace(go.Scatter(x=x, y=act["actual"] * 100, mode="lines", name="realized return", line=dict(width=2, color=TEXT),
                             hovertemplate="%{x|%Y-%m} realized %{y:.2f}%<extra></extra>"))
    for m in models:
        p = g[g["model"] == m].sort_values("target_month")
        fig.add_trace(go.Scatter(x=_to_ts(pd.PeriodIndex(p["target_month"])), y=p["prediction"] * 100, mode="lines",
                                 name=m, line=dict(width=2, color=MODEL_COLORS.get(m, NEUTRAL)),
                                 hovertemplate=f"{m} %{{x|%Y-%m}} forecast %{{y:.2f}}%<extra></extra>"))
    fig.update_yaxes(title="next-month return (%)")
    return _style(fig, f"{symbol}: out-of-sample forecasts vs realized (target month)", 380, synthetic=synthetic)


def cumulative_sse_gap(fc: pd.DataFrame, models, baseline="hist_mean", synthetic=False):
    """Cumulative (baseline SSE − model SSE) over matched dates: rising = model beating baseline."""
    from .forecast import matched
    f = matched(fc)
    f = f.sort_values("target_month")
    b = f[f["model"] == baseline].groupby("target_month")["error"].apply(lambda e: (e ** 2).sum())
    fig = go.Figure()
    for m in models:
        if m == baseline:
            continue
        s = f[f["model"] == m].groupby("target_month")["error"].apply(lambda e: (e ** 2).sum())
        gap = (b - s).cumsum()
        fig.add_trace(go.Scatter(x=_to_ts(pd.PeriodIndex(gap.index)), y=gap.values, mode="lines", name=m,
                                 line=dict(width=2, color=MODEL_COLORS.get(m, NEUTRAL)),
                                 hovertemplate=f"{m} %{{x|%Y-%m}}: %{{y:.4f}}<extra></extra>"))
    fig.update_yaxes(title="cumulative SSE(baseline) − SSE(model)")
    return _style(fig, "Cumulative squared-error advantage over the historical-mean baseline (above 0 = better)", 340,
                  synthetic=synthetic)


def error_distribution(fc: pd.DataFrame, models, synthetic=False):
    fig = go.Figure()
    for m in models:
        e = fc.loc[fc["model"] == m, "error"] * 100
        fig.add_trace(go.Histogram(x=e, name=m, opacity=0.55, nbinsx=40, marker_color=MODEL_COLORS.get(m, NEUTRAL),
                                   hovertemplate=f"{m} error %{{x}} pp: %{{y}}<extra></extra>"))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="forecast error = realized − predicted (pp)")
    return _style(fig, "Distribution of out-of-sample forecast errors", 320, synthetic=synthetic)


def actual_vs_pred(fc: pd.DataFrame, model: str, colors: dict, synthetic=False):
    g = fc[fc["model"] == model]
    fig = go.Figure()
    for s, gs in g.groupby("symbol"):
        fig.add_trace(go.Scatter(x=gs["prediction"] * 100, y=gs["actual"] * 100, mode="markers", name=s,
                                 marker=dict(size=8, color=colors.get(s, NEUTRAL), opacity=0.7, line=dict(color=SURFACE, width=1)),
                                 text=gs["target_month"].astype(str),
                                 hovertemplate=f"{s} %{{text}}<br>pred %{{x:.2f}}%, realized %{{y:.2f}}%<extra></extra>"))
    lim = float(np.nanmax(np.abs(g[["prediction", "actual"]].to_numpy()))) * 100 if len(g) else 1
    fig.add_trace(go.Scatter(x=[-lim, lim], y=[-lim, lim], mode="lines", line=dict(color=NEUTRAL, width=1, dash="dot"),
                             name="45° line", hoverinfo="skip"))
    fig.update_xaxes(title="predicted (%)")
    fig.update_yaxes(title="realized (%)")
    return _style(fig, f"Realized vs predicted next-month return — {model}", 400, synthetic=synthetic)


# --------------------------------------------------------------------------- portfolio
def weights_bar(w: pd.DataFrame, synthetic=False):
    fig = go.Figure()
    for i, c in enumerate(w.columns):
        fig.add_trace(go.Bar(x=w.index, y=w[c] * 100, name=c, marker=dict(color=SERIES[i], line=dict(color=SURFACE, width=2)),
                             hovertemplate=f"{c} %{{x}}: %{{y:.1f}}%<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.25)
    fig.update_yaxes(title="weight (%)")
    return _style(fig, "Current allocation calculation (not a recommendation)", 340, synthetic=synthetic)


def growth_plot(res: pd.DataFrame, kind="net", synthetic=False):
    fig = go.Figure()
    cmap = color_map(sorted(res["strategy"].unique()))
    for s, g in res.groupby("strategy"):
        g = g.sort_values("execution")
        v = (1 + g[f"{kind}_return"]).cumprod()
        x = list(g["execution"].iloc[:1]) + list(g["next_execution"])
        y = [1.0] + list(v)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=s, line=dict(width=2, color=cmap[s]),
                                 hovertemplate=f"{s} %{{x|%Y-%m-%d}}: %{{y:.3f}}<extra></extra>"))
    fig.update_yaxes(title=f"growth of 1 ({kind} of costs)")
    return _style(fig, f"Historical evaluation, execution-to-execution ({kind})", 380, synthetic=synthetic)


def drawdown_plot(res: pd.DataFrame, synthetic=False):
    fig = go.Figure()
    cmap = color_map(sorted(res["strategy"].unique()))
    for s, g in res.groupby("strategy"):
        g = g.sort_values("execution")
        v = (1 + g["net_return"]).cumprod()
        dd = v / v.cummax() - 1
        fig.add_trace(go.Scatter(x=g["next_execution"], y=dd * 100, mode="lines", name=s, line=dict(width=2, color=cmap[s]),
                                 hovertemplate=f"{s} %{{x|%Y-%m-%d}}: %{{y:.1f}}%<extra></extra>"))
    fig.update_yaxes(title="drawdown from peak (%)")
    return _style(fig, "Drawdowns (net)", 300, synthetic=synthetic)


# --------------------------------------------------------------------------- attribution
COMPONENT_LABELS = {"market": "Market", "oil": "Oil", "rate": "Interest rate", "fx": "USD/CAD",
                    "control": "AI-leader control (NVDA)", "y10": "10-yr yield", "ffr": "Fed funds", "vix": "VIX",
                    "usd": "US dollar",
                    "stock_specific": "Stock-specific (unexplained)"}


def attribution_bar(att: pd.DataFrame, components, title: str, synthetic=False):
    """Horizontal stacked bars: factor contributions + stock-specific remainder; a marker for the actual return."""
    fig = go.Figure()
    comps = [c for c in components if c in att] + ["stock_specific"]
    for i, c in enumerate(comps):
        color = NEUTRAL if c == "stock_specific" else SERIES[i]
        fig.add_trace(go.Bar(y=att["symbol"], x=att[c] * 100, name=COMPONENT_LABELS.get(c, c), orientation="h",
                             marker=dict(color=color, line=dict(color=SURFACE, width=2)),
                             hovertemplate=f"%{{y}} · {COMPONENT_LABELS.get(c, c)}: %{{x:+.2f}} pp<extra></extra>"))
    fig.add_trace(go.Scatter(y=att["symbol"], x=att["actual"] * 100, mode="markers", name="Actual return",
                             marker=dict(symbol="diamond", size=11, color=TEXT, line=dict(color=SURFACE, width=2)),
                             hovertemplate="%{y} actual return: %{x:+.2f}%<extra></extra>"))
    fig.update_layout(barmode="relative")
    fig.update_xaxes(title="contribution to return (percentage points)", zeroline=True, zerolinecolor="#9a998f")
    return _style(fig, title, 90 + 46 * len(att), synthetic=synthetic)
