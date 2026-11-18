"""Chronological evaluation: no look-ahead, training-only preprocessing, matched-date metrics."""
import numpy as np
import pandas as pd
import pytest

from mfd import forecast as F

FEATS = ["lag_own", "lag_market", "lag_oil", "lag_debt_to_assets"]


def _sym(synth_panel, s="SYN-EPA"):
    p = synth_panel["panel"]
    return p[p.symbol == s].set_index("month")


def test_training_labels_realized_before_decision(synth_panel):
    fc, _ = F.walk_forward(_sym(synth_panel), FEATS, "SYN-EPA", min_train=60, retune_every=6)
    assert len(fc)
    # the last training row's target month (training_end + 1) must be <= decision month
    assert ((fc["training_end"] + 1) <= fc["decision_month"]).all()
    assert (fc["target_month"] == fc["decision_month"] + 1).all()
    assert (fc["n_train"] >= 60).all()


def test_changing_future_observations_leaves_earlier_forecasts_unchanged(synth_panel):
    d = _sym(synth_panel)
    base, _ = F.walk_forward(d, FEATS, "SYN-EPA", min_train=60, retune_every=1, interaction=("lag_oil", "lag_debt_to_assets"),
                             impute="train_median")
    cut = pd.Period("2021-06", "M")
    d2 = d.copy()
    future = d2.index > cut
    d2.loc[future, "ret"] = d2.loc[future, "ret"] * 5 + 0.3
    d2.loc[future, "lag_own"] = d2.loc[future, "ret"]
    d2.loc[future, "lag_oil"] = -d2.loc[future, "lag_oil"]
    d2["target_next"] = d2["ret"].shift(-1)
    alt, _ = F.walk_forward(d2, FEATS, "SYN-EPA", min_train=60, retune_every=1, interaction=("lag_oil", "lag_debt_to_assets"),
                            impute="train_median")
    # forecasts made at decisions <= cut-1 use only labels <= cut-1+... (target_month <= cut)
    early = lambda f: f[f["decision_month"] < cut].sort_values(["model", "decision_month"])["prediction"].to_numpy()
    assert len(early(base)) > 50
    assert np.allclose(early(base), early(alt))
    late = lambda f: f[f["decision_month"] > cut + 1].sort_values(["model", "decision_month"])["prediction"].to_numpy()
    assert not np.allclose(late(base), late(alt))


def test_preprocessing_fitted_on_training_rows_only(synth_panel):
    d = _sym(synth_panel)
    fc, _ = F.walk_forward(d, FEATS, "SYN-EPA", min_train=60, interaction=("lag_oil", "lag_debt_to_assets"),
                           impute="train_median", retune_every=12)
    r = fc[fc.model == "ols"].iloc[10]
    tr = d[(d.index + 1 <= r["decision_month"]) & d["target_next"].notna()]
    assert r["settings"]["center_lag_debt_to_assets"] == pytest.approx(tr["lag_debt_to_assets"].fillna(tr["lag_debt_to_assets"].median()).mean())
    assert r["settings"]["medians"]["lag_oil"] == pytest.approx(tr["lag_oil"].median())


def test_hist_mean_baseline_is_training_mean(synth_panel):
    d = _sym(synth_panel)
    fc, _ = F.walk_forward(d, FEATS, "SYN-EPA", min_train=60, retune_every=12)
    r = fc[fc.model == "hist_mean"].iloc[5]
    tr = d[(d.index + 1 <= r["decision_month"]) & d["target_next"].notna()].dropna(subset=FEATS)
    assert r["prediction"] == pytest.approx(tr["target_next"].mean())


def _toy():
    rows = []
    for i, m in enumerate(pd.period_range("2020-01", periods=6, freq="M")):
        y = [0.02, -0.01, 0.03, 0.00, -0.02, 0.01][i]
        for model, p in (("hist_mean", 0.005), ("zero", 0.0), ("ols", -y), ("ridge", 0.004)):
            rows.append(dict(symbol="A", target_month=m, model=model, prediction=p, actual=y, error=y - p))
    rows = [r for r in rows if not (r["model"] == "ridge" and r["target_month"] == pd.Period("2020-06", "M"))]
    return pd.DataFrame(rows)


def test_metrics_use_matching_dates_and_keep_negative_r2():
    fc = _toy()
    m = F.metrics(fc).set_index("model")
    assert (m["n_forecasts"] == 5).all()  # ridge missing June → June dropped for all models
    f5 = fc[fc.target_month < pd.Period("2020-06", "M")]
    sse = lambda mm: (f5[f5.model == mm]["error"] ** 2).sum()
    assert m.loc["ols", "oos_r2_vs_hist_mean"] == pytest.approx(1 - sse("ols") / sse("hist_mean"))
    assert m.loc["ols", "oos_r2_vs_hist_mean"] < -1  # much worse than baseline, preserved
    assert m.loc["hist_mean", "oos_r2_vs_hist_mean"] == pytest.approx(0.0)
    assert np.isnan(m.loc["zero", "directional_accuracy"])


def test_empirical_intervals_use_only_past_errors():
    fc = _toy()
    fc = fc[fc.model == "ols"].assign(decision_month=lambda d: d["target_month"] - 1)
    out = F.add_empirical_intervals(fc, 0.8, min_errors=2)
    out = out.sort_values("decision_month")
    # forecast for target 2020-04 (decided 2020-03) may use errors with target <= 2020-03 (3 errors)
    row = out[out.target_month == pd.Period("2020-04", "M")].iloc[0]
    past = out[out.target_month <= pd.Period("2020-03", "M")]["error"]
    assert row["pi_low"] == pytest.approx(row["prediction"] + past.quantile(0.1))
    assert np.isnan(out.iloc[0]["pi_low"])
