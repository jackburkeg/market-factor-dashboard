import math

import numpy as np
import pandas as pd
import pytest

from mfd.live import attribute, period_moves


def test_period_moves_and_attribution():
    idx = pd.to_datetime(["2026-10-29", "2026-10-30", "2026-11-02", "2026-11-03"])
    series = {"A": pd.Series([10, 10, 10.5, 11.0], index=idx), "CL=F": pd.Series([60, 60, 63, 66.0], index=idx)}
    pm = period_moves(series, pd.Timestamp("2026-10-30")).set_index("symbol")
    assert pm.loc["A", "period_return"] == pytest.approx(0.10)
    assert pm.loc["A", "day_change"] == pytest.approx(11 / 10.5 - 1)
    assert pm.loc["CL=F", "period_log"] == pytest.approx(math.log(1.1))
    betas = pd.DataFrame({"market": [1.0], "oil": [0.5]}, index=["A"])
    att = attribute(pd.Series({"A": 0.10}), betas, {"market": 0.02, "oil": math.log(1.1)}).set_index("symbol")
    assert att.loc["A", "oil"] == pytest.approx(0.5 * math.log(1.1))
    assert att.loc["A", "explained_by_factors"] + att.loc["A", "stock_specific"] == pytest.approx(0.10)
    assert np.isclose(att.loc["A", "market"], 0.02)
