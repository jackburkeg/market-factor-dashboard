import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mfd.calendar import tsx_sessions  # noqa: E402
from mfd.config import MARKET_TZ  # noqa: E402


@pytest.fixture(scope="session")
def synth():
    from mfd.providers.synthetic import generate
    return generate()


@pytest.fixture(scope="session")
def synth_panel(synth):
    from mfd.prep import build_panel
    return build_panel(synth["prices"], synth["macro"], synth["fundamentals"], list(synth["truth"]["symbol"]), "SYN-MKT")


def make_prices(symbols, start="2015-01-01", end="2018-12-31", seed=0, drift=0.0005, vol=0.01):
    """Simple adjusted-close price frame on TSX sessions."""
    rng = np.random.default_rng(seed)
    s = tsx_sessions(start, end)
    rows = []
    for k, sym in enumerate(symbols):
        r = rng.normal(drift, vol, len(s))
        px = 50 * np.exp(np.cumsum(r))
        rows.append(pd.DataFrame({"symbol": sym, "session_date": s, "adjusted_close": px, "close": px, "dividend": 0.0,
                                  "split_ratio": 1.0, "close_basis": "split_adjusted", "currency": "CAD",
                                  "exchange": "TEST", "source": "test", "retrieved_at": "test"}))
    return pd.concat(rows, ignore_index=True)


def ts(s):
    return pd.Timestamp(s).tz_localize(MARKET_TZ)
