from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from backtesting import Strategy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from bt3 import run_backtest  # noqa: E402


class SpreadCaptureStrategy(Strategy):
    spread_price: float = 0.0
    recorded_spreads: list[float] = []

    def __init__(self, broker, data, params):
        super().__init__(broker, data, params)
        spread = float(params.get("spread_price", 0.0) or 0.0)
        self.spread_price = spread
        SpreadCaptureStrategy.recorded_spreads.append(spread)

    def init(self):
        pass

    def next(self):
        pass


def _make_df() -> pd.DataFrame:
    periods = 50
    idx = pd.date_range("2022-01-01", periods=periods, freq="h")
    base = np.linspace(1.0, 2.0, periods)
    df = pd.DataFrame(
        {
            "Open": base + 0.01,
            "High": base + 0.02,
            "Low": base - 0.02,
            "Close": base,
            "Volume": np.ones(periods),
        },
        index=idx,
    )
    df.attrs["symbol"] = "USDJPY"
    return df


def test_spread_price_does_not_leak_between_runs() -> None:
    df = _make_df()
    SpreadCaptureStrategy.recorded_spreads = []

    run_backtest(df, SpreadCaptureStrategy, spread_pips=30, pip_size=0.01, commission=0.0)
    run_backtest(df, SpreadCaptureStrategy, spread_pips=None, commission=0.0)

    assert SpreadCaptureStrategy.recorded_spreads == [0.3, 0.0]
