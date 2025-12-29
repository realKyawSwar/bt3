from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from alligator_fractal import AlligatorFractal  # noqa: E402
from bt3 import run_backtest  # noqa: E402


def _make_synthetic_df() -> pd.DataFrame:
    n = 300
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    trend = np.concatenate([
        np.linspace(0, 5, 100),
        np.linspace(5, -5, 100),
        np.linspace(-5, 5, 100),
    ])
    close = 100 + trend + 2 * np.sin(np.linspace(0, 20, n))
    open_ = close + 0.1 * np.sin(np.linspace(0, 10, n))
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    volume = np.full(n, 1000.0)
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=idx,
    )


def _trade_count(stats: dict) -> int:
    return int(stats.get("# Trades", stats.get("Trades", 0)))


def test_filters_trade_count_regression() -> None:
    df = _make_synthetic_df()

    warnings.filterwarnings("ignore", category=UserWarning)

    baseline = run_backtest(
        df,
        AlligatorFractal,
        cash=10000,
        commission=0.0,
        strategy_params={"use_htf_bias": False, "use_vol_filter": False},
    )
    baseline_trades = _trade_count(baseline)
    assert baseline_trades == 3

    filtered = run_backtest(
        df,
        AlligatorFractal,
        cash=10000,
        commission=0.0,
        strategy_params={
            "use_htf_bias": True,
            "use_vol_filter": True,
            "htf_tf": "4h",
            "atr_period": 14,
            "atr_long": 100,
        },
    )
    filtered_trades = _trade_count(filtered)
    assert filtered_trades < baseline_trades
