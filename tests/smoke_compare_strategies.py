from __future__ import annotations

import numpy as np
import pandas as pd
import sys
from pathlib import Path

from backtesting import Backtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alligator_fractal import AlligatorFractal, AlligatorFractalClassic


def _make_sample_data(rows: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range(start="2020-01-01", periods=rows, freq="H")

    price = 100 + rng.standard_normal(rows).cumsum()
    open_ = price + rng.standard_normal(rows) * 0.1
    close = price + rng.standard_normal(rows) * 0.1
    high = np.maximum(open_, close) + np.abs(rng.standard_normal(rows) * 0.2)
    low = np.minimum(open_, close) - np.abs(rng.standard_normal(rows) * 0.2)

    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": rng.integers(100, 1000, size=rows),
        },
        index=dates,
    )


def _run_strategy(strategy) -> dict:
    data = _make_sample_data()
    bt = Backtest(data, strategy, cash=10000, commission=0.0, exclusive_orders=True)
    return bt.run()


def main() -> None:
    strict_stats = _run_strategy(AlligatorFractal)
    classic_stats = _run_strategy(AlligatorFractalClassic)

    required_keys = ["Return [%]", "# Trades", "Win Rate [%]"]
    for key in required_keys:
        if key not in strict_stats or key not in classic_stats:
            raise AssertionError(f"Missing required stat '{key}' in output")

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
