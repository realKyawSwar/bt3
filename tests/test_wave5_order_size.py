from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from bt3 import run_backtest  # noqa: E402
from wave5_ao import Wave5AODivergenceStrategy  # noqa: E402


def _make_df() -> pd.DataFrame:
    periods = 80
    idx = pd.date_range("2023-01-01", periods=periods, freq="h")
    base = np.linspace(100.0, 105.0, periods)
    noise = 0.05 * np.sin(np.linspace(0, 6.28, periods))
    close = base + noise
    open_ = close + 0.01
    high = np.maximum(open_, close) + 0.1
    low = np.minimum(open_, close) - 0.1
    volume = np.ones(periods)
    return pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }, index=idx)


class Wave5OrderSizeProbe(Wave5AODivergenceStrategy):
    """Minimal subclass to place deterministic orders for size checks."""

    def init(self):
        super().init()
        self._placed = False

    def next(self):
        if self._placed or len(self.data) < 10:
            return

        price = float(self.data.Close[-1])
        sl = price - 1.0
        tp = price + 1.5
        base = float(self.order_size)

        if bool(self.tp_split):
            ratio = float(self.tp_split_ratio)
            self.buy(sl=sl, tp=tp, size=base * ratio)
            self.buy(sl=sl, tp=tp + 0.5, size=base * (1.0 - ratio))
        else:
            self.buy(sl=sl, tp=tp, size=base)

        self._placed = True


def test_wave5_order_size_default_applied() -> None:
    df = _make_df()

    stats = run_backtest(
        df,
        Wave5OrderSizeProbe,
        cash=10000,
        commission=0.0,
        strategy_params={"tp_split": False},
    )

    sizes = np.abs(stats._trades["Size"].to_numpy())
    assert len(sizes) == 1
    assert sizes[0] == pytest.approx(0.2)


def test_wave5_order_size_tp_split_scales() -> None:
    df = _make_df()
    order_size = 0.2
    ratio = 0.3

    stats = run_backtest(
        df,
        Wave5OrderSizeProbe,
        cash=10000,
        commission=0.0,
        strategy_params={"tp_split": True, "tp_split_ratio": ratio, "order_size": order_size},
    )

    sizes = np.sort(np.abs(stats._trades["Size"].to_numpy()))
    expected = np.sort(np.array([order_size * ratio, order_size * (1.0 - ratio)]))

    assert len(sizes) == 2
    assert np.allclose(sizes, expected)
    assert np.isclose(float(sizes.sum()), order_size)
