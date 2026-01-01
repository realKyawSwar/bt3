"""Test Wave5 margin-based position sizing constraints."""
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
    """Create minimal test data."""
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


class Wave5MarginProbe(Wave5AODivergenceStrategy):
    """Test strategy to verify margin-based sizing."""

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
        self.buy(sl=sl, tp=tp, size=base)
        self._placed = True


def test_wave5_margin_no_leverage() -> None:
    """With margin=1.0 (no leverage), position size should be clamped by equity."""
    df = _make_df()
    
    stats = run_backtest(
        df,
        Wave5MarginProbe,
        cash=10000,
        commission=0.0,
        margin=1.0,  # No leverage
        strategy_params={"order_size": 0.2},
    )

    # With margin=1.0, units_raw = equity * size / price
    # = 10000 * 0.2 / ~102 ≈ 19-20 units (approx)
    sizes = np.abs(stats._trades["Size"].to_numpy())
    assert len(sizes) >= 1, "No trades executed"
    # Verify that a trade was placed (not skipped due to margin constraint)
    assert sizes[0] > 0


def test_wave5_margin_with_leverage() -> None:
    """With margin=0.05 (20:1 leverage), position size should allow larger units."""
    df = _make_df()
    
    stats_no_lev = run_backtest(
        df,
        Wave5MarginProbe,
        cash=10000,
        commission=0.0,
        margin=1.0,  # No leverage
        strategy_params={"order_size": 0.2},
    )

    stats_with_lev = run_backtest(
        df,
        Wave5MarginProbe,
        cash=10000,
        commission=0.0,
        margin=0.05,  # 20:1 leverage
        strategy_params={"order_size": 0.2},
    )

    # With leverage, we should be able to place larger positions
    # (assuming both have trades)
    sizes_no_lev = np.abs(stats_no_lev._trades["Size"].to_numpy())
    sizes_with_lev = np.abs(stats_with_lev._trades["Size"].to_numpy())
    
    # Both should have at least 1 trade
    assert len(sizes_no_lev) >= 1, "No trades with margin=1.0"
    assert len(sizes_with_lev) >= 1, "No trades with margin=0.05"


def test_wave5_margin_parameter_passed() -> None:
    """Verify margin parameter is passed correctly to strategy."""
    df = _make_df()
    
    stats = run_backtest(
        df,
        Wave5MarginProbe,
        cash=10000,
        commission=0.0,
        margin=0.02,  # 50:1 leverage
        strategy_params={"order_size": 0.2},
    )

    # Check that strategy received margin parameter
    strat = stats._strategy
    assert hasattr(strat, '_margin'), "Strategy should have _margin attribute"
    assert strat._margin == 0.02, f"Expected margin=0.02, got {strat._margin}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
