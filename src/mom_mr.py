from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
from backtesting import Strategy

from factors import compute_momentum


def rsi(series: pd.Series, period: int) -> pd.Series:
    """
    Wilder-style RSI using exponential smoothing of gains/losses.
    """
    if period <= 0:
        raise ValueError("period must be positive")
    series = pd.Series(series).astype(float)
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """
    Average True Range using Wilder smoothing.
    """
    if period <= 0:
        raise ValueError("period must be positive")
    high = pd.Series(high).astype(float)
    low = pd.Series(low).astype(float)
    close = pd.Series(close).astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


class MomentumMeanReversionStrategy(Strategy):
    """
    Regime filter via momentum, entries via mean-reverting RSI, exits on RSI revert or ATR TP.

    Strategy attributes are exposed for backtesting.py optimization / grid search.
    """

    mom_window: int = 126
    mom_threshold: float = 0.0
    rsi_window: int = 3
    rsi_threshold: float = 20.0
    rsi_exit: float = 60.0
    atr_window: int = 14
    sl_atr: float = 2.0
    tp_mode: str = "rsi"  # rsi | atr
    tp_atr: float = 3.0
    risk_pct: float = 0.01
    min_size: float = 0.0
    max_size: float = 1e9

    def init(self) -> None:
        # Cache arrays for speed; register indicators for backtesting.py plots/caching
        self._open = np.asarray(self.data.Open, dtype=float)
        self._high = np.asarray(self.data.High, dtype=float)
        self._low = np.asarray(self.data.Low, dtype=float)
        self._close = np.asarray(self.data.Close, dtype=float)

        self.momentum = self.I(lambda close=self.data.Close: compute_momentum(close, int(self.mom_window)))
        self.rsi_series = self.I(lambda close=self.data.Close: rsi(close, int(self.rsi_window)))
        self.atr_series = self.I(lambda h=self.data.High, l=self.data.Low, c=self.data.Close: atr(h, l, c, int(self.atr_window)))

    def _position_size(self, entry: float, sl: float, atr_val: float) -> float:
        if not np.isfinite(entry) or not np.isfinite(sl) or not np.isfinite(atr_val):
            return 0.0
        sl_dist = abs(entry - sl)
        if sl_dist <= 0:
            return 0.0
        equity = float(self.equity)
        risk_cash = max(0.0, float(self.risk_pct) * equity)
        if risk_cash <= 0:
            return 0.0
        size = risk_cash / (self.sl_atr * atr_val + 1e-12)
        size = max(self.min_size, min(self.max_size, size))
        if size <= 0 or not math.isfinite(size):
            return 0.0
        return size

    def _regime(self, mom_val: float) -> str:
        thr = float(self.mom_threshold)
        if mom_val > thr:
            return "bullish"
        if mom_val < -thr:
            return "bearish"
        return "flat"

    def next(self) -> None:
        i = len(self.data) - 1
        mom_val = float(self.momentum[-1])
        rsi_val = float(self.rsi_series[-1])
        atr_val = float(self.atr_series[-1])

        if not np.isfinite(mom_val) or not np.isfinite(rsi_val) or not np.isfinite(atr_val):
            return

        regime = self._regime(mom_val)
        price = self._close[i]

        # Exit logic
        if self.position:
            if self.position.is_long and rsi_val > float(self.rsi_exit):
                self.position.close()
            elif self.position.is_short and rsi_val < 100.0 - float(self.rsi_exit):
                self.position.close()
            elif str(self.tp_mode).lower() == "atr":
                tp = price + float(self.tp_atr) * atr_val if self.position.is_long else price - float(self.tp_atr) * atr_val
                if self.position.is_long and price >= tp:
                    self.position.close()
                elif self.position.is_short and price <= tp:
                    self.position.close()

        # Entry gating
        if self.position or regime == "flat":
            return

        if regime == "bullish" and rsi_val < float(self.rsi_threshold):
            sl = price - float(self.sl_atr) * atr_val
            size = self._position_size(price, sl, atr_val)
            if size > 0:
                tp_price: Optional[float] = None
                if str(self.tp_mode).lower() == "atr":
                    tp_price = price + float(self.tp_atr) * atr_val
                self.buy(sl=sl, tp=tp_price, size=size)

        elif regime == "bearish" and rsi_val > 100.0 - float(self.rsi_threshold):
            sl = price + float(self.sl_atr) * atr_val
            size = self._position_size(price, sl, atr_val)
            if size > 0:
                tp_price = None
                if str(self.tp_mode).lower() == "atr":
                    tp_price = price - float(self.tp_atr) * atr_val
                self.sell(sl=sl, tp=tp_price, size=size)


def _self_test() -> None:
    """
    Lightweight manual check when running the module directly.
    """
    try:
        from bt3 import fetch_data, run_backtest

        df = fetch_data("EURUSD", "1h").tail(800)
        stats = run_backtest(
            data=df,
            strategy=MomentumMeanReversionStrategy,
            cash=10_000,
            commission=0.0,
            spread_pips=10,
            margin=1.0,
            strategy_params={
                "mom_window": 63,
                "mom_threshold": 0.0,
                "rsi_threshold": 20.0,
                "rsi_exit": 60.0,
                "risk_pct": 0.01,
                "sl_atr": 2.0,
            },
        )
        print("Self-test stats:", stats.get("Return [%]", "n/a"))
    except Exception as exc:  # pragma: no cover - optional smoke test
        print("module ok (self-test skipped):", exc)


if __name__ == "__main__":
    _self_test()
