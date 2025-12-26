"""
Alligator + Fractal breakout strategy for backtesting.py

Implements Bill Williams' Alligator (SMMA of median price with shifts)
and 5-bar fractals, with breakout entries and simple bracket exits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from backtesting import Strategy


@dataclass
class AlligatorParams:
    jaw_period: int = 13
    jaw_shift: int = 8
    teeth_period: int = 8
    teeth_shift: int = 5
    lips_period: int = 5
    lips_shift: int = 3

    min_spread_factor: float = 0.0005
    spread_lookback: int = 5
    tp_rr: float = 2.0
    
    def keys(self):
        """For compatibility with backtesting.py stats display"""
        return self.__dataclass_fields__.keys()
    
    def values(self):
        """For compatibility with backtesting.py stats display"""
        return (getattr(self, k) for k in self.keys())


def _smma_np(arr: np.ndarray, length: int) -> np.ndarray:
    out = np.full_like(arr, np.nan, dtype=float)
    if length <= 0 or len(arr) < length:
        return out

    # first value: SMA of first length items
    first = np.nanmean(arr[:length])
    out[length - 1] = first
    prev = first
    for i in range(length, len(arr)):
        x = arr[i]
        if np.isnan(x):
            out[i] = prev
        else:
            prev = (prev * (length - 1) + x) / length
            out[i] = prev
    return out


def _shift_forward(values: np.ndarray, shift: int) -> np.ndarray:
    if shift <= 0:
        return values.copy()
    res = np.full(len(values), np.nan)
    if shift < len(values):
        res[shift:] = values[:-shift]
    return res


def compute_alligator_ohlc(df: pd.DataFrame, params: AlligatorParams) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    mp = (df['High'].astype(float).to_numpy() + df['Low'].astype(float).to_numpy()) / 2.0

    jaw_raw = _smma_np(mp, params.jaw_period)
    teeth_raw = _smma_np(mp, params.teeth_period)
    lips_raw = _smma_np(mp, params.lips_period)

    out['jaw'] = _shift_forward(jaw_raw, params.jaw_shift)
    out['teeth'] = _shift_forward(teeth_raw, params.teeth_shift)
    out['lips'] = _shift_forward(lips_raw, params.lips_shift)
    return out


def compute_fractals_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    bullish = np.zeros(n, dtype=bool)
    bearish = np.zeros(n, dtype=bool)
    high = df['High'].to_numpy()
    low = df['Low'].to_numpy()

    # 5-bar fractals
    for i in range(2, n - 2):
        h = high[i]
        l = low[i]
        if (
            h > high[i - 1] and h > high[i - 2] and
            h > high[i + 1] and h > high[i + 2]
        ):
            bullish[i] = True
        if (
            l < low[i - 1] and l < low[i - 2] and
            l < low[i + 1] and l < low[i + 2]
        ):
            bearish[i] = True

    out = pd.DataFrame(index=df.index)
    out['bullish_fractal'] = bullish
    out['bearish_fractal'] = bearish
    out['bullish_fractal_price'] = df['High'].where(out['bullish_fractal'])
    out['bearish_fractal_price'] = df['Low'].where(out['bearish_fractal'])
    out['bullish_fractal_confirmed'] = out['bullish_fractal_price'].shift(2)
    out['bearish_fractal_confirmed'] = out['bearish_fractal_price'].shift(2)
    return out


def _alligator_state(
    jaw: float, teeth: float, lips: float,
    closes: np.ndarray, jaws: np.ndarray, teeths: np.ndarray, lipss: np.ndarray,
    params: AlligatorParams
) -> str:
    # Unknown if insufficient lookback or NaNs
    look = params.spread_lookback
    if len(closes) < look:
        return 'unknown'
    window_jaw = jaws[-look:]
    window_teeth = teeths[-look:]
    window_lips = lipss[-look:]
    window_close = closes[-look:]
    if (np.isnan(window_jaw).any() or np.isnan(window_teeth).any() or
            np.isnan(window_lips).any() or np.isnan(window_close).any()):
        return 'unknown'

    spreads = (
        np.abs(window_lips - window_teeth) +
        np.abs(window_teeth - window_jaw) +
        np.abs(window_lips - window_jaw)
    ) / (3 * np.abs(window_close))
    mean_spread = np.nanmean(spreads)
    if mean_spread < params.min_spread_factor:
        return 'sleeping'

    if lips > teeth > jaw:
        return 'eating_up'
    if lips < teeth < jaw:
        return 'eating_down'
    return 'crossing'


class AlligatorFractal(Strategy):
    jaw_period = 13
    jaw_shift = 8
    teeth_period = 8
    teeth_shift = 5
    lips_period = 5
    lips_shift = 3
    min_spread_factor = 0.0005
    spread_lookback = 5
    tp_rr = 2.0
    enable_tp = False

    def init(self):
        df = pd.DataFrame({
            'High': pd.Series(self.data.High),
            'Low': pd.Series(self.data.Low),
            'Close': pd.Series(self.data.Close),
        })

        # Cache params once
        self._params = AlligatorParams(
            jaw_period=self.jaw_period,
            jaw_shift=self.jaw_shift,
            teeth_period=self.teeth_period,
            teeth_shift=self.teeth_shift,
            lips_period=self.lips_period,
            lips_shift=self.lips_shift,
            min_spread_factor=self.min_spread_factor,
            spread_lookback=self.spread_lookback,
            tp_rr=self.tp_rr,
        )

        alligator = compute_alligator_ohlc(df, self._params)
        fractals = compute_fractals_ohlc(df)
        self.jaw = self.I(lambda x=alligator['jaw'].to_numpy(): x)
        self.teeth = self.I(lambda x=alligator['teeth'].to_numpy(): x)
        self.lips = self.I(lambda x=alligator['lips'].to_numpy(): x)
        self.bullish_fractal = self.I(lambda x=fractals['bullish_fractal_confirmed'].to_numpy(): x)
        self.bearish_fractal = self.I(lambda x=fractals['bearish_fractal_confirmed'].to_numpy(): x)

        # Cache numpy arrays
        self._closes = np.asarray(self.data.Close)
        self._highs = np.asarray(self.data.High)
        self._lows = np.asarray(self.data.Low)

        # Track pending stop orders to avoid duplicates
        self._last_long_stop = None
        self._last_short_stop = None
        # Arm SL/TP after stop entry fills
        self._arm_brackets = False
        self._next_sl = None
        self._next_tp = None
        self._prev_position_open = False

    def next(self):
        i = len(self.data.Close) - 1
        jaw = self.jaw[-1]
        teeth = self.teeth[-1]
        lips = self.lips[-1]
        if np.isnan(jaw) or np.isnan(teeth) or np.isnan(lips):
            return

        # If a new position just opened, arm SL/TP now to avoid same-bar contingent order warning
        if self._arm_brackets and self.position and not self._prev_position_open:
            if self._next_sl is not None:
                self.position.sl = self._next_sl
            if self.enable_tp and self._next_tp is not None:
                self.position.tp = self._next_tp
            self._arm_brackets = False
            self._next_sl = None
            self._next_tp = None

        # Use cached arrays
        jaws = np.asarray(self.jaw)
        teeths = np.asarray(self.teeth)
        lipss = np.asarray(self.lips)

        state = _alligator_state(jaw, teeth, lips, self._closes, jaws, teeths, lipss, self._params)

        # Manage exits on structure loss
        if self.position:
            # Reset pending order trackers when position exists
            self._last_long_stop = None
            self._last_short_stop = None
            
            if state == 'sleeping':
                self.position.close()
                return
            # close if structure lost
            if self.position.is_long and not (lips > teeth > jaw):
                self.position.close()
                return
            if self.position.is_short and not (lips < teeth < jaw):
                self.position.close()
                return
            # Position open, no new entry orders
            self._prev_position_open = True
            return

        # No position - check if we just closed one and reset trackers
        if not self.position and (self._last_long_stop is not None or self._last_short_stop is not None):
            # Position was closed, reset trackers
            if not any(trade.is_open for trade in self.trades if hasattr(trade, 'is_open')):
                self._last_long_stop = None
                self._last_short_stop = None

        if state == 'sleeping' or state == 'crossing' or state == 'unknown':
            self._prev_position_open = bool(self.position)
            return

        last_bull = np.nan
        last_bear = np.nan
        # scan backwards for last confirmed fractals
        bf = np.asarray(self.bullish_fractal)
        af = np.asarray(self.bearish_fractal)
        for k in range(i, -1, -1):
            if np.isnan(last_bull) and not np.isnan(bf[k]):
                last_bull = bf[k]
            if np.isnan(last_bear) and not np.isnan(af[k]):
                last_bear = af[k]
            if not np.isnan(last_bull) and not np.isnan(last_bear):
                break

        if state == 'eating_up' and not self.position and not np.isnan(last_bull):
            # Require fractal above alligator
            if last_bull <= max(jaw, teeth, lips):
                self._prev_position_open = bool(self.position)
                return
            
            # Use stop order at fractal level
            eps = max(1e-4, abs(last_bull) * 1e-6)
            entry_stop = last_bull + eps
            
            # Check if we already have a pending order at this level
            if self._last_long_stop is not None and abs(entry_stop - self._last_long_stop) < eps:
                self._prev_position_open = bool(self.position)
                return  # Don't place duplicate order
            
            # Build SL/TP
            sl = None
            tp = None
            if not np.isnan(last_bear):
                sl = min(last_bear, entry_stop - eps)
                risk = max(entry_stop - sl, eps)
                if self.enable_tp:
                    tp = entry_stop + self._params.tp_rr * risk
                    # Ensure ordering: sl < entry_stop < tp
                    if not (sl + eps < entry_stop < tp - eps):
                        tp = None  # Invalid ordering, drop TP
                # Ensure sl < entry_stop
                if sl >= entry_stop:
                    sl = entry_stop - eps
            
            # Place stop order without contingent SL/TP; arm brackets after fill
            try:
                self.buy(stop=entry_stop)
                self._last_long_stop = entry_stop
                self._next_sl = sl
                self._next_tp = tp if self.enable_tp else None
                self._arm_brackets = True
            except Exception:
                self._arm_brackets = False
                self._next_sl = None
                self._next_tp = None

        if state == 'eating_down' and not self.position and not np.isnan(last_bear):
            # Require fractal below alligator
            if last_bear >= min(jaw, teeth, lips):
                self._prev_position_open = bool(self.position)
                return
            
            # Use stop order at fractal level
            eps = max(1e-4, abs(last_bear) * 1e-6)
            entry_stop = last_bear - eps
            
            # Check if we already have a pending order at this level
            if self._last_short_stop is not None and abs(entry_stop - self._last_short_stop) < eps:
                self._prev_position_open = bool(self.position)
                return  # Don't place duplicate order
            
            # Build SL/TP
            sl = None
            tp = None
            if not np.isnan(last_bull):
                sl = max(last_bull, entry_stop + eps)
                risk = max(sl - entry_stop, eps)
                if self.enable_tp:
                    tp = entry_stop - self._params.tp_rr * risk
                    # Ensure ordering: tp < entry_stop < sl
                    if not (tp + eps < entry_stop < sl - eps):
                        tp = None  # Invalid ordering, drop TP
                # Ensure entry_stop < sl
                if entry_stop >= sl:
                    sl = entry_stop + eps
            
            # Place stop order without contingent SL/TP; arm brackets after fill
            try:
                self.sell(stop=entry_stop)
                self._last_short_stop = entry_stop
                self._next_sl = sl
                self._next_tp = tp if self.enable_tp else None
                self._arm_brackets = True
            except Exception:
                self._arm_brackets = False
                self._next_sl = None
                self._next_tp = None

        self._prev_position_open = bool(self.position)
