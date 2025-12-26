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

        params = AlligatorParams(
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

        alligator = compute_alligator_ohlc(df, params)
        fractals = compute_fractals_ohlc(df)
        self.jaw = self.I(lambda x=alligator['jaw'].to_numpy(): x)
        self.teeth = self.I(lambda x=alligator['teeth'].to_numpy(): x)
        self.lips = self.I(lambda x=alligator['lips'].to_numpy(): x)
        self.bullish_fractal = self.I(lambda x=fractals['bullish_fractal_confirmed'].to_numpy(): x)
        self.bearish_fractal = self.I(lambda x=fractals['bearish_fractal_confirmed'].to_numpy(): x)

    def next(self):
        i = len(self.data.Close) - 1
        jaw = self.jaw[-1]
        teeth = self.teeth[-1]
        lips = self.lips[-1]
        if np.isnan(jaw) or np.isnan(teeth) or np.isnan(lips):
            return

        closes = np.asarray(self.data.Close)
        highs = np.asarray(self.data.High)
        lows = np.asarray(self.data.Low)
        jaws = np.asarray(self.jaw)
        teeths = np.asarray(self.teeth)
        lipss = np.asarray(self.lips)

        params = AlligatorParams(
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

        state = _alligator_state(jaw, teeth, lips, closes, jaws, teeths, lipss, params)

        # Manage exits on structure loss
        if self.position:
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

        if state == 'sleeping' or state == 'crossing' or state == 'unknown':
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
                return
            prev_high = highs[i - 1] if i - 1 >= 0 else np.nan
            if not np.isnan(prev_high) and highs[i] >= last_bull and prev_high < last_bull:
                entry = closes[i]
                # Sanitize protective levels
                eps = max(1e-4, abs(entry) * 1e-6)
                if not np.isnan(last_bear):
                    sl = min(last_bear, entry - eps)
                    risk = max(entry - sl, eps)
                    if self.enable_tp:
                        tp = entry + params.tp_rr * risk
                        # Use current close as conservative reference for broker's entry check
                        entry_ref = float(self.data.Close[-1])
                        # Ensure ordering SL < ENTRY < TP with buffer
                        if sl + eps < entry_ref < tp - eps:
                            try:
                                self.buy(sl=sl, tp=tp)
                            except Exception:
                                # Broker rejected TP ordering; fall back to SL-only
                                self.buy(sl=sl)
                            return
                    else:
                        if sl < entry:
                            self.buy(sl=sl)
                            return
                # Fallback to market order without brackets
                self.buy()

        if state == 'eating_down' and not self.position and not np.isnan(last_bear):
            # Require fractal below alligator
            if last_bear >= min(jaw, teeth, lips):
                return
            prev_low = lows[i - 1] if i - 1 >= 0 else np.nan
            if not np.isnan(prev_low) and lows[i] <= last_bear and prev_low > last_bear:
                entry = closes[i]
                # Sanitize protective levels
                eps = max(1e-4, abs(entry) * 1e-6)
                if not np.isnan(last_bull):
                    sl = max(last_bull, entry + eps)
                    risk = max(sl - entry, eps)
                    if self.enable_tp:
                        tp = entry - params.tp_rr * risk
                        # Use current close as conservative reference for broker's entry check
                        entry_ref = float(self.data.Close[-1])
                        # Ensure ordering TP < ENTRY < SL with buffer
                        if tp + eps < entry_ref < sl - eps:
                            try:
                                self.sell(sl=sl, tp=tp)
                            except Exception:
                                # Broker rejected TP ordering; fall back to SL-only
                                self.sell(sl=sl)
                            return
                    else:
                        if entry < sl:
                            self.sell(sl=sl)
                            return
                # Fallback to market order without brackets
                self.sell()
