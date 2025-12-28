# alligator_fractal.py
"""
Alligator + Fractal breakout strategy for backtesting.py

Implements Bill Williams' Alligator (SMMA of median price with shifts)
and 5-bar fractals, with stop-order breakout entries and bracket exits.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    mp = (df["High"].astype(float).to_numpy() + df["Low"].astype(float).to_numpy()) / 2.0

    jaw_raw = _smma_np(mp, params.jaw_period)
    teeth_raw = _smma_np(mp, params.teeth_period)
    lips_raw = _smma_np(mp, params.lips_period)

    out["jaw"] = _shift_forward(jaw_raw, params.jaw_shift)
    out["teeth"] = _shift_forward(teeth_raw, params.teeth_shift)
    out["lips"] = _shift_forward(lips_raw, params.lips_shift)
    return out


def compute_fractals_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    bullish = np.zeros(n, dtype=bool)
    bearish = np.zeros(n, dtype=bool)
    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()

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
    out["bullish_fractal"] = bullish
    out["bearish_fractal"] = bearish
    out["bullish_fractal_price"] = df["High"].where(out["bullish_fractal"])
    out["bearish_fractal_price"] = df["Low"].where(out["bearish_fractal"])
    # confirm 2 bars later (avoid lookahead)
    out["bullish_fractal_confirmed"] = out["bullish_fractal_price"].shift(2)
    out["bearish_fractal_confirmed"] = out["bearish_fractal_price"].shift(2)
    return out


def _alligator_state(
    jaw: float, teeth: float, lips: float,
    closes: np.ndarray, jaws: np.ndarray, teeths: np.ndarray, lipss: np.ndarray,
    params: AlligatorParams
) -> str:
    look = params.spread_lookback
    if len(closes) < look:
        return "unknown"

    window_jaw = jaws[-look:]
    window_teeth = teeths[-look:]
    window_lips = lipss[-look:]
    window_close = closes[-look:]
    if (np.isnan(window_jaw).any() or np.isnan(window_teeth).any() or
            np.isnan(window_lips).any() or np.isnan(window_close).any()):
        return "unknown"

    spreads = (
        np.abs(window_lips - window_teeth) +
        np.abs(window_teeth - window_jaw) +
        np.abs(window_lips - window_jaw)
    ) / (3 * np.abs(window_close))
    mean_spread = float(np.nanmean(spreads))
    if mean_spread < params.min_spread_factor:
        return "sleeping"

    if lips > teeth > jaw:
        return "eating_up"
    if lips < teeth < jaw:
        return "eating_down"
    return "crossing"


class AlligatorFractal(Strategy):
    # Tunables (optimize-friendly)
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
    eps = None

    # Spread model (price units). bt3.run_backtest can set this.
    spread_price = 0.0

    # Optional position sizing (backtesting.py uses `size` if defined)
    # size = 0.1

    def init(self):
        # Cache OHLC arrays once
        self._closes = np.asarray(self.data.Close, dtype=float)
        self._highs = np.asarray(self.data.High, dtype=float)
        self._lows = np.asarray(self.data.Low, dtype=float)

        # Cache params once
        self._cfg = AlligatorParams(
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

        df = pd.DataFrame({
            "High": pd.Series(self._highs),
            "Low": pd.Series(self._lows),
            "Close": pd.Series(self._closes),
        })

        alligator = compute_alligator_ohlc(df, self._cfg)
        fractals = compute_fractals_ohlc(df)

        # Register indicators
        self.jaw = self.I(lambda x=alligator["jaw"].to_numpy(): x)
        self.teeth = self.I(lambda x=alligator["teeth"].to_numpy(): x)
        self.lips = self.I(lambda x=alligator["lips"].to_numpy(): x)
        self.bullish_fractal = self.I(lambda x=fractals["bullish_fractal_confirmed"].to_numpy(): x)
        self.bearish_fractal = self.I(lambda x=fractals["bearish_fractal_confirmed"].to_numpy(): x)

        # Internal state for stop-order realism & bracket arming
        self._last_long_stop = None
        self._last_short_stop = None
        self._arm_brackets = False
        self._next_sl = None
        self._next_tp = None
        self._prev_position_open = False

    def _arm_if_opened(self):
        """Apply SL/TP only after the stop order has turned into a trade (avoid same-bar ambiguity warning)."""
        if self._arm_brackets and self.position and not self._prev_position_open:
            if self._next_sl is not None:
                self.position.sl = self._next_sl
            if self.enable_tp and self._next_tp is not None:
                self.position.tp = self._next_tp
            self._arm_brackets = False
            self._next_sl = None
            self._next_tp = None

    def next(self):
        i = len(self.data.Close) - 1

        # Apply bracket orders right after position opens
        self._arm_if_opened()

        jaw = float(self.jaw[-1])
        teeth = float(self.teeth[-1])
        lips = float(self.lips[-1])
        if np.isnan(jaw) or np.isnan(teeth) or np.isnan(lips):
            self._prev_position_open = bool(self.position)
            return

        jaws = np.asarray(self.jaw, dtype=float)
        teeths = np.asarray(self.teeth, dtype=float)
        lipss = np.asarray(self.lips, dtype=float)

        state = _alligator_state(jaw, teeth, lips, self._closes[: i + 1], jaws[: i + 1], teeths[: i + 1], lipss[: i + 1], self._cfg)

        # Exits: close on sleeping or structure loss
        if self.position:
            self._last_long_stop = None
            self._last_short_stop = None

            if state == "sleeping":
                self.position.close()
                self._prev_position_open = bool(self.position)
                return

            if self.position.is_long and not (lips > teeth > jaw):
                self.position.close()
                self._prev_position_open = bool(self.position)
                return

            if self.position.is_short and not (lips < teeth < jaw):
                self.position.close()
                self._prev_position_open = bool(self.position)
                return

            self._prev_position_open = True
            return

        # No new entries in non-trending states
        if state in ("sleeping", "crossing", "unknown"):
            self._prev_position_open = False
            return

        # Find last confirmed fractals
        last_bull = np.nan
        last_bear = np.nan
        bf = np.asarray(self.bullish_fractal, dtype=float)
        af = np.asarray(self.bearish_fractal, dtype=float)

        for k in range(i, -1, -1):
            if np.isnan(last_bull) and not np.isnan(bf[k]):
                last_bull = bf[k]
            if np.isnan(last_bear) and not np.isnan(af[k]):
                last_bear = af[k]
            if not np.isnan(last_bull) and not np.isnan(last_bear):
                break

        # Spread model (price units)
        spr = float(getattr(self, "spread_price", 0.0))
        half_spread = spr / 2.0

        # Long setup: eating_up + bullish fractal above alligator
        if state == "eating_up" and not np.isnan(last_bull):
            if last_bull <= max(jaw, teeth, lips):
                self._prev_position_open = False
                return

            eps = self.eps if self.eps is not None else max(1e-6, abs(last_bull) * 1e-6)
            entry_stop = float(last_bull) + eps + half_spread

            # Don’t re-place identical stops every bar
            if self._last_long_stop is not None and abs(entry_stop - self._last_long_stop) < eps:
                self._prev_position_open = False
                return

            sl = None
            tp = None
            if not np.isnan(last_bear):
                sl = min(float(last_bear), entry_stop - eps)
                # Ensure SL < entry
                if sl >= entry_stop:
                    sl = entry_stop - eps
                risk = max(entry_stop - sl, eps)

                if self.enable_tp:
                    tp = entry_stop + self._cfg.tp_rr * risk
                    # Ensure ordering
                    if not (sl + eps < entry_stop < tp - eps):
                        tp = None

            # Place stop entry (NO sl/tp here to avoid same-bar contingent warning)
            try:
                self.buy(stop=entry_stop)
                self._last_long_stop = entry_stop

                # Arm bracket for when position opens
                if sl is not None or (self.enable_tp and tp is not None):
                    self._next_sl = sl
                    self._next_tp = tp
                    self._arm_brackets = True
            except Exception:
                self._arm_brackets = False
                self._next_sl = None
                self._next_tp = None

        # Short setup: eating_down + bearish fractal below alligator
        if state == "eating_down" and not np.isnan(last_bear):
            if last_bear >= min(jaw, teeth, lips):
                self._prev_position_open = False
                return

            eps = self.eps if self.eps is not None else max(1e-6, abs(last_bear) * 1e-6)
            entry_stop = float(last_bear) - eps - half_spread

            if self._last_short_stop is not None and abs(entry_stop - self._last_short_stop) < eps:
                self._prev_position_open = False
                return

            sl = None
            tp = None
            if not np.isnan(last_bull):
                sl = max(float(last_bull), entry_stop + eps)
                if entry_stop >= sl:
                    sl = entry_stop + eps
                risk = max(sl - entry_stop, eps)

                if self.enable_tp:
                    tp = entry_stop - self._cfg.tp_rr * risk
                    if not (tp + eps < entry_stop < sl - eps):
                        tp = None

            try:
                self.sell(stop=entry_stop)
                self._last_short_stop = entry_stop

                if sl is not None or (self.enable_tp and tp is not None):
                    self._next_sl = sl
                    self._next_tp = tp
                    self._arm_brackets = True
            except Exception:
                self._arm_brackets = False
                self._next_sl = None
                self._next_tp = None

        self._prev_position_open = bool(self.position)


class AlligatorFractalClassic(AlligatorFractal):
    """
    Classic rules variant:
    - Entry requires 5 consecutive closes above/below teeth.
    - Fractal only needs to be above/below teeth (not all lines).
    - Exits on alligator line cross (ordering break), no "sleeping" exit.
    """

    consecutive_teeth_lookback = 5

    def _has_consecutive_closes_above_teeth(self, i: int) -> bool:
        look = self.consecutive_teeth_lookback
        if i < look:
            return False
        closes = self._closes[i - look:i]
        teeths = np.asarray(self.teeth, dtype=float)[i - look:i]
        if np.isnan(closes).any() or np.isnan(teeths).any():
            return False
        return np.all(closes > teeths)

    def _has_consecutive_closes_below_teeth(self, i: int) -> bool:
        look = self.consecutive_teeth_lookback
        if i < look:
            return False
        closes = self._closes[i - look:i]
        teeths = np.asarray(self.teeth, dtype=float)[i - look:i]
        if np.isnan(closes).any() or np.isnan(teeths).any():
            return False
        return np.all(closes < teeths)

    def next(self):
        i = len(self.data.Close) - 1

        # Apply bracket orders right after position opens
        self._arm_if_opened()

        jaw = float(self.jaw[-1])
        teeth = float(self.teeth[-1])
        lips = float(self.lips[-1])
        if np.isnan(jaw) or np.isnan(teeth) or np.isnan(lips):
            self._prev_position_open = bool(self.position)
            return

        # Exits: close when ordering breaks (no sleeping exit)
        if self.position:
            self._last_long_stop = None
            self._last_short_stop = None

            if self.position.is_long and (lips < teeth or teeth < jaw):
                self.position.close()
                self._prev_position_open = bool(self.position)
                return

            if self.position.is_short and (lips > teeth or teeth > jaw):
                self.position.close()
                self._prev_position_open = bool(self.position)
                return

            self._prev_position_open = True
            return

        # Find last confirmed fractals
        last_bull = np.nan
        last_bear = np.nan
        bf = np.asarray(self.bullish_fractal, dtype=float)
        af = np.asarray(self.bearish_fractal, dtype=float)

        for k in range(i, -1, -1):
            if np.isnan(last_bull) and not np.isnan(bf[k]):
                last_bull = bf[k]
            if np.isnan(last_bear) and not np.isnan(af[k]):
                last_bear = af[k]
            if not np.isnan(last_bull) and not np.isnan(last_bear):
                break

        # Spread model (price units)
        spr = float(getattr(self, "spread_price", 0.0))
        half_spread = spr / 2.0

        # Long setup: closes above teeth + bullish fractal above teeth
        if not np.isnan(last_bull):
            if last_bull <= teeth:
                self._prev_position_open = False
            elif not self._has_consecutive_closes_above_teeth(i):
                self._prev_position_open = False
            else:
                eps = self.eps if self.eps is not None else max(1e-6, abs(last_bull) * 1e-6)
                entry_stop = float(last_bull) + eps + half_spread

                if self._last_long_stop is not None and abs(entry_stop - self._last_long_stop) < eps:
                    self._prev_position_open = False
                else:
                    sl = None
                    tp = None
                    if not np.isnan(last_bear):
                        sl = min(float(last_bear), entry_stop - eps)
                        if sl >= entry_stop:
                            sl = entry_stop - eps
                        risk = max(entry_stop - sl, eps)

                        if self.enable_tp:
                            tp = entry_stop + self._cfg.tp_rr * risk
                            if not (sl + eps < entry_stop < tp - eps):
                                tp = None

                    try:
                        self.buy(stop=entry_stop)
                        self._last_long_stop = entry_stop

                        if sl is not None or (self.enable_tp and tp is not None):
                            self._next_sl = sl
                            self._next_tp = tp
                            self._arm_brackets = True
                    except Exception:
                        self._arm_brackets = False
                        self._next_sl = None
                        self._next_tp = None

        # Short setup: closes below teeth + bearish fractal below teeth
        if not np.isnan(last_bear):
            if last_bear >= teeth:
                self._prev_position_open = False
            elif not self._has_consecutive_closes_below_teeth(i):
                self._prev_position_open = False
            else:
                eps = self.eps if self.eps is not None else max(1e-6, abs(last_bear) * 1e-6)
                entry_stop = float(last_bear) - eps - half_spread

                if self._last_short_stop is not None and abs(entry_stop - self._last_short_stop) < eps:
                    self._prev_position_open = False
                else:
                    sl = None
                    tp = None
                    if not np.isnan(last_bull):
                        sl = max(float(last_bull), entry_stop + eps)
                        if entry_stop >= sl:
                            sl = entry_stop + eps
                        risk = max(sl - entry_stop, eps)

                        if self.enable_tp:
                            tp = entry_stop - self._cfg.tp_rr * risk
                            if not (tp + eps < entry_stop < sl - eps):
                                tp = None

                    try:
                        self.sell(stop=entry_stop)
                        self._last_short_stop = entry_stop

                        if sl is not None or (self.enable_tp and tp is not None):
                            self._next_sl = sl
                            self._next_tp = tp
                            self._arm_brackets = True
                    except Exception:
                        self._arm_brackets = False
                        self._next_sl = None
                        self._next_tp = None

        self._prev_position_open = bool(self.position)
