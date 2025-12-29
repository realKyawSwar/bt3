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


def compute_atr_ohlc(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def compute_htf_bias(
    df_ltf: pd.DataFrame,
    htf_rule: str = "4H",
    params: AlligatorParams | None = None,
) -> pd.Series:
    if not isinstance(df_ltf.index, pd.DatetimeIndex):
        raise ValueError("df_ltf must use a DatetimeIndex for HTF resampling.")

    ohlc = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    rule = htf_rule.lower()
    df_htf = df_ltf.resample(rule, label="right", closed="right").agg(ohlc)
    df_htf = df_htf.dropna(subset=["Open", "High", "Low", "Close"])

    cfg = params or AlligatorParams()
    cfg_bias = AlligatorParams(
        jaw_period=cfg.jaw_period,
        jaw_shift=0,
        teeth_period=cfg.teeth_period,
        teeth_shift=0,
        lips_period=cfg.lips_period,
        lips_shift=0,
        min_spread_factor=cfg.min_spread_factor,
        spread_lookback=cfg.spread_lookback,
        tp_rr=cfg.tp_rr,
    )
    alligator = compute_alligator_ohlc(df_htf, cfg_bias)

    lips = alligator["lips"]
    teeth = alligator["teeth"]
    jaw = alligator["jaw"]

    bias = pd.Series("neutral", index=df_htf.index, dtype=object)
    bullish = (lips > teeth) & (teeth > jaw)
    bearish = (lips < teeth) & (teeth < jaw)
    bias = bias.where(~bullish, "bullish")
    bias = bias.where(~bearish, "bearish")

    bias = bias.shift(1)
    bias = bias.reindex(df_ltf.index, method="ffill")
    return bias.fillna("neutral")


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

    htf_tf = "4H"
    use_htf_bias = True
    use_vol_filter = True
    atr_period = 14
    atr_long = 100

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

        index = pd.Index(self.data.index)
        volumes = getattr(self.data, "Volume", np.zeros(len(self.data)))
        df = pd.DataFrame(
            {
                "Open": pd.Series(self.data.Open, index=index, dtype=float),
                "High": pd.Series(self._highs, index=index, dtype=float),
                "Low": pd.Series(self._lows, index=index, dtype=float),
                "Close": pd.Series(self._closes, index=index, dtype=float),
                "Volume": pd.Series(volumes, index=index, dtype=float),
            },
            index=index,
        )

        alligator = compute_alligator_ohlc(df, self._cfg)
        fractals = compute_fractals_ohlc(df)

        # Register indicators
        self.jaw = self.I(lambda x=alligator["jaw"].to_numpy(): x)
        self.teeth = self.I(lambda x=alligator["teeth"].to_numpy(): x)
        self.lips = self.I(lambda x=alligator["lips"].to_numpy(): x)
        self.bullish_fractal = self.I(lambda x=fractals["bullish_fractal_confirmed"].to_numpy(): x)
        self.bearish_fractal = self.I(lambda x=fractals["bearish_fractal_confirmed"].to_numpy(): x)

        self._htf_bias = None
        if self.use_htf_bias:
            htf_bias = compute_htf_bias(df, self.htf_tf, self._cfg)
            self._htf_bias = htf_bias.to_numpy(dtype=object)

        self._vol_ok = None
        if self.use_vol_filter:
            atr = compute_atr_ohlc(df, self.atr_period)
            atr_sma = atr.rolling(self.atr_long, min_periods=self.atr_long).mean()
            self._vol_ok = (atr > 0.9 * atr_sma).fillna(False).to_numpy()

        bias_series = (
            pd.Series(self._htf_bias, index=df.index, dtype=object)
            if self._htf_bias is not None
            else pd.Series("neutral", index=df.index, dtype=object)
        )
        bias_counts = bias_series.value_counts(normalize=True)
        bias_dist = {
            "bullish": float(bias_counts.get("bullish", 0.0)),
            "bearish": float(bias_counts.get("bearish", 0.0)),
            "neutral": float(bias_counts.get("neutral", 0.0)),
        }
        print(f"HTF bias %: {bias_dist}")

        vol_ok_series = (
            pd.Series(self._vol_ok, index=df.index, dtype=bool)
            if self._vol_ok is not None
            else pd.Series(True, index=df.index, dtype=bool)
        )
        print(f"VOL ok %: {vol_ok_series.mean():.2f}")
        both_ok = (bias_series != "neutral") & vol_ok_series
        print(f"BOTH ok %: {both_ok.mean():.2f}")

        # Internal state for stop-order realism & bracket arming
        self._last_long_stop = None
        self._last_short_stop = None
        self._arm_brackets = False
        self._next_sl = None
        self._next_tp = None
        self._prev_position_open = False

    def _entry_permissions(self, i: int) -> tuple[bool, bool]:
        allow_long = True
        allow_short = True

        if self.use_htf_bias and self._htf_bias is not None:
            bias = self._htf_bias[i]
            if bias == "bullish":
                allow_short = False
            elif bias == "bearish":
                allow_long = False
            else:
                allow_long = False
                allow_short = False

        if self.use_vol_filter and self._vol_ok is not None:
            if not bool(self._vol_ok[i]):
                allow_long = False
                allow_short = False

        return allow_long, allow_short

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

        allow_long, allow_short = self._entry_permissions(i)
        if not allow_long and not allow_short:
            self._prev_position_open = False
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
        if allow_long and state == "eating_up" and not np.isnan(last_bull):
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
        if allow_short and state == "eating_down" and not np.isnan(last_bear):
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

        allow_long, allow_short = self._entry_permissions(i)
        if not allow_long and not allow_short:
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

        # Long setup: closes above teeth + bullish fractal above teeth
        if allow_long and not np.isnan(last_bull):
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
        if allow_short and not np.isnan(last_bear):
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


class AlligatorFractalPullback(AlligatorFractal):
    """
    Trend + Pullback variant (Structure-First):
    - Requires pullback into Teeth zone before allowing the strict fractal breakout entry.
    - Uses existing HTF bias + vol filter gating via _entry_permissions().
    - Keeps Strict exits/trailing/brackets unchanged.
    """

    pullback_k_atr = 0.5
    require_touch_teeth = False

    def init(self):
        super().init()

        index = pd.Index(self.data.index)
        df = pd.DataFrame(
            {
                "Open": pd.Series(self.data.Open, index=index, dtype=float),
                "High": pd.Series(self._highs, index=index, dtype=float),
                "Low": pd.Series(self._lows, index=index, dtype=float),
                "Close": pd.Series(self._closes, index=index, dtype=float),
            },
            index=index,
        )
        atr = compute_atr_ohlc(df, self.atr_period)
        self._atr = atr.to_numpy(dtype=float)

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

        allow_long, allow_short = self._entry_permissions(i)
        if not allow_long and not allow_short:
            self._prev_position_open = False
            return

        # No new entries in non-trending states
        if state in ("sleeping", "crossing", "unknown"):
            self._prev_position_open = False
            return

        atr_now = self._atr[i] if i < len(self._atr) else np.nan
        if np.isnan(atr_now):
            self._prev_position_open = False
            return

        if self.require_touch_teeth:
            long_pullback = self._lows[i] <= teeth
            short_pullback = self._highs[i] >= teeth
        else:
            long_pullback = self._lows[i] <= teeth + self.pullback_k_atr * atr_now
            short_pullback = self._highs[i] >= teeth - self.pullback_k_atr * atr_now

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

        # Long setup: eating_up + bullish fractal above alligator + pullback
        if allow_long and state == "eating_up" and long_pullback and not np.isnan(last_bull):
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

        # Short setup: eating_down + bearish fractal below alligator + pullback
        if allow_short and state == "eating_down" and short_pullback and not np.isnan(last_bear):
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
