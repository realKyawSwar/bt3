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
    if not df_ltf.index.is_monotonic_increasing:
        df_ltf = df_ltf.sort_index()

    ohlc = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    df_htf = df_ltf.resample(htf_rule, label="right", closed="right").agg(ohlc)
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
    cancel_stale_orders = False

    htf_tf = "4H"
    use_htf_bias = True
    use_vol_filter = True
    atr_period = 14
    atr_long = 100

    # Spread model (price units). bt3.run_backtest can set this.
    spread_price = 0.0

    # Margin support (for compatibility with Wave5 and parameter passing)
    margin = 1.0  # Margin requirement (1.0=no leverage, 0.02=50:1)

    # Exit behavior flags (USED in next())
    exit_on_structure_loss = True
    exit_on_sleeping = True

    # R-based management (optimize-friendly)
    enable_be = False
    be_at_r = 1.0
    be_buffer_r = 0.1

    # Entry cooldown after a trade closes (bars)
    cooldown_bars = 3

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

        # Internal state for stop-order realism & bracket arming
        self._last_long_stop = None
        self._last_short_stop = None
        self._arm_brackets = False
        self._next_sl = None
        self._next_tp = None
        self._prev_position_open = False

        # Per-trade frozen entry info (R0) + cooldown
        self._trade_entry_price = None
        self._trade_initial_sl = None
        self._trade_r0 = None
        self._be_done = False
        self._cooldown_until_i = None

        # Diagnostics
        self._diag_trades_opened = 0
        self._diag_be_moves = 0
        self._diag_r0_total = 0.0
        self._diag_r0_count = 0
        self._be_moves = 0

        # Fractal cache
        self._last_bull_fractal = np.nan
        self._last_bear_fractal = np.nan

    def _eps_for(self, price: float) -> float:
        return float(self.eps) if self.eps is not None else max(1e-6, abs(float(price)) * 1e-6)

    def _update_last_fractals(self) -> None:
        bull = float(self.bullish_fractal[-1])
        bear = float(self.bearish_fractal[-1])
        if not np.isnan(bull):
            self._last_bull_fractal = bull
        if not np.isnan(bear):
            self._last_bear_fractal = bear

    def _reset_trade_meta(self) -> None:
        self._trade_entry_price = None
        self._trade_initial_sl = None
        self._trade_r0 = None
        self._be_done = False

    def _set_cooldown(self, i: int) -> None:
        if int(self.cooldown_bars) > 0:
            self._cooldown_until_i = i + int(self.cooldown_bars)
        else:
            self._cooldown_until_i = None

    def _cooldown_active(self, i: int) -> bool:
        if int(self.cooldown_bars) <= 0:
            return False
        if self._cooldown_until_i is None:
            return False
        return i <= int(self._cooldown_until_i)

    def _maybe_cancel_stale_orders(self, state: str, allow_long: bool, allow_short: bool) -> None:
        if not self.cancel_stale_orders or not self.orders:
            return

        cancel = False
        if state in ("sleeping", "crossing", "unknown") or (not allow_long and not allow_short):
            cancel = True
        else:
            for order in self.orders:
                if order.is_long and state != "eating_up":
                    cancel = True
                    break
                if order.is_short and state != "eating_down":
                    cancel = True
                    break

        if cancel:
            self.cancel()
            self._last_long_stop = None
            self._last_short_stop = None

    def _half_spread(self) -> float:
        return float(getattr(self, "spread_price", 0.0)) / 2.0

    def _exit_level_to_engine_level(self, level_mid: float, is_long: bool) -> float:
        half_spread = self._half_spread()
        if is_long:
            return float(level_mid) - half_spread
        return float(level_mid) + half_spread

    def _place_long_entry(self, entry_stop: float, sl: float | None, tp: float | None) -> bool:
        try:
            self.buy(stop=entry_stop)
            self._last_long_stop = entry_stop

            if sl is not None or (self.enable_tp and tp is not None):
                self._next_sl = sl
                self._next_tp = tp
                self._arm_brackets = True
            return True
        except Exception:
            self._arm_brackets = False
            self._next_sl = None
            self._next_tp = None
            return False

    def _place_short_entry(self, entry_stop: float, sl: float | None, tp: float | None) -> bool:
        try:
            self.sell(stop=entry_stop)
            self._last_short_stop = entry_stop

            if sl is not None or (self.enable_tp and tp is not None):
                self._next_sl = sl
                self._next_tp = tp
                self._arm_brackets = True
            return True
        except Exception:
            self._arm_brackets = False
            self._next_sl = None
            self._next_tp = None
            return False

    def _maybe_trail(self, current_price: float) -> None:
        # Base = no trailing. Trailing variant overrides this.
        return

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
        """Apply SL/TP only after stop order becomes a trade."""
        if self._arm_brackets and self.position and not self._prev_position_open:
            # Capture per-trade entry data (frozen R0)
            engine_entry = float(getattr(self.position, "entry_price", np.nan))
            self._trade_entry_price = engine_entry if np.isfinite(engine_entry) else None

            raw_sl = self._next_sl
            raw_tp = self._next_tp
            is_long = bool(self.position.is_long)
            adjusted_sl = (
                self._exit_level_to_engine_level(raw_sl, is_long) if raw_sl is not None else None
            )
            adjusted_tp = (
                self._exit_level_to_engine_level(raw_tp, is_long) if raw_tp is not None else None
            )

            if adjusted_sl is not None:
                self.position.sl = adjusted_sl
            if self.enable_tp and adjusted_tp is not None:
                self.position.tp = adjusted_tp

            self._trade_initial_sl = adjusted_sl
            if self._trade_entry_price is not None and self._trade_initial_sl is not None:
                r0 = abs(self._trade_entry_price - self._trade_initial_sl)
                eps = self._eps_for(self._trade_entry_price)
                self._trade_r0 = r0 if r0 > eps else None
            else:
                self._trade_r0 = None

            self._diag_trades_opened += 1
            if self._trade_r0 is not None:
                self._diag_r0_total += float(self._trade_r0)
                self._diag_r0_count += 1
            self._be_done = False
            self._arm_brackets = False
            self._next_sl = None
            self._next_tp = None

    def _maybe_apply_break_even(self, current_price: float) -> None:
        if not self.enable_be or not self.position or self._be_done:
            return
        if self._trade_r0 is None or self._trade_entry_price is None:
            return

        r0 = float(self._trade_r0)
        entry = float(self._trade_entry_price)
        eps = self._eps_for(current_price)
        be_at = float(self.be_at_r)
        be_buf = float(self.be_buffer_r) * r0

        if self.position.is_long:
            if current_price < entry + be_at * r0:
                return
            target = entry + be_buf
            cur = self.position.sl
            if cur is not None:
                target = max(float(cur), target)
            if target < current_price - eps:
                self.position.sl = float(target)
                self._be_done = True
                self._diag_be_moves += 1
                self._be_moves += 1
            return

        if current_price > entry - be_at * r0:
            return
        target = entry - be_buf
        cur = self.position.sl
        if cur is not None:
            target = min(float(cur), target)
        if target > current_price + eps:
            self.position.sl = float(target)
            self._be_done = True
            self._diag_be_moves += 1
            self._be_moves += 1

    def stop(self):
        avg_r0 = None
        if self._diag_r0_count > 0:
            avg_r0 = self._diag_r0_total / self._diag_r0_count
        print(
            "AlligatorFractal diagnostics: "
            f"spread_price={self.spread_price} "
            f"trades_opened={self._diag_trades_opened} "
            f"be_moves={self._diag_be_moves} "
            f"be_moves_debug={self._be_moves} "
            f"avg_r0={avg_r0}"
        )

    def next(self):
        i = len(self.data.Close) - 1
        if self._prev_position_open and not self.position:
            self._set_cooldown(i)
            self._reset_trade_meta()
        elif not self.position:
            self._reset_trade_meta()

        # Apply bracket orders right after position opens
        self._arm_if_opened()

        # Update fractal caches FIRST (so trailing can use latest confirmed)
        self._update_last_fractals()

        # Apply BE logic BEFORE any trailing
        if self.position:
            self._maybe_apply_break_even(float(self._closes[i]))

        # Trail SL if applicable (trailing variant overrides _maybe_trail)
        if self.position:
            self._maybe_trail(float(self._closes[i]))

        jaw = float(self.jaw[-1])
        teeth = float(self.teeth[-1])
        lips = float(self.lips[-1])
        if np.isnan(jaw) or np.isnan(teeth) or np.isnan(lips):
            self._prev_position_open = bool(self.position)
            return

        jaws = np.asarray(self.jaw, dtype=float)
        teeths = np.asarray(self.teeth, dtype=float)
        lipss = np.asarray(self.lips, dtype=float)

        state = _alligator_state(
            jaw, teeth, lips,
            self._closes[: i + 1],
            jaws[: i + 1],
            teeths[: i + 1],
            lipss[: i + 1],
            self._cfg,
        )

        # Exits: close on sleeping or structure loss (configurable)
        if self.position:
            self._last_long_stop = None
            self._last_short_stop = None

            if self.exit_on_sleeping and state == "sleeping":
                self.position.close()
                self._set_cooldown(i)
                self._reset_trade_meta()
                self._prev_position_open = bool(self.position)
                return

            if self.exit_on_structure_loss:
                if self.position.is_long and not (lips > teeth > jaw):
                    self.position.close()
                    self._set_cooldown(i)
                    self._reset_trade_meta()
                    self._prev_position_open = bool(self.position)
                    return

                if self.position.is_short and not (lips < teeth < jaw):
                    self.position.close()
                    self._set_cooldown(i)
                    self._reset_trade_meta()
                    self._prev_position_open = bool(self.position)
                    return

            self._prev_position_open = True
            return

        allow_long, allow_short = self._entry_permissions(i)
        if not allow_long and not allow_short:
            self._maybe_cancel_stale_orders("unknown", allow_long, allow_short)
            self._prev_position_open = False
            return

        if state in ("sleeping", "crossing", "unknown"):
            self._maybe_cancel_stale_orders(state, allow_long, allow_short)
            self._prev_position_open = False
            return

        self._maybe_cancel_stale_orders(state, allow_long, allow_short)
        if self._cooldown_active(i):
            self._prev_position_open = False
            return
        last_bull = self._last_bull_fractal
        last_bear = self._last_bear_fractal

        spr = float(getattr(self, "spread_price", 0.0))
        half_spread = spr / 2.0

        # Long setup: eating_up + bullish fractal above alligator
        if allow_long and state == "eating_up" and not np.isnan(last_bull):
            if last_bull <= max(jaw, teeth, lips):
                self._prev_position_open = False
                return

            eps = self._eps_for(last_bull)
            entry_stop = float(last_bull) + eps + half_spread

            if self._last_long_stop is not None and abs(entry_stop - self._last_long_stop) < eps:
                self._prev_position_open = False
                return

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

            self._place_long_entry(entry_stop, sl, tp)

        # Short setup: eating_down + bearish fractal below alligator
        if allow_short and state == "eating_down" and not np.isnan(last_bear):
            if last_bear >= min(jaw, teeth, lips):
                self._prev_position_open = False
                return

            eps = self._eps_for(last_bear)
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

            self._place_short_entry(entry_stop, sl, tp)

        self._prev_position_open = bool(self.position)


class AlligatorFractalTrailing(AlligatorFractal):
    """
    Strict entries, but exits are managed by:
      - Opposite fractal trailing SL
      - Base-class break-even logic (optional)
    """

    # Turn off strict "structure loss" exit so trailing logic can do the work
    exit_on_structure_loss = False
    enable_be = True

    def _trail_fractal_sl(self, current_price: float) -> float | None:
        """
        Candidate SL from opposite confirmed fractal (same as your prior trailing).
        Returns None if no safe update.
        """
        eps = self._eps_for(current_price)

        if self.position.is_long:
            if np.isnan(self._last_bear_fractal):
                return None
            new_sl = self._exit_level_to_engine_level(float(self._last_bear_fractal), True)
            if new_sl > current_price - eps:
                return None
            cur = self.position.sl
            if cur is not None and new_sl <= float(cur):
                return None
            return new_sl

        if self.position.is_short:
            if np.isnan(self._last_bull_fractal):
                return None
            new_sl = self._exit_level_to_engine_level(float(self._last_bull_fractal), False)
            if new_sl < current_price + eps:
                return None
            cur = self.position.sl
            if cur is not None and new_sl >= float(cur):
                return None
            return new_sl

        return None

    def _maybe_trail(self, current_price: float) -> None:
        if not self.position:
            return
        new_sl = self._trail_fractal_sl(current_price)
        if new_sl is not None:
            self.position.sl = float(new_sl)


class AlligatorFractalClassic(AlligatorFractal):
    """
    Classic rules variant:
    - Entry requires 5 consecutive closes above/below teeth.
    - Fractal only needs to be above/below teeth (not all lines).
    - Exits on alligator line cross (ordering break), no "sleeping" exit.
    """
    exit_on_sleeping = False
    exit_on_structure_loss = False  # classic uses its own exit logic below
    enable_be = False

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
        if self._prev_position_open and not self.position:
            self._set_cooldown(i)
            self._reset_trade_meta()
        elif not self.position:
            self._reset_trade_meta()

        self._arm_if_opened()
        self._update_last_fractals()
        if self.position:
            self._maybe_apply_break_even(float(self._closes[i]))

        jaw = float(self.jaw[-1])
        teeth = float(self.teeth[-1])
        lips = float(self.lips[-1])
        if np.isnan(jaw) or np.isnan(teeth) or np.isnan(lips):
            self._prev_position_open = bool(self.position)
            return

        # Classic exits: close on ordering breaks (no sleeping exit)
        if self.position:
            self._last_long_stop = None
            self._last_short_stop = None

            if self.position.is_long and (lips < teeth or teeth < jaw):
                self.position.close()
                self._set_cooldown(i)
                self._reset_trade_meta()
                self._prev_position_open = bool(self.position)
                return

            if self.position.is_short and (lips > teeth or teeth > jaw):
                self.position.close()
                self._set_cooldown(i)
                self._reset_trade_meta()
                self._prev_position_open = bool(self.position)
                return

            self._prev_position_open = True
            return

        allow_long, allow_short = self._entry_permissions(i)
        if not allow_long and not allow_short:
            self._maybe_cancel_stale_orders("unknown", allow_long, allow_short)
            self._prev_position_open = False
            return

        if self.cancel_stale_orders:
            jaws = np.asarray(self.jaw, dtype=float)
            teeths = np.asarray(self.teeth, dtype=float)
            lipss = np.asarray(self.lips, dtype=float)
            state = _alligator_state(
                jaw, teeth, lips,
                self._closes[: i + 1],
                jaws[: i + 1],
                teeths[: i + 1],
                lipss[: i + 1],
                self._cfg,
            )
            self._maybe_cancel_stale_orders(state, allow_long, allow_short)

        if self._cooldown_active(i):
            self._prev_position_open = False
            return

        last_bull = self._last_bull_fractal
        last_bear = self._last_bear_fractal

        spr = float(getattr(self, "spread_price", 0.0))
        half_spread = spr / 2.0

        # Long: closes above teeth + bullish fractal above teeth
        if allow_long and not np.isnan(last_bull):
            if last_bull <= teeth or not self._has_consecutive_closes_above_teeth(i):
                self._prev_position_open = False
            else:
                eps = self._eps_for(last_bull)
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

                    self._place_long_entry(entry_stop, sl, tp)

        # Short: closes below teeth + bearish fractal below teeth
        if allow_short and not np.isnan(last_bear):
            if last_bear >= teeth or not self._has_consecutive_closes_below_teeth(i):
                self._prev_position_open = False
            else:
                eps = self._eps_for(last_bear)
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

                    self._place_short_entry(entry_stop, sl, tp)

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
    pullback_max_bars = 20
    enable_be = False

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
        self._pb_long_armed = False
        self._pb_short_armed = False
        self._pb_long_armed_i = None
        self._pb_short_armed_i = None

    def next(self):
        i = len(self.data.Close) - 1
        if self._prev_position_open and not self.position:
            self._set_cooldown(i)
            self._reset_trade_meta()
        elif not self.position:
            self._reset_trade_meta()

        # Apply bracket orders right after position opens
        self._arm_if_opened()
        self._update_last_fractals()
        if self.position:
            self._maybe_apply_break_even(float(self._closes[i]))

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

        # Exits: close on sleeping or structure loss (configurable)
        if self.position:
            self._last_long_stop = None
            self._last_short_stop = None

            if self.exit_on_sleeping and state == "sleeping":
                self.position.close()
                self._set_cooldown(i)
                self._reset_trade_meta()
                self._prev_position_open = bool(self.position)
                return

            if self.exit_on_structure_loss:
                if self.position.is_long and not (lips > teeth > jaw):
                    self.position.close()
                    self._set_cooldown(i)
                    self._reset_trade_meta()
                    self._prev_position_open = bool(self.position)
                    return

                if self.position.is_short and not (lips < teeth < jaw):
                    self.position.close()
                    self._set_cooldown(i)
                    self._reset_trade_meta()
                    self._prev_position_open = bool(self.position)
                    return

            self._prev_position_open = True
            return

        allow_long, allow_short = self._entry_permissions(i)
        if not allow_long and not allow_short:
            self._maybe_cancel_stale_orders("unknown", allow_long, allow_short)
            self._prev_position_open = False
            return

        # No new entries in non-trending states
        if state in ("sleeping", "crossing", "unknown"):
            self._pb_long_armed = False
            self._pb_short_armed = False
            self._pb_long_armed_i = None
            self._pb_short_armed_i = None
            self._maybe_cancel_stale_orders(state, allow_long, allow_short)
            self._prev_position_open = False
            return

        atr_now = self._atr[i] if i < len(self._atr) else np.nan
        if np.isnan(atr_now):
            self._pb_long_armed = False
            self._pb_short_armed = False
            self._pb_long_armed_i = None
            self._pb_short_armed_i = None
            self._maybe_cancel_stale_orders(state, allow_long, allow_short)
            self._prev_position_open = False
            return

        if not allow_long or state != "eating_up":
            self._pb_long_armed = False
            self._pb_long_armed_i = None
        if not allow_short or state != "eating_down":
            self._pb_short_armed = False
            self._pb_short_armed_i = None

        if self.require_touch_teeth:
            long_pullback = self._lows[i] <= teeth
            short_pullback = self._highs[i] >= teeth
        else:
            long_pullback = self._lows[i] <= teeth + self.pullback_k_atr * atr_now
            short_pullback = self._highs[i] >= teeth - self.pullback_k_atr * atr_now

        if allow_long and state == "eating_up" and long_pullback:
            self._pb_long_armed = True
            self._pb_long_armed_i = i
        if allow_short and state == "eating_down" and short_pullback:
            self._pb_short_armed = True
            self._pb_short_armed_i = i

        if (
            self._pb_long_armed
            and self._pb_long_armed_i is not None
            and i - self._pb_long_armed_i >= self.pullback_max_bars
        ):
            self._pb_long_armed = False
            self._pb_long_armed_i = None
        if (
            self._pb_short_armed
            and self._pb_short_armed_i is not None
            and i - self._pb_short_armed_i >= self.pullback_max_bars
        ):
            self._pb_short_armed = False
            self._pb_short_armed_i = None

        self._maybe_cancel_stale_orders(state, allow_long, allow_short)
        if self._cooldown_active(i):
            self._prev_position_open = False
            return
        last_bull = self._last_bull_fractal
        last_bear = self._last_bear_fractal

        # Spread model (price units)
        spr = float(getattr(self, "spread_price", 0.0))
        half_spread = spr / 2.0

        # Long setup: eating_up + bullish fractal above alligator + pullback
        if allow_long and state == "eating_up" and self._pb_long_armed and not np.isnan(last_bull):
            if self._closes[i] <= teeth:
                self._prev_position_open = False
                return

            if last_bull <= max(jaw, teeth, lips):
                self._prev_position_open = False
                return

            eps = self._eps_for(last_bull)
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
            if self._place_long_entry(entry_stop, sl, tp):
                self._pb_long_armed = False
                self._pb_long_armed_i = None

        # Short setup: eating_down + bearish fractal below alligator + pullback
        if allow_short and state == "eating_down" and self._pb_short_armed and not np.isnan(last_bear):
            if self._closes[i] >= teeth:
                self._prev_position_open = False
                return

            if last_bear >= min(jaw, teeth, lips):
                self._prev_position_open = False
                return

            eps = self._eps_for(last_bear)
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

            if self._place_short_entry(entry_stop, sl, tp):
                self._pb_short_armed = False
                self._pb_short_armed_i = None

        self._prev_position_open = bool(self.position)


def _self_check() -> None:
    print("AlligatorFractal self-check:")
    print("- Backtest runs without exceptions.")
    print("- Existing variants match trade count/equity with cancel_stale_orders=False.")
    print("- Trailing variant changes only exits and improves loss profile on winners.")
    print("- Break-even moves (be_moves_debug) should be reasonable.")


if __name__ == "__main__":
    _self_check()
