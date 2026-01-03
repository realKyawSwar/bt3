#wave5_ao.py
from backtesting import Strategy
import numpy as np


def _sma(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return x.astype(float)
    out = np.full_like(x, np.nan, dtype=float)
    c = np.cumsum(x, dtype=float)
    out[n-1:] = (c[n-1:] - np.concatenate(([0.0], c[:-n])) ) / n
    return out


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return _sma(tr, n)


def _percentile_rank(window: np.ndarray, x: float) -> float:
    """Return percentile rank of x within window using mean(window <= x)."""
    finite = window[np.isfinite(window)]
    if finite.size < 30:
        return 0.5
    return float(np.mean(finite <= x))


class Wave5AODivergenceStrategy(Strategy):
    swing_window = 2
    fib_levels = (1.272, 1.618)
    fib_tol_atr = 0.25
    overlap_tol_atr = 0.0  # NEW: allow small overlap tolerance
    fib_tol_mode = "fixed"  # fixed|atr_pct
    fib_tol_lookback = 500
    fib_tol_lo = 0.35
    fib_tol_hi = 1.25
    ao_div_min = 0.0
    require_zero_cross = True
    trigger = ('engulfing', 'pin')
    entry_mode = 'close'          # close|break
    tp_r = 2.0
    order_size = 0.2  # Position size fraction (0-1]
    pin_ratio = 2.0
    min_bars_between_signals = 5
    max_trigger_lag = 3            # Max bars after H5/L5 to trigger entry
    min_w3_atr = 1.0               # Min wave3 length in ATR units
    break_buffer_atr = 0.10        # Buffer distance in ATR for break stop placement
    max_body_atr = 1.0             # Max candle body size (in ATR) to allow break entry
    asset = 'UNKNOWN'              # Asset symbol for labeling/reference
    spread_price = 0.0             # Spread in price terms injected by runner
    debug = False
    require_ext_touch = False  # if True, require H5/L5 extreme also tagged the zone
    sl_at_wave5_extreme = True  # If True, SL uses H5/L5 extreme instead of trigger candle
    tp_mode = "hybrid"  # "rr" (2R TP), "wave4" (Wave4 level), or "hybrid" (closer of the two)
    zone_mode = "trigger"  # trigger|extreme|either
    
    # Upgrade 1: Wave5 AO decay exhaustion confirmation
    wave5_ao_decay = False  # Require AO decay at Wave5 extreme
    ao_decay_mode = "strict"  # strict|soft
    
    # Upgrade 2: Wave5 minimum extension
    min_w5_ext = 1.272  # Minimum Wave5 extension relative to Wave3
    
    # Upgrade 3: Partial TP with split orders
    tp_split = False  # Enable split TP (tp1 at Wave4, tp2 at 0.618 retrace)
    tp_split_ratio = 0.5  # Position size ratio for first order (0-1)
    
    # Upgrade 4: ATR expansion regime filter
    atr_long = 100  # ATR long SMA period
    atr_expand_k = 1.5  # Skip trades when ATR > k * atr_sma

    # Margin support for position sizing (used only for sizing clamp, not broker)
    sizing_margin = 1.0  # Margin fraction used to cap position size (1.0=no leverage, 0.02=50x)
    exec_margin = 1.0  # Execution margin for broker logging/debug (not used for sizing)
    margin = 1.0  # Backtesting param compatibility; not used for sizing

    # Probabilistic scoring
    use_scoring = False
    debug_trace = True  # enable per-bar trace printing when debug=True
    score_threshold = 0.60
    w_zone = 1.0
    w_div = 1.0
    w_candle = 0.7
    w_lag = 0.5
    w_regime = 0.3
    w_zero = 0.2
    w_decay = 0.2
    zone_k = 3.0
    div_scale = 0.5
    regime_r = 0.6
    enable_size_by_score = False
    min_size_mult = 0.5

    def __init__(self, broker, data, params):
        super().__init__(broker, data, params)
        self.asset = params.get('asset', 'UNKNOWN')
        self.spread_price = float(params.get('spread_price', 0.0) or 0.0)
        osize = float(params.get('order_size', getattr(self, 'order_size', 0.2)))
        if osize <= 0 or osize > 1:
            raise ValueError("order_size must satisfy 0 < size <= 1")
        self.order_size = osize
        sizing_margin_val = params.get('sizing_margin', getattr(self, 'sizing_margin', 1.0))
        sizing_margin_val = 1.0 if sizing_margin_val is None else float(sizing_margin_val)
        if sizing_margin_val <= 0:
            sizing_margin_val = 1.0
        self.sizing_margin = sizing_margin_val
        exec_margin_val = params.get('exec_margin', getattr(self, 'exec_margin', 1.0))
        exec_margin_val = 1.0 if exec_margin_val is None else float(exec_margin_val)
        if exec_margin_val <= 0:
            exec_margin_val = 1.0
        self.exec_margin = exec_margin_val
        self.debug_trace = bool(params.get('debug_trace', getattr(self, 'debug_trace', True)))

    def init(self):
        # Numpy arrays for speed & correctness
        self._open = np.asarray(self.data.Open, dtype=float)
        self._high = np.asarray(self.data.High, dtype=float)
        self._low  = np.asarray(self.data.Low, dtype=float)
        self._close= np.asarray(self.data.Close, dtype=float)

        median = (self._high + self._low) / 2.0
        ao = _sma(median, 5) - _sma(median, 34)
        atr = _atr(self._high, self._low, self._close, 14)
        atr_sma = _sma(atr, int(self.atr_long))

        # Register with backtesting for plots/compat
        self.ao = self.I(lambda x=ao: x)
        self.atr = self.I(lambda x=atr: x)
        self.atr_sma = self.I(lambda x=atr_sma: x)

        # Swing state
        self.swing_highs = []   # list of (idx, price)
        self.swing_lows = []
        self._added_high = set()
        self._added_low = set()
        self.last_confirmed = -1
        self.last_signal_idx = -10_000
        self.summary = {
            "bars_seen": 0,
            "wave_candidates": 0,
            "zone_pass": 0,
            "trigger_pass": 0,
            "entry_pass": 0,
            "order_attempts": 0,
            "orders_placed": 0,
            "exceptions_count": 0,
        }
        self._final_summary_printed = False

        if self.debug:
            self.counters = {
                'swings_count': 0,
                'type_match': 0,
                'elliott_pass': 0,
                'zone_fail': 0,
                'w3_size_fail': 0,
                'w3_short_fail': 0,
                'divergence_pass': 0,
                'div_fail': 0,
                'zero_cross_fail': 0,
                'lag_fail': 0,
                'candle_fail': 0,
                'break_body_fail': 0,
                'entries': 0,
                'ao_decay_fail': 0,
                'ao_decay_pass': 0,
                'w5_ext_fail': 0,
                'atr_regime_fail': 0,
                'same_bar_ambiguous_fail': 0,
                'zone_trigger_pass': 0,
                'zone_extreme_pass': 0,
                'score_fail': 0,
                'score_pass': 0,
            }
            self._debug_zone_mode = 'trigger'
            self._debug_zone_trigger = 0
            self._debug_zone_extreme = 0

    def _dbg(self, msg: str) -> None:
        if self.debug:
            print(msg)

    def _print_counters(self, i: int, direction: str) -> None:
        """Print debug counters before a trade entry."""
        if self.debug:
            print(f"[{i}] {direction.upper()} setup: swings={self.counters['swings_count']} "
                  f"type_match={self.counters['type_match']} elliott={self.counters['elliott_pass']} "
                  f"zone_fail={self.counters['zone_fail']} w3_size_fail={self.counters['w3_size_fail']} "
                  f"w3_short_fail={self.counters['w3_short_fail']} div_fail={self.counters['div_fail']} "
                  f"zero_cross_fail={self.counters['zero_cross_fail']} "
                  f"lag_fail={self.counters['lag_fail']} candle_fail={self.counters['candle_fail']} "
                  f"break_body_fail={self.counters['break_body_fail']} "
                  f"ao_decay_fail={self.counters['ao_decay_fail']} ao_decay_pass={self.counters['ao_decay_pass']} "
                  f"ao_decay_mode={self._get_ao_decay_mode()} "
                  f"w5_ext_fail={self.counters['w5_ext_fail']} "
                  f"atr_regime_fail={self.counters['atr_regime_fail']} "
                  f"same_bar_ambiguous_fail={self.counters['same_bar_ambiguous_fail']} "
                  f"entries={self.counters['entries']} "
                  f"zone_mode={self._debug_zone_mode} "
                  f"zone_trigger={self._debug_zone_trigger} "
                  f"zone_extreme={self._debug_zone_extreme}")

    def _emit_trace(self, trace: dict) -> None:
        if not self.debug or not getattr(self, "debug_trace", True):
            return
        msg = (
            f"[W5 TRACE] ts={trace.get('ts')} "
            f"c={trace.get('close'):.5f} "
            f"ao={trace.get('ao'):.5f} "
            f"ao_signal={trace.get('ao_signal')} "
            f"swing_ready={int(bool(trace.get('swing_ready', 0)))} "
            f"type={trace.get('wave_type')} "
            f"zone={int(bool(trace.get('zone_ok', 0)))} "
            f"trigger={int(bool(trace.get('trigger_ok', 0)))} "
            f"entry={int(bool(trace.get('entry_ok', 0)))} "
            f"sl={int(bool(trace.get('sl_ok', 0)))} "
            f"size={int(bool(trace.get('size_ok', 0)))} "
            f"reason=\"{trace.get('reason', '')}\""
        )
        if "fib_tol_eff" in trace and np.isfinite(trace.get("fib_tol_eff", np.nan)):
            msg += f" fib_tol={trace.get('fib_tol_eff'):.4f}"
        if "fib_pct" in trace and np.isfinite(trace.get("fib_pct", np.nan)):
            msg += f" fib_pct={trace.get('fib_pct'):.3f}"
        if "score" in trace and np.isfinite(trace.get("score", np.nan)):
            msg += f" score={trace.get('score'):.3f}"
        if "zone_score" in trace and np.isfinite(trace.get("zone_score", np.nan)):
            msg += f" z={trace.get('zone_score'):.3f}"
        if "div_score" in trace and np.isfinite(trace.get("div_score", np.nan)):
            msg += f" d={trace.get('div_score'):.3f}"
        if "lag_score" in trace and np.isfinite(trace.get("lag_score", np.nan)):
            msg += f" lag_s={trace.get('lag_score'):.3f}"
        if "regime_score" in trace and np.isfinite(trace.get("regime_score", np.nan)):
            msg += f" reg_s={trace.get('regime_score'):.3f}"
        if "candle_score" in trace and np.isfinite(trace.get("candle_score", np.nan)):
            msg += f" cndl_s={trace.get('candle_score'):.3f}"
        self._dbg(msg)

    def _emit_final_summary(self) -> None:
        if self._final_summary_printed or not self.debug:
            return
        self._final_summary_printed = True
        summary = {
            "bars_seen": self.summary.get("bars_seen", 0),
            "wave_candidates": self.summary.get("wave_candidates", 0),
            "zone_pass": self.summary.get("zone_pass", 0),
            "trigger_pass": self.summary.get("trigger_pass", 0),
            "entry_pass": self.summary.get("entry_pass", 0),
            "order_attempts": self.summary.get("order_attempts", 0),
            "orders_placed": self.summary.get("orders_placed", 0),
            "exceptions_count": self.summary.get("exceptions_count", 0),
        }
        self._dbg(f"[W5 SUMMARY] {summary}")

    # -------------------------
    # Swings
    # -------------------------
    def is_swing_high(self, j: int) -> bool:
        w = int(self.swing_window)
        if j < w or j >= len(self._high) - w:
            return False
        h = self._high[j]
        return h > np.max(self._high[j-w:j]) and h > np.max(self._high[j+1:j+1+w])

    def is_swing_low(self, j: int) -> bool:
        w = int(self.swing_window)
        if j < w or j >= len(self._low) - w:
            return False
        l = self._low[j]
        return l < np.min(self._low[j-w:j]) and l < np.min(self._low[j+1:j+1+w])

    def _update_swings(self, i: int) -> None:
        w = int(self.swing_window)
        start = max(0, self.last_confirmed + 1)
        end = i - w
        if end < start:
            return

        for j in range(start, end + 1):
            if self.is_swing_high(j) and j not in self._added_high:
                self.swing_highs.append((j, self._high[j]))
                self._added_high.add(j)
                if self.debug:
                    self.counters['swings_count'] += 1
            if self.is_swing_low(j) and j not in self._added_low:
                self.swing_lows.append((j, self._low[j]))
                self._added_low.add(j)
                if self.debug:
                    self.counters['swings_count'] += 1

        self.last_confirmed = end

        # keep recent only
        self.swing_highs = self.swing_highs[-50:]
        self.swing_lows = self.swing_lows[-50:]

    # -------------------------
    # Wave sequence via merged pivots (time-ordered)
    # -------------------------
    def _latest_pivots(self, limit: int = 40):
        piv = [(idx, 'H', px) for idx, px in self.swing_highs] + [(idx, 'L', px) for idx, px in self.swing_lows]
        piv.sort(key=lambda x: x[0])
        return piv[-limit:]

    def find_uptrend_sequence(self, i: int):
        piv = self._latest_pivots()
        # pattern: L H L H L (in time order)
        # scan from end
        for t in range(len(piv) - 5, -1, -1):
            a,b,c,d,e = piv[t:t+5]
            if (a[1],b[1],c[1],d[1],e[1]) != ('L','H','L','H','L'):
                continue
            L0_idx, _, L0_p = a
            H1_idx, _, H1_p = b
            L2_idx, _, L2_p = c
            H3_idx, _, H3_p = d
            L4_idx, _, L4_p = e

            # structure checks
            if not (L0_idx < H1_idx < L2_idx < H3_idx < L4_idx):
                continue
            if not (H3_p > H1_p and L4_p > L2_p):  # higher high / higher low
                continue

            # Elliott-ish rules
            if L2_p < L0_p:
                continue

            atr_val = float(self.atr[i])
            overlap_tol = self.overlap_tol_atr * atr_val
            if L4_p < (H1_p - overlap_tol):
                continue

            if self.debug:
                self.counters['elliott_pass'] += 1

            return {
                'L0_idx': L0_idx, 'L0_p': L0_p,
                'H1_idx': H1_idx, 'H1_p': H1_p,
                'L2_idx': L2_idx, 'L2_p': L2_p,
                'H3_idx': H3_idx, 'H3_p': H3_p,
                'L4_idx': L4_idx, 'L4_p': L4_p,
            }
        return None

    def find_downtrend_sequence(self, i: int):
        piv = self._latest_pivots()
        # pattern: H L H L H
        for t in range(len(piv) - 5, -1, -1):
            a,b,c,d,e = piv[t:t+5]
            if (a[1],b[1],c[1],d[1],e[1]) != ('H','L','H','L','H'):
                continue
            H0_idx, _, H0_p = a
            L1_idx, _, L1_p = b
            H2_idx, _, H2_p = c
            L3_idx, _, L3_p = d
            H4_idx, _, H4_p = e

            if not (H0_idx < L1_idx < H2_idx < L3_idx < H4_idx):
                continue
            if not (L3_p < L1_p and H4_p < H2_p):  # lower low / lower high
                continue

            # Elliott-ish rules
            if H2_p > H0_p:
                continue

            atr_val = float(self.atr[i])
            overlap_tol = self.overlap_tol_atr * atr_val
            if H4_p > (L1_p + overlap_tol):
                continue

            if self.debug:
                self.counters['elliott_pass'] += 1

            return {
                'H0_idx': H0_idx, 'H0_p': H0_p,
                'L1_idx': L1_idx, 'L1_p': L1_p,
                'H2_idx': H2_idx, 'H2_p': H2_p,
                'L3_idx': L3_idx, 'L3_p': L3_p,
                'H4_idx': H4_idx, 'H4_p': H4_p,
            }
        return None

    # -------------------------
    # Helpers
    # -------------------------
    def has_zero_cross(self, start: int, end: int, direction: str) -> bool:
        ao = np.asarray(self.ao, dtype=float)
        s = max(start + 1, 1)
        e = min(end, len(ao) - 1)
        if e < s:
            return False
        if direction == 'bullish':
            return np.any((ao[s-1:e] <= 0) & (ao[s:e+1] > 0))
        else:
            return np.any((ao[s-1:e] >= 0) & (ao[s:e+1] < 0))

    def _candle_signal(self, i: int, direction: str):
        """Return candle reversal type and score for scoring logic."""
        if i < 1:
            return "none", 0.0
        o, c, h, l = self._open[i], self._close[i], self._high[i], self._low[i]
        o1, c1 = self._open[i-1], self._close[i-1]
        body = max(abs(o - c), 1e-12)

        if direction == 'bearish':
            if 'engulfing' in self.trigger and (c < o) and (c1 > o1) and (o > c1) and (c < o1):
                return "engulfing", 1.0
            if 'pin' in self.trigger:
                upper = h - max(o, c)
                if upper > self.pin_ratio * body and c < (o + l) / 2:
                    return "pin", 0.6
        else:
            if 'engulfing' in self.trigger and (c > o) and (c1 < o1) and (o < c1) and (c > o1):
                return "engulfing", 1.0
            if 'pin' in self.trigger:
                lower = min(o, c) - l
                if lower > self.pin_ratio * body and c > (o + h) / 2:
                    return "pin", 0.6

        return "none", 0.0

    def is_reversal_candle(self, i: int, direction: str) -> bool:
        _, score = self._candle_signal(i, direction)
        return score > 0

    def _get_candle_body_atr(self, i: int) -> float:
        """Return candle body size in ATR units."""
        atr_val = float(self.atr[i])
        if atr_val <= 0:
            return 0.0
        body = abs(self._close[i] - self._open[i])
        return body / atr_val

    @staticmethod
    def _clip01(x: float) -> float:
        return float(np.clip(x, 0.0, 1.0))

    def _normalize_zone_mode(self) -> str:
        mode = str(getattr(self, "zone_mode", "trigger")).strip().lower()
        if mode not in {"trigger", "extreme", "either"}:
            return "trigger"
        return mode

    def _get_ao_decay_mode(self) -> str:
        mode = str(getattr(self, "ao_decay_mode", "strict")).strip().lower()
        if mode not in {"strict", "soft"}:
            return "strict"
        return mode

    def _fib_tolerance(self, i: int, atr_val: float) -> tuple[float, float]:
        mode = str(getattr(self, "fib_tol_mode", "fixed")).strip().lower()
        if mode != "atr_pct":
            return float(self.fib_tol_atr), 0.5
        lookback = max(int(getattr(self, "fib_tol_lookback", 0)), 1)
        start = max(0, i - lookback)
        atr_window = np.asarray(self.atr, dtype=float)[start:i+1]
        p = _percentile_rank(atr_window, atr_val)
        lo = float(getattr(self, "fib_tol_lo", 0.35))
        hi = float(getattr(self, "fib_tol_hi", 1.25))
        tol = lo + p * (hi - lo)
        return float(tol), float(p)

    def _combine_scores(self, components: dict) -> float:
        weights = {
            "zone": float(getattr(self, "w_zone", 1.0)),
            "div": float(getattr(self, "w_div", 1.0)),
            "candle": float(getattr(self, "w_candle", 0.7)),
            "lag": float(getattr(self, "w_lag", 0.5)),
            "regime": float(getattr(self, "w_regime", 0.3)),
            "zero": float(getattr(self, "w_zero", 0.2)),
            "decay": float(getattr(self, "w_decay", 0.2)),
        }
        num = 0.0
        den = 0.0
        for key, w in weights.items():
            if w <= 0:
                continue
            num += w * float(components.get(key, 0.0))
            den += w
        if den <= 0:
            return 0.0
        return num / den

    def _evaluate_zone(self, zone_mode: str, in_zone_trigger: bool, in_zone_extreme: bool) -> bool:
        if self.debug:
            if in_zone_trigger:
                self.counters['zone_trigger_pass'] += 1
            if in_zone_extreme:
                self.counters['zone_extreme_pass'] += 1
            self._debug_zone_mode = zone_mode
            self._debug_zone_trigger = int(bool(in_zone_trigger))
            self._debug_zone_extreme = int(bool(in_zone_extreme))

        if bool(self.require_ext_touch) and not in_zone_extreme:
            return False

        if zone_mode == "trigger":
            return in_zone_trigger
        if zone_mode == "extreme":
            return in_zone_extreme
        return in_zone_trigger or in_zone_extreme

    def _can_place_order(self):
        if self.position and getattr(self.position, "size", 0) != 0:
            return False, "already_in_position"
        pending = len(self.orders) if getattr(self, "orders", None) else 0
        if pending > 0:
            return False, "pending_order_exists"
        return True, ""

    def _final_units(self, base_size_frac: float, entry_price: float, sl_price: float, return_details: bool = False):
        """Convert fractional size to units with sizing_margin cap and 1+ unit minimum."""
        eq = float(self.equity)
        entry_price = float(entry_price)
        sl_price = float(sl_price)
        size = float(base_size_frac)
        sizing_margin = float(getattr(self, "sizing_margin", 1.0)) or 1.0
        exec_margin = float(getattr(self, "exec_margin", 1.0)) or 1.0

        details = {
            "equity": eq,
            "entry": entry_price,
            "sl": sl_price,
            "size_frac": size,
            "sizing_margin": sizing_margin,
            "exec_margin": exec_margin,
            "risk_cash": 0.0,
            "risk_units": 0,
            "max_units_cap": 0,
            "final_units": 0,
            "fail_reason": "",
        }

        if (
            entry_price <= 0
            or not np.isfinite(entry_price)
            or not np.isfinite(sl_price)
            or not np.isfinite(size)
        ):
            details["fail_reason"] = "sl_invalid"
            if self.debug:
                self._dbg("[WAVE5 REJECT] reason=non_finite_inputs")
            return (0, details) if return_details else 0

        if size <= 0:
            details["fail_reason"] = "size_zero"
            if self.debug:
                self._dbg("[WAVE5 REJECT] reason=invalid_risk_fraction")
            return (0, details) if return_details else 0

        sl_dist = abs(entry_price - sl_price)
        if sl_dist <= 0 or not np.isfinite(sl_dist):
            details["fail_reason"] = "sl_invalid"
            if self.debug:
                self._dbg("[WAVE5 REJECT] reason=invalid_sl_distance")
            return (0, details) if return_details else 0

        risk_cash = eq * min(max(size, 0.0), 1.0)
        risk_units = int(np.floor(risk_cash / sl_dist))
        max_units = int(np.floor(eq / (entry_price * sizing_margin))) if sizing_margin > 0 else 0

        details.update({
            "risk_cash": risk_cash,
            "risk_units": risk_units,
            "max_units_cap": max_units,
        })

        if max_units <= 0:
            details["fail_reason"] = "size_zero"
            if self.debug:
                self._dbg("[WAVE5 REJECT] reason=insufficient_margin_cap")
            return (0, details) if return_details else 0

        if risk_units >= 1 and max_units >= 1:
            final_units = max(1, min(risk_units, max_units))
        else:
            details["fail_reason"] = "size_zero"
            final_units = 0
        details["final_units"] = final_units

        if self.debug and final_units <= 0:
            self._dbg(
                "[WAVE5 SIZE] "
                f"equity={eq:.2f} entry={entry_price:.5f} sl={sl_price:.5f} sl_dist={sl_dist:.5f} "
                f"risk_cash={risk_cash:.2f} risk_units={risk_units} max_units_cap={max_units} "
                f"final_units={final_units} sizing_margin={sizing_margin:.4f} exec_margin={exec_margin:.2f}"
            )

        return (final_units, details) if return_details else final_units

    # -------------------------
    # Main loop
    # -------------------------
    def next(self):
        i = len(self.data) - 1
        if i < 60:  # warm-up
            return

        ts = self.data.index[-1] if hasattr(self.data, 'index') else i
        ao_val = float(self.ao[i])
        trace = {
            "ts": ts,
            "close": float(self._close[i]),
            "ao": ao_val,
            "ao_signal": int(np.sign(ao_val)),
            "swing_ready": 0,
            "wave_type": "none",
            "zone_ok": 0,
            "trigger_ok": 0,
            "entry_ok": 0,
            "sl_ok": 0,
            "size_ok": 0,
            "reason": "",
        }
        self.summary["bars_seen"] += 1
        trace["use_scoring"] = int(bool(getattr(self, "use_scoring", False)))

        if self.debug:
            num_orders = len(self.orders) if hasattr(self, 'orders') else 0
            num_trades = len(self.trades) if hasattr(self, 'trades') else 0
            if hasattr(self, '_prev_num_orders'):
                if self._prev_num_orders > 0 and num_orders == 0 and num_trades == getattr(self, '_prev_num_trades', 0):
                    self._dbg(f"[WAVE5 ALERT] order_disappeared i={i-1} -> i={i}, investigate broker logs above")
            self._prev_num_orders = num_orders
            self._prev_num_trades = num_trades

        # Upgrade 4: ATR regime filter - skip when ATR is expanding
        atr_val = float(self.atr[i])
        atr_sma_val = float(self.atr_sma[i])
        use_scoring = bool(getattr(self, "use_scoring", False))
        if np.isfinite(atr_val) and np.isfinite(atr_sma_val) and atr_sma_val > 0:
            if not use_scoring and atr_val > float(self.atr_expand_k) * atr_sma_val:
                trace["reason"] = "trigger_fail"
                if self.debug:
                    self.counters['atr_regime_fail'] += 1
                self._emit_trace(trace)
                return

        self._update_swings(i)
        swings_ready = (len(self.swing_highs) + len(self.swing_lows)) >= 5
        trace["swing_ready"] = int(swings_ready)
        if not swings_ready:
            trace["reason"] = "no_swings"
            self._emit_trace(trace)
            return

        if i - self.last_signal_idx < int(self.min_bars_between_signals):
            trace["reason"] = "trigger_fail"
            self._emit_trace(trace)
            return

        up_seq = self.find_uptrend_sequence(i)
        if up_seq:
            if self.debug:
                self.counters['type_match'] += 1
            trace = self._handle_sell(i, up_seq, trace)
        down_seq = self.find_downtrend_sequence(i)
        if not up_seq and down_seq:
            if self.debug:
                self.counters['type_match'] += 1
            trace = self._handle_buy(i, down_seq, trace)

        if not up_seq and not down_seq:
            trace["reason"] = "no_wave_match"

        self._emit_trace(trace)
        if i >= len(self._close) - 1:
            self._emit_final_summary()

    def _handle_sell(self, i: int, seq: dict, trace: dict):
        trace["wave_type"] = "sell"
        use_scoring = bool(getattr(self, "use_scoring", False))
        self.summary["wave_candidates"] += 1

        w3_len = seq['H3_p'] - seq['L2_p']
        if not np.isfinite(w3_len) or w3_len <= 0:
            trace["reason"] = "entry_fail"
            return trace

        atr_val = float(self.atr[i])
        if not np.isfinite(atr_val) or atr_val <= 0:
            trace["reason"] = "entry_fail"
            return trace

        fib_tol_eff, fib_pct = self._fib_tolerance(i, atr_val)
        trace["fib_tol_eff"] = fib_tol_eff
        trace["fib_pct"] = fib_pct

        if w3_len < self.min_w3_atr * atr_val:
            if self.debug:
                self.counters['w3_size_fail'] += 1
            trace["reason"] = "entry_fail"
            return trace

        ext_levels = [seq['L4_p'] + fib * w3_len for fib in self.fib_levels]
        post_start = seq['L4_idx'] + 1
        if post_start >= i:
            trace["reason"] = "entry_fail"
            return trace

        highs = self._high[post_start:i+1]
        max_pos = int(np.argmax(highs))
        H5_idx = post_start + max_pos
        H5_p = float(highs[max_pos])
        trigger_px = self._close[i] if self.entry_mode == "close" else self._low[i]
        zone_dist = np.inf
        if ext_levels:
            zone_dist = min(abs(trigger_px - ext) for ext in ext_levels) / (atr_val + 1e-12)
        zone_score = float(np.exp(-float(self.zone_k) * zone_dist)) if np.isfinite(zone_dist) else 0.0
        in_zone_trigger = any(abs(trigger_px - ext) <= fib_tol_eff * atr_val for ext in ext_levels)
        in_zone_extreme = any(abs(H5_p - ext) <= fib_tol_eff * atr_val for ext in ext_levels)

        zone_mode = self._normalize_zone_mode()
        in_zone = self._evaluate_zone(zone_mode, in_zone_trigger, in_zone_extreme)
        trace["zone_ok"] = int(bool(in_zone))
        trace["zone_score"] = self._clip01(zone_score)
        if in_zone:
            self.summary["zone_pass"] += 1
        else:
            if self.debug:
                self.counters['zone_fail'] += 1
            if not use_scoring:
                trace["reason"] = "zone_fail"
                return trace

        if self.debug:
            self.counters['elliott_pass'] += 1

        w1_len = seq['H1_p'] - seq['L0_p']
        w5_len = H5_p - seq['L4_p']
        if w3_len < min(w1_len, w5_len):
            if self.debug:
                self.counters['w3_short_fail'] += 1
            trace["reason"] = "entry_fail"
            return trace

        if w5_len < float(self.min_w5_ext) * w3_len:
            if self.debug:
                self.counters['w5_ext_fail'] += 1
            trace["reason"] = "entry_fail"
            return trace

        ao = np.asarray(self.ao, dtype=float)
        ao_h3 = ao[seq['H3_idx']]
        ao_h5 = ao[H5_idx]

        ao_decay_ok = True
        if bool(self.wave5_ao_decay):
            mode = self._get_ao_decay_mode()
            ao_decay_ok = False
            if mode == "strict":
                if H5_idx >= 2:
                    ao_h5_m1 = ao[H5_idx - 1]
                    ao_h5_m2 = ao[H5_idx - 2]
                    ao_decay_ok = bool(ao_h5 < ao_h5_m1 < ao_h5_m2)
            else:
                if H5_idx >= 1:
                    ao_h5_m1 = ao[H5_idx - 1]
                    ao_decay_ok = bool(ao_h5 < ao_h5_m1)

            if not ao_decay_ok:
                if self.debug:
                    self.counters['ao_decay_fail'] += 1
                if not use_scoring:
                    trace["reason"] = "trigger_fail"
                    return trace
            elif self.debug:
                self.counters['ao_decay_pass'] += 1

        div_ok = (H5_p > seq['H3_p'] and ao_h5 < ao_h3 - self.ao_div_min)
        div_raw = (ao_h3 - ao_h5 - self.ao_div_min) / (abs(ao_h3) + 1e-12)
        div_scale = max(float(getattr(self, "div_scale", 0.5)), 1e-12)
        div_score = self._clip01(div_raw / div_scale)
        if not div_ok:
            if self.debug:
                self.counters['div_fail'] += 1
            if not use_scoring:
                trace["reason"] = "trigger_fail"
                return trace
        elif self.debug:
            self.counters['divergence_pass'] += 1

        zero_ok = True
        if self.require_zero_cross:
            zero_ok = self.has_zero_cross(seq['H3_idx'], H5_idx, 'bullish')
            if not zero_ok and self.debug:
                self.counters['zero_cross_fail'] += 1
            if not zero_ok and not use_scoring:
                trace["reason"] = "trigger_fail"
                return trace

        allowed_lag = int(self.max_trigger_lag)
        lag_val = max(0, i - H5_idx)
        lag_ok = lag_val <= allowed_lag
        lag_score = 1.0 - min(1.0, lag_val / max(allowed_lag, 1))
        if not lag_ok:
            if self.debug:
                self.counters['lag_fail'] += 1
            if not use_scoring:
                trace["reason"] = "trigger_fail"
                return trace

        candle_type, candle_score = self._candle_signal(i, 'bearish')
        candle_ok = candle_score > 0
        if not candle_ok:
            if self.debug:
                self.counters['candle_fail'] += 1
            if not use_scoring:
                trace["reason"] = "trigger_fail"
                return trace

        atr_sma_val = float(self.atr_sma[i])
        regime_score = 0.5
        if np.isfinite(atr_val) and np.isfinite(atr_sma_val) and atr_sma_val > 0:
            ratio = atr_val / atr_sma_val
            regime_score = self._clip01(1 - max(0.0, (ratio - 1.0) / float(getattr(self, "regime_r", 0.6))))
            trace["regime_ratio"] = ratio
        trace["regime_score"] = regime_score

        zero_score = 1.0 if (not self.require_zero_cross or zero_ok) else 0.0
        decay_score = 1.0 if (not self.wave5_ao_decay or ao_decay_ok) else 0.0
        trace["div_score"] = div_score
        trace["lag_score"] = lag_score
        trace["candle_score"] = candle_score
        trace["zero_score"] = zero_score
        trace["decay_score"] = decay_score

        score = 1.0
        if use_scoring:
            components = {
                "zone": trace.get("zone_score", 0.0),
                "div": div_score,
                "candle": candle_score,
                "lag": lag_score,
                "regime": regime_score,
                "zero": zero_score,
                "decay": decay_score,
            }
            score = self._combine_scores(components)
            trace["score"] = score
            if score < float(getattr(self, "score_threshold", 0.6)):
                trace["reason"] = "score_fail"
                if self.debug:
                    self.counters['score_fail'] += 1
                return trace
            if self.debug:
                self.counters['score_pass'] += 1
        else:
            self.summary["trigger_pass"] += 1
            trace["trigger_ok"] = 1

        self.summary["trigger_pass"] += int(use_scoring)
        if use_scoring:
            trace["trigger_ok"] = 1

        allowed, gate_reason = self._can_place_order()
        if not allowed:
            trace["reason"] = gate_reason
            return trace
        trace["entry_ok"] = 1

        buffer = float(getattr(self, 'spread_price', 0.0) or 0.0)
        trigger_high = self._high[i]
        trigger_low = self._low[i]

        if bool(getattr(self, "sl_at_wave5_extreme", True)):
            sl = float(H5_p) + buffer
        else:
            sl = float(trigger_high) + buffer

        if self.entry_mode == 'close':
            entry = self._close[i]
        else:
            candle_body = self._get_candle_body_atr(i)
            if candle_body > float(self.max_body_atr):
                if self.debug:
                    self.counters['break_body_fail'] += 1
                trace["reason"] = "entry_fail"
                return trace

            atr_val = float(self.atr[i])
            entry = trigger_low
            sl = trigger_high + buffer + float(self.break_buffer_atr) * atr_val

        trace["sl_ok"] = 1
        base_size = float(getattr(self, "order_size", 0.2))
        if use_scoring and bool(getattr(self, "enable_size_by_score", False)):
            score_adj = self._clip01(score)
            min_mult = float(getattr(self, "min_size_mult", 0.5))
            size_mult = min(1.0, max(min_mult, min_mult + (1.0 - min_mult) * score_adj))
            base_size *= size_mult
            trace["size_mult"] = size_mult
        if base_size <= 0:
            trace["reason"] = "size_zero"
            return trace
        entry_price_for_size = float(entry)

        placed_any = False
        order_attempts = 0

        if bool(self.tp_split):
            tp1 = float(seq['L4_p'])
            tp2 = seq['L4_p'] - 0.618 * (H5_p - seq['L0_p'])
            tp2 = min(tp2, tp1)

            if self.entry_mode == 'break':
                if self._high[i] >= sl or self._low[i] <= tp2:
                    if self.debug:
                        self.counters['same_bar_ambiguous_fail'] += 1
                    trace["reason"] = "entry_fail"
                    return trace

            split_ratio = float(self.tp_split_ratio)
            size1 = base_size * split_ratio
            size2 = base_size * (1.0 - split_ratio)
            order_size1, size_dbg1 = self._final_units(size1, entry_price_for_size, sl, return_details=True)
            order_size2, size_dbg2 = self._final_units(size2, entry_price_for_size, sl, return_details=True)

            if order_size1 < 1 and order_size2 < 1:
                fail_reason = size_dbg1.get("fail_reason") or size_dbg2.get("fail_reason") or "size_zero"
                trace["reason"] = fail_reason if fail_reason in {"size_zero", "sl_invalid"} else "size_zero"
                if trace["reason"] == "sl_invalid":
                    trace["sl_ok"] = 0
                trace["size_ok"] = 0
                return trace

            if self.debug:
                self._dbg(f"[SELL SPLIT] entry={entry:.5f} sl={sl:.5f} tp1={tp1:.5f} tp2={tp2:.5f} sizes={size1:.2f}/{size2:.2f}")

            if self.entry_mode == 'close':
                self.summary["order_attempts"] += 1
                order_attempts += 1
                try:
                    o1 = self.sell(sl=sl, tp=tp1, size=order_size1)
                    if o1 is not None:
                        placed_any = True
                        self.summary["orders_placed"] += 1
                except (ValueError, AssertionError, RuntimeError) as e:
                    self.summary["exceptions_count"] += 1
                    self._dbg(f"[W5 ORDER FAIL] exc={repr(e)}")
                self.summary["order_attempts"] += 1
                order_attempts += 1
                try:
                    o2 = self.sell(sl=sl, tp=tp2, size=order_size2)
                    if o2 is not None:
                        placed_any = True
                        self.summary["orders_placed"] += 1
                except (ValueError, AssertionError, RuntimeError) as e:
                    self.summary["exceptions_count"] += 1
                    self._dbg(f"[W5 ORDER FAIL] exc={repr(e)}")
            else:
                self.summary["order_attempts"] += 1
                order_attempts += 1
                try:
                    o1 = self.sell(stop=trigger_low, sl=sl, tp=tp1, size=order_size1)
                    if o1 is not None:
                        placed_any = True
                        self.summary["orders_placed"] += 1
                except (ValueError, AssertionError, RuntimeError) as e:
                    self.summary["exceptions_count"] += 1
                    self._dbg(f"[W5 ORDER FAIL] exc={repr(e)}")
                self.summary["order_attempts"] += 1
                order_attempts += 1
                try:
                    o2 = self.sell(stop=trigger_low, sl=sl, tp=tp2, size=order_size2)
                    if o2 is not None:
                        placed_any = True
                        self.summary["orders_placed"] += 1
                except (ValueError, AssertionError, RuntimeError) as e:
                    self.summary["exceptions_count"] += 1
                    self._dbg(f"[W5 ORDER FAIL] exc={repr(e)}")

        else:
            tp_mode = str(getattr(self, "tp_mode", "hybrid")).lower()
            tp_2r = entry - self.tp_r * (sl - entry)

            if tp_mode == "rr":
                tp = tp_2r
            elif tp_mode == "wave4":
                tp = float(seq["L4_p"])
            else:
                tp_wave4 = float(seq["L4_p"])
                valid_candidates = []
                if tp_2r < entry:
                    valid_candidates.append(("2R", tp_2r))
                if tp_wave4 < entry:
                    valid_candidates.append(("Wave4", tp_wave4))
                if len(valid_candidates) == 0:
                    tp = tp_2r
                elif len(valid_candidates) == 1:
                    _, tp = valid_candidates[0]
                else:
                    candidates_with_distance = [
                        (source, tp_val, entry - tp_val) for source, tp_val in valid_candidates
                    ]
                    _, tp, _ = min(candidates_with_distance, key=lambda x: x[2])

            if self.entry_mode == 'break':
                if self._high[i] >= sl or self._low[i] <= tp:
                    if self.debug:
                        self.counters['same_bar_ambiguous_fail'] += 1
                    trace["reason"] = "entry_fail"
                    return trace

            final_size, size_dbg = self._final_units(base_size, entry_price_for_size, sl, return_details=True)
            if final_size < 1:
                fail_reason = size_dbg.get("fail_reason") or "size_zero"
                trace["reason"] = fail_reason if fail_reason in {"size_zero", "sl_invalid"} else "size_zero"
                if trace["reason"] == "sl_invalid":
                    trace["sl_ok"] = 0
                trace["size_ok"] = 0
                return trace

            self.summary["order_attempts"] += 1
            order_attempts += 1
            try:
                if self.entry_mode == 'close':
                    order = self.sell(sl=sl, tp=tp, size=final_size)
                else:
                    order = self.sell(stop=trigger_low, sl=sl, tp=tp, size=final_size)
                if order is not None:
                    placed_any = True
                    self.summary["orders_placed"] += 1
            except (ValueError, AssertionError, RuntimeError) as e:
                self.summary["exceptions_count"] += 1
                self._dbg(f"[W5 ORDER FAIL] exc={repr(e)}")

        if placed_any:
            trace["size_ok"] = 1
            trace["reason"] = trace.get("reason", "")
            self.summary["entry_pass"] += 1
            self.last_signal_idx = i
            if self.debug:
                self.counters['entries'] += 1
        else:
            trace["reason"] = trace.get("reason") or "entry_fail"
        return trace

    def _handle_buy(self, i: int, seq: dict, trace: dict):
        trace["wave_type"] = "buy"
        use_scoring = bool(getattr(self, "use_scoring", False))
        self.summary["wave_candidates"] += 1

        w3_len = seq['H2_p'] - seq['L3_p']
        if not np.isfinite(w3_len) or w3_len <= 0:
            trace["reason"] = "entry_fail"
            return trace

        atr_val = float(self.atr[i])
        if not np.isfinite(atr_val) or atr_val <= 0:
            trace["reason"] = "entry_fail"
            return trace

        fib_tol_eff, fib_pct = self._fib_tolerance(i, atr_val)
        trace["fib_tol_eff"] = fib_tol_eff
        trace["fib_pct"] = fib_pct

        if w3_len < self.min_w3_atr * atr_val:
            if self.debug:
                self.counters['w3_size_fail'] += 1
            trace["reason"] = "entry_fail"
            return trace

        ext_levels = [seq['H4_p'] - fib * w3_len for fib in self.fib_levels]
        post_start = seq['H4_idx'] + 1
        if post_start >= i:
            trace["reason"] = "entry_fail"
            return trace

        lows = self._low[post_start:i+1]
        min_pos = int(np.argmin(lows))
        L5_idx = post_start + min_pos
        L5_p = float(lows[min_pos])
        trigger_px = self._close[i] if self.entry_mode == "close" else self._high[i]
        zone_dist = np.inf
        if ext_levels:
            zone_dist = min(abs(trigger_px - ext) for ext in ext_levels) / (atr_val + 1e-12)
        zone_score = float(np.exp(-float(self.zone_k) * zone_dist)) if np.isfinite(zone_dist) else 0.0
        in_zone_trigger = any(abs(trigger_px - ext) <= fib_tol_eff * atr_val for ext in ext_levels)
        in_zone_extreme = any(abs(L5_p - ext) <= fib_tol_eff * atr_val for ext in ext_levels)

        zone_mode = self._normalize_zone_mode()
        in_zone = self._evaluate_zone(zone_mode, in_zone_trigger, in_zone_extreme)
        trace["zone_ok"] = int(bool(in_zone))
        trace["zone_score"] = self._clip01(zone_score)
        if in_zone:
            self.summary["zone_pass"] += 1
        if not in_zone:
            if self.debug:
                self.counters['zone_fail'] += 1
            if not use_scoring:
                trace["reason"] = "zone_fail"
                return trace

        if self.debug:
            self.counters['elliott_pass'] += 1

        w1_len = seq['H0_p'] - seq['L1_p']
        w5_len = seq['H4_p'] - L5_p
        if w3_len < min(w1_len, w5_len):
            if self.debug:
                self.counters['w3_short_fail'] += 1
            trace["reason"] = "entry_fail"
            return trace

        if w5_len < float(self.min_w5_ext) * w3_len:
            if self.debug:
                self.counters['w5_ext_fail'] += 1
            trace["reason"] = "entry_fail"
            return trace

        ao = np.asarray(self.ao, dtype=float)
        ao_l3 = ao[seq['L3_idx']]
        ao_l5 = ao[L5_idx]

        ao_decay_ok = True
        if bool(self.wave5_ao_decay):
            mode = self._get_ao_decay_mode()
            ao_decay_ok = False
            if mode == "strict":
                if L5_idx >= 2:
                    ao_l5_m1 = ao[L5_idx - 1]
                    ao_l5_m2 = ao[L5_idx - 2]
                    ao_decay_ok = bool(ao_l5 > ao_l5_m1 > ao_l5_m2)
            else:
                if L5_idx >= 1:
                    ao_l5_m1 = ao[L5_idx - 1]
                    ao_decay_ok = bool(ao_l5 > ao_l5_m1)

            if not ao_decay_ok:
                if self.debug:
                    self.counters['ao_decay_fail'] += 1
                if not use_scoring:
                    trace["reason"] = "trigger_fail"
                    return trace
            elif self.debug:
                self.counters['ao_decay_pass'] += 1
        div_ok = (L5_p < seq['L3_p'] and ao_l5 > ao_l3 + self.ao_div_min)
        div_raw = (ao_l5 - ao_l3 - self.ao_div_min) / (abs(ao_l3) + 1e-12)
        div_scale = max(float(getattr(self, "div_scale", 0.5)), 1e-12)
        div_score = self._clip01(div_raw / div_scale)
        if not div_ok:
            if self.debug:
                self.counters['div_fail'] += 1
            if not use_scoring:
                trace["reason"] = "trigger_fail"
                return trace
        elif self.debug:
            self.counters['divergence_pass'] += 1

        zero_ok = True
        if self.require_zero_cross:
            zero_ok = self.has_zero_cross(seq['L3_idx'], L5_idx, 'bearish')
            if not zero_ok and self.debug:
                self.counters['zero_cross_fail'] += 1
            if not zero_ok and not use_scoring:
                trace["reason"] = "trigger_fail"
                return trace

        allowed_lag = int(self.max_trigger_lag)
        lag_val = max(0, i - L5_idx)
        lag_ok = lag_val <= allowed_lag
        lag_score = 1.0 - min(1.0, lag_val / max(allowed_lag, 1))
        if not lag_ok:
            if self.debug:
                self.counters['lag_fail'] += 1
            if not use_scoring:
                trace["reason"] = "trigger_fail"
                return trace

        candle_type, candle_score = self._candle_signal(i, 'bullish')
        candle_ok = candle_score > 0
        if not candle_ok:
            if self.debug:
                self.counters['candle_fail'] += 1
            if not use_scoring:
                trace["reason"] = "trigger_fail"
                return trace

        atr_sma_val = float(self.atr_sma[i])
        regime_score = 0.5
        if np.isfinite(atr_val) and np.isfinite(atr_sma_val) and atr_sma_val > 0:
            ratio = atr_val / atr_sma_val
            regime_score = self._clip01(1 - max(0.0, (ratio - 1.0) / float(getattr(self, "regime_r", 0.6))))
            trace["regime_ratio"] = ratio
        trace["regime_score"] = regime_score

        zero_score = 1.0 if (not self.require_zero_cross or zero_ok) else 0.0
        decay_score = 1.0 if (not self.wave5_ao_decay or ao_decay_ok) else 0.0
        trace["div_score"] = div_score
        trace["lag_score"] = lag_score
        trace["candle_score"] = candle_score
        trace["zero_score"] = zero_score
        trace["decay_score"] = decay_score

        score = 1.0
        if use_scoring:
            components = {
                "zone": trace.get("zone_score", 0.0),
                "div": div_score,
                "candle": candle_score,
                "lag": lag_score,
                "regime": regime_score,
                "zero": zero_score,
                "decay": decay_score,
            }
            score = self._combine_scores(components)
            trace["score"] = score
            if score < float(getattr(self, "score_threshold", 0.6)):
                trace["reason"] = "score_fail"
                if self.debug:
                    self.counters['score_fail'] += 1
                return trace
            if self.debug:
                self.counters['score_pass'] += 1
        else:
            self.summary["trigger_pass"] += 1
            trace["trigger_ok"] = 1

        self.summary["trigger_pass"] += int(use_scoring)
        if use_scoring:
            trace["trigger_ok"] = 1

        allowed, gate_reason = self._can_place_order()
        if not allowed:
            trace["reason"] = gate_reason
            return trace
        trace["entry_ok"] = 1

        buffer = float(getattr(self, 'spread_price', 0.0) or 0.0)

        trigger_low = self._low[i]
        trigger_high = self._high[i]

        if bool(getattr(self, "sl_at_wave5_extreme", True)):
            sl = float(L5_p) - buffer
        else:
            sl = float(trigger_low) - buffer

        if self.entry_mode == 'close':
            entry = self._close[i]
        else:
            candle_body = self._get_candle_body_atr(i)
            if candle_body > float(self.max_body_atr):
                if self.debug:
                    self.counters['break_body_fail'] += 1
                trace["reason"] = "entry_fail"
                return trace

            atr_val = float(self.atr[i])
            entry = trigger_high
            sl = trigger_low - buffer - float(self.break_buffer_atr) * atr_val

        trace["sl_ok"] = 1
        base_size = float(getattr(self, "order_size", 0.2))
        if use_scoring and bool(getattr(self, "enable_size_by_score", False)):
            score_adj = self._clip01(score)
            min_mult = float(getattr(self, "min_size_mult", 0.5))
            size_mult = min(1.0, max(min_mult, min_mult + (1.0 - min_mult) * score_adj))
            base_size *= size_mult
            trace["size_mult"] = size_mult
        if base_size <= 0:
            trace["reason"] = "size_zero"
            return trace
        entry_price_for_size = float(entry)

        placed_any = False
        order_attempts = 0

        if bool(self.tp_split):
            tp1 = float(seq['H4_p'])
            tp2 = seq['H4_p'] + 0.618 * (seq['H0_p'] - L5_p)
            tp2 = max(tp2, tp1)

            if self.entry_mode == 'break':
                if self._low[i] <= sl or self._high[i] >= tp2:
                    if self.debug:
                        self.counters['same_bar_ambiguous_fail'] += 1
                    trace["reason"] = "entry_fail"
                    return trace

            split_ratio = float(self.tp_split_ratio)
            size1 = base_size * split_ratio
            size2 = base_size * (1.0 - split_ratio)
            order_size1, size_dbg1 = self._final_units(size1, entry_price_for_size, sl, return_details=True)
            order_size2, size_dbg2 = self._final_units(size2, entry_price_for_size, sl, return_details=True)

            if order_size1 < 1 and order_size2 < 1:
                fail_reason = size_dbg1.get("fail_reason") or size_dbg2.get("fail_reason") or "size_zero"
                trace["reason"] = fail_reason if fail_reason in {"size_zero", "sl_invalid"} else "size_zero"
                if trace["reason"] == "sl_invalid":
                    trace["sl_ok"] = 0
                trace["size_ok"] = 0
                return trace

            if self.debug:
                self._dbg(f"[BUY SPLIT] entry={entry:.5f} sl={sl:.5f} tp1={tp1:.5f} tp2={tp2:.5f} sizes={size1:.2f}/{size2:.2f}")

            if self.entry_mode == 'close':
                self.summary["order_attempts"] += 1
                order_attempts += 1
                try:
                    o1 = self.buy(sl=sl, tp=tp1, size=order_size1)
                    if o1 is not None:
                        placed_any = True
                        self.summary["orders_placed"] += 1
                except (ValueError, AssertionError, RuntimeError) as e:
                    self.summary["exceptions_count"] += 1
                    self._dbg(f"[W5 ORDER FAIL] exc={repr(e)}")
                self.summary["order_attempts"] += 1
                order_attempts += 1
                try:
                    o2 = self.buy(sl=sl, tp=tp2, size=order_size2)
                    if o2 is not None:
                        placed_any = True
                        self.summary["orders_placed"] += 1
                except (ValueError, AssertionError, RuntimeError) as e:
                    self.summary["exceptions_count"] += 1
                    self._dbg(f"[W5 ORDER FAIL] exc={repr(e)}")
            else:
                self.summary["order_attempts"] += 1
                order_attempts += 1
                try:
                    o1 = self.buy(stop=trigger_high, sl=sl, tp=tp1, size=order_size1)
                    if o1 is not None:
                        placed_any = True
                        self.summary["orders_placed"] += 1
                except (ValueError, AssertionError, RuntimeError) as e:
                    self.summary["exceptions_count"] += 1
                    self._dbg(f"[W5 ORDER FAIL] exc={repr(e)}")
                self.summary["order_attempts"] += 1
                order_attempts += 1
                try:
                    o2 = self.buy(stop=trigger_high, sl=sl, tp=tp2, size=order_size2)
                    if o2 is not None:
                        placed_any = True
                        self.summary["orders_placed"] += 1
                except (ValueError, AssertionError, RuntimeError) as e:
                    self.summary["exceptions_count"] += 1
                    self._dbg(f"[W5 ORDER FAIL] exc={repr(e)}")

        else:
            tp_mode = str(getattr(self, "tp_mode", "hybrid")).lower()
            tp_2r = entry + self.tp_r * (entry - sl)

            if tp_mode == "rr":
                tp = tp_2r
            elif tp_mode == "wave4":
                tp = float(seq["H4_p"])
            else:
                tp_wave4 = float(seq["H4_p"])
                valid_candidates = []
                if tp_2r > entry:
                    valid_candidates.append(("2R", tp_2r))
                if tp_wave4 > entry:
                    valid_candidates.append(("Wave4", tp_wave4))
                if len(valid_candidates) == 0:
                    tp = tp_2r
                elif len(valid_candidates) == 1:
                    _, tp = valid_candidates[0]
                else:
                    candidates_with_distance = [
                        (source, tp_val, tp_val - entry) for source, tp_val in valid_candidates
                    ]
                    _, tp, _ = min(candidates_with_distance, key=lambda x: x[2])

            if self.entry_mode == 'break':
                if self._low[i] <= sl or self._high[i] >= tp:
                    if self.debug:
                        self.counters['same_bar_ambiguous_fail'] += 1
                    trace["reason"] = "entry_fail"
                    return trace

            final_size, size_dbg = self._final_units(base_size, entry_price_for_size, sl, return_details=True)
            if final_size < 1:
                fail_reason = size_dbg.get("fail_reason") or "size_zero"
                trace["reason"] = fail_reason if fail_reason in {"size_zero", "sl_invalid"} else "size_zero"
                if trace["reason"] == "sl_invalid":
                    trace["sl_ok"] = 0
                trace["size_ok"] = 0
                return trace

            self.summary["order_attempts"] += 1
            order_attempts += 1
            try:
                if self.entry_mode == 'close':
                    order = self.buy(sl=sl, tp=tp, size=final_size)
                else:
                    order = self.buy(stop=trigger_high, sl=sl, tp=tp, size=final_size)
                if order is not None:
                    placed_any = True
                    self.summary["orders_placed"] += 1
            except (ValueError, AssertionError, RuntimeError) as e:
                self.summary["exceptions_count"] += 1
                self._dbg(f"[W5 ORDER FAIL] exc={repr(e)}")

        if placed_any:
            trace["size_ok"] = 1
            trace["reason"] = trace.get("reason", "")
            self.summary["entry_pass"] += 1
            self.last_signal_idx = i
            if self.debug:
                self.counters['entries'] += 1
        else:
            trace["reason"] = trace.get("reason") or "entry_fail"
        return trace
