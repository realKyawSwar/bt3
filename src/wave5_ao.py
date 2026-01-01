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


class Wave5AODivergenceStrategy(Strategy):
    swing_window = 2
    fib_levels = (1.272, 1.618)
    fib_tol_atr = 0.25
    overlap_tol_atr = 0.0  # NEW: allow small overlap tolerance
    ao_div_min = 0.0
    require_zero_cross = True
    trigger = ('engulfing', 'pin')
    entry_mode = 'close'          # close|break
    tp_r = 2.0
    pin_ratio = 2.0
    min_bars_between_signals = 5
    max_trigger_lag = 3            # Max bars after H5/L5 to trigger entry
    min_w3_atr = 1.0               # Min wave3 length in ATR units
    asset = 'UNKNOWN'              # Asset symbol for labeling/reference
    debug = False
    require_ext_touch = False  # if True, require H5/L5 extreme also tagged the zone

    def __init__(self, broker, data, params):
        super().__init__(broker, data, params)
        self.asset = params.get('asset', 'UNKNOWN')

    def init(self):
        # Numpy arrays for speed & correctness
        self._open = np.asarray(self.data.Open, dtype=float)
        self._high = np.asarray(self.data.High, dtype=float)
        self._low  = np.asarray(self.data.Low, dtype=float)
        self._close= np.asarray(self.data.Close, dtype=float)

        median = (self._high + self._low) / 2.0
        ao = _sma(median, 5) - _sma(median, 34)
        atr = _atr(self._high, self._low, self._close, 14)

        # Register with backtesting for plots/compat
        self.ao = self.I(lambda x=ao: x)
        self.atr = self.I(lambda x=atr: x)

        # Swing state
        self.swing_highs = []   # list of (idx, price)
        self.swing_lows = []
        self._added_high = set()
        self._added_low = set()
        self.last_confirmed = -1
        self.last_signal_idx = -10_000

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
                'entries': 0,
            }

    def _print_counters(self, i: int, direction: str) -> None:
        """Print debug counters before a trade entry."""
        if self.debug:
            print(f"[{i}] {direction.upper()} setup: swings={self.counters['swings_count']} "
                  f"type_match={self.counters['type_match']} elliott={self.counters['elliott_pass']} "
                  f"zone_fail={self.counters['zone_fail']} w3_size_fail={self.counters['w3_size_fail']} "
                  f"w3_short_fail={self.counters['w3_short_fail']} div_fail={self.counters['div_fail']} "
                  f"zero_cross_fail={self.counters['zero_cross_fail']} "
                  f"lag_fail={self.counters['lag_fail']} candle_fail={self.counters['candle_fail']} "
                  f"entries={self.counters['entries']}")

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

    def find_uptrend_sequence(self):
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

            atr_val = float(self.atr[-1])
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

    def find_downtrend_sequence(self):
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

            atr_val = float(self.atr[-1])
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

    def is_reversal_candle(self, i: int, direction: str) -> bool:
        if i < 1:
            return False
        o, c, h, l = self._open[i], self._close[i], self._high[i], self._low[i]
        o1, c1 = self._open[i-1], self._close[i-1]
        body = max(abs(o - c), 1e-12)

        if direction == 'bearish':
            if 'engulfing' in self.trigger and (c < o) and (c1 > o1) and (o > c1) and (c < o1):
                return True
            if 'pin' in self.trigger:
                upper = h - max(o, c)
                if upper > self.pin_ratio * body and c < (o + l) / 2:
                    return True

        if direction == 'bullish':
            if 'engulfing' in self.trigger and (c > o) and (c1 < o1) and (o < c1) and (c > o1):
                return True
            if 'pin' in self.trigger:
                lower = min(o, c) - l
                if lower > self.pin_ratio * body and c > (o + h) / 2:
                    return True

        return False

    # -------------------------
    # Main loop
    # -------------------------
    def next(self):
        i = len(self.data) - 1
        if i < 60:  # warm-up
            return

        self._update_swings(i)

        # Avoid repeated entries too close
        if i - self.last_signal_idx < int(self.min_bars_between_signals):
            return

        up_seq = self.find_uptrend_sequence()
        if up_seq:
            if self.debug:
                self.counters['type_match'] += 1
            self._handle_sell(i, up_seq)

        down_seq = self.find_downtrend_sequence()
        if down_seq:
            if self.debug:
                self.counters['type_match'] += 1
            self._handle_buy(i, down_seq)

    def _handle_sell(self, i: int, seq: dict):
        w3_len = seq['H3_p'] - seq['L2_p']
        if not np.isfinite(w3_len) or w3_len <= 0:
            return

        atr_val = float(self.atr[-1])
        if not np.isfinite(atr_val) or atr_val <= 0:
            return

        # Check wave3 minimum size requirement
        if w3_len < self.min_w3_atr * atr_val:
            if self.debug:
                self.counters['w3_size_fail'] += 1
            return

        ext_levels = [seq['L4_p'] + fib * w3_len for fib in self.fib_levels]
        post_start = seq['L4_idx'] + 1
        if post_start >= i:
            return

        highs = self._high[post_start:i+1]
        max_pos = int(np.argmax(highs))
        H5_idx = post_start + max_pos
        H5_p = float(highs[max_pos])
        # Zone check based on trigger candle location
        trigger_px = self._close[i] if self.entry_mode == "close" else self._high[i]
        in_zone_trigger = any(abs(trigger_px - ext) <= self.fib_tol_atr * atr_val for ext in ext_levels)

        # Optional: also require the Wave5 extreme to have tagged the zone
        in_zone_extreme = any(abs(H5_p - ext) <= self.fib_tol_atr * atr_val for ext in ext_levels)

        if self.require_ext_touch:
            in_zone = in_zone_trigger and in_zone_extreme
        else:
            in_zone = in_zone_trigger

        if not in_zone:
            if self.debug:
                self.counters['zone_fail'] += 1
            return

        if self.debug:
            self.counters['elliott_pass'] += 1

        # Wave3 not shortest among 1,3,5 (approx)
        w1_len = seq['H1_p'] - seq['L0_p']
        w5_len = H5_p - seq['L4_p']
        if w3_len < min(w1_len, w5_len):
            if self.debug:
                self.counters['w3_short_fail'] += 1
            return

        ao = np.asarray(self.ao, dtype=float)
        ao_h3 = ao[seq['H3_idx']]
        ao_h5 = ao[H5_idx]
        if not (H5_p > seq['H3_p'] and ao_h5 < ao_h3 - self.ao_div_min):
            if self.debug:
                self.counters['div_fail'] += 1
            return
        if self.debug:
            self.counters['divergence_pass'] += 1

        if self.require_zero_cross and not self.has_zero_cross(seq['H3_idx'], H5_idx, 'bullish'):
            if self.debug:
                self.counters['zero_cross_fail'] += 1
            return

        # Check trigger lag: must occur within max_trigger_lag bars after H5
        if i - H5_idx > int(self.max_trigger_lag):
            if self.debug:
                self.counters['lag_fail'] += 1
            return

        if not self.is_reversal_candle(i, 'bearish'):
            if self.debug:
                self.counters['candle_fail'] += 1
            return
        if self.debug:
            self._print_counters(i, 'sell')

        buffer = float(getattr(self, 'spread_price', 0.0) or 0.0)
        trigger_high = self._high[i]
        trigger_low = self._low[i]
        sl = trigger_high + buffer

        if self.entry_mode == 'close':
            entry = self._close[i]
            tp = entry - self.tp_r * (sl - entry)
            self.sell(sl=sl, tp=tp)
        else:
            entry = trigger_low
            tp = entry - self.tp_r * (sl - entry)
            self.sell(stop=entry, sl=sl, tp=tp)

        self.last_signal_idx = i
        if self.debug:
            self.counters['entries'] += 1

    def _handle_buy(self, i: int, seq: dict):
        w3_len = seq['H2_p'] - seq['L3_p']
        if not np.isfinite(w3_len) or w3_len <= 0:
            return

        atr_val = float(self.atr[-1])
        if not np.isfinite(atr_val) or atr_val <= 0:
            return

        # Check wave3 minimum size requirement
        if w3_len < self.min_w3_atr * atr_val:
            if self.debug:
                self.counters['w3_size_fail'] += 1
            return

        ext_levels = [seq['H4_p'] - fib * w3_len for fib in self.fib_levels]
        post_start = seq['H4_idx'] + 1
        if post_start >= i:
            return

        lows = self._low[post_start:i+1]
        min_pos = int(np.argmin(lows))
        L5_idx = post_start + min_pos
        L5_p = float(lows[min_pos])
        trigger_px = self._close[i] if self.entry_mode == "close" else self._low[i]
        in_zone_trigger = any(abs(trigger_px - ext) <= self.fib_tol_atr * atr_val for ext in ext_levels)
        in_zone_extreme = any(abs(L5_p - ext) <= self.fib_tol_atr * atr_val for ext in ext_levels)

        if self.require_ext_touch:
            in_zone = in_zone_trigger and in_zone_extreme
        else:
            in_zone = in_zone_trigger

        if not in_zone:
            if self.debug:
                self.counters['zone_fail'] += 1
            return

        if self.debug:
            self.counters['elliott_pass'] += 1

        w1_len = seq['H0_p'] - seq['L1_p']
        w5_len = seq['H4_p'] - L5_p
        if w3_len < min(w1_len, w5_len):
            if self.debug:
                self.counters['w3_short_fail'] += 1
            return

        ao = np.asarray(self.ao, dtype=float)
        ao_l3 = ao[seq['L3_idx']]
        ao_l5 = ao[L5_idx]
        if not (L5_p < seq['L3_p'] and ao_l5 > ao_l3 + self.ao_div_min):
            if self.debug:
                self.counters['div_fail'] += 1
            return
        if self.debug:
            self.counters['divergence_pass'] += 1

        if self.require_zero_cross and not self.has_zero_cross(seq['L3_idx'], L5_idx, 'bearish'):
            if self.debug:
                self.counters['zero_cross_fail'] += 1
            return

        # Check trigger lag: must occur within max_trigger_lag bars after L5
        if i - L5_idx > int(self.max_trigger_lag):
            if self.debug:
                self.counters['lag_fail'] += 1
            return

        if not self.is_reversal_candle(i, 'bullish'):
            if self.debug:
                self.counters['candle_fail'] += 1
            return
        if self.debug:
            self._print_counters(i, 'buy')

        buffer = float(getattr(self, 'spread_price', 0.0) or 0.0)
        trigger_low = self._low[i]
        trigger_high = self._high[i]
        sl = trigger_low - buffer

        if self.entry_mode == 'close':
            entry = self._close[i]
            tp = entry + self.tp_r * (entry - sl)
            self.buy(sl=sl, tp=tp)
        else:
            entry = trigger_high
            tp = entry + self.tp_r * (entry - sl)
            self.buy(stop=entry, sl=sl, tp=tp)

        self.last_signal_idx = i
        if self.debug:
            self.counters['entries'] += 1
