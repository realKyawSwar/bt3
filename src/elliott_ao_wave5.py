#elliott_ao_wave5.py
import numpy as np
import pandas as pd
from backtesting import Strategy

WAVE5_DEFAULT_PARAMS = {
    "pivot_len": 3,
    "tol": 0.10,
    "stop_pad_atr": 0.10,
    "tp_r": 2.0,
    "min_swing_atr": 0.6,
    "imp_mode": "w1",
    "overlap_mode": "soft",
    "trigger_bars": 20,
}

WAVE5_SYMBOL_PRESETS = {
    "XAUUSD": {"min_swing_atr": 1.0, "tol": 0.12, "stop_pad_atr": 0.15},
    "EURUSD": {"min_swing_atr": 0.6, "tol": 0.10, "stop_pad_atr": 0.10},
    "GBPUSD": {"min_swing_atr": 0.6, "tol": 0.10, "stop_pad_atr": 0.10},
}


def resolve_wave5_params(symbol: str | None, overrides: dict | None = None) -> dict:
    params = dict(WAVE5_DEFAULT_PARAMS)
    if symbol:
        params.update(WAVE5_SYMBOL_PRESETS.get(symbol.upper(), {}))
    if overrides:
        params.update({k: v for k, v in overrides.items() if v is not None})
    return params


def _sanitize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Volume" not in out.columns:
        out["Volume"] = 0
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if isinstance(out.index, pd.DatetimeIndex):
        out = out[~out.index.duplicated(keep="first")]
        if not out.index.is_monotonic_increasing:
            out = out.sort_index()
    return out

def sma(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).mean()

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def awesome_oscillator(df: pd.DataFrame) -> pd.Series:
    median = (df["High"] + df["Low"]) / 2.0
    return sma(median, 5) - sma(median, 34)

def pivots(df: pd.DataFrame, pivot_len: int = 3):
    """Return arrays pivot_high, pivot_low as booleans."""
    h = df["High"].values
    l = df["Low"].values
    n = len(df)
    ph = np.zeros(n, dtype=bool)
    pl = np.zeros(n, dtype=bool)

    for i in range(pivot_len, n - pivot_len):
        w_h = h[i - pivot_len:i + pivot_len + 1]
        w_l = l[i - pivot_len:i + pivot_len + 1]
        if h[i] == w_h.max():
            ph[i] = True
        if l[i] == w_l.min():
            pl[i] = True
    return ph, pl

def build_swings(df: pd.DataFrame, pivot_len: int = 3, min_swing_atr: float = 0.0):
    """Return list of (idx, price, 'H'/'L') alternating."""
    AO = df["AO"].values
    ATR = df["ATR"].values

    ph, pl = pivots(df, pivot_len=pivot_len)
    swings = []
    last_type = None

    for i in range(len(df)):
        if not (ph[i] or pl[i]):
            continue
        t = "H" if ph[i] else "L"
        price = df["High"].iloc[i] if t == "H" else df["Low"].iloc[i]

        # ignore weak swings
        if swings and min_swing_atr > 0:
            if abs(price - swings[-1][1]) < min_swing_atr * (ATR[i] if not np.isnan(ATR[i]) else 0):
                continue

        if not swings:
            swings.append((i, float(price), t))
            last_type = t
            continue

        # enforce alternation: if same type, keep more extreme
        if t == last_type:
            prev_i, prev_p, prev_t = swings[-1]
            if t == "H" and price > prev_p:
                swings[-1] = (i, float(price), t)
            elif t == "L" and price < prev_p:
                swings[-1] = (i, float(price), t)
            continue

        swings.append((i, float(price), t))
        last_type = t

    return swings

def is_bear_engulf(df: pd.DataFrame, i: int) -> bool:
    o, c = df["Open"].iloc[i], df["Close"].iloc[i]
    o1, c1 = df["Open"].iloc[i-1], df["Close"].iloc[i-1]
    return (c < o) and (c <= o1) and (o >= c1)

def is_bear_pin(df: pd.DataFrame, i: int) -> bool:
    h, l, o, c = df["High"].iloc[i], df["Low"].iloc[i], df["Open"].iloc[i], df["Close"].iloc[i]
    upper = h - max(o, c)
    body = abs(c - o)
    rng = h - l
    return (rng > 0) and (upper >= 2 * max(body, 1e-9)) and (upper >= 0.6 * rng)

def is_bull_engulf(df: pd.DataFrame, i: int) -> bool:
    o, c = df["Open"].iloc[i], df["Close"].iloc[i]
    o1, c1 = df["Open"].iloc[i-1], df["Close"].iloc[i-1]
    return (c > o) and (c >= o1) and (o <= c1)

def is_bull_pin(df: pd.DataFrame, i: int) -> bool:
    h, l, o, c = df["High"].iloc[i], df["Low"].iloc[i], df["Open"].iloc[i], df["Close"].iloc[i]
    lower = min(o, c) - l
    body = abs(c - o)
    rng = h - l
    return (rng > 0) and (lower >= 2 * max(body, 1e-9)) and (lower >= 0.6 * rng)

def _impulse_size(
    mode: str,
    L0: tuple,
    H1: tuple,
    L2: tuple,
    H3: tuple,
    H0: tuple,
    L1: tuple,
    H2: tuple,
    L3: tuple,
) -> float:
    if mode == "w1":
        return abs(H1[1] - L0[1]) if L0 else abs(L1[1] - H0[1])
    if mode == "w3":
        return abs(H3[1] - L2[1]) if L2 else abs(L3[1] - H2[1])
    return abs(H3[1] - L0[1]) if L0 else abs(L3[1] - H0[1])

def _overlap_ok_up(mode: str, L4: tuple, H1: tuple, atr_val: float) -> bool:
    if mode == "strict":
        return L4[1] > H1[1]
    return L4[1] > H1[1] - 0.2 * atr_val

def _overlap_ok_down(mode: str, H4: tuple, L1: tuple, atr_val: float) -> bool:
    if mode == "strict":
        return H4[1] < L1[1]
    return H4[1] < L1[1] + 0.2 * atr_val

def wave5_signals(
    df: pd.DataFrame,
    pivot_len: int = 3,
    tol: float = 0.10,
    stop_pad_atr: float = 0.10,
    tp_r: float = 2.0,
    min_swing_atr: float = 0.0,
    max_windows: int = 2000,
    imp_mode: str = "w1",
    overlap_mode: str = "soft",
    trigger_bars: int = 20,
    debug: bool = False,
    debug_top_n: int = 20
) -> pd.DataFrame:
    """
    Output columns:
      signal: 1 long, -1 short, 0 none
      sl, tp, entry_price
    """
    out = _sanitize_ohlcv(df)
    out["AO"] = awesome_oscillator(out)
    out["ATR"] = atr(out, 14)
    out["signal"] = 0
    out["sl"] = np.nan
    out["tp"] = np.nan
    out["entry_price"] = np.nan

    swings = build_swings(out, pivot_len=pivot_len, min_swing_atr=min_swing_atr)
    debug_info = {
        "swings_count": len(swings),
        "windows_checked": 0,
        "type_match": 0,
        "elliott_pass": 0,
        "zone_pass": 0,
        "div_pass": 0,
        "trigger_pass": 0,
        "signals_emitted_short": 0,
        "signals_emitted_long": 0,
        "fail_wave2": 0,
        "fail_wave3": 0,
        "fail_overlap": 0,
        "fail_zone": 0,
        "fail_div": 0,
        "fail_trigger": 0,
    }

    # Helper to fetch AO at pivot idx
    def ao_at(idx): 
        return out["AO"].iloc[idx]

    # Scan recent impulse windows
    for k in range(max(0, len(swings) - max_windows), len(swings) - 5):
        debug_info["windows_checked"] += 1
        window = swings[k:k+6]
        types = "".join([w[2] for w in window])
        if types not in {"LHLHLH", "HLHLHL"}:
            continue
        debug_info["type_match"] += 1

        if types == "LHLHLH":
            L0, H1, L2, H3, L4, H5 = window
            if not (L2[1] > L0[1]):
                debug_info["fail_wave2"] += 1
                continue
            if not (H3[1] > H1[1]):
                debug_info["fail_wave3"] += 1
                continue
            atr_val = out["ATR"].iloc[L4[0]]
            atr_val = float(atr_val) if np.isfinite(atr_val) else 0.0
            if not _overlap_ok_up(overlap_mode, L4, H1, atr_val):
                debug_info["fail_overlap"] += 1
                continue
            debug_info["elliott_pass"] += 1

            imp = _impulse_size(imp_mode, L0, H1, L2, H3, None, None, None, None)
            if imp <= 0:
                debug_info["fail_zone"] += 1
                continue
            t1 = L4[1] + 1.272 * imp
            zone_ok = (H5[1] >= (t1 - tol * imp))
            if not zone_ok:
                debug_info["fail_zone"] += 1
                continue
            debug_info["zone_pass"] += 1

            if not (H5[1] > H3[1] and ao_at(H5[0]) < ao_at(H3[0])):
                debug_info["fail_div"] += 1
                continue
            debug_info["div_pass"] += 1

            confirm_i = H5[0] + pivot_len
            start_i = confirm_i + 1
            if start_i >= len(out) - 1:
                debug_info["fail_trigger"] += 1
                continue
            end_i = min(len(out) - 1, start_i + trigger_bars)

            found = False
            for i in range(start_i, end_i + 1):
                if i < 1:
                    continue
                atr_val = out["ATR"].iloc[i]
                if not np.isfinite(atr_val):
                    continue
                if is_bear_engulf(out, i) or is_bear_pin(out, i):
                    entry = out["Close"].iloc[i]
                    sl = out["High"].iloc[i] + stop_pad_atr * atr_val
                    r = sl - entry
                    if not (r > 0 and np.isfinite(r)):
                        break
                    tp = entry - tp_r * r

                    out.at[out.index[i], "signal"] = -1
                    out.at[out.index[i], "entry_price"] = entry
                    out.at[out.index[i], "sl"] = sl
                    out.at[out.index[i], "tp"] = tp
                    found = True
                    debug_info["trigger_pass"] += 1
                    debug_info["signals_emitted_short"] += 1
                    break

            if not found:
                debug_info["fail_trigger"] += 1
            continue

        H0, L1, H2, L3, H4, L5 = window
        if not (H2[1] < H0[1]):
            debug_info["fail_wave2"] += 1
            continue
        if not (L3[1] < L1[1]):
            debug_info["fail_wave3"] += 1
            continue
        atr_val = out["ATR"].iloc[H4[0]]
        atr_val = float(atr_val) if np.isfinite(atr_val) else 0.0
        if not _overlap_ok_down(overlap_mode, H4, L1, atr_val):
            debug_info["fail_overlap"] += 1
            continue
        debug_info["elliott_pass"] += 1

        imp = _impulse_size(imp_mode, None, None, None, None, H0, L1, H2, L3)
        if imp <= 0:
            debug_info["fail_zone"] += 1
            continue
        t1 = H4[1] - 1.272 * imp
        zone_ok = (L5[1] <= (t1 + tol * imp))
        if not zone_ok:
            debug_info["fail_zone"] += 1
            continue
        debug_info["zone_pass"] += 1

        if not (L5[1] < L3[1] and ao_at(L5[0]) > ao_at(L3[0])):
            debug_info["fail_div"] += 1
            continue
        debug_info["div_pass"] += 1

        confirm_i = L5[0] + pivot_len
        start_i = confirm_i + 1
        if start_i >= len(out) - 1:
            debug_info["fail_trigger"] += 1
            continue
        end_i = min(len(out) - 1, start_i + trigger_bars)

        found = False
        for i in range(start_i, end_i + 1):
            if i < 1:
                continue
            atr_val = out["ATR"].iloc[i]
            if not np.isfinite(atr_val):
                continue
            if is_bull_engulf(out, i) or is_bull_pin(out, i):
                entry = out["Close"].iloc[i]
                sl = out["Low"].iloc[i] - stop_pad_atr * atr_val
                r = entry - sl
                if not (r > 0 and np.isfinite(r)):
                    break
                tp = entry + tp_r * r

                out.at[out.index[i], "signal"] = 1
                out.at[out.index[i], "entry_price"] = entry
                out.at[out.index[i], "sl"] = sl
                out.at[out.index[i], "tp"] = tp
                found = True
                debug_info["trigger_pass"] += 1
                debug_info["signals_emitted_long"] += 1
                break

        if not found:
            debug_info["fail_trigger"] += 1

    out.attrs["wave5_debug"] = debug_info
    if debug:
        print("Wave5 debug counters:")
        for key, value in debug_info.items():
            print(f"  {key}: {value}")
        signal_idx = out.index[out["signal"] != 0]
        if len(signal_idx) == 0:
            print("Wave5 signals: none")
        else:
            sample_first = list(signal_idx[:3])
            sample_last = list(signal_idx[-3:])
            if len(signal_idx) > debug_top_n:
                print(f"Wave5 signals (showing first/last 3 of {len(signal_idx)}):")
            else:
                print(f"Wave5 signals ({len(signal_idx)} total):")
            print(f"  first: {sample_first}")
            print(f"  last:  {sample_last}")

    return out


class ElliottAOWave5Strategy(Strategy):
    pivot_len = 3
    tol = 0.10
    stop_pad_atr = 0.10
    tp_r = 2.0
    min_swing_atr = 0.6
    imp_mode = "w1"
    overlap_mode = "soft"
    trigger_bars = 20
    max_windows = None

    def init(self):
        raw_df = self.data.df
        cleaned_df = _sanitize_ohlcv(raw_df)
        signals = wave5_signals(
            cleaned_df,
            pivot_len=int(self.pivot_len),
            tol=float(self.tol),
            stop_pad_atr=float(self.stop_pad_atr),
            tp_r=float(self.tp_r),
            min_swing_atr=float(self.min_swing_atr),
            imp_mode=str(self.imp_mode),
            overlap_mode=str(self.overlap_mode),
            trigger_bars=int(self.trigger_bars),
            max_windows=int(self.max_windows) if self.max_windows is not None else None,
        )
        signals = signals.reindex(raw_df.index)
        self._signal = signals["signal"].fillna(0).astype(int).to_numpy()
        self._sl = signals["sl"].to_numpy(dtype=float)
        self._tp = signals["tp"].to_numpy(dtype=float)

    def _half_spread(self) -> float:
        return float(getattr(self, "spread_price", 0.0)) / 2.0

    def _force_sl_first(self) -> bool:
        if not self.position:
            return False
        sl = getattr(self.position, "sl", None)
        tp = getattr(self.position, "tp", None)
        if sl is None or tp is None:
            return False
        high = float(self.data.High[-1])
        low = float(self.data.Low[-1])
        sl = float(sl)
        tp = float(tp)
        if low <= sl <= high and low <= tp <= high:
            self.position.close()
            return True
        return False

    def next(self):
        i = len(self.data) - 1
        if self.position:
            self._force_sl_first()
            return
        if i >= len(self._signal):
            return
        signal = int(self._signal[i])
        if signal == 0:
            return
        sl = self._sl[i]
        tp = self._tp[i]
        if not np.isfinite(sl) or not np.isfinite(tp):
            return

        half_spread = self._half_spread()
        if signal == 1:
            sl_engine = sl - half_spread
            tp_engine = tp - half_spread
            self.buy(sl=sl_engine, tp=tp_engine)
        elif signal == -1:
            sl_engine = sl + half_spread
            tp_engine = tp + half_spread
            self.sell(sl=sl_engine, tp=tp_engine)
