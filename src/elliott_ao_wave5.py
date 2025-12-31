import numpy as np
import pandas as pd
from backtesting import Strategy

WAVE5_DEFAULT_PARAMS = {
    "pivot_len": 3,
    "tol": 0.10,
    "stop_pad_atr": 0.10,
    "tp_r": 2.0,
    "min_swing_atr": 0.6,
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

def wave5_signals(
    df: pd.DataFrame,
    pivot_len: int = 3,
    tol: float = 0.10,
    stop_pad_atr: float = 0.10,
    tp_r: float = 2.0,
    min_swing_atr: float = 0.0,
    max_windows: int = 60
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

    # Helper to fetch AO at pivot idx
    def ao_at(idx): 
        return out["AO"].iloc[idx]

    # Scan recent impulse windows
    # We'll attempt SHORT setups (end of UP impulse) first; add LONG similarly after.
    for k in range(max(0, len(swings) - max_windows), len(swings) - 5):
        window = swings[k:k+6]
        types = "".join([w[2] for w in window])

        # Up impulse candidate: L H L H L H
        if types != "LHLHLH":
            continue

        L0, H1, L2, H3, L4, H5 = window

        # Elliott hard rules (strict)
        if not (L2[1] > L0[1]):  # wave2 not 100%
            continue
        if not (H3[1] > H1[1]):  # wave3 beyond wave1
            continue
        if not (L4[1] > H1[1]):  # wave4 no overlap
            continue

        # Wave5 fib extension zone
        imp = H3[1] - L0[1]
        if imp <= 0:
            continue
        t1 = L4[1] + 1.272 * imp
        # t2 = L4[1] + 1.618 * imp  # optional second target
        zone_ok = (H5[1] >= (t1 - tol * imp))
        if not zone_ok:
            continue

        # AO divergence: price HH but AO LH
        if not (H5[1] > H3[1] and ao_at(H5[0]) < ao_at(H3[0])):
            continue

        # Trigger candle AFTER H5 pivot (we can require i > H5.idx)
        # Since pivots confirm late, simplest: search next N bars after H5 index
        start_i = H5[0] + 1
        end_i = min(len(out) - 1, start_i + 10)

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
                break  # one signal per impulse

    return out


class ElliottAOWave5Strategy(Strategy):
    pivot_len = 3
    tol = 0.10
    stop_pad_atr = 0.10
    tp_r = 2.0
    min_swing_atr = 0.6

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
