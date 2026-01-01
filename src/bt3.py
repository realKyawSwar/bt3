# bt3.py
from __future__ import annotations

import urllib.request
import urllib.error
from io import StringIO
from typing import Optional, Type

import numpy as np
import pandas as pd
from backtesting import Backtest


SUPPORTED_SYMBOLS = {
    "AUDJPY", "AUDUSD", "EURCHF", "EURGBP", "EURJPY", "EURUSD",
    "GBPJPY", "GBPUSD", "USDCAD", "USDCHF", "USDJPY", "XAUUSD",
}


def _map_timeframe_suffix(timeframe: str) -> str:
    """Map user timeframe like '1d','4h','1h' to ejtrader suffix like 'd1','h4','h1'."""
    tf = timeframe.strip().lower()
    mapping = {
        "1d": "d1", "d": "d1",
        "4h": "h4",
        "1h": "h1",
    }
    if tf not in mapping:
        raise ValueError(f"Unsupported timeframe '{timeframe}'. Use one of: {sorted(mapping)}")
    return mapping[tf]


def _target_range_for_symbol(symbol: str) -> tuple[float, float]:
    s = symbol.upper()
    if s.endswith("JPY"):
        return (50.0, 300.0)
    if s == "XAUUSD":
        return (800.0, 4000.0)
    return (0.5, 3.0)


def _interval_distance(x: float, lo: float, hi: float) -> float:
    if lo <= x <= hi:
        return 0.0
    if x < lo:
        return lo - x
    return x - hi


def _choose_price_divisor(symbol: str, median_close: float) -> int:
    """Choose divisor from [1,10,100,1000,10000] that puts median into plausible range."""
    lo, hi = _target_range_for_symbol(symbol)
    mid = (lo + hi) / 2.0
    candidates = [1, 10, 100, 1000, 10000, 100000]


    best_d = 1
    best_score = float("inf")

    for d in candidates:
        scaled = median_close / d
        # Primary: within range => score = distance to midpoint
        if lo <= scaled <= hi:
            score = abs(scaled - mid)
        else:
            # Secondary: distance to interval (prefer closer to range)
            score = 10_000 + _interval_distance(scaled, lo, hi)
        if score < best_score:
            best_score = score
            best_d = d

    return best_d


def _ohlc_sanity_ok(df: pd.DataFrame) -> bool:
    """Basic OHLC sanity: High >= max(Open,Close) and Low <= min(Open,Close) for most rows."""
    o = df["Open"].to_numpy()
    h = df["High"].to_numpy()
    l = df["Low"].to_numpy()
    c = df["Close"].to_numpy()

    valid = ~(np.isnan(o) | np.isnan(h) | np.isnan(l) | np.isnan(c))
    if valid.sum() < 100:
        return True  # too little data to judge

    o = o[valid]; h = h[valid]; l = l[valid]; c = c[valid]
    bad = (h < np.maximum(o, c)) | (l > np.minimum(o, c)) | (h < l)
    bad_ratio = bad.mean()
    return bad_ratio <= 0.20  # allow some noisy rows


def fetch_data(symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Fetch historical OHLCV from ejtraderLabs repo, standardize columns/index,
    and apply robust auto-scaling for integer-scaled FX prices.

    Returns DataFrame with DateTimeIndex and columns: Open, High, Low, Close, Volume
    """
    symbol_upper = symbol.strip().upper()
    if symbol_upper not in SUPPORTED_SYMBOLS:
        raise ValueError(f"Unsupported symbol '{symbol}'. Supported: {sorted(SUPPORTED_SYMBOLS)}")

    suffix = _map_timeframe_suffix(timeframe)
    url = (
        "https://raw.githubusercontent.com/ejtraderLabs/historical-data/main/"
        f"{symbol_upper}/{symbol_upper}{suffix}.csv"
    )

    try:
        print(f"Fetching data from: {url}")
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read().decode("utf-8")

        df = pd.read_csv(StringIO(data))

        # Standardize column names
        column_mapping = {}
        for col in df.columns:
            cl = col.lower()
            if cl == "open":
                column_mapping[col] = "Open"
            elif cl == "high":
                column_mapping[col] = "High"
            elif cl == "low":
                column_mapping[col] = "Low"
            elif cl == "close":
                column_mapping[col] = "Close"
            elif cl in ("volume", "vol"):
                column_mapping[col] = "Volume"
        if column_mapping:
            df = df.rename(columns=column_mapping)

        # Parse datetime index
        date_cols = ["Date", "date", "timestamp", "Timestamp", "time", "Time", "datetime", "Datetime"]
        idx_set = False
        for col in date_cols:
            if col in df.columns:
                dt = pd.to_datetime(df[col], errors="coerce")
                if dt.notna().sum() > 0:
                    df[col] = dt
                    df = df.set_index(col)
                    idx_set = True
                    break
        if not idx_set:
            # fall back to first column
            first_col = df.columns[0]
            df[first_col] = pd.to_datetime(df[first_col], errors="coerce")
            df = df.set_index(first_col)

        df = df[~df.index.isna()]

        # Sort & de-dup index
        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()

        required_cols = ["Open", "High", "Low", "Close"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if "Volume" not in df.columns:
            df["Volume"] = 0

        # Ensure numeric floats
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # Robust auto-scaling
        median_close = float(pd.Series(df["Close"]).dropna().median())
        if median_close and np.isfinite(median_close):
            divisor = _choose_price_divisor(symbol_upper, median_close)
            if divisor != 1:
                preview = median_close / divisor
                for c in ["Open", "High", "Low", "Close"]:
                    df[c] = df[c] / float(divisor)

                # Sanity check; if broken, revert
                if not _ohlc_sanity_ok(df):
                    for c in ["Open", "High", "Low", "Close"]:
                        df[c] = df[c] * float(divisor)
                    divisor = 1
                else:
                    print(
                        f"Auto-scaled prices for {symbol_upper} by /{divisor} "
                        f"(median_close {median_close:g} -> {preview:g})"
                    )

        # Enforce exact OHLCV columns
        df = df[["Open", "High", "Low", "Close", "Volume"]]

        # Store symbol so run_backtest can infer it if needed
        df.attrs["symbol"] = symbol_upper
        df.attrs["timeframe"] = timeframe

        print(f"Successfully loaded {len(df)} rows of data")
        return df

    except urllib.error.HTTPError as e:
        raise Exception(f"HTTP Error {e.code}: Failed to fetch data from {url}.")
    except urllib.error.URLError as e:
        raise Exception(f"URL Error: Failed to fetch data from {url}. Reason: {e.reason}")
    except Exception as e:
        raise Exception(f"Failed to fetch data from {url}: {str(e)}")


def _default_pip_size(symbol: Optional[str]) -> float:
    if not symbol:
        return 0.0001
    s = symbol.upper()
    if s.endswith("JPY"):
        return 0.01
    if s == "XAUUSD":
        return 0.1
    return 0.0001


def run_backtest(
    data: pd.DataFrame,
    strategy: Type,
    cash: float = 100000.0,
    commission: float = 0.0002,
    strategy_params: Optional[dict] = None,
    spread_pips: Optional[float] = None,
    pip_size: Optional[float] = None,
    symbol: Optional[str] = None,
    **kwargs
) -> dict:
    """Run a backtest using backtesting.py.

    If spread_pips is provided:
      - spread_price is injected via run(**params)
      - recommend commission=0.0 (FX is primarily spread, not % commission)
    """
    sym = symbol or data.attrs.get("symbol")

    bt = Backtest(data, strategy, cash=cash, commission=commission, finalize_trades=True, **kwargs)
    params = dict(strategy_params or {})
    if spread_pips is not None:
        ps = pip_size if pip_size is not None else _default_pip_size(sym)
        params["spread_price"] = float(spread_pips) * float(ps)
    stats = bt.run(**params)
    return stats
