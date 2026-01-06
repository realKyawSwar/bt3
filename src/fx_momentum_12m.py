from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


def _standardize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize common OHLC column names and datetime index similar to compare_strategies._load_data.
    """
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

    date_cols = ["Date", "date", "timestamp", "Timestamp", "time", "Time", "datetime", "Datetime"]
    idx_set = False
    for col in date_cols:
        if col in df.columns:
            dt = pd.to_datetime(df[col], errors="coerce")
            if dt.notna().sum() > 0:
                df = df.set_index(dt)
                idx_set = True
                break
    if not idx_set:
        first_col = df.columns[0]
        df = df.set_index(pd.to_datetime(df[first_col], errors="coerce"))

    if "Volume" not in df.columns:
        df["Volume"] = 0
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[~df.index.isna()]
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


def load_fx_prices_from_csv(path_or_paths: Union[str, Path, Iterable[Union[str, Path]]]) -> Dict[str, pd.DataFrame]:
    """
    Load one or many CSVs containing FX pair prices.

    Returns dict keyed by inferred pair symbol (file stem uppercased).
    """
    paths: Sequence[Union[str, Path]]
    if isinstance(path_or_paths, (str, Path)):
        paths = [path_or_paths]
    else:
        paths = list(path_or_paths)

    result: Dict[str, pd.DataFrame] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            inner = sorted(path.glob("*.csv"))
            if inner:
                result.update(load_fx_prices_from_csv(inner))
            continue
        if not path.exists():
            raise FileNotFoundError(f"Price file not found: {path}")
        df = pd.read_csv(path)
        df = _standardize_ohlc(df)
        pair = path.stem.upper()
        result[pair] = df
    return result


def resample_to_month_end_close(df: pd.DataFrame, price_col: str = "Close") -> pd.Series:
    """
    Resample OHLCV data to month-end close price.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("resample_to_month_end_close expects a DatetimeIndex")
    if price_col not in df.columns:
        raise ValueError(f"Column '{price_col}' not found in dataframe")
    return df[price_col].resample("M").last().dropna()


def normalize_pairs_to_usd_per_ccy(prices_by_pair: Mapping[str, pd.Series]) -> Dict[str, pd.Series]:
    """
    Convert pair prices to USD-per-currency series.
    Pair names must be USD crosses (XXXUSD or USDXXX).
    """
    normalized: Dict[str, pd.Series] = {}
    for pair, series in prices_by_pair.items():
        if series is None:
            continue
        p = pair.upper()
        if p.endswith("USD") and len(p) == 6:
            ccy = p[:3]
            normalized[ccy] = series
        elif p.startswith("USD") and len(p) == 6:
            ccy = p[3:]
            normalized[ccy] = 1.0 / series
    return normalized


def build_currency_panel(prices_by_pair: Mapping[str, pd.Series]) -> pd.DataFrame:
    """
    Build a panel of USD-per-currency monthly series from pair closes.
    """
    normalized = normalize_pairs_to_usd_per_ccy(prices_by_pair)
    if not normalized:
        raise ValueError("No USD-cross pairs provided; cannot build currency panel.")
    panel = pd.DataFrame(normalized).sort_index()
    return panel


def compute_momentum_scores(panel: pd.DataFrame, carry_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Compute 12-month sum of log returns, optionally adding carry differential.
    carry_df should have same index frequency and columns for currencies (including USD).
    """
    log_rets = np.log(panel / panel.shift(1))
    momentum = log_rets.rolling(12, min_periods=12).sum()

    if carry_df is not None and not carry_df.empty:
        carry = carry_df.copy()
        carry.columns = [c.upper() for c in carry.columns]
        carry = carry.reindex(momentum.index).ffill()
        if "USD" in carry.columns:
            carry_spread = carry.sub(carry["USD"], axis=0)
        else:
            carry_spread = carry
        carry_component = (carry_spread / 12.0).rolling(12, min_periods=12).sum()
        carry_component = carry_component.reindex(columns=momentum.columns)
        momentum = momentum.add(carry_component, fill_value=0.0)

    return momentum


def build_weights(scores: pd.DataFrame, k_top: int = 2, k_bottom: int = 2) -> pd.DataFrame:
    """
    Equal-weight long top k and short bottom k currencies by score. Net 0 weights.
    """
    weights = []
    for _, row in scores.iterrows():
        row_clean = row.dropna()
        w = pd.Series(0.0, index=scores.columns, dtype=float)
        if not row_clean.empty:
            sorted_scores = row_clean.sort_values(ascending=False)
            top_n = min(int(k_top), len(sorted_scores))
            top = sorted_scores.head(top_n)
            bottom_candidates = sorted_scores.tail(max(0, int(k_bottom)))
            bottom = bottom_candidates[~bottom_candidates.index.isin(top.index)]
            if len(top) > 0:
                w[top.index] = 1.0 / float(len(top))
            if len(bottom) > 0:
                w[bottom.index] = -1.0 / float(len(bottom))
        weights.append(w)
    return pd.DataFrame(weights, index=scores.index).sort_index()


def map_currency_weights_to_pairs(weights: pd.DataFrame, available_pairs: Sequence[str]) -> pd.DataFrame:
    """
    Map currency weights to tradable USD-cross pairs.
    """
    pairs_upper = [p.upper() for p in available_pairs]
    data = []
    for _, row in weights.iterrows():
        pair_w = {p: 0.0 for p in pairs_upper}
        for ccy, w in row.items():
            if pd.isna(w) or w == 0:
                continue
            pair1 = f"{ccy}USD"
            pair2 = f"USD{ccy}"
            if pair1 in pair_w:
                pair_w[pair1] += float(w)
            elif pair2 in pair_w:
                pair_w[pair2] += float(-w)
        data.append(pd.Series(pair_w))
    return pd.DataFrame(data, index=weights.index).sort_index().loc[:, pairs_upper]


def compute_portfolio_returns(
    pair_weights: pd.DataFrame,
    pair_monthly_returns: pd.DataFrame,
    target_vol: Optional[float] = None,
    vol_lookback: int = 12,
    max_leverage: float = 3.0,
    vol_floor: float = 1e-6,
) -> Tuple[pd.Series, pd.DataFrame, pd.Series]:
    """
    Apply shift(1) weights to monthly returns. If target_vol is set, apply volatility targeting.
    Returns tuple of (portfolio_returns, levered_pair_weights, leverage_series).
    """
    common_idx = pair_weights.index.intersection(pair_monthly_returns.index)
    pair_weights = pair_weights.reindex(common_idx).sort_index()
    pair_monthly_returns = pair_monthly_returns.reindex(common_idx).sort_index()

    if pair_weights.empty or len(pair_weights) < 2:
        return pd.Series(dtype=float), pair_weights, pd.Series(dtype=float)

    levered_weights = pair_weights.copy()
    leverage_series = pd.Series(index=pair_weights.index, dtype=float)
    realized = []
    idx_list = list(pair_weights.index)

    for i, reb_date in enumerate(idx_list):
        window = realized[-vol_lookback:]
        if target_vol is None:
            lev = 1.0
        else:
            if window:
                vol_annual = float(np.std(window, ddof=0)) * np.sqrt(12)
            else:
                vol_annual = None
            if vol_annual is None or not np.isfinite(vol_annual) or vol_annual <= 0:
                lev = 1.0
            else:
                lev = float(np.clip(target_vol / max(vol_annual, vol_floor), 0.0, max_leverage))
        leverage_series.loc[reb_date] = lev
        levered_weights.loc[reb_date] = pair_weights.loc[reb_date] * lev

        if i == 0:
            continue

        ret_date = idx_list[i]
        period_ret = float(
            (levered_weights.loc[idx_list[i - 1]].fillna(0.0) * pair_monthly_returns.loc[ret_date].fillna(0.0)).sum()
        )
        realized.append(period_ret)

    portfolio_returns = pd.Series(realized, index=idx_list[1:])
    return portfolio_returns, levered_weights, leverage_series


def prepare_monthly_closes(prices_by_pair: Mapping[str, pd.DataFrame], price_col: str = "Close") -> pd.DataFrame:
    """
    Convert raw OHLCV dict to DataFrame of month-end closes (columns per pair).
    """
    monthly = {}
    for pair, df in prices_by_pair.items():
        monthly[pair.upper()] = resample_to_month_end_close(df, price_col=price_col)
    df_out = pd.DataFrame(monthly)
    df_out = df_out.dropna(how="all").dropna(axis=1, how="all").sort_index()
    return df_out
