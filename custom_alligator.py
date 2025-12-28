from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from alligator_fractal import AlligatorFractal, AlligatorFractalClassic
from bt3 import fetch_data, run_backtest
from reporting import export_equity_curve_csv, export_trades_csv


def _parse_timeframe(tf: str) -> str:
    tf_norm = tf.strip().lower()
    mapping = {
        "1d": "1D",
        "d1": "1D",
        "1h": "1H",
        "h1": "1H",
        "4h": "4H",
        "h4": "4H",
        "15m": "15T",
        "m15": "15T",
        "30m": "30T",
        "m30": "30T",
        "5m": "5T",
        "m5": "5T",
    }
    if tf_norm not in mapping:
        raise ValueError(f"Unsupported timeframe '{tf}'. Use one of: {sorted(mapping)}")
    return mapping[tf_norm]


def _load_data(data_path: Optional[str], asset: Optional[str], tf: Optional[str]) -> pd.DataFrame:
    if data_path:
        if data_path.startswith("http://") or data_path.startswith("https://"):
            df = pd.read_csv(data_path)
        else:
            path = Path(data_path)
            if not path.exists():
                raise FileNotFoundError(f"Data file not found: {data_path}")
            if path.suffix.lower() in {".parquet", ".pq"}:
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path)

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
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                df = df.set_index(col)
                break
        if not isinstance(df.index, pd.DatetimeIndex):
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
            df = df.set_index(df.columns[0])

        if "Volume" not in df.columns:
            df["Volume"] = 0
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df[~df.index.duplicated(keep="first")].sort_index()
        return df

    if asset and tf:
        return fetch_data(asset, tf)

    raise ValueError("Provide --data <path|url> or both --asset and --tf for remote fetch.")


def _resample_ohlcv(df: pd.DataFrame, tf: Optional[str]) -> pd.DataFrame:
    if not tf:
        return df
    rule = _parse_timeframe(tf)
    ohlc = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    return df.resample(rule).agg(ohlc)


def _ensure_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    required = ["Open", "High", "Low", "Close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if "Volume" not in df.columns:
        df["Volume"] = 0
    return df


def _filter_range(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if start:
        df = df.loc[pd.to_datetime(start):]
    if end:
        df = df.loc[:pd.to_datetime(end)]
    return df


def _assert_identical(df_a: pd.DataFrame, df_b: pd.DataFrame) -> None:
    if len(df_a) != len(df_b):
        raise AssertionError("DataFrames differ in length.")
    if not df_a.index.equals(df_b.index):
        raise AssertionError("DataFrames differ in index.")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if not np.allclose(df_a[col].to_numpy(), df_b[col].to_numpy(), equal_nan=True):
            raise AssertionError(f"DataFrames differ in column '{col}'.")


def _stats_to_json(stats) -> dict:
    result = {}
    for key, value in stats.items():
        if str(key).startswith("_"):
            continue
        if isinstance(value, (np.floating, np.integer)):
            result[key] = value.item()
        elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
            result[key] = str(value)
        elif isinstance(value, (float, int, str, bool)) or value is None:
            result[key] = value
        else:
            result[key] = str(value)
    return result


def _metric_value(stats, keys: Iterable[str]):
    for key in keys:
        if key in stats:
            return stats[key]
    return None


def _is_number(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value)


def _comparison_table(strict_stats, classic_stats) -> pd.DataFrame:
    metrics = [
        ("Return %", ["Return [%]", "Return %"]),
        ("Buy&Hold %", ["Buy & Hold Return [%]", "Buy&Hold Return [%]", "Buy&Hold %"]),
        ("# Trades", ["# Trades", "Trades"]),
        ("Win Rate %", ["Win Rate [%]", "Win Rate %"]),
        ("Profit Factor", ["Profit Factor"]),
        ("Expectancy %", ["Expectancy [%]", "Expectancy %"]),
        ("Max Drawdown %", ["Max. Drawdown [%]", "Max Drawdown [%]", "Max Drawdown %"]),
        ("Avg Trade %", ["Avg. Trade [%]", "Avg Trade [%]", "Avg Trade %"]),
        ("Avg Duration", ["Avg. Trade Duration", "Avg Trade Duration"]),
        ("Sharpe", ["Sharpe Ratio", "Sharpe"]),
    ]

    rows = []
    for label, keys in metrics:
        strict_val = _metric_value(strict_stats, keys)
        classic_val = _metric_value(classic_stats, keys)
        if _is_number(strict_val) and _is_number(classic_val):
            delta = classic_val - strict_val
        else:
            delta = None
        rows.append({
            "metric": label,
            "strict": strict_val,
            "classic": classic_val,
            "delta": delta,
        })

    return pd.DataFrame(rows)


def run_comparison(
    data: pd.DataFrame,
    *,
    cash: float,
    commission: float,
    spread_pips: Optional[float],
    outdir: Path,
    eps: Optional[float] = None,
    exclusive_orders: bool = False,
    export: bool = True,
    print_table: bool = True,
) -> dict:
    strict_df = data.copy()
    classic_df = data.copy()
    _assert_identical(strict_df, classic_df)

    strategy_params = {"eps": eps} if eps is not None else None

    strict_stats = run_backtest(
        data=strict_df,
        strategy=AlligatorFractal,
        cash=cash,
        commission=commission,
        spread_pips=spread_pips,
        exclusive_orders=exclusive_orders,
        strategy_params=strategy_params,
    )

    classic_stats = run_backtest(
        data=classic_df,
        strategy=AlligatorFractalClassic,
        cash=cash,
        commission=commission,
        spread_pips=spread_pips,
        exclusive_orders=exclusive_orders,
        strategy_params=strategy_params,
    )

    comparison = _comparison_table(strict_stats, classic_stats)

    if export:
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "strict_stats.json").write_text(json.dumps(_stats_to_json(strict_stats), indent=2))
        (outdir / "classic_stats.json").write_text(json.dumps(_stats_to_json(classic_stats), indent=2))

        export_trades_csv(strict_stats, outdir / "strict_trades.csv")
        export_equity_curve_csv(strict_stats, outdir / "strict_equity.csv")
        export_trades_csv(classic_stats, outdir / "classic_trades.csv")
        export_equity_curve_csv(classic_stats, outdir / "classic_equity.csv")
        comparison.to_csv(outdir / "comparison.csv", index=False)

    if print_table:
        print("=" * 70)
        print("Strict Strategy Stats")
        print("=" * 70)
        print(strict_stats)
        print("=" * 70)
        print("Classic Strategy Stats")
        print("=" * 70)
        print(classic_stats)
        print("\nComparison (Strict vs Classic)")
        print(comparison.to_string(index=False))

    return {
        "strict": strict_stats,
        "classic": classic_stats,
        "comparison": comparison,
    }


if __name__ == "__main__":
    currency = "USDJPY"
    timeframe = "1h"
    data_path = None

    data = _load_data(data_path, currency, timeframe)
    data = _ensure_ohlc(data)
    data = _filter_range(data, start=None, end=None)
    data = _resample_ohlcv(data, timeframe)
    data = data.dropna(subset=["Open", "High", "Low", "Close"])

    if data.empty:
        raise ValueError("No data available after filtering/resampling.")

    run_comparison(
        data,
        cash=10000.0,
        commission=0.0,
        spread_pips=1.5,
        outdir=Path("reports/usdjpy_h1"),
        eps=None,
        exclusive_orders=True,
        export=True,
        print_table=True,
    )
