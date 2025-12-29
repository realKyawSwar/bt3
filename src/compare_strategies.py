from __future__ import annotations

import argparse
import json
from datetime import datetime
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
    df_tf = df.attrs.get("timeframe")
    if df_tf is not None and str(df_tf).strip().lower() == tf.strip().lower():
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


def _data_fingerprint(df: pd.DataFrame) -> str:
    start = df.index.min() if not df.empty else None
    end = df.index.max() if not df.empty else None
    try:
        infer_freq = pd.infer_freq(df.index)
    except ValueError:
        infer_freq = None
    median_close = df["Close"].median() if "Close" in df.columns and not df.empty else None
    attrs = dict(df.attrs) if df.attrs else {}
    return (
        "DATA "
        f"rows={len(df)} "
        f"start={start} "
        f"end={end} "
        f"infer_freq={infer_freq} "
        f"median_close={median_close} "
        f"attrs={attrs}"
    )


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


def _print_stats(label: str, stats) -> None:
    print("=" * 70)
    print(label)
    print("=" * 70)
    print(stats)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare strict vs classic Alligator+Fractal strategies.")
    parser.add_argument("--data", help="Path or URL to CSV/Parquet data.")
    parser.add_argument("--asset", help="Symbol for remote fetch via bt3.fetch_data.")
    parser.add_argument("--tf", help="Timeframe (e.g. H4, H1, 15m).")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", help="End date (YYYY-MM-DD).")
    parser.add_argument("--cash", type=float, default=10000.0)
    parser.add_argument("--commission", type=float, default=0.0)
    parser.add_argument("--spread", type=float, default=None, help="FX spread in pips.")
    parser.add_argument("--eps", type=float, default=None, help="Optional epsilon override for stop entries.")
    parser.add_argument("--exclusive_orders", action="store_true", default=False)
    parser.add_argument("--outdir", default="reports/", help="Output directory for reports.")

    args = parser.parse_args()

    df = _load_data(args.data, args.asset, args.tf)
    df = _ensure_ohlc(df)
    df = _filter_range(df, args.start, args.end)
    df = _resample_ohlcv(df, args.tf)
    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    if df.empty:
        raise ValueError("No data available after filtering/resampling.")

    print(_data_fingerprint(df))

    strict_df = df.copy()
    classic_df = df.copy()
    _assert_identical(strict_df, classic_df)

    strategy_params = {"eps": args.eps} if args.eps is not None else None

    strict_stats = run_backtest(
        data=strict_df,
        strategy=AlligatorFractal,
        cash=args.cash,
        commission=args.commission,
        spread_pips=args.spread,
        exclusive_orders=args.exclusive_orders,
        strategy_params=strategy_params,
    )

    classic_stats = run_backtest(
        data=classic_df,
        strategy=AlligatorFractalClassic,
        cash=args.cash,
        commission=args.commission,
        spread_pips=args.spread,
        exclusive_orders=args.exclusive_orders,
        strategy_params=strategy_params,
    )

    _print_stats("Strict Strategy Stats", strict_stats)
    _print_stats("Classic Strategy Stats", classic_stats)

    # Generate timestamp for unique output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create dynamic output directory name based on asset and timeframe
    asset_name = args.asset if args.asset else "data"
    tf_name = args.tf if args.tf else "custom"
    run_dir = f"{asset_name}_{tf_name}_{timestamp}"
    
    outdir = Path(args.outdir) / run_dir
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    stats_dir = outdir / "stats"
    trades_dir = outdir / "trades"
    equity_dir = outdir / "equity"
    
    stats_dir.mkdir(exist_ok=True)
    trades_dir.mkdir(exist_ok=True)
    equity_dir.mkdir(exist_ok=True)

    # Save stats to stats subdirectory
    (stats_dir / "strict_stats.json").write_text(json.dumps(_stats_to_json(strict_stats), indent=2))
    (stats_dir / "classic_stats.json").write_text(json.dumps(_stats_to_json(classic_stats), indent=2))

    # Save trades to trades subdirectory
    export_trades_csv(strict_stats, trades_dir / "strict_trades.csv")
    export_trades_csv(classic_stats, trades_dir / "classic_trades.csv")
    
    # Save equity curves to equity subdirectory
    export_equity_curve_csv(strict_stats, equity_dir / "strict_equity.csv")
    export_equity_curve_csv(classic_stats, equity_dir / "classic_equity.csv")

    # Save comparison to root of run directory
    comparison = _comparison_table(strict_stats, classic_stats)
    comparison.to_csv(outdir / "comparison.csv", index=False)

    print("\nComparison (Strict vs Classic)")
    print(comparison.to_string(index=False))
    print(f"\nAll reports saved to: {outdir}")


if __name__ == "__main__":
    main()
