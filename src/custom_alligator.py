from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from alligator_fractal import AlligatorFractal, AlligatorFractalClassic, AlligatorFractalPullback
from bt3 import fetch_data, run_backtest
from reporting import export_equity_curve_csv, export_trades_csv

# --- Optimized defaults (used when CLI args not provided) ---
OPT_DEFAULTS = dict(
    currency="GBPJPY",
    timeframe="1h",
    spread_pips=1.5,
    cash=10000.0,
    commission=0.0,
    exclusive_orders=True,
    use_htf_bias=True,
    use_vol_filter=True,
    htf_tf="4h",
    atr_period=14,
    atr_long=50,         # <-- optimized value from your latest run
    eps=None,
    outdir=Path("reports/gbpjpy_h1_opt"),
    pullback_k_atr=0.5,  # optional: only applies to pullback strategy
    require_touch_teeth=False,
)

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


def _comparison_table(stats_a, stats_b, label_a: str = "strict", label_b: str = "classic") -> pd.DataFrame:
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
        val_a = _metric_value(stats_a, keys)
        val_b = _metric_value(stats_b, keys)
        if _is_number(val_a) and _is_number(val_b):
            delta = val_b - val_a
        else:
            delta = None
        rows.append({
            "metric": label,
            label_a: val_a,
            label_b: val_b,
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
    use_htf_bias: bool = True,
    use_vol_filter: bool = True,
    htf_tf: str = "4h",
    atr_period: int = 14,
    atr_long: int = 100,
    pullback_k_atr: float = 0.5,
    require_touch_teeth: bool = False,
    exclusive_orders: bool = False,
    export: bool = True,
    print_table: bool = True,
) -> dict:
    print(_data_fingerprint(data))
    strict_df = data.copy()
    classic_df = data.copy()
    _assert_identical(strict_df, classic_df)

    strategy_params = {
        "use_htf_bias": use_htf_bias,
        "use_vol_filter": use_vol_filter,
        "htf_tf": htf_tf,
        "atr_period": atr_period,
        "atr_long": atr_long,
    }
    if eps is not None:
        strategy_params["eps"] = eps

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

    pullback_stats = run_backtest(
        data=data.copy(),
        strategy=AlligatorFractalPullback,
        cash=cash,
        commission=commission,
        spread_pips=spread_pips,
        exclusive_orders=exclusive_orders,
        strategy_params={
            **strategy_params,
            "pullback_k_atr": pullback_k_atr,
            "require_touch_teeth": require_touch_teeth,
        },
    )

    # comparisons
    comparison_sc = _comparison_table(strict_stats, classic_stats, "strict", "classic")
    comparison_sp = _comparison_table(strict_stats, pullback_stats, "strict", "pullback")
    comparison_cp = _comparison_table(classic_stats, pullback_stats, "classic", "pullback")

    if export:
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "strict_stats.json").write_text(json.dumps(_stats_to_json(strict_stats), indent=2))
        (outdir / "classic_stats.json").write_text(json.dumps(_stats_to_json(classic_stats), indent=2))
        (outdir / "pullback_stats.json").write_text(json.dumps(_stats_to_json(pullback_stats), indent=2))

        export_trades_csv(strict_stats, outdir / "strict_trades.csv")
        export_equity_curve_csv(strict_stats, outdir / "strict_equity.csv")
        export_trades_csv(classic_stats, outdir / "classic_trades.csv")
        export_equity_curve_csv(classic_stats, outdir / "classic_equity.csv")
        export_trades_csv(pullback_stats, outdir / "pullback_trades.csv")
        export_equity_curve_csv(pullback_stats, outdir / "pullback_equity.csv")
        comparison_sc.to_csv(outdir / "comparison_strict_classic.csv", index=False)
        comparison_sp.to_csv(outdir / "comparison_strict_pullback.csv", index=False)
        comparison_cp.to_csv(outdir / "comparison_classic_pullback.csv", index=False)

    pb_trades = pullback_stats.get("# Trades")
    strict_trades = strict_stats.get("# Trades")
    if _is_number(pb_trades) and _is_number(strict_trades):
        if pb_trades <= 0 or pb_trades < 0.5 * strict_trades:
            print(
                "WARNING: Pullback trade count collapsed; likely gating bug. "
                f"pullback={pb_trades}, strict={strict_trades}"
            )

    if print_table:
        print("=" * 70)
        print("Strict Strategy Stats")
        print("=" * 70)
        print(strict_stats)
        print("=" * 70)
        print("Classic Strategy Stats")
        print("=" * 70)
        print(classic_stats)
        print("=" * 70)
        print("Pullback Strategy Stats")
        print("=" * 70)
        print(pullback_stats)
        print("\nComparison (Strict vs Classic)")
        print(comparison_sc.to_string(index=False))
        print("\nComparison (Strict vs Pullback)")
        print(comparison_sp.to_string(index=False))
        print("\nComparison (Classic vs Pullback)")
        print(comparison_cp.to_string(index=False))

    return {
        "strict": strict_stats,
        "classic": classic_stats,
        "pullback": pullback_stats,
        "comparison_sc": comparison_sc,
        "comparison_sp": comparison_sp,
        "comparison_cp": comparison_cp,
    }


def run_single(
    data: pd.DataFrame,
    *,
    strategy_name: str,
    cash: float,
    commission: float,
    spread_pips: Optional[float],
    outdir: Path,
    eps: Optional[float] = None,
    use_htf_bias: bool = True,
    use_vol_filter: bool = True,
    htf_tf: str = "4h",
    atr_period: int = 14,
    atr_long: int = 100,
    exclusive_orders: bool = False,
    export: bool = True,
    print_table: bool = True,
    pullback_k_atr: Optional[float] = None,
    require_touch_teeth: bool = False,
) -> dict:
    print(_data_fingerprint(data))

    strategy_map = {
        "strict": AlligatorFractal,
        "classic": AlligatorFractalClassic,
        "pullback": AlligatorFractalPullback,
    }
    if strategy_name not in strategy_map:
        raise ValueError(f"Unsupported strategy: {strategy_name}")

    strategy_params = {
        "use_htf_bias": use_htf_bias,
        "use_vol_filter": use_vol_filter,
        "htf_tf": htf_tf,
        "atr_period": atr_period,
        "atr_long": atr_long,
    }
    if eps is not None:
        strategy_params["eps"] = eps
    if strategy_name == "pullback":
        if pullback_k_atr is not None:
            strategy_params["pullback_k_atr"] = pullback_k_atr
        if require_touch_teeth:
            strategy_params["require_touch_teeth"] = True

    stats = run_backtest(
        data=data.copy(),
        strategy=strategy_map[strategy_name],
        cash=cash,
        commission=commission,
        spread_pips=spread_pips,
        exclusive_orders=exclusive_orders,
        strategy_params=strategy_params,
    )

    if export:
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / f"{strategy_name}_stats.json").write_text(json.dumps(_stats_to_json(stats), indent=2))
        export_trades_csv(stats, outdir / f"{strategy_name}_trades.csv")
        export_equity_curve_csv(stats, outdir / f"{strategy_name}_equity.csv")

    if print_table:
        print("=" * 70)
        print(f"{strategy_name.title()} Strategy Stats")
        print("=" * 70)
        print(stats)

    return {
        strategy_name: stats,
    }


if __name__ == "__main__":

    
    import argparse

    parser = argparse.ArgumentParser(description="Compare or run Alligator+Fractal strategies.")

    # mode
    parser.add_argument("--strategy", choices=["strict", "classic", "pullback"], default=None,
                        help="If omitted, runs comparison (strict vs classic vs pullback).")

    # data selection
    parser.add_argument("--data", help="Path/URL to CSV/Parquet data.")
    parser.add_argument("--asset", help="Symbol for remote fetch via bt3.fetch_data.")
    parser.add_argument("--tf", help="Timeframe (e.g. 1h, 4h, 15m).")

    # backtest controls
    parser.add_argument("--start", help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", help="End date (YYYY-MM-DD).")
    parser.add_argument("--cash", type=float, default=None)
    parser.add_argument("--commission", type=float, default=None)
    parser.add_argument("--spread", type=float, default=None)
    parser.add_argument("--exclusive_orders", action="store_true", default=None)
    parser.add_argument("--eps", type=float, default=None)

    # filters
    parser.add_argument("--no-htf-bias", action="store_true", default=False)
    parser.add_argument("--no-vol-filter", action="store_true", default=False)
    parser.add_argument("--htf", default=None, help="HTF timeframe (e.g. 4h).")
    parser.add_argument("--atr", type=int, default=None)
    parser.add_argument("--atr-long", type=int, default=None)

    # pullback params (only used for pullback)
    parser.add_argument("--pullback-k", type=float, default=None)
    parser.add_argument("--touch-teeth", action="store_true", default=False)

    # output
    parser.add_argument("--outdir", default=None)

    args = parser.parse_args()

    # --- resolve defaults + overrides ---
    currency = args.asset or OPT_DEFAULTS["currency"]
    timeframe = args.tf or OPT_DEFAULTS["timeframe"]
    spread_pips = args.spread if args.spread is not None else OPT_DEFAULTS["spread_pips"]
    cash = args.cash if args.cash is not None else OPT_DEFAULTS["cash"]
    commission = args.commission if args.commission is not None else OPT_DEFAULTS["commission"]

    # exclusive_orders: default to optimized True unless explicitly disabled
    if args.exclusive_orders is None:
        exclusive_orders = OPT_DEFAULTS["exclusive_orders"]
    else:
        exclusive_orders = bool(args.exclusive_orders)

    use_htf_bias = OPT_DEFAULTS["use_htf_bias"] and (not args.no_htf_bias)
    use_vol_filter = OPT_DEFAULTS["use_vol_filter"] and (not args.no_vol_filter)

    htf_tf = args.htf or OPT_DEFAULTS["htf_tf"]
    atr_period = args.atr if args.atr is not None else OPT_DEFAULTS["atr_period"]
    atr_long = args.atr_long if args.atr_long is not None else OPT_DEFAULTS["atr_long"]

    eps = args.eps if args.eps is not None else OPT_DEFAULTS["eps"]

    outdir = Path(args.outdir) if args.outdir else OPT_DEFAULTS["outdir"]

    # pullback strategy tuning
    pullback_k_atr = args.pullback_k if args.pullback_k is not None else OPT_DEFAULTS["pullback_k_atr"]
    require_touch_teeth = bool(args.touch_teeth) or OPT_DEFAULTS["require_touch_teeth"]

    # --- load & prep data ---
    # data = _load_data(args.data, currency, timeframe)
    # Use optimized defaults when flags not provided
    asset = args.asset or "GBPJPY"
    tf = args.tf or "1h"

    df = _load_data(args.data, asset, tf)

    data = _ensure_ohlc(df)
    data = _filter_range(data, start=args.start, end=args.end)
    data = _resample_ohlcv(data, timeframe)
    data = data.dropna(subset=["Open", "High", "Low", "Close"])

    if data.empty:
        raise ValueError("No data available after filtering/resampling.")

    # --- run ---
    if args.strategy is None:
        # comparison default: strict vs classic vs pullback using optimized params
        run_comparison(
            data,
            cash=cash,
            commission=commission,
            spread_pips=spread_pips,
            outdir=outdir,
            eps=eps,
            use_htf_bias=use_htf_bias,
            use_vol_filter=use_vol_filter,
            htf_tf=htf_tf,
            atr_period=atr_period,
            atr_long=atr_long,
            pullback_k_atr=pullback_k_atr,
            require_touch_teeth=require_touch_teeth,
            exclusive_orders=exclusive_orders,
            export=True,
            print_table=True,
        )
    else:
        # single strategy mode (strict/classic/pullback)
        run_single(
            data,
            strategy_name=args.strategy,
            cash=cash,
            commission=commission,
            spread_pips=spread_pips,
            outdir=outdir,
            eps=eps,
            use_htf_bias=use_htf_bias,
            use_vol_filter=use_vol_filter,
            htf_tf=htf_tf,
            atr_period=atr_period,
            atr_long=atr_long,
            exclusive_orders=exclusive_orders,
            export=True,
            print_table=True,
            pullback_k_atr=pullback_k_atr,
            require_touch_teeth=require_touch_teeth,
        )
