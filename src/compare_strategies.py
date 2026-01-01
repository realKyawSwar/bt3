from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from alligator_fractal import (
    AlligatorFractal,
    AlligatorFractalClassic,
    AlligatorFractalPullback,
    AlligatorParams,
    compute_atr_ohlc,
    compute_htf_bias,
)
from bt3 import fetch_data, run_backtest
from reporting import export_equity_curve_csv, export_trades_csv
from wave5_ao import Wave5AODivergenceStrategy


STRATEGY_REGISTRY = {
    "wave5": Wave5AODivergenceStrategy,
}


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


def _print_stats(label: str, stats) -> None:
    print("=" * 70)
    print(label)
    print("=" * 70)
    print(stats)


def _summary_table(stats_by_name: dict[str, dict]) -> pd.DataFrame:
    metrics = [
        ("Return %", ["Return [%]", "Return %"]),
        ("Max DD %", ["Max. Drawdown [%]", "Max Drawdown [%]", "Max Drawdown %"]),
        ("Sharpe", ["Sharpe Ratio", "Sharpe"]),
        ("Profit Factor", ["Profit Factor"]),
        ("Expectancy %", ["Expectancy [%]", "Expectancy %"]),
        ("# Trades", ["# Trades", "Trades"]),
    ]
    rows = []
    for name, stats in stats_by_name.items():
        row = {"strategy": name}
        for label, keys in metrics:
            row[label] = _metric_value(stats, keys)
        rows.append(row)
    return pd.DataFrame(rows)


def compute_gating_debug(
    df: pd.DataFrame,
    use_htf_bias: bool,
    use_vol_filter: bool,
    htf_tf: str,
    atr_period: int,
    atr_long: int,
    cfg: AlligatorParams,
) -> dict:
    if use_htf_bias:
        bias_series = compute_htf_bias(df, htf_tf, cfg)
    else:
        bias_series = pd.Series("neutral", index=df.index, dtype=object)

    bias_counts = bias_series.value_counts(normalize=True)
    bias_dist = {
        "bullish": float(bias_counts.get("bullish", 0.0)),
        "bearish": float(bias_counts.get("bearish", 0.0)),
        "neutral": float(bias_counts.get("neutral", 0.0)),
    }

    if use_vol_filter:
        atr = compute_atr_ohlc(df, atr_period)
        atr_sma = atr.rolling(atr_long, min_periods=atr_long).mean()
        vol_ok_series = (atr > 0.9 * atr_sma).fillna(False)
    else:
        vol_ok_series = pd.Series(True, index=df.index, dtype=bool)

    both_ok = (bias_series != "neutral") & vol_ok_series
    return {
        "bias_dist": bias_dist,
        "vol_ok": float(vol_ok_series.mean()),
        "both_ok": float(both_ok.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare strict vs classic Alligator+Fractal strategies.")
    parser.add_argument(
        "--mode",
        choices=["alligator", *sorted(STRATEGY_REGISTRY)],
        default="alligator",
        help="Strategy mode to run (default: alligator compare).",
    )
    parser.add_argument("--data", help="Path or URL to CSV/Parquet data.")
    parser.add_argument("--asset", help="Symbol for remote fetch via bt3.fetch_data.")
    parser.add_argument("--tf", help="Timeframe (e.g. H4, H1, 15m).")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", help="End date (YYYY-MM-DD).")
    parser.add_argument("--cash", type=float, default=10000.0)
    parser.add_argument("--commission", type=float, default=0.0)
    parser.add_argument("--spread", type=float, default=None, help="FX spread in pips.")
    parser.add_argument("--eps", type=float, default=None, help="Optional epsilon override for stop entries.")
    parser.add_argument("--no-htf-bias", action="store_true", default=False, help="Disable H4 bias filter.")
    parser.add_argument("--no-vol-filter", action="store_true", default=False, help="Disable ATR regime filter.")
    parser.add_argument("--htf", default="4h", help="Higher timeframe for bias (e.g. 4h, 1d).")
    parser.add_argument("--atr", type=int, default=14, help="ATR period for volatility filter.")
    parser.add_argument("--atr-long", type=int, default=100, help="ATR long SMA length.")
    parser.add_argument("--exclusive_orders", action="store_true", default=False)
    parser.add_argument("--outdir", default="reports/", help="Output directory for reports.")
    parser.add_argument("--pullback-k", type=float, default=None, help="Override pullback_k_atr for pullback strategy.")
    parser.add_argument("--touch-teeth", action="store_true", default=False, help="Require pullback to touch teeth.")

    parser.add_argument("--wave5-swing-window", type=int, default=Wave5AODivergenceStrategy.swing_window)
    parser.add_argument("--wave5-fib-tol", type=float, default=Wave5AODivergenceStrategy.fib_tol_atr)
    parser.add_argument("--wave5-div-threshold", type=float, default=Wave5AODivergenceStrategy.ao_div_min)
    parser.add_argument("--wave5-entry-mode", choices=["close", "break"], default=Wave5AODivergenceStrategy.entry_mode)
    parser.add_argument("--wave5-tp-r", type=float, default=Wave5AODivergenceStrategy.tp_r)
    parser.add_argument("--wave5-tp-mode", choices=["rr", "wave4", "hybrid"], default=Wave5AODivergenceStrategy.tp_mode, help="TP mode: rr (2R TP), wave4 (Wave4 level), or hybrid (closer of the two).")
    parser.add_argument("--wave5-min-w3-atr", type=float, default=Wave5AODivergenceStrategy.min_w3_atr, help="Min wave3 length in ATR units.")
    parser.add_argument("--wave5-break-buffer-atr", type=float, default=Wave5AODivergenceStrategy.break_buffer_atr, help="Buffer distance in ATR for break stop placement.")
    parser.add_argument("--wave5-max-body-atr", type=float, default=Wave5AODivergenceStrategy.max_body_atr, help="Max candle body size in ATR units to allow break entry.")
    parser.add_argument("--wave5-debug", action="store_true", default=Wave5AODivergenceStrategy.debug)
    parser.add_argument("--wave5-trigger-lag", type=int, default=Wave5AODivergenceStrategy.max_trigger_lag, help="Max bars after H5/L5 to allow trigger.")
    parser.add_argument("--wave5-require-zero-cross", dest="wave5_require_zero_cross", action="store_true")
    parser.add_argument("--wave5-no-require-zero-cross", dest="wave5_require_zero_cross", action="store_false")
    parser.set_defaults(wave5_require_zero_cross=Wave5AODivergenceStrategy.require_zero_cross)
    parser.add_argument("--wave5-sl-extreme", dest="wave5_sl_extreme", action="store_true")
    parser.add_argument("--wave5-sl-trigger", dest="wave5_sl_extreme", action="store_false")
    parser.set_defaults(wave5_sl_extreme=True)
    parser.add_argument("--wave5-require-ext-touch", action="store_true", default=Wave5AODivergenceStrategy.require_ext_touch, help="Require Wave5 extreme to touch fib zone.")
    parser.add_argument(
        "--wave5-zone-mode",
        choices=["trigger", "extreme", "either"],
        default=Wave5AODivergenceStrategy.zone_mode,
        help=(
            "Fib zone test price source: trigger=entry trigger price, extreme=Wave5 extreme (H5/L5), either="
            "pass if trigger OR extreme is inside zone"
        ),
    )
    
    # Upgrade 1: Wave5 AO decay exhaustion
    parser.add_argument("--wave5-ao-decay", action="store_true", default=Wave5AODivergenceStrategy.wave5_ao_decay, help="Require AO decay at Wave5 extreme.")
    
    # Upgrade 2: Wave5 minimum extension
    parser.add_argument("--wave5-min-w5-ext", type=float, default=Wave5AODivergenceStrategy.min_w5_ext, help="Minimum Wave5 extension relative to Wave3.")
    
    # Upgrade 3: Partial TP with split orders
    parser.add_argument("--wave5-tp-split", action="store_true", default=Wave5AODivergenceStrategy.tp_split, help="Enable partial TP with split orders (tp1 at Wave4, tp2 at 0.618 retrace).")
    parser.add_argument("--wave5-tp-split-ratio", type=float, default=Wave5AODivergenceStrategy.tp_split_ratio, help="Position size ratio for first order (0-1).")
    
    # Upgrade 4: ATR expansion regime filter
    parser.add_argument("--wave5-atr-long", type=int, default=Wave5AODivergenceStrategy.atr_long, help="ATR long SMA period for regime filter.")
    parser.add_argument("--wave5-atr-expand-k", type=float, default=Wave5AODivergenceStrategy.atr_expand_k, help="Skip trades when ATR > k * atr_sma.")

    args = parser.parse_args()

    df = _load_data(args.data, args.asset, args.tf)
    df = _ensure_ohlc(df)
    df = _filter_range(df, args.start, args.end)
    df = _resample_ohlcv(df, args.tf)
    df = _sanitize_ohlcv(df)
    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    if df.empty:
        raise ValueError("No data available after filtering/resampling.")

    print(_data_fingerprint(df))

    if args.mode == "wave5":
        wave5_params = {
            "swing_window": args.wave5_swing_window,
            "fib_tol_atr": args.wave5_fib_tol,
            "ao_div_min": args.wave5_div_threshold,
            "require_zero_cross": args.wave5_require_zero_cross,
            "entry_mode": args.wave5_entry_mode,
            "tp_r": args.wave5_tp_r,
            "tp_mode": args.wave5_tp_mode,
            "debug": args.wave5_debug,
            "min_w3_atr": args.wave5_min_w3_atr,
            "max_trigger_lag": args.wave5_trigger_lag,
            "break_buffer_atr": args.wave5_break_buffer_atr,
            "max_body_atr": args.wave5_max_body_atr,
            "asset": args.asset or df.attrs.get("symbol"),
            "sl_at_wave5_extreme": args.wave5_sl_extreme,
            "require_ext_touch": args.wave5_require_ext_touch,
            # Upgrade 1: Wave5 AO decay exhaustion
            "wave5_ao_decay": args.wave5_ao_decay,
            # Upgrade 2: Wave5 minimum extension
            "min_w5_ext": args.wave5_min_w5_ext,
            # Upgrade 3: Partial TP with split orders
            "tp_split": args.wave5_tp_split,
            "tp_split_ratio": args.wave5_tp_split_ratio,
            # Upgrade 4: ATR expansion regime filter
            "atr_long": args.wave5_atr_long,
            "atr_expand_k": args.wave5_atr_expand_k,
            "zone_mode": args.wave5_zone_mode,
        }

        # Force exclusive_orders=False when tp_split is enabled to allow placing 2 orders
        # This ensures deterministic behavior for split TP mode
        wave5_exclusive = False if args.wave5_tp_split else args.exclusive_orders

        wave5_stats = run_backtest(
            data=df,
            strategy=STRATEGY_REGISTRY["wave5"],
            cash=args.cash,
            commission=args.commission,
            spread_pips=args.spread,
            exclusive_orders=wave5_exclusive,
            strategy_params=wave5_params,
        )

        # Print final debug counters if debug mode is enabled
        strat = wave5_stats._strategy
        if getattr(strat, "debug", False) and hasattr(strat, "counters"):
            print("FINAL COUNTERS:", strat.counters)

        _print_stats("Wave5 Strategy Stats", wave5_stats)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        asset_name = args.asset if args.asset else "data"
        tf_name = args.tf if args.tf else "custom"
        run_dir = f"{asset_name}_{tf_name}_wave5_{timestamp}"

        outdir = Path(args.outdir) / run_dir
        outdir.mkdir(parents=True, exist_ok=True)

        stats_dir = outdir / "stats"
        trades_dir = outdir / "trades"
        equity_dir = outdir / "equity"

        stats_dir.mkdir(exist_ok=True)
        trades_dir.mkdir(exist_ok=True)
        equity_dir.mkdir(exist_ok=True)

        (stats_dir / "wave5_stats.json").write_text(json.dumps(_stats_to_json(wave5_stats), indent=2))
        export_trades_csv(wave5_stats, trades_dir / "wave5_trades.csv")
        export_equity_curve_csv(wave5_stats, equity_dir / "wave5_equity.csv")

        print(f"\nAll reports saved to: {outdir}")
        return

    gate_debug = compute_gating_debug(
        df,
        use_htf_bias=not args.no_htf_bias,
        use_vol_filter=not args.no_vol_filter,
        htf_tf=_parse_timeframe(args.htf),
        atr_period=args.atr,
        atr_long=args.atr_long,
        cfg=AlligatorParams(),
    )
    print(f"HTF bias %: {gate_debug['bias_dist']}")
    print(f"VOL ok %: {gate_debug['vol_ok']:.2f}")
    print(f"BOTH ok %: {gate_debug['both_ok']:.2f}")

    strict_df = df.copy()
    classic_df = df.copy()
    pullback_df = df.copy()
    _assert_identical(strict_df, classic_df)
    _assert_identical(strict_df, pullback_df)

    base_params = {
        "use_htf_bias": not args.no_htf_bias,
        "use_vol_filter": not args.no_vol_filter,
        "htf_tf": _parse_timeframe(args.htf),
        "atr_period": args.atr,
        "atr_long": args.atr_long,
    }
    if args.eps is not None:
        base_params["eps"] = args.eps

    pullback_params = dict(base_params)
    if args.pullback_k is not None:
        pullback_params["pullback_k_atr"] = args.pullback_k
    if args.touch_teeth:
        pullback_params["require_touch_teeth"] = True

    strict_stats = run_backtest(
        data=strict_df,
        strategy=AlligatorFractal,
        cash=args.cash,
        commission=args.commission,
        spread_pips=args.spread,
        exclusive_orders=args.exclusive_orders,
        strategy_params=base_params,
    )

    classic_stats = run_backtest(
        data=classic_df,
        strategy=AlligatorFractalClassic,
        cash=args.cash,
        commission=args.commission,
        spread_pips=args.spread,
        exclusive_orders=args.exclusive_orders,
        strategy_params=base_params,
    )

    pullback_stats = run_backtest(
        data=pullback_df,
        strategy=AlligatorFractalPullback,
        cash=args.cash,
        commission=args.commission,
        spread_pips=args.spread,
        exclusive_orders=args.exclusive_orders,
        strategy_params=pullback_params,
    )

    _print_stats("Strict Strategy Stats", strict_stats)
    _print_stats("Classic Strategy Stats", classic_stats)
    _print_stats("Pullback Strategy Stats", pullback_stats)

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
    (stats_dir / "pullback_stats.json").write_text(json.dumps(_stats_to_json(pullback_stats), indent=2))

    # Save trades to trades subdirectory
    export_trades_csv(strict_stats, trades_dir / "strict_trades.csv")
    export_trades_csv(classic_stats, trades_dir / "classic_trades.csv")
    export_trades_csv(pullback_stats, trades_dir / "pullback_trades.csv")
    
    # Save equity curves to equity subdirectory
    export_equity_curve_csv(strict_stats, equity_dir / "strict_equity.csv")
    export_equity_curve_csv(classic_stats, equity_dir / "classic_equity.csv")
    export_equity_curve_csv(pullback_stats, equity_dir / "pullback_equity.csv")

    # Save comparisons to root of run directory
    comparison_strict_classic = _comparison_table(strict_stats, classic_stats, "strict", "classic")
    comparison_strict_pullback = _comparison_table(strict_stats, pullback_stats, "strict", "pullback")
    comparison_classic_pullback = _comparison_table(classic_stats, pullback_stats, "classic", "pullback")
    comparison_strict_classic.to_csv(outdir / "comparison_strict_classic.csv", index=False)
    comparison_strict_pullback.to_csv(outdir / "comparison_strict_pullback.csv", index=False)
    comparison_classic_pullback.to_csv(outdir / "comparison_classic_pullback.csv", index=False)

    summary = _summary_table({
        "strict": strict_stats,
        "classic": classic_stats,
        "pullback": pullback_stats,
    })
    print("\nSummary")
    print(summary.to_string(index=False))

    print("\nComparison (Strict vs Classic)")
    print(comparison_strict_classic.to_string(index=False))
    print("\nComparison (Strict vs Pullback)")
    print(comparison_strict_pullback.to_string(index=False))
    print("\nComparison (Classic vs Pullback)")
    print(comparison_classic_pullback.to_string(index=False))
    print(f"\nAll reports saved to: {outdir}")


if __name__ == "__main__":
    main()
