from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from alligator_fractal import (
    AlligatorFractal,
    AlligatorFractalClassic,
    AlligatorFractalPullback,
    AlligatorFractalTrailing,
    AlligatorParams,
    compute_atr_ohlc,
    compute_htf_bias,
)
from bt3 import fetch_data, run_backtest
from reporting import export_equity_curve_csv, export_trades_csv


# --- Optimized defaults (used when CLI args not provided) ---
OPT_DEFAULTS = dict(
    currency="GBPJPY",
    timeframe="1h",
    spread_pips=3.5,
    cash=10000.0,
    commission=0.0,
    exclusive_orders=True,
    use_htf_bias=True,
    use_vol_filter=True,
    htf_tf="4h",
    atr_period=14,
    atr_long=50,          # optimized value
    eps=None,
    cancel_stale_orders=False,
    outdir=Path("reports/gbpjpy_h1_opt"),
    pullback_k_atr=0.5,   # only applies to pullback strategy
    require_touch_teeth=False,
)


def _parse_timeframe(tf: str) -> str:
    tf_norm = tf.strip().lower().replace(" ", "")
    mapping = {
        "1d": "1D",
        "d1": "1D",
        "daily": "1D",
        "1h": "1H",
        "h1": "1H",
        "60m": "1H",
        "1hour": "1H",
        "4h": "4H",
        "h4": "4H",
        "240m": "4H",
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
    # If a local/URL path is provided, load from it.
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

        # Standardize column names
        column_mapping = {}
        for col in df.columns:
            cl = str(col).lower()
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
            # fallback: find first column that looks like datetime (but avoid Open/High/Low/Close)
            for col in df.columns:
                if str(col) in ("Open", "High", "Low", "Close", "Volume"):
                    continue
                dt = pd.to_datetime(df[col], errors="coerce")
                if dt.notna().sum() > 0 and dt.notna().mean() > 0.5:
                    df[col] = dt
                    df = df.set_index(col)
                    idx_set = True
                    break

        if not isinstance(df.index, pd.DatetimeIndex):
            # last resort: try first column
            first_col = df.columns[0]
            dt = pd.to_datetime(df[first_col], errors="coerce")
            df[first_col] = dt
            df = df.set_index(first_col)

        if "Volume" not in df.columns:
            df["Volume"] = 0

        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df[~df.index.duplicated(keep="first")].sort_index()

        # Preserve attrs for bt3.run_backtest pip sizing
        if asset:
            df.attrs["symbol"] = asset.strip().upper()
        if tf:
            df.attrs["timeframe"] = tf.strip()

        return df

    # Otherwise, fetch remote
    if asset and tf:
        return fetch_data(asset, tf)

    raise ValueError("Provide --data <path|url> or both --asset and --tf for remote fetch.")


def _resample_ohlcv(df: pd.DataFrame, tf: Optional[str]) -> pd.DataFrame:
    if not tf:
        return df

    df_tf = df.attrs.get("timeframe")
    if df_tf is not None and str(df_tf).strip().lower() == tf.strip().lower():
        return df

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex to resample.")

    saved_attrs = dict(df.attrs) if df.attrs else {}
    rule = _parse_timeframe(tf)

    ohlc = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    out = df.resample(rule).agg(ohlc)
    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    out = out[~out.index.duplicated(keep="first")].sort_index()
    out.attrs.update(saved_attrs)  # preserve symbol/timeframe for bt3.run_backtest
    out.attrs["timeframe"] = tf
    return out


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


def _comparison_table(stats_a, stats_b, label_a: str = "A", label_b: str = "B") -> pd.DataFrame:
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
        rows.append({"metric": label, label_a: val_a, label_b: val_b, "delta": delta})
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


def _json_safe(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def _export_manifest(
    outdir: Path,
    *,
    resolved: dict,
    gate_debug: dict,
    data_fp: str,
) -> None:
    manifest = {
        "resolved": resolved,
        "gating_debug": gate_debug,
        "data_fingerprint": data_fp,
        "opt_defaults": {k: _json_safe(v) for k, v in OPT_DEFAULTS.items()},
    }
    (outdir / "run_config.json").write_text(json.dumps(manifest, indent=2, default=_json_safe))


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
    cancel_stale_orders: bool = False,
) -> dict:
    data_fp = _data_fingerprint(data)
    print(data_fp)

    # Keep attrs for bt3.run_backtest pip sizing
    base_attrs = dict(data.attrs) if data.attrs else {}

    strict_df = data.copy()
    classic_df = data.copy()
    pullback_df = data.copy()
    trailing_df = data.copy()
    for d in (strict_df, classic_df, pullback_df, trailing_df):
        d.attrs.update(base_attrs)

    _assert_identical(strict_df, classic_df)

    gate_debug = compute_gating_debug(
        data,
        use_htf_bias=use_htf_bias,
        use_vol_filter=use_vol_filter,
        htf_tf=htf_tf,
        atr_period=atr_period,
        atr_long=atr_long,
        cfg=AlligatorParams(),
    )
    print(f"HTF bias %: {gate_debug['bias_dist']}")
    print(f"VOL ok %: {gate_debug['vol_ok']:.2f}")
    print(f"BOTH ok %: {gate_debug['both_ok']:.2f}")

    strategy_params = {
        "use_htf_bias": use_htf_bias,
        "use_vol_filter": use_vol_filter,
        "htf_tf": htf_tf,
        "atr_period": atr_period,
        "atr_long": atr_long,
        "cancel_stale_orders": cancel_stale_orders,
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
        data=pullback_df,
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

    trailing_stats = run_backtest(
        data=trailing_df,
        strategy=AlligatorFractalTrailing,
        cash=cash,
        commission=commission,
        spread_pips=spread_pips,
        exclusive_orders=exclusive_orders,
        strategy_params=strategy_params,
    )

    # comparisons
    comparison_sc = _comparison_table(strict_stats, classic_stats, "strict", "classic")
    comparison_sp = _comparison_table(strict_stats, pullback_stats, "strict", "pullback")
    comparison_st = _comparison_table(strict_stats, trailing_stats, "strict", "trailing")
    comparison_ct = _comparison_table(classic_stats, trailing_stats, "classic", "trailing")
    comparison_pt = _comparison_table(pullback_stats, trailing_stats, "pullback", "trailing")

    if export:
        outdir.mkdir(parents=True, exist_ok=True)

        # Manifest for reproducibility
        resolved = {
            "mode": "comparison",
            "symbol": data.attrs.get("symbol"),
            "timeframe": data.attrs.get("timeframe"),
            "cash": cash,
            "commission": commission,
            "spread_pips": spread_pips,
            "exclusive_orders": exclusive_orders,
            "use_htf_bias": use_htf_bias,
            "use_vol_filter": use_vol_filter,
            "htf_tf": htf_tf,
            "atr_period": atr_period,
            "atr_long": atr_long,
            "eps": eps,
            "cancel_stale_orders": cancel_stale_orders,
            "pullback_k_atr": pullback_k_atr,
            "require_touch_teeth": require_touch_teeth,
        }
        _export_manifest(outdir, resolved=resolved, gate_debug=gate_debug, data_fp=data_fp)

        # Stats JSON
        (outdir / "strict_stats.json").write_text(json.dumps(_stats_to_json(strict_stats), indent=2))
        (outdir / "classic_stats.json").write_text(json.dumps(_stats_to_json(classic_stats), indent=2))
        (outdir / "pullback_stats.json").write_text(json.dumps(_stats_to_json(pullback_stats), indent=2))
        (outdir / "trailing_stats.json").write_text(json.dumps(_stats_to_json(trailing_stats), indent=2))

        # Trades & equity
        export_trades_csv(strict_stats, outdir / "strict_trades.csv")
        export_equity_curve_csv(strict_stats, outdir / "strict_equity.csv")

        export_trades_csv(classic_stats, outdir / "classic_trades.csv")
        export_equity_curve_csv(classic_stats, outdir / "classic_equity.csv")

        export_trades_csv(pullback_stats, outdir / "pullback_trades.csv")
        export_equity_curve_csv(pullback_stats, outdir / "pullback_equity.csv")

        export_trades_csv(trailing_stats, outdir / "trailing_trades.csv")
        export_equity_curve_csv(trailing_stats, outdir / "trailing_equity.csv")

        # Comparisons
        comparison_sc.to_csv(outdir / "comparison_strict_classic.csv", index=False)
        comparison_sp.to_csv(outdir / "comparison_strict_pullback.csv", index=False)
        comparison_st.to_csv(outdir / "comparison_strict_trailing.csv", index=False)
        comparison_ct.to_csv(outdir / "comparison_classic_trailing.csv", index=False)
        comparison_pt.to_csv(outdir / "comparison_pullback_trailing.csv", index=False)

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

        print("=" * 70)
        print("Trailing Strategy Stats")
        print("=" * 70)
        print(trailing_stats)

        print("\nComparison (Strict vs Classic)")
        print(comparison_sc.to_string(index=False))
        print("\nComparison (Strict vs Pullback)")
        print(comparison_sp.to_string(index=False))
        print("\nComparison (Strict vs Trailing)")
        print(comparison_st.to_string(index=False))
        print("\nComparison (Classic vs Trailing)")
        print(comparison_ct.to_string(index=False))
        print("\nComparison (Pullback vs Trailing)")
        print(comparison_pt.to_string(index=False))

    return {
        "strict": strict_stats,
        "classic": classic_stats,
        "pullback": pullback_stats,
        "trailing": trailing_stats,
        "comparison_sc": comparison_sc,
        "comparison_sp": comparison_sp,
        "comparison_st": comparison_st,
        "comparison_ct": comparison_ct,
        "comparison_pt": comparison_pt,
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
    cancel_stale_orders: bool = False,
) -> dict:
    data_fp = _data_fingerprint(data)
    print(data_fp)

    gate_debug = compute_gating_debug(
        data,
        use_htf_bias=use_htf_bias,
        use_vol_filter=use_vol_filter,
        htf_tf=htf_tf,
        atr_period=atr_period,
        atr_long=atr_long,
        cfg=AlligatorParams(),
    )
    print(f"HTF bias %: {gate_debug['bias_dist']}")
    print(f"VOL ok %: {gate_debug['vol_ok']:.2f}")
    print(f"BOTH ok %: {gate_debug['both_ok']:.2f}")

    strategy_map = {
        "strict": AlligatorFractal,
        "classic": AlligatorFractalClassic,
        "pullback": AlligatorFractalPullback,
        "trailing": AlligatorFractalTrailing,
    }
    if strategy_name not in strategy_map:
        raise ValueError(f"Unsupported strategy: {strategy_name}")

    strategy_params = {
        "use_htf_bias": use_htf_bias,
        "use_vol_filter": use_vol_filter,
        "htf_tf": htf_tf,
        "atr_period": atr_period,
        "atr_long": atr_long,
        "cancel_stale_orders": cancel_stale_orders,
    }
    if eps is not None:
        strategy_params["eps"] = eps

    if strategy_name == "pullback":
        if pullback_k_atr is not None:
            strategy_params["pullback_k_atr"] = pullback_k_atr
        strategy_params["require_touch_teeth"] = bool(require_touch_teeth)

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

        resolved = {
            "mode": "single",
            "strategy": strategy_name,
            "symbol": data.attrs.get("symbol"),
            "timeframe": data.attrs.get("timeframe"),
            "cash": cash,
            "commission": commission,
            "spread_pips": spread_pips,
            "exclusive_orders": exclusive_orders,
            "use_htf_bias": use_htf_bias,
            "use_vol_filter": use_vol_filter,
            "htf_tf": htf_tf,
            "atr_period": atr_period,
            "atr_long": atr_long,
            "eps": eps,
            "cancel_stale_orders": cancel_stale_orders,
            "pullback_k_atr": pullback_k_atr,
            "require_touch_teeth": require_touch_teeth,
        }
        _export_manifest(outdir, resolved=resolved, gate_debug=gate_debug, data_fp=data_fp)

        (outdir / f"{strategy_name}_stats.json").write_text(json.dumps(_stats_to_json(stats), indent=2))
        export_trades_csv(stats, outdir / f"{strategy_name}_trades.csv")
        export_equity_curve_csv(stats, outdir / f"{strategy_name}_equity.csv")

    if print_table:
        print("=" * 70)
        print(f"{strategy_name.title()} Strategy Stats")
        print("=" * 70)
        print(stats)

    return {strategy_name: stats}


if __name__ == "__main__":
    """
    Validation checklist:
    - No-flag run uses OPT_DEFAULTS values.
    - --exclusive-orders and --no-exclusive-orders correctly flip.
    - After resample/copy, data.attrs still contains symbol/timeframe.
    - Comparison mode runs 4 strategies (strict/classic/pullback/trailing) and exports all files.
    """

    parser = argparse.ArgumentParser(description="Compare or run Alligator+Fractal strategies.")

    # mode
    parser.add_argument(
        "--strategy",
        choices=["strict", "classic", "pullback", "trailing"],
        default=None,
        help="If omitted, runs comparison (strict vs classic vs pullback vs trailing).",
    )

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
    parser.add_argument("--eps", type=float, default=None)

    parser.add_argument("--exclusive-orders", dest="exclusive_orders", action="store_true")
    parser.add_argument("--no-exclusive-orders", dest="exclusive_orders", action="store_false")
    parser.set_defaults(exclusive_orders=OPT_DEFAULTS["exclusive_orders"])

    parser.add_argument("--cancel-stale-orders", dest="cancel_stale_orders", action="store_true")
    parser.add_argument("--no-cancel-stale-orders", dest="cancel_stale_orders", action="store_false")
    parser.set_defaults(cancel_stale_orders=OPT_DEFAULTS["cancel_stale_orders"])

    # filters
    parser.add_argument("--no-htf-bias", action="store_true", default=False)
    parser.add_argument("--no-vol-filter", action="store_true", default=False)
    parser.add_argument("--htf", default=None, help="HTF timeframe (e.g. 4h).")
    parser.add_argument("--atr", type=int, default=None)
    parser.add_argument("--atr-long", type=int, default=None)

    # pullback params
    parser.add_argument("--pullback-k", type=float, default=None)

    parser.add_argument("--touch-teeth", dest="touch_teeth", action="store_true")
    parser.add_argument("--no-touch-teeth", dest="touch_teeth", action="store_false")
    parser.set_defaults(touch_teeth=OPT_DEFAULTS["require_touch_teeth"])

    # output
    parser.add_argument("--outdir", default=None)

    args = parser.parse_args()

    # --- resolve defaults + overrides ---
    currency = args.asset or OPT_DEFAULTS["currency"]
    timeframe = args.tf or OPT_DEFAULTS["timeframe"]

    spread_pips = args.spread if args.spread is not None else OPT_DEFAULTS["spread_pips"]
    cash = args.cash if args.cash is not None else OPT_DEFAULTS["cash"]
    commission = args.commission if args.commission is not None else OPT_DEFAULTS["commission"]

    exclusive_orders = bool(args.exclusive_orders)
    cancel_stale_orders = bool(args.cancel_stale_orders)

    use_htf_bias = OPT_DEFAULTS["use_htf_bias"] and (not args.no_htf_bias)
    use_vol_filter = OPT_DEFAULTS["use_vol_filter"] and (not args.no_vol_filter)

    htf_tf = args.htf or OPT_DEFAULTS["htf_tf"]
    atr_period = args.atr if args.atr is not None else OPT_DEFAULTS["atr_period"]
    atr_long = args.atr_long if args.atr_long is not None else OPT_DEFAULTS["atr_long"]

    eps = args.eps if args.eps is not None else OPT_DEFAULTS["eps"]

    outdir = Path(args.outdir) if args.outdir else OPT_DEFAULTS["outdir"]

    pullback_k_atr = args.pullback_k if args.pullback_k is not None else OPT_DEFAULTS["pullback_k_atr"]
    require_touch_teeth = bool(args.touch_teeth)

    # --- load & prep data ---
    asset = args.asset or currency
    tf = args.tf or timeframe

    df = _load_data(args.data, asset, tf)
    data = _ensure_ohlc(df)
    data = _filter_range(data, start=args.start, end=args.end)
    data = _resample_ohlcv(data, timeframe)
    data = data.dropna(subset=["Open", "High", "Low", "Close"])

    if data.empty:
        raise ValueError("No data available after filtering/resampling.")

    # --- run ---
    if args.strategy is None:
        print(f'spread_pips={spread_pips}')
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
            cancel_stale_orders=cancel_stale_orders,
            export=True,
            print_table=True,
        )
    else:
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
            cancel_stale_orders=cancel_stale_orders,
            export=True,
            print_table=True,
            pullback_k_atr=pullback_k_atr,
            require_touch_teeth=require_touch_teeth,
        )
