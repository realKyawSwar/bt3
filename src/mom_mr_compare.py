"""
Grid-search runner for Momentum + Mean Reversion strategy.

Example commands:
python src/mom_mr_compare.py --asset XAUUSD --tf 1h --mom-window-grid 63,126 --rsi-threshold-grid 15,20,25 --sl-atr-grid 2.0,2.5 --outdir reports/mom_mr/
python src/mom_mr_compare.py --asset EURUSD --tf 1h --spread 10 --rsi-exit-grid 50,60,70 --points 5000
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

from bt3 import fetch_data, run_backtest
from mom_mr import MomentumMeanReversionStrategy
from reporting import export_equity_curve_csv, export_trades_csv, plot_equity_curve


def _parse_list(arg: str | None, cast=float) -> List:
    if not arg:
        return []
    return [cast(x.strip()) for x in arg.split(",") if x.strip()]


def _metric_value(stats, keys: Iterable[str]):
    for key in keys:
        if key in stats:
            return stats[key]
    return None


def _extract_metrics(stats) -> dict:
    return {
        "return": _metric_value(stats, ["Return [%]", "Return %", "Return"]),
        "sharpe": _metric_value(stats, ["Sharpe Ratio", "Sharpe"]),
        "maxdd": _metric_value(stats, ["Max. Drawdown [%]", "Max Drawdown [%]", "Max Drawdown %"]),
        "winrate": _metric_value(stats, ["Win Rate [%]", "Win Rate %"]),
        "trades": _metric_value(stats, ["# Trades", "Trades"]),
        "exposure": _metric_value(stats, ["Exposure [%]", "Exposure %"]),
    }


def _safe_number(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (pd.Timestamp, pd.Timedelta)):
        return str(x)
    if isinstance(x, (float, int)):
        return x
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return x


def _stats_to_json(stats) -> dict:
    result = {}
    for key, value in stats.items():
        if str(key).startswith("_"):
            continue
        result[key] = _safe_number(value)
    return result


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]
    df = df.sort_index()
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid search for Momentum + Mean Reversion strategy.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python src/mom_mr_compare.py --asset XAUUSD --tf 1h --mom-window-grid 63,126 --rsi-threshold-grid 15,20,25 --sl-atr-grid 2.0,2.5 --outdir reports/mom_mr/\n"
            "  python src/mom_mr_compare.py --asset EURUSD --tf 1h --spread 10 --rsi-exit-grid 50,60,70 --points 5000\n"
        ),
    )
    parser.add_argument("--asset", required=True, help="Symbol (e.g., EURUSD, XAUUSD).")
    parser.add_argument("--tf", required=True, help="Timeframe (e.g., 1h, 4h).")
    parser.add_argument("--spread", type=float, default=None, help="FX spread in pips.")
    parser.add_argument("--cash", type=float, default=10000.0)
    parser.add_argument("--commission", type=float, default=0.0)
    parser.add_argument("--outdir", default="reports/", help="Output directory (default: reports/)")
    parser.add_argument("--points", type=int, default=None, help="Optional cap on number of bars (tail) for faster sweeps.")
    parser.add_argument("--mom-window-grid", help="Comma-separated momentum windows, e.g., 63,126,252")
    parser.add_argument("--rsi-threshold-grid", help="Comma-separated RSI entry thresholds, e.g., 10,15,20,25")
    parser.add_argument("--sl-atr-grid", help="Comma-separated SL multiples of ATR, e.g., 1.5,2.0,2.5")
    parser.add_argument("--rsi-exit-grid", help="Comma-separated RSI exit thresholds, e.g., 50,60,70")

    args = parser.parse_args()

    df = fetch_data(args.asset, args.tf)
    df = _ensure_datetime_index(df)
    if args.points and args.points > 0:
        df = df.tail(int(args.points))
    if df.empty:
        raise ValueError("Loaded data is empty.")

    mom_windows = _parse_list(args.mom_window_grid, int) or [MomentumMeanReversionStrategy.mom_window]
    rsi_thresholds = _parse_list(args.rsi_threshold_grid, float) or [MomentumMeanReversionStrategy.rsi_threshold]
    sl_atrs = _parse_list(args.sl_atr_grid, float) or [MomentumMeanReversionStrategy.sl_atr]
    rsi_exits = _parse_list(args.rsi_exit_grid, float) or [MomentumMeanReversionStrategy.rsi_exit]

    grid = list(product(mom_windows, rsi_thresholds, sl_atrs, rsi_exits))
    if not grid:
        raise ValueError("No grid combinations generated.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.outdir) / f"{args.asset}_{args.tf}_mommr_grid_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    best_stats = None
    best_label = ""
    best_rank = (-float("inf"), -float("inf"))

    for i, (mw, rsi_thr, sl_atr_val, rsi_exit_val) in enumerate(grid):
        params = {
            "mom_window": int(mw),
            "mom_threshold": MomentumMeanReversionStrategy.mom_threshold,
            "rsi_window": MomentumMeanReversionStrategy.rsi_window,
            "rsi_threshold": float(rsi_thr),
            "rsi_exit": float(rsi_exit_val),
            "atr_window": MomentumMeanReversionStrategy.atr_window,
            "sl_atr": float(sl_atr_val),
            "tp_mode": MomentumMeanReversionStrategy.tp_mode,
            "tp_atr": MomentumMeanReversionStrategy.tp_atr,
            "risk_pct": MomentumMeanReversionStrategy.risk_pct,
            "min_size": MomentumMeanReversionStrategy.min_size,
            "max_size": MomentumMeanReversionStrategy.max_size,
        }
        name = f"mw={mw}_rsi={rsi_thr}_sl={sl_atr_val}_exit={rsi_exit_val}"
        print(f"[{i+1}/{len(grid)}] Running {name}")

        try:
            stats = run_backtest(
                data=df,
                strategy=MomentumMeanReversionStrategy,
                cash=args.cash,
                commission=args.commission,
                spread_pips=args.spread,
                margin=1.0,
                exclusive_orders=False,
                strategy_params=params,
            )
            metrics = _extract_metrics(stats)
            error = ""
        except Exception as exc:  # pylint: disable=broad-except
            stats = None
            metrics = {}
            error = str(exc)
            print(f"  Error: {error}")

        sharpe_val = float(metrics.get("sharpe") or -float("inf"))
        ret_val = float(metrics.get("return") or -float("inf"))
        rank_key = (sharpe_val, ret_val)
        if rank_key > best_rank and stats is not None:
            best_rank = rank_key
            best_stats = stats
            best_label = name

        row = {
            "name": name,
            "params_json": json.dumps(params),
            "Return[%]": _safe_number(metrics.get("return")),
            "Sharpe": _safe_number(metrics.get("sharpe")),
            "MaxDD[%]": _safe_number(metrics.get("maxdd")),
            "WinRate[%]": _safe_number(metrics.get("winrate")),
            "#Trades": _safe_number(metrics.get("trades")),
            "Exposure[%]": _safe_number(metrics.get("exposure")),
            "error": error,
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = run_dir / "mom_mr_compare_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    ranked_df = summary_df.copy()
    ranked_df["_Sharpe"] = pd.to_numeric(ranked_df["Sharpe"], errors="coerce")
    ranked_df["_Return"] = pd.to_numeric(ranked_df["Return[%]"], errors="coerce")
    ranked_df = ranked_df.sort_values(["_Sharpe", "_Return"], ascending=[False, False])
    ranked_csv = run_dir / "mom_mr_compare_ranked.csv"
    ranked_df.to_csv(ranked_csv, index=False)

    print(f"\nWrote summary to {summary_csv}")
    print(f"Wrote ranked to {ranked_csv}")

    top = ranked_df.head(10)
    if not top.empty:
        print("\nTop 10 (by Sharpe then Return):")
        cols = ["name", "Return[%]", "Sharpe", "MaxDD[%]", "WinRate[%]", "#Trades", "Exposure[%]"]
        print(top[cols].to_string(index=False))

    if best_stats is not None:
        best_dir = run_dir / "best"
        best_dir.mkdir(exist_ok=True)
        (best_dir / "name.txt").write_text(best_label)
        (best_dir / "stats.json").write_text(json.dumps(_stats_to_json(best_stats), indent=2))
        export_trades_csv(best_stats, best_dir / "trades.csv")
        export_equity_curve_csv(best_stats, best_dir / "equity.csv")
        try:
            plot_equity_curve(best_stats, title=f"Best: {best_label}", save_path=best_dir / "equity.png", show=False)
        except Exception as exc:  # pragma: no cover - plotting optional
            print(f"Equity plot skipped: {exc}")


if __name__ == "__main__":
    main()
