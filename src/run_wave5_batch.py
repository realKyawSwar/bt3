from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from bt3 import fetch_data, run_backtest
from elliott_ao_wave5 import ElliottAOWave5Strategy, resolve_wave5_params
from reporting import export_equity_curve_csv, export_trades_csv


def _split_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _filter_range(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if start:
        df = df.loc[pd.to_datetime(start):]
    if end:
        df = df.loc[:pd.to_datetime(end)]
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


def _metric_value(stats, keys: list[str]):
    for key in keys:
        if key in stats:
            return stats[key]
    return None


def _profit_factor(stats) -> float | None:
    pf = _metric_value(stats, ["Profit Factor"])
    if pf is not None and isinstance(pf, (int, float, np.integer, np.floating)):
        return float(pf)
    trades = stats.get("_trades")
    if trades is None:
        return None
    if not isinstance(trades, pd.DataFrame):
        trades = pd.DataFrame(trades)
    if "PnL" not in trades.columns:
        return None
    pnl = pd.to_numeric(trades["PnL"], errors="coerce")
    gains = pnl[pnl > 0].sum()
    losses = pnl[pnl < 0].sum()
    if losses == 0:
        return None
    return float(gains / abs(losses))


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-run Elliott AO Wave5 strategy.")
    parser.add_argument("--symbols", default="XAUUSD,EURUSD,GBPUSD")
    parser.add_argument("--tfs", default="1h,4h")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", help="End date (YYYY-MM-DD).")
    parser.add_argument("--cash", type=float, default=10000.0)
    parser.add_argument("--commission", type=float, default=0.0)
    parser.add_argument("--spread", type=float, default=None, help="FX spread in pips.")
    parser.add_argument("--outdir", default="reports/", help="Output directory for reports.")
    parser.add_argument("--pivot-len", type=int, default=None)
    parser.add_argument("--tol", type=float, default=None)
    parser.add_argument("--stop-pad-atr", type=float, default=None)
    parser.add_argument("--tp-r", type=float, default=None)
    parser.add_argument("--min-swing-atr", type=float, default=None)

    args = parser.parse_args()

    symbols = [s.upper() for s in _split_list(args.symbols)]
    tfs = [t.lower() for t in _split_list(args.tfs)]

    overrides = {
        "pivot_len": args.pivot_len,
        "tol": args.tol,
        "stop_pad_atr": args.stop_pad_atr,
        "tp_r": args.tp_r,
        "min_swing_atr": args.min_swing_atr,
    }

    summary_rows = []
    failures = []

    for symbol in symbols:
        for tf in tfs:
            try:
                df = fetch_data(symbol, tf)
                df = _filter_range(df, args.start, args.end)
                df = _sanitize_ohlcv(df)
                df = df.dropna(subset=["Open", "High", "Low", "Close"])
                if df.empty:
                    raise ValueError("No data available after filtering.")

                params = resolve_wave5_params(symbol, overrides)
                stats = run_backtest(
                    data=df,
                    strategy=ElliottAOWave5Strategy,
                    cash=args.cash,
                    commission=args.commission,
                    spread_pips=args.spread,
                    exclusive_orders=True,
                    strategy_params=params,
                )

                print(f"\n=== {symbol} {tf} ===")
                print(stats)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_dir = f"{symbol}_{tf}_{timestamp}"
                outdir = Path(args.outdir) / run_dir
                outdir.mkdir(parents=True, exist_ok=True)

                stats_dir = outdir / "stats"
                trades_dir = outdir / "trades"
                equity_dir = outdir / "equity"
                stats_dir.mkdir(exist_ok=True)
                trades_dir.mkdir(exist_ok=True)
                equity_dir.mkdir(exist_ok=True)

                (stats_dir / "wave5_stats.json").write_text(json.dumps(_stats_to_json(stats), indent=2))
                export_trades_csv(stats, trades_dir / "wave5_trades.csv")
                export_equity_curve_csv(stats, equity_dir / "wave5_equity.csv")

                trades = _metric_value(stats, ["# Trades", "Trades"])
                winrate = _metric_value(stats, ["Win Rate [%]", "Win Rate %"])
                ret = _metric_value(stats, ["Return [%]", "Return %"])
                max_dd = _metric_value(stats, ["Max. Drawdown [%]", "Max Drawdown [%]", "Max Drawdown %"])
                profit_factor = _profit_factor(stats)

                summary_rows.append(
                    {
                        "symbol": symbol,
                        "tf": tf,
                        "trades": trades,
                        "winrate": winrate,
                        "return_%": ret,
                        "max_dd_%": max_dd,
                        "profit_factor": profit_factor,
                    }
                )
            except Exception as exc:
                failures.append(f"{symbol} {tf}: {exc}")
                print(f"Error running {symbol} {tf}: {exc}")

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        print("\nSummary")
        print(summary.to_string(index=False))

    if failures:
        print("\nFailures")
        for failure in failures:
            print(f"- {failure}")


if __name__ == "__main__":
    main()
