"""
Grid-search runner for Wave5 strategy parameter sweeps.

Example commands:
python src/wave5_compare.py --asset XAUUSD --tf 1h --spread 30 --wave5-use-scoring --wave5-grid "min_w5_ext=1.14:1.20:0.02,score_threshold=0.56|0.58|0.60,w_candle=0.05|0.1" --save-artifacts --outdir reports/
python src/wave5_compare.py --asset GBPJPY --tf 4h --spread 15 --wave5-grid "max_trigger_lag=2|3|4,zone_mode=trigger|either" --top 5
"""
from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime
from decimal import Decimal, getcontext
from itertools import product
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

from bt3 import fetch_data, run_backtest
from reporting import export_equity_curve_csv, export_trades_csv
from wave5_ao import Wave5AODivergenceStrategy
from broker_debug import install_all_broker_hooks


getcontext().prec = 28


def _metric_value(stats, keys: Iterable[str]):
    for key in keys:
        if key in stats:
            return stats[key]
    return None


def _extract_metrics(stats) -> dict:
    return {
        "return": _metric_value(stats, ["Return [%]", "Return %", "Return"]),
        "maxdd": _metric_value(stats, ["Max. Drawdown [%]", "Max Drawdown [%]", "Max Drawdown %"]),
        "pf": _metric_value(stats, ["Profit Factor"]),
        "trades": _metric_value(stats, ["# Trades", "Trades"]),
        "sharpe": _metric_value(stats, ["Sharpe Ratio", "Sharpe"]),
        "winrate": _metric_value(stats, ["Win Rate [%]", "Win Rate %"]),
        "expectancy": _metric_value(stats, ["Expectancy [%]", "Expectancy %"]),
        "exposure": _metric_value(stats, ["Exposure [%]", "Exposure %"]),
        "cagr": _metric_value(stats, ["CAGR [%]", "CAGR", "Annual Return [%]", "Annual Return"]),
    }


def _safe_number(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (pd.Timestamp, pd.Timedelta)):
        return str(x)
    if isinstance(x, (float, int)):
        return x
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return x


def _stats_to_json(stats) -> dict:
    result = {}
    for key, value in stats.items():
        if str(key).startswith("_"):
            continue
        result[key] = _safe_number(value)
    return result


def _objective_score(metrics: dict, objective: str) -> float:
    ret = float(metrics.get("return") or 0.0)
    pf = float(metrics.get("pf") or 0.0)
    trades = float(metrics.get("trades") or 0.0)
    cagr = float(metrics.get("cagr") or 0.0)
    maxdd = abs(float(metrics.get("maxdd") or 0.0))

    if math.isnan(ret):
        ret = 0.0
    if math.isnan(pf):
        pf = 0.0
    if math.isnan(trades):
        trades = 0.0
    if math.isnan(cagr):
        cagr = 0.0
    if math.isnan(maxdd):
        maxdd = 0.0

    denom = max(maxdd, 0.0) + 1e-9
    if objective == "cagr_dd_pf":
        return cagr * pf / denom
    return ret * pf / denom * math.log1p(trades)


def _infer_literal(token: str):
    t = token.strip()
    tl = t.lower()
    if tl == "true":
        return True
    if tl == "false":
        return False
    try:
        if t.startswith(("0x", "0b", "0o")):
            raise ValueError
        if "." not in t and "e" not in tl:
            return int(t)
    except ValueError:
        pass
    try:
        return float(t)
    except ValueError:
        return t


def _coerce_by_base(value, base):
    if base is None:
        return value
    if isinstance(base, bool):
        return bool(value)
    if isinstance(base, int) and not isinstance(base, bool):
        try:
            return int(round(float(value)))
        except Exception:
            return base
    if isinstance(base, float):
        try:
            return float(value)
        except Exception:
            return base
    return value


def _range_values(expr: str) -> List[float]:
    parts = expr.split(":")
    if len(parts) != 3:
        raise ValueError(f"Range spec must be start:stop:step, got '{expr}'")
    start_d, stop_d, step_d = map(Decimal, parts)
    if step_d <= 0:
        raise ValueError("Range step must be > 0")
    values: List[float] = []
    current = start_d
    tol = Decimal("1e-12")
    while current <= stop_d + tol:
        values.append(float(current))
        current += step_d
    return values


def parse_grid_spec(spec: str) -> List[Tuple[str, List]]:
    if not spec:
        return []
    pairs = []
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Grid token missing '=': '{part}'")
        key, expr = part.split("=", 1)
        key = key.strip()
        expr = expr.strip()
        if not key:
            raise ValueError(f"Empty key in grid token '{part}'")
        values: List
        if ":" in expr:
            values = _range_values(expr)
        elif "|" in expr:
            values = [_infer_literal(x) for x in expr.split("|")]
        else:
            values = [_infer_literal(expr)]
        pairs.append((key, values))
    return pairs


def expand_grid(pairs: List[Tuple[str, List]]) -> List[dict]:
    if not pairs:
        return [{}]
    keys = [k for k, _ in pairs]
    value_lists = [v for _, v in pairs]
    combos = []
    for prod_vals in product(*value_lists):
        combos.append({k: v for k, v in zip(keys, prod_vals)})
    return combos


def _format_name(template: str, i: int, params: dict) -> str:
    try:
        return template.format(i=i, **params)
    except Exception:
        return f"{i:03d}"


def _build_base_wave5_params(args) -> dict:
    entry_mode = args.wave5_entry_mode
    if entry_mode == "limit":
        entry_mode = "break"
    params = {
        "use_scoring": args.wave5_use_scoring,
        "score_threshold": args.wave5_score_threshold,
        "w_candle": args.wave5_w_candle,
        "max_trigger_lag": args.wave5_trigger_lag,
        "zone_mode": args.wave5_zone_mode,
        "fib_tol_mode": args.wave5_fib_tol_mode,
        "require_zero_cross": args.wave5_require_zero_cross,
        "entry_mode": entry_mode,
        "break_buffer_atr": args.wave5_break_buffer_atr,
        "max_body_atr": args.wave5_max_body_atr,
        "min_w5_ext": args.wave5_min_w5_ext,
        "debug": args.wave5_debug,
        "debug_trace": not args.wave5_no_trace,
        "asset": args.asset,
        "sizing_margin": 1.0,
        "exec_margin": 1.0,
    }
    return params


def _print_table(rows: List[dict], top: int | None) -> None:
    if not rows:
        print("No results to display.")
        return
    limit = top if top is not None else 10
    headers = ["name", "Return", "MaxDD", "PF", "Trades", "objective_score"]
    print("\nTop results:")
    print(" | ".join(f"{h:>14}" for h in headers))
    print("-" * 90)
    def _num(val, default=float("nan")):
        try:
            if val is None:
                return default
            return float(val)
        except Exception:
            return default
    for row in rows[:limit]:
        print(
            f"{row.get('name',''):>14} | "
            f"{_num(row.get('Return[%]')):>8.2f} | "
            f"{_num(row.get('MaxDD[%]')):>8.2f} | "
            f"{_num(row.get('PF')):>6.2f} | "
            f"{_num(row.get('#Trades'), 0.0):>6.0f} | "
            f"{_num(row.get('objective_score')):>14.4f}"
        )


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]
    df = df.sort_index()
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wave5 parameter grid comparison",
        epilog=(
            "Examples:\n"
            "  python src/wave5_compare.py --asset XAUUSD --tf 1h --spread 30 "
            "--wave5-use-scoring --wave5-grid \"min_w5_ext=1.14:1.20:0.02,score_threshold=0.56|0.58|0.60,w_candle=0.05|0.1\" --save-artifacts\n"
            "  python src/wave5_compare.py --asset GBPJPY --tf 4h --spread 15 "
            "--wave5-grid \"max_trigger_lag=2|3|4,zone_mode=trigger|either\" --top 5"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--asset", required=True, help="Symbol (e.g., XAUUSD, GBPJPY).")
    parser.add_argument("--tf", required=True, help="Timeframe (e.g., 1h, 4h).")
    parser.add_argument("--spread", type=float, default=None, help="FX spread in pips.")
    parser.add_argument("--cash", type=float, default=10000.0)
    parser.add_argument("--commission", type=float, default=0.0)
    parser.add_argument("--outdir", default="reports/", help="Output directory (default: reports/)")
    parser.add_argument("--objective", choices=["cagr_dd_pf", "return_dd_trades"], default="cagr_dd_pf")
    parser.add_argument("--save-artifacts", action="store_true", help="Save per-run stats/trades/equity.")
    parser.add_argument("--top", type=int, default=None, help="Print only top K rows (default 10).")
    parser.add_argument("--wave5-use-scoring", action="store_true", default=Wave5AODivergenceStrategy.use_scoring)
    parser.add_argument("--wave5-score-threshold", type=float, default=Wave5AODivergenceStrategy.score_threshold)
    parser.add_argument("--wave5-w-candle", type=float, default=Wave5AODivergenceStrategy.w_candle)
    parser.add_argument("--wave5-trigger-lag", type=int, default=Wave5AODivergenceStrategy.max_trigger_lag)
    parser.add_argument(
        "--wave5-zone-mode",
        choices=["either", "trigger", "extreme"],
        default=Wave5AODivergenceStrategy.zone_mode,
    )
    parser.add_argument(
        "--wave5-fib-tol-mode",
        choices=["atr_pct", "fixed"],
        default=Wave5AODivergenceStrategy.fib_tol_mode,
    )
    parser.add_argument("--wave5-require-zero-cross", dest="wave5_require_zero_cross", action="store_true")
    parser.add_argument("--wave5-no-require-zero-cross", dest="wave5_require_zero_cross", action="store_false")
    parser.set_defaults(wave5_require_zero_cross=Wave5AODivergenceStrategy.require_zero_cross)
    parser.add_argument(
        "--wave5-entry-mode",
        choices=["break", "limit", "close"],
        default=Wave5AODivergenceStrategy.entry_mode,
        help="Entry mode for Wave5 triggers.",
    )
    parser.add_argument("--wave5-break-buffer-atr", type=float, default=Wave5AODivergenceStrategy.break_buffer_atr)
    parser.add_argument("--wave5-max-body-atr", type=float, default=Wave5AODivergenceStrategy.max_body_atr)
    parser.add_argument("--wave5-min-w5-ext", type=float, default=Wave5AODivergenceStrategy.min_w5_ext)
    parser.add_argument("--wave5-debug", action="store_true", default=Wave5AODivergenceStrategy.debug)
    parser.add_argument("--wave5-no-trace", action="store_true", help="Disable per-bar trace when debug is on.")
    parser.add_argument("--wave5-grid", help="Grid spec string, e.g., \"min_w5_ext=1.1:1.3:0.05,w_candle=0.05|0.1\"")
    parser.add_argument("--grid-max", type=int, default=500, help="Maximum total grid combinations (default 500).")
    parser.add_argument("--grid-sample", type=int, default=None, help="Sample N combinations if total exceeds max.")
    parser.add_argument("--grid-seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument(
        "--name-template",
        default="{i:03d}_minw5={min_w5_ext}_thr={score_threshold}_wc={w_candle}",
        help="Template for run names using format fields from params.",
    )

    args = parser.parse_args()

    df = fetch_data(args.asset, args.tf)
    df = _ensure_datetime_index(df)

    if df.empty:
        raise ValueError("Loaded data is empty.")

    base_params = _build_base_wave5_params(args)

    grid_pairs = parse_grid_spec(args.wave5_grid or "")
    for key, _ in grid_pairs:
        if key not in base_params:
            raise ValueError(f"Grid key '{key}' not in Wave5 params: {sorted(base_params)}")

    combos = expand_grid(grid_pairs)
    total_combos = len(combos)
    grid_max = int(args.grid_max)
    if total_combos > grid_max:
        if args.grid_sample is None:
            raise ValueError(f"Total combinations {total_combos} exceed grid-max {grid_max}. Use --grid-sample to subsample.")
        sample_n = int(args.grid_sample)
        if sample_n > grid_max:
            raise ValueError("--grid-sample must be <= --grid-max")
        rng = random.Random(int(args.grid_seed))
        combos = rng.sample(combos, min(sample_n, total_combos))
        print(f"Sampled {len(combos)} combos out of {total_combos} (grid-max={grid_max})")
    else:
        print(f"Running {total_combos} combinations")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.outdir) / f"{args.asset}_{args.tf}_wave5_grid_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    if args.wave5_debug:
        if not args.wave5_no_trace:
            install_all_broker_hooks(debug=True)
        else:
            print("[wave5_compare] Debug counters enabled (trace suppressed)")

    for i, combo in enumerate(combos):
        params = dict(base_params)
        for key, value in combo.items():
            params[key] = _coerce_by_base(value, base_params.get(key))

        name = _format_name(args.name_template, i=i, params=params)
        print(f"[{i+1}/{len(combos)}] Running {name} with overrides {combo}")

        try:
            stats = run_backtest(
                data=df,
                strategy=Wave5AODivergenceStrategy,
                cash=args.cash,
                commission=args.commission,
                spread_pips=args.spread,
                margin=1.0,
                exclusive_orders=False,
                strategy_params=params,
            )
            metrics = _extract_metrics(stats)
            objective_score = _objective_score(metrics, args.objective)
            error = ""
        except Exception as exc:  # pylint: disable=broad-except
            metrics = {}
            stats = None
            objective_score = -float("inf")
            error = str(exc)
            print(f"  Error: {error}")

        row = {
            "name": name,
            "params_json": json.dumps(params),
            "Return[%]": _safe_number(metrics.get("return")),
            "MaxDD[%]": _safe_number(metrics.get("maxdd")),
            "PF": _safe_number(metrics.get("pf")),
            "#Trades": _safe_number(metrics.get("trades")),
            "Sharpe": _safe_number(metrics.get("sharpe")),
            "WinRate[%]": _safe_number(metrics.get("winrate")),
            "Expectancy[%]": _safe_number(metrics.get("expectancy")),
            "Exposure[%]": _safe_number(metrics.get("exposure")),
            "CAGR[%]": _safe_number(metrics.get("cagr")),
            "objective_score": objective_score,
            "error": error,
        }
        summary_rows.append(row)

        if args.save_artifacts and stats is not None:
            run_subdir = run_dir / name
            run_subdir.mkdir(parents=True, exist_ok=True)
            (run_subdir / "stats.json").write_text(json.dumps(_stats_to_json(stats), indent=2))
            export_trades_csv(stats, run_subdir / "trades.csv")
            export_equity_curve_csv(stats, run_subdir / "equity.csv")

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = run_dir / "wave5_compare_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    ranked_df = summary_df.sort_values("objective_score", ascending=False)
    ranked_csv = run_dir / "wave5_compare_ranked.csv"
    ranked_df.to_csv(ranked_csv, index=False)

    print(f"\nWrote summary to {summary_csv}")
    print(f"Wrote ranked to {ranked_csv}")

    ranked_rows = ranked_df.to_dict(orient="records")
    _print_table(ranked_rows, args.top)


if __name__ == "__main__":
    main()
