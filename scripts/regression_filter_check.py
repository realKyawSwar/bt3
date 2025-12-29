from __future__ import annotations

from typing import Iterable

from alligator_fractal import AlligatorFractal
from bt3 import fetch_data, run_backtest


def _metric_value(stats, keys: Iterable[str]):
    for key in keys:
        if key in stats:
            return stats[key]
    return None


def _trade_count(stats) -> int:
    value = _metric_value(stats, ["# Trades", "Trades"])
    if value is None:
        raise ValueError("Could not find trade count in stats.")
    return int(value)


def main() -> None:
    data = fetch_data("GBPJPY", "1h")

    baseline_stats = run_backtest(
        data,
        AlligatorFractal,
        strategy_params={
            "use_htf_bias": False,
            "use_vol_filter": False,
        },
    )
    baseline_trades = _trade_count(baseline_stats)

    filtered_stats = run_backtest(
        data,
        AlligatorFractal,
        strategy_params={
            "use_htf_bias": True,
            "use_vol_filter": True,
            "htf_tf": "4H",
        },
    )
    filtered_trades = _trade_count(filtered_stats)

    if baseline_trades < 10:
        print(
            "Baseline trade count too small; skipping ratio assertion. "
            f"baseline={baseline_trades}, filtered={filtered_trades}"
        )
        return

    ratio = filtered_trades / baseline_trades if baseline_trades else 0.0
    print(
        "Trade counts: "
        f"baseline={baseline_trades}, filtered={filtered_trades}, ratio={ratio:.2f}"
    )

    assert filtered_trades < baseline_trades, "Filtered trades should be lower than baseline."
    assert filtered_trades >= int(0.6 * baseline_trades), (
        "Filtered trades fell below 60% of baseline."
    )


if __name__ == "__main__":
    main()
