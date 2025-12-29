from __future__ import annotations

from alligator_fractal import AlligatorFractal, AlligatorFractalPullback
from bt3 import fetch_data, run_backtest


def main() -> None:
    data = fetch_data("GBPJPY", "1h")

    base_params = {
        "use_htf_bias": True,
        "use_vol_filter": True,
        "htf_tf": "4h",
        "atr_period": 14,
        "atr_long": 50,
    }

    strict_stats = run_backtest(
        data=data.copy(),
        strategy=AlligatorFractal,
        cash=10000,
        commission=0,
        spread_pips=1.5,
        exclusive_orders=True,
        strategy_params=base_params,
    )

    pullback_stats = run_backtest(
        data=data.copy(),
        strategy=AlligatorFractalPullback,
        cash=10000,
        commission=0,
        spread_pips=1.5,
        exclusive_orders=True,
        strategy_params={
            **base_params,
            "pullback_k_atr": 0.5,
            "require_touch_teeth": False,
        },
    )

    strict_trades = strict_stats.get("# Trades")
    pullback_trades = pullback_stats.get("# Trades")
    ratio = pullback_trades / strict_trades if strict_trades else 0.0

    print(f"Strict trades: {strict_trades}")
    print(f"Pullback trades: {pullback_trades}")
    print(f"Pullback/Strict ratio: {ratio:.2f}")

    assert pullback_trades >= 0.60 * strict_trades, (
        "Pullback trades dropped too far relative to strict trades. "
        f"strict={strict_trades}, pullback={pullback_trades}, ratio={ratio:.2f}"
    )


if __name__ == "__main__":
    main()
