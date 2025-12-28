from alligator_fractal import AlligatorFractal
from bt3 import fetch_data, run_backtest
from reporting import plot_equity_curve, export_trades_csv, export_equity_curve_csv

class MyAlligator(AlligatorFractal):
    enable_tp = True
    tp_rr = 1.5  # 1.5R
    size = 0.03  # 1% of equity per trade
    jaw_period = 13
    teeth_period = 8
    lips_period = 5

if __name__ == "__main__":
    currency = "USDJPY"
    timeframe = "1h"
    data = fetch_data(currency, timeframe)

    stats = run_backtest(
        data=data,
        strategy=MyAlligator,
        cash=10000,
        spread_pips= 1.5,      # FX-style cost (recommended)
        symbol=currency,
        commission=0.0,
        exclusive_orders=True,
        margin=1.0,
    )


    print(stats)

    # --- exports ---
    export_trades_csv(stats, "out_trades_gbpjpy_h1.csv")
    export_equity_curve_csv(stats, "out_equity_gbpjpy_h1.csv")

    # --- plot ---
    plot_equity_curve(stats, title=f"{currency} {timeframe} - AlligatorFractal Equity", save_path=f"equity_{currency.lower()}_{timeframe}.png", show=True)