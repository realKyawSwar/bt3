from alligator_fractal import AlligatorFractal
from bt3 import fetch_data, run_backtest

# Create custom version with TP enabled
class MyAlligator(AlligatorFractal):
    enable_tp = True  # Enable take profit
    tp_rr = 2.0       # 3:1 risk/reward
    jaw_period = 13
    teeth_period = 8
    lips_period = 5

data = fetch_data('GBPJPY', '1h')
stats = run_backtest(data, MyAlligator, cash=10000, commission=0.0002)
print(stats)