# BT3 - Simple Backtesting Framework

A simple backtesting framework built on top of the [backtesting.py](https://github.com/kernc/backtesting.py) library, designed to work seamlessly with historical forex data from [ejtraderLabs/historical-data](https://github.com/ejtraderLabs/historical-data) repository.

## Features

- 🚀 Easy data fetching from ejtraderLabs historical forex data repository
- 📊 Simple wrapper around backtesting.py library
- 💹 Support for multiple trading strategies
- 📈 Comprehensive backtest statistics
- 🎯 Clean and intuitive API

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install backtesting pandas numpy
```

## Quick Start

For a ready-to-run comparison between strict and classic Alligator+Fractal strategies, call the comparison runner from the repo root (or inside `src/`):

```bash
# From repo root
python src/compare_strategies.py --asset GBPJPY --tf 1h --spread 1.5 --exclusive_orders
```

You can also point to your own CSV/Parquet data instead of remote fetch:

```bash
python src/compare_strategies.py --data "path/to/data.csv" --tf 4h --start 2023-01-01 --end 2024-12-31 --cash 50000
```

## Data Source

The framework fetches forex data from the ejtraderLabs historical data repository using the URL pattern:

```
https://raw.githubusercontent.com/ejtraderLabs/historical-data/main/{symbol}/{symbol}{suffix}.csv
```

Timeframe suffix mapping:
- Daily: `1d` or `d1` → `d1` (e.g., `GBPJPY/GBPJPYd1.csv`)
- Hourly: `1h`/`h1` → `h1`, `4h`/`h4` → `h4`
- Minutes: `5m`/`m5`, `15m`/`m15`, `30m`/`m30`
- Weekly: `1w`/`w1` → `w1`

Supported forex symbols (no crypto):
`AUDJPY`, `AUDUSD`, `EURCHF`, `EURGBP`, `EURJPY`, `EURUSD`, `GBPJPY`, `GBPUSD`, `USDCAD`, `USDCHF`, `USDJPY`, `XAUUSD`

## API Reference

### `fetch_data(symbol, timeframe)`

Fetches historical OHLCV data from the ejtraderLabs repository.

**Parameters:**
- `symbol` (str): Forex trading symbol (see supported list above)
- `timeframe` (str): Timeframe for the data (e.g., "1d", "4h", "1h")

**Returns:**
- `pd.DataFrame`: DataFrame with OHLCV data indexed by datetime

**Example:**
```python
data = fetch_data("GBPJPY", "1d")
```

### `run_backtest(data, strategy, cash=10000, commission=0.001, **kwargs)`

Runs a backtest using the backtesting.py library.

**Parameters:**
- `data` (pd.DataFrame): OHLCV data with datetime index
- `strategy` (Strategy class): Strategy class inheriting from backtesting.Strategy
- `cash` (float): Initial cash amount (default: 10000)
- `commission` (float): Commission rate per trade (default: 0.001 = 0.1%)
- `**kwargs`: Additional arguments to pass to Backtest

**Returns:**
- `dict`: Backtest statistics including returns, drawdowns, trade metrics, etc.

**Example:**
```python
stats = run_backtest(data, MyStrategy, cash=50000, commission=0.002)
```

## Alligator + Fractal Strategy

Implements Bill Williams' Alligator (SMMA of median price with forward shifts) and 5-bar fractals. **Now using realistic stop orders at fractal breakout levels** instead of immediate market entry.

**Key Features:**
- ✅ **Stop Order Entry**: Places stop orders at fractal levels (`last_bull + eps` for longs, `last_bear - eps` for shorts)
- ✅ **Performance Optimized**: Caches parameters and numpy arrays in `init()` to avoid repeated creation in `next()`
- ✅ **Smart Order Management**: Prevents duplicate pending orders by tracking last submitted stop levels
- ✅ **Bracket Orders**: Automatic stop-loss using opposite fractal, optional take-profit at customizable risk:reward ratio
- ✅ **R-Based Management**: Optional break-even move based on frozen entry risk (`R0`) plus post-exit cooldown to reduce churn
- ✅ **Structure-Based Exits**: Closes positions when Alligator structure is lost (lips > teeth > jaw for longs)

**Entry Logic:**
- Long entries: Stop order above last bullish fractal when Alligator is "eating up" (lips > teeth > jaw)
- Short entries: Stop order below last bearish fractal when Alligator is "eating down" (lips < teeth < jaw)
- Fractals must be positioned correctly relative to Alligator lines

**Usage:**

```python
from alligator_fractal import AlligatorFractal
from bt3 import fetch_data, run_backtest

# Basic usage (SL-only, no TP)
data = fetch_data('GBPJPY', '1h')
stats = run_backtest(
    data,
    AlligatorFractal,
    cash=10000,
    commission=0.0002,
    exclusive_orders=True  # Prevent duplicate pending orders
)
print(stats)

# With take profit enabled
class AlligatorTP(AlligatorFractal):
    enable_tp = True  # Enable take profit
    tp_rr = 2.0       # 2:1 risk/reward ratio

# Example: adjust break-even + cooldown behavior
class AlligatorBE(AlligatorFractal):
    enable_be = True
    be_at_r = 0.5
    be_buffer_r = 0.0
    cooldown_bars = 3

stats = run_backtest(data, AlligatorTP, cash=10000, commission=0.0002)
```

**Backtest Performance (GBPJPY 1H, 2012-2022):**
- Return: 35.04%
- Sharpe Ratio: 0.40
- Max Drawdown: -15.46%
- Win Rate: 37.91%
- Number of Trades: 1,807

### Compare strict vs classic Alligator+Fractal

Use the comparison runner to execute both the repo-strict strategy and the classic rule variant on the **same** data and timeframe. It exports per-strategy stats, trades, equity curves, plus a side-by-side comparison table.

**Example:**

```bash
python src/compare_strategies.py --asset GBPJPY --tf 1h --spread 1.5 --exclusive_orders
```

**Output layout:**

```
reports/
└── {asset}_{timeframe}_{timestamp}/
    ├── stats/
    │   ├── strict_stats.json
    │   └── classic_stats.json
    ├── trades/
    │   ├── strict_trades.csv
    │   └── classic_trades.csv
    ├── equity/
    │   ├── strict_equity.csv
    │   └── classic_equity.csv
    └── comparison.csv
```

**Key arguments:**
- `--asset` and `--tf` fetch remote data (e.g., `GBPJPY`, `1h`, `4h`, `15m`)
- `--data` lets you supply a CSV/Parquet file instead of remote fetch
- `--spread` FX spread in pips (e.g., `1.5`)
- `--exclusive_orders` prevents overlapping pending orders
- `--no-htf-bias` disables the higher-timeframe (default H4) Alligator bias gate
- `--no-vol-filter` disables the ATR(14) > SMA(100) volatility regime gate
- `--htf` sets the higher-timeframe rule for bias (e.g., `4h`, `1d`)
- `--atr` sets ATR period for volatility filter (default `14`)
- `--atr-long` sets ATR SMA length for volatility filter (default `100`)
- `--outdir` base output folder (default `reports/`)

### Run the Pullback variant

Use the pullback strategy class to require a pullback into the Alligator Teeth zone before the strict fractal breakout entry.

```python
from alligator_fractal import AlligatorFractalPullback
from bt3 import fetch_data, run_backtest

data = fetch_data("GBPJPY", "1h")
stats = run_backtest(
    data,
    AlligatorFractalPullback,
    cash=10000,
    commission=0.0002,
    exclusive_orders=True,
)
print(stats)
```

Tune pullback behavior via class attributes:

```python
class PullbackTuned(AlligatorFractalPullback):
    pullback_k_atr = 0.35
    require_touch_teeth = False
```

### Run with `custom_alligator.py`

Use the interactive runner to select strict/classic/pullback strategies (plus your own tuned subclasses) and output reports:

```bash
# From repo root
python src/custom_alligator.py --asset GBPJPY --tf 1h --spread 1.5 --exclusive_orders
```

You can also point it to local data:

```bash
python src/custom_alligator.py --data "path/to/data.csv" --tf 4h --cash 50000
```

## Regression Checks

Run the lightweight filter-coverage check to ensure enabling HTF bias + volatility filter reduces trades moderately:

```bash
python scripts/regression_filter_check.py
```

## Strategy Development

Strategies should inherit from `backtesting.Strategy` and implement two methods:

```python
from backtesting import Strategy

class MyStrategy(Strategy):
    # Define strategy parameters
    param1 = 20
    param2 = 50
    
    def init(self):
        # Initialize indicators (called once)
        # Use self.I() to register indicators
        pass
    
    def next(self):
        # Define trading logic (called for each bar)
        # Use self.buy() or self.position.close()
        pass
```

## Backtest Results

The `run_backtest()` function returns comprehensive statistics:

- **Performance**: Return %, CAGR, Sharpe Ratio, Sortino Ratio
- **Risk**: Max Drawdown, Volatility, Beta, Alpha
- **Trade Statistics**: Win Rate, # of Trades, Profit Factor
- **Execution**: Exposure Time, Commissions

## “baseline v1”

-HTF bias ON
-ATR period 14
-ATR long 50
-Spread 1.5

Exclusive orders ON

Default:

python src/compare_strategies.py
runs GBPJPY 1h, spread 1.5, HTF 4h, ATR 14/50, exclusive orders ON.

Override example:

python src/compare_strategies.py --asset EURUSD --tf 1h --atr-long 100 --spread 1.0

Pullback:

python src/compare_strategies.py --strategy pullback --pullback-k 0.25

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

MIT License

## Acknowledgments

- [backtesting.py](https://github.com/kernc/backtesting.py) - Awesome backtesting library
- [ejtraderLabs/historical-data](https://github.com/ejtraderLabs/historical-data) - Historical data source
