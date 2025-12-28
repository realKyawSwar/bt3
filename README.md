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

### Basic Usage

```python
from backtesting import Strategy
from backtesting.test import SMA
from bt3 import fetch_data, run_backtest

# Define your strategy
class SMAStrategy(Strategy):
    fast_period = 10
    slow_period = 30
    
    def init(self):
        self.sma_fast = self.I(SMA, self.data.Close, self.fast_period)
        self.sma_slow = self.I(SMA, self.data.Close, self.slow_period)
    
    def next(self):
        if self.sma_fast[-1] > self.sma_slow[-1]:
            if not self.position:
                self.buy()
        elif self.sma_fast[-1] < self.sma_slow[-1]:
            if self.position:
                self.position.close()

# Fetch forex data from ejtraderLabs (forex only)
data = fetch_data(symbol="GBPJPY", timeframe="1d")

# Run backtest
stats = run_backtest(data, SMAStrategy, cash=100000, commission=0.0002)
print(stats)
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

## Examples

The repository includes `example.py` with three complete strategy examples:

1. **SMA Crossover Strategy** - Moving average crossover
2. **RSI Strategy** - Mean reversion based on RSI
3. **Breakout Strategy** - Channel breakout
4. **Alligator + Fractal** - Bill Williams Alligator with fractal breakouts

Run the examples:

```bash
python example.py
```

Or run the built-in demo:

```bash
python bt3.py
```

### Alligator + Fractal Strategy

Implements Bill Williams' Alligator (SMMA of median price with forward shifts) and 5-bar fractals. **Now using realistic stop orders at fractal breakout levels** instead of immediate market entry.

**Key Features:**
- ✅ **Stop Order Entry**: Places stop orders at fractal levels (`last_bull + eps` for longs, `last_bear - eps` for shorts)
- ✅ **Performance Optimized**: Caches parameters and numpy arrays in `init()` to avoid repeated creation in `next()`
- ✅ **Smart Order Management**: Prevents duplicate pending orders by tracking last submitted stop levels
- ✅ **Bracket Orders**: Automatic stop-loss using opposite fractal, optional take-profit at customizable risk:reward ratio
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

stats = run_backtest(data, AlligatorTP, cash=10000, commission=0.0002)
```

**Backtest Performance (GBPJPY 1H, 2012-2022):**
- Return: 35.04%
- Sharpe Ratio: 0.40
- Max Drawdown: -15.46%
- Win Rate: 37.91%
- Number of Trades: 1,807

### Compare strict vs classic Alligator+Fractal

Use the `custom_alligator.py` runner to execute both the repo-strict strategy and the classic rule variant on the **same** data and timeframe. It exports per-strategy stats, trades, equity curves, plus a side-by-side comparison table.

**Example:**

```bash
python custom_alligator.py
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

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

MIT License

## Acknowledgments

- [backtesting.py](https://github.com/kernc/backtesting.py) - Awesome backtesting library
- [ejtraderLabs/historical-data](https://github.com/ejtraderLabs/historical-data) - Historical data source
