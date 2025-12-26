# BT3 - Simple Backtesting Framework

A simple backtesting framework built on top of the [backtesting.py](https://github.com/kernc/backtesting.py) library, designed to work seamlessly with historical data from [ejtraderLabs/historical-data](https://github.com/ejtraderLabs/historical-data) repository.

## Features

- 🚀 Easy data fetching from ejtraderLabs historical data repository
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

# Fetch data from ejtraderLabs
data = fetch_data(symbol="BTCUSDT", timeframe="1d")

# Run backtest
stats = run_backtest(data, SMAStrategy, cash=10000, commission=0.001)
print(stats)
```

## Data Source

The framework fetches data from the ejtraderLabs historical data repository using the URL pattern:

```
https://raw.githubusercontent.com/ejtraderLabs/historical-data/main/{symbol}/{symbol}{timeframe}.csv
```

Examples:
- `BTCUSDT` with `1d` timeframe → `BTCUSDT/BTCUSDT1d.csv`
- `ETHUSDT` with `4h` timeframe → `ETHUSDT/ETHUSDT4h.csv`

## API Reference

### `fetch_data(symbol, timeframe)`

Fetches historical OHLCV data from the ejtraderLabs repository.

**Parameters:**
- `symbol` (str): Trading symbol (e.g., "BTCUSDT", "ETHUSDT")
- `timeframe` (str): Timeframe for the data (e.g., "1d", "4h", "1h")

**Returns:**
- `pd.DataFrame`: DataFrame with OHLCV data indexed by datetime

**Example:**
```python
data = fetch_data("BTCUSDT", "1d")
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

Run the examples:

```bash
python example.py
```

Or run the built-in demo:

```bash
python bt3.py
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