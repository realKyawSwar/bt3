"""
Example usage of the bt3 backtesting framework

This example demonstrates how to:
1. Fetch data from ejtraderLabs historical data repository
2. Define a custom trading strategy
3. Run a backtest and analyze results
"""

from backtesting import Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
from bt3 import fetch_data, run_backtest
import pandas as pd
import numpy as np


# Example 1: Simple Moving Average Crossover Strategy
class SMAXStrategy(Strategy):
    """
    Simple Moving Average Crossover Strategy
    - Buy when fast MA crosses above slow MA
    - Sell when fast MA crosses below slow MA
    """
    
    # Strategy parameters (can be optimized)
    fast_period = 10
    slow_period = 30
    
    def init(self):
        # Pre-calculate indicators
        self.sma_fast = self.I(SMA, self.data.Close, self.fast_period)
        self.sma_slow = self.I(SMA, self.data.Close, self.slow_period)
    
    def next(self):
        # Trading logic
        if crossover(self.sma_fast, self.sma_slow):
            # Fast MA crosses above slow MA - Buy signal
            if not self.position:
                self.buy()
        elif crossover(self.sma_slow, self.sma_fast):
            # Fast MA crosses below slow MA - Sell signal
            if self.position:
                self.position.close()


# Example 2: RSI Strategy
class RSIStrategy(Strategy):
    """
    RSI-based mean reversion strategy
    - Buy when RSI < oversold level
    - Sell when RSI > overbought level
    """
    
    rsi_period = 14
    oversold = 30
    overbought = 70
    
    def init(self):
        # Calculate RSI indicator
        close = self.data.Close
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.rsi_period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        self.rsi = self.I(lambda: 100 - (100 / (1 + rs)))
    
    def next(self):
        if self.rsi[-1] < self.oversold:
            if not self.position:
                self.buy()
        elif self.rsi[-1] > self.overbought:
            if self.position:
                self.position.close()


# Example 3: Breakout Strategy
class BreakoutStrategy(Strategy):
    """
    Breakout strategy based on channel breakout
    - Buy when price breaks above upper channel
    - Sell when price breaks below lower channel
    """
    
    lookback = 20
    
    def init(self):
        # Calculate channel bounds
        self.upper_channel = self.I(lambda: pd.Series(self.data.High).rolling(self.lookback).max())
        self.lower_channel = self.I(lambda: pd.Series(self.data.Low).rolling(self.lookback).min())
    
    def next(self):
        if len(self.data.Close) < self.lookback + 1:
            return
        
        if self.data.Close[-1] > self.upper_channel[-2]:
            if not self.position:
                self.buy()
        elif self.data.Close[-1] < self.lower_channel[-2]:
            if self.position:
                self.position.close()


def main():
    print("=" * 70)
    print("BT3 - Simple Backtesting Framework Examples")
    print("=" * 70)
    
    # Try to fetch real data from remote source
    try:
        print("\nFetching data from remote source...")
        data = fetch_data(symbol="BTCUSDT", timeframe="1d")
        print(f"Successfully loaded {len(data)} rows of real data")
    except Exception as e:
        print(f"Could not fetch real data: {e}")
        print("\nGenerating sample data for demonstration...")
        
        # Create realistic sample data
        dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
        np.random.seed(42)
        
        price = 100
        prices = []
        for _ in range(200):
            price += np.random.randn() * 2
            prices.append(max(price, 1))  # Ensure positive prices
        
        data = pd.DataFrame({
            'Open': prices,
            'High': [p + abs(np.random.randn()) for p in prices],
            'Low': [max(p - abs(np.random.randn()), 0.1) for p in prices],
            'Close': [p + np.random.randn() * 0.5 for p in prices],
            'Volume': np.random.randint(1000, 10000, 200)
        }, index=dates)
    
    # Example 1: SMA Crossover Strategy
    print("\n" + "=" * 70)
    print("Example 1: Simple Moving Average Crossover Strategy")
    print("=" * 70)
    
    stats = run_backtest(
        data=data,
        strategy=SMAXStrategy,
        cash=10000.0,
        commission=0.001
    )
    
    print(stats)
    
    # Example 2: RSI Strategy
    print("\n" + "=" * 70)
    print("Example 2: RSI Mean Reversion Strategy")
    print("=" * 70)
    
    stats = run_backtest(
        data=data,
        strategy=RSIStrategy,
        cash=10000.0,
        commission=0.001
    )
    
    print(stats)
    
    # Example 3: Breakout Strategy
    print("\n" + "=" * 70)
    print("Example 3: Channel Breakout Strategy")
    print("=" * 70)
    
    stats = run_backtest(
        data=data,
        strategy=BreakoutStrategy,
        cash=10000.0,
        commission=0.001
    )
    
    print(stats)
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
