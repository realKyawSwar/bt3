#!/usr/bin/env python3
"""
Quick Start Guide for BT3 Backtesting Framework

This script shows you how to get started with bt3 quickly.
"""

from backtesting import Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import pandas as pd
import numpy as np


def main():
    print("=" * 70)
    print("BT3 Quick Start Guide")
    print("=" * 70)
    
    # Step 1: Import the framework
    print("\n1. Import bt3 functions:")
    print("   from bt3 import fetch_data, run_backtest")
    from bt3 import fetch_data, run_backtest
    
    # Step 2: Define your strategy
    print("\n2. Define your trading strategy:")
    print("   class MyStrategy(Strategy): ...")
    
    class MyStrategy(Strategy):
        """Simple Moving Average Crossover Strategy"""
        
        # Parameters that can be optimized
        fast_ma = 10
        slow_ma = 20
        
        def init(self):
            """Initialize indicators"""
            # Calculate moving averages
            self.ma1 = self.I(SMA, self.data.Close, self.fast_ma)
            self.ma2 = self.I(SMA, self.data.Close, self.slow_ma)
        
        def next(self):
            """Trading logic executed on each bar"""
            # Buy when fast MA crosses above slow MA
            if crossover(self.ma1, self.ma2):
                if not self.position:
                    self.buy()
            
            # Sell when fast MA crosses below slow MA
            elif crossover(self.ma2, self.ma1):
                if self.position:
                    self.position.close()
    
    # Step 3: Get historical data
    print("\n3. Fetch historical data:")
    print("   data = fetch_data('BTCUSDT', '1d')")
    
    try:
        # Try to fetch real data
        data = fetch_data(symbol="BTCUSDT", timeframe="1d")
        print(f"   ✓ Loaded {len(data)} rows from remote source")
    except Exception as e:
        # Use sample data if remote fetch fails
        print(f"   ⚠ Remote fetch failed, using sample data")
        print(f"   Error: {e}")
        
        # Generate sample data
        MIN_PRICE = 1.0  # Minimum price threshold for realistic data
        dates = pd.date_range(start='2023-01-01', periods=150, freq='D')
        np.random.seed(42)
        
        price = 100
        opens, highs, lows, closes = [], [], [], []
        
        for _ in range(150):
            open_price = price
            close_price = price + np.random.randn() * 2
            high_price = max(open_price, close_price) + abs(np.random.randn() * 0.5)
            low_price = min(open_price, close_price) - abs(np.random.randn() * 0.5)
            
            opens.append(open_price)
            highs.append(high_price)
            lows.append(max(low_price, MIN_PRICE))
            closes.append(close_price)
            
            price = close_price
        
        data = pd.DataFrame({
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': closes,
            'Volume': np.random.randint(1000, 10000, 150)
        }, index=dates)
        
        print(f"   ✓ Generated {len(data)} rows of sample data")
    
    # Step 4: Run the backtest
    print("\n4. Run backtest:")
    print("   stats = run_backtest(data, MyStrategy)")
    
    stats = run_backtest(
        data=data,
        strategy=MyStrategy,
        cash=10000,        # Starting capital
        commission=0.001   # 0.1% commission per trade
    )
    
    # Step 5: View results
    print("\n5. View results:")
    print("=" * 70)
    
    # Display key metrics
    key_metrics = [
        'Start', 'End', 'Duration',
        'Return [%]', 'Buy & Hold Return [%]',
        'Sharpe Ratio', 'Max. Drawdown [%]',
        '# Trades', 'Win Rate [%]',
        'Best Trade [%]', 'Worst Trade [%]'
    ]
    
    for metric in key_metrics:
        if metric in stats:
            print(f"{metric:.<25} {stats[metric]}")
    
    print("=" * 70)
    
    # Tips
    print("\n" + "=" * 70)
    print("Tips:")
    print("=" * 70)
    print("• View full results: print(stats)")
    print("• Plot results: stats.plot()")
    print("• Optimize parameters: bt.optimize()")
    print("• See example.py for more strategy examples")
    print("• Read README.md for complete documentation")
    print("=" * 70)
    
    print("\n✓ Quick start complete! Happy backtesting!")


if __name__ == "__main__":
    main()
