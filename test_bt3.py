"""
Simple test to verify the bt3 framework works correctly
"""

import pandas as pd
import numpy as np
from backtesting import Strategy
from backtesting.test import SMA
from bt3 import fetch_data, run_backtest


def create_sample_data():
    """Create sample OHLCV data for testing"""
    MIN_PRICE = 1.0  # Minimum price threshold for realistic data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Generate realistic OHLC data
    opens = []
    highs = []
    lows = []
    closes = []
    
    price = 100
    for _ in range(100):
        open_price = price
        close_price = price + np.random.randn() * 2
        
        # Ensure high/low make sense
        high_price = max(open_price, close_price) + abs(np.random.randn() * 0.5)
        low_price = min(open_price, close_price) - abs(np.random.randn() * 0.5)
        low_price = max(low_price, MIN_PRICE)  # Ensure positive
        
        opens.append(open_price)
        highs.append(high_price)
        lows.append(low_price)
        closes.append(close_price)
        
        price = close_price
    
    data = pd.DataFrame({
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    return data


def test_simple_strategy():
    """Test a simple buy-and-hold strategy"""
    
    class BuyAndHold(Strategy):
        def init(self):
            pass
        
        def next(self):
            if not self.position:
                self.buy()
    
    data = create_sample_data()
    stats = run_backtest(data, BuyAndHold, cash=10000, commission=0.001)
    
    # Basic assertions - just verify the framework runs
    assert 'Return [%]' in stats, "Should have return percentage"
    assert 'Sharpe Ratio' in stats, "Should have Sharpe ratio"
    
    print("✓ test_simple_strategy passed")


def test_sma_strategy():
    """Test SMA crossover strategy"""
    
    class SimpleStrategy(Strategy):
        fast_period = 5
        slow_period = 10
        
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
    
    data = create_sample_data()
    stats = run_backtest(data, SimpleStrategy, cash=10000, commission=0.001)
    
    # Basic assertions
    assert 'Return [%]' in stats, "Should have return percentage"
    assert 'Sharpe Ratio' in stats, "Should have Sharpe ratio"
    assert 'Max. Drawdown [%]' in stats, "Should have max drawdown"
    
    print("✓ test_sma_strategy passed")


def test_data_structure():
    """Test that sample data has correct structure"""
    data = create_sample_data()
    
    # Check required columns
    assert 'Open' in data.columns, "Should have Open column"
    assert 'High' in data.columns, "Should have High column"
    assert 'Low' in data.columns, "Should have Low column"
    assert 'Close' in data.columns, "Should have Close column"
    assert 'Volume' in data.columns, "Should have Volume column"
    
    # Check index is datetime
    assert isinstance(data.index, pd.DatetimeIndex), "Index should be DatetimeIndex"
    
    # Check data validity
    assert len(data) > 0, "Should have data"
    assert (data['High'] >= data['Low']).all(), "High should be >= Low"
    assert (data['High'] >= data['Close']).all(), "High should be >= Close"
    assert (data['High'] >= data['Open']).all(), "High should be >= Open"
    assert (data['Low'] <= data['Close']).all(), "Low should be <= Close"
    assert (data['Low'] <= data['Open']).all(), "Low should be <= Open"
    
    print("✓ test_data_structure passed")


def main():
    print("=" * 50)
    print("Running BT3 Framework Tests")
    print("=" * 50)
    
    try:
        test_data_structure()
        test_simple_strategy()
        test_sma_strategy()
        
        print("\n" + "=" * 50)
        print("All tests passed! ✓")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
