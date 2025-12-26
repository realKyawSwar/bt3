"""
Test the updated AlligatorFractal strategy with stop orders
"""

import pandas as pd
import numpy as np
from bt3 import fetch_data, run_backtest
from alligator_fractal import AlligatorFractal


def test_alligator_fractal():
    """Test AlligatorFractal strategy with real forex data"""
    
    print("Fetching GBPJPY data...")
    data = fetch_data(symbol="GBPJPY", timeframe="1h")
    
    print(f"Data loaded: {len(data)} bars")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    print("\n" + "=" * 70)
    print("Running AlligatorFractal Backtest (with stop orders)")
    print("=" * 70)
    
    # Run backtest with default parameters
    stats = run_backtest(
        data, 
        AlligatorFractal,
        cash=10000,
        commission=0.0002,  # 2 pips
        exclusive_orders=True  # Prevent duplicate pending orders
    )
    
    print("\n" + "=" * 70)
    print("Backtest Results")
    print("=" * 70)
    print(stats)
    
    # Print key metrics
    print("\n" + "=" * 70)
    print("Key Metrics")
    print("=" * 70)
    print(f"Return: {stats['Return [%]']:.2f}%")
    print(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
    print(f"Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%")
    print(f"Win Rate: {stats['Win Rate [%]']:.2f}%")
    print(f"Number of Trades: {stats['# Trades']}")
    
    return stats


def test_alligator_with_tp():
    """Test AlligatorFractal strategy with take profit enabled"""
    
    print("\n" + "=" * 70)
    print("Running AlligatorFractal with Take Profit")
    print("=" * 70)
    
    data = fetch_data(symbol="EURUSD", timeframe="4h")
    
    print(f"Data loaded: {len(data)} bars")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Create a modified strategy class with TP enabled
    class AlligatorFractalTP(AlligatorFractal):
        enable_tp = True
        tp_rr = 2.0
    
    # Run backtest with TP enabled
    stats = run_backtest(
        data, 
        AlligatorFractalTP,
        cash=10000,
        commission=0.0002,
        exclusive_orders=True
    )
    
    print("\n" + "=" * 70)
    print("Results with TP enabled")
    print("=" * 70)
    print(f"Return: {stats['Return [%]']:.2f}%")
    print(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
    print(f"Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%")
    print(f"Win Rate: {stats['Win Rate [%]']:.2f}%")
    print(f"Number of Trades: {stats['# Trades']}")
    
    return stats


def test_strategy_validation():
    """Validate that the strategy is using stop orders correctly"""
    
    print("\n" + "=" * 70)
    print("Validating Strategy Implementation")
    print("=" * 70)
    
    # Check that the strategy class has the required cached attributes
    from alligator_fractal import AlligatorFractal
    import inspect
    
    # Get the init method source
    init_source = inspect.getsource(AlligatorFractal.init)
    next_source = inspect.getsource(AlligatorFractal.next)
    
    # Validate caching
    print("\n[OK] Checking for cached parameters...")
    assert 'self._params' in init_source, "Should cache params in init()"
    print("  [OK] self._params found in init()")
    
    assert 'self._closes' in init_source, "Should cache closes array"
    assert 'self._highs' in init_source, "Should cache highs array"
    assert 'self._lows' in init_source, "Should cache lows array"
    print("  [OK] Cached numpy arrays found in init()")
    
    assert 'self._last_long_stop' in init_source, "Should track long stop orders"
    assert 'self._last_short_stop' in init_source, "Should track short stop orders"
    print("  [OK] Pending order trackers found in init()")
    
    # Validate stop orders
    print("\n[OK] Checking for stop order usage...")
    assert 'stop=' in next_source, "Should use stop orders"
    print("  [OK] stop= parameter found in next()")
    
    # Validate that we're NOT recreating params in next()
    print("\n[OK] Checking that params are not recreated in next()...")
    assert 'AlligatorParams(' not in next_source, "Should NOT recreate AlligatorParams in next()"
    print("  [OK] AlligatorParams not recreated in next()")
    
    # Validate using cached arrays
    assert 'self._closes' in next_source or 'self._params' in next_source, "Should use cached values"
    print("  [OK] Cached values used in next()")
    
    print("\n" + "=" * 70)
    print("[OK] All validation checks passed!")
    print("=" * 70)


def main():
    print("=" * 70)
    print("AlligatorFractal Strategy Test Suite")
    print("=" * 70)
    
    try:
        # Validate implementation
        test_strategy_validation()
        
        # Run basic test
        stats1 = test_alligator_fractal()
        
        # Run test with TP
        stats2 = test_alligator_with_tp()
        
        print("\n" + "=" * 70)
        print("[OK] All tests completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
