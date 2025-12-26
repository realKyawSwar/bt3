"""
Test script to demonstrate the improvements to bt3:
1. Robust auto-scaling for different symbol types
2. Spread modeling with pips
3. Updated AlligatorFractal strategy with spread_price
"""

from bt3 import fetch_data, run_backtest
from alligator_fractal import AlligatorFractal

print("=" * 70)
print("Testing bt3 Improvements")
print("=" * 70)

# Test 1: Auto-scaling for JPY pair
print("\n1. Testing auto-scaling for GBPJPY (JPY pair)...")
try:
    data_jpy = fetch_data('GBPJPY', '1d')
    print(f"   Sample prices: Open={data_jpy['Open'].iloc[-1]:.4f}, Close={data_jpy['Close'].iloc[-1]:.4f}")
    print(f"   Median Close: {data_jpy['Close'].median():.4f}")
except Exception as e:
    print(f"   Error: {e}")

# Test 2: Auto-scaling for EURUSD (non-JPY pair)
print("\n2. Testing auto-scaling for EURUSD (non-JPY pair)...")
try:
    data_eur = fetch_data('EURUSD', '1d')
    print(f"   Sample prices: Open={data_eur['Open'].iloc[-1]:.4f}, Close={data_eur['Close'].iloc[-1]:.4f}")
    print(f"   Median Close: {data_eur['Close'].median():.4f}")
except Exception as e:
    print(f"   Error: {e}")

# Test 3: Run backtest with spread modeling
print("\n3. Testing spread modeling with AlligatorFractal strategy...")
print("   Using 1.5 pip spread for GBPJPY...")
try:
    # Custom strategy with spread_price enabled
    class MyAlligator(AlligatorFractal):
        enable_tp = True
        tp_rr = 2.0
    
    stats = run_backtest(
        data=data_jpy,
        strategy=MyAlligator,
        cash=10000,
        spread_pips=1.5,  # 1.5 pip spread
        symbol='GBPJPY',  # For pip_size auto-detection (0.01 for JPY)
        commission=0.0    # Explicitly set to 0 when using spread
    )
    
    print(f"\n   Backtest completed!")
    print(f"   Return: {stats['Return [%]']:.2f}%")
    print(f"   # Trades: {stats['# Trades']}")
    print(f"   Win Rate: {stats['Win Rate [%]']:.2f}%")
    
except Exception as e:
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Traditional backtest without spread (backward compatibility)
print("\n4. Testing backward compatibility (no spread, commission-based)...")
try:
    stats_traditional = run_backtest(
        data=data_jpy,
        strategy=MyAlligator,
        cash=10000,
        commission=0.0002  # Traditional 0.02% commission
    )
    
    print(f"   Backtest completed!")
    print(f"   Return: {stats_traditional['Return [%]']:.2f}%")
    print(f"   # Trades: {stats_traditional['# Trades']}")
    
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "=" * 70)
print("All tests completed!")
print("=" * 70)
