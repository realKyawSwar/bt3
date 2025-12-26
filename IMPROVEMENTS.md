# BT3 Improvements Summary

## Changes Implemented

### 1. Robust Auto-Scaling Selector in `fetch_data()`

**Previous Issue:** Simple heuristic that only divided by 1000 when median > 1000, which was too simplistic and error-prone.

**New Implementation:**
- Symbol-aware target ranges:
  - JPY pairs (GBPJPY, USDJPY, etc.): target range [50, 300]
  - XAUUSD: target range [800, 4000]
  - Other FX pairs: target range [0.5, 3.0]
  
- Evaluates divisors [1, 10, 100, 1000, 10000] and selects the one that:
  - Places scaled median within target range (if possible)
  - Otherwise minimizes distance to target range
  - Chooses divisor yielding median closest to target midpoint
  
- **Sanity check:** Validates OHLC integrity after scaling:
  - Ensures High >= max(Open, Close) for >80% of rows
  - Ensures Low <= min(Open, Close) for >80% of rows
  - Reverts to divisor=1 if validation fails
  
- Only prints scaling message when divisor != 1

### 2. Spread/Pip Cost Modeling in `run_backtest()`

**New Parameters:**
- `spread_pips` (float, optional): Spread cost in pips (e.g., 1.5)
- `pip_size` (float, optional): Size of one pip in price units
- `symbol` (str, optional): For pip_size auto-detection

**Auto-detection of pip_size:**
- JPY pairs: 0.01
- XAUUSD: 0.1
- Others: 0.0001

**Behavior:**
- When `spread_pips` is provided:
  - Converts spread to price units: `spread_price = spread_pips * pip_size`
  - Injects `spread_price` into strategy via `strategy_params`
  - Sets `commission=0` (unless explicitly overridden)
  
**Backward Compatible:** 
- Existing code without spread_pips continues to work with commission-based costs

### 3. AlligatorFractal Strategy Updates

**New class attribute:**
```python
spread_price = 0.0  # Spread cost in price units
```

**Entry adjustments:**
- Long buy stop: `entry_stop = last_bull + eps + spread_price / 2.0`
- Short sell stop: `entry_stop = last_bear - eps - spread_price / 2.0`

This models the cost of crossing the spread at entry.

### 4. Quality Fixes in `fetch_data()`

1. **Index sorting:** `df.sort_index(inplace=True)`
2. **Deduplication:** `df = df[~df.index.duplicated(keep='first')]`
3. **Dtype enforcement:** All OHLC columns converted to float64
4. **Volume handling:** Kept as numeric (float/int)

## Usage Examples

### Example 1: Basic usage with auto-scaling
```python
from bt3 import fetch_data, run_backtest
from alligator_fractal import AlligatorFractal

# Auto-scaling will detect JPY pair and normalize appropriately
data = fetch_data('GBPJPY', '4h')
stats = run_backtest(data, AlligatorFractal, cash=10000)
```

### Example 2: Using spread modeling
```python
# Spread modeling with 1.5 pip spread
stats = run_backtest(
    data=data,
    strategy=AlligatorFractal,
    cash=10000,
    spread_pips=1.5,      # 1.5 pip spread
    symbol='GBPJPY',      # Auto-detects pip_size=0.01
    commission=0.0        # Spread replaces commission
)
```

### Example 3: Custom pip_size
```python
# Explicitly set pip_size
stats = run_backtest(
    data=data,
    strategy=AlligatorFractal,
    cash=10000,
    spread_pips=2.0,
    pip_size=0.015,       # Custom pip size
)
```

## Technical Notes

### Auto-Scaling Algorithm
1. Load data and convert OHLC to numeric floats
2. Compute median of Close column
3. Determine target range based on symbol type
4. Evaluate divisors and score each based on:
   - In-range: distance to midpoint
   - Out-of-range: distance to nearest boundary
5. Apply best divisor
6. Validate OHLC integrity (High/Low constraints)
7. Revert if validation fails (<80% valid rows)

### Spread Cost Implementation
- Spread is modeled at entry via adjusted stop prices
- Does NOT modify historical OHLC data
- Strategy receives `spread_price` as parameter
- Compatible with existing strategies (spread_price defaults to 0.0)

### Backward Compatibility
- All existing function signatures remain compatible
- New parameters are optional with sensible defaults
- Strategies without spread_price support still work (defaults to 0.0)
- Commission-based costing still available when spread_pips not provided

## Testing

Run the test script to verify all improvements:
```bash
python test_improvements.py
```

This tests:
1. Auto-scaling for JPY pairs
2. Auto-scaling for non-JPY pairs
3. Spread modeling with AlligatorFractal
4. Backward compatibility with commission-based costs
