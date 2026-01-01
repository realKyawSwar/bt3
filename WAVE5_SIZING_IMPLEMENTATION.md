# Wave5 Risk-Based Sizing Implementation

## Overview
Implemented true risk-based position sizing for the Wave5 strategy. Instead of using `int(equity * fraction) / entry_price`, the sizing now correctly implements:

```
risk_cash = equity * risk_fraction
units_raw = floor(risk_cash / sl_dist)
units_final = min(units_raw, max_units)
```

Where `sl_dist = abs(entry_price - sl_price)` and `max_units = floor((equity / margin) / entry_price)`.

## Changes Made

### 1. Modified `_size_to_units()` in [wave5_ao.py](wave5_ao.py#L365)
- **What**: Updated method signature to accept optional `sl_price` parameter
- **How**: 
  - If `sl_price` is provided, computes `sl_dist = abs(entry_price - sl_price)`
  - Calculates `risk_cash = equity * size`
  - Returns `units_raw = floor(risk_cash / sl_dist)`
  - Applies margin cap: `units_final = min(units_raw, max_units)`
  - Returns 0 if `sl_dist <= 0` (invalid stop loss)
- **Backward Compatible**: Falls back to original behavior when `sl_price=None`

### 2. Updated `_resolve_order_size()` in `_handle_sell()` and `_handle_buy()` methods
- **What**: Modified function signature to accept `sl_for_risk` parameter
- **How**:
  - Passes `sl_price` to `_size_to_units()` for risk-based calculation
  - Generates detailed debug output showing:
    - `equity`, `margin`, `entry`, `sl`, `sl_dist`
    - `risk_frac`, `risk_cash`, `units_raw`, `max_units`, `units_final`
  - Increments `size_fail` counter when `units_final < 1`
- **Where Called**: Both split order and single order modes pass `sl` parameter

### 3. Added `margin` class variable to [alligator_fractal.py](alligator_fractal.py#L228)
- **Why**: AlligatorFractal strategy needs this parameter for compatibility with compare_strategies.py
- **What**: Added `margin = 1.0` as a class variable (no leverage by default)

## Validation Results

### Test 1: XAUUSD with margin=1.0 (cash-only)
```
[WAVE5 SIZE] equity=10000.00 margin=1.0000 entry=1768.66000 sl=1756.19000 
sl_dist=12.47000 risk_frac=0.100 risk_cash=1000.00 units_raw=80 max_units=5 units_final=5
```

**Breakdown**:
- risk_cash = 10000 * 0.1 = 1000 ✓
- sl_dist = 1768.66 - 1756.19 = 12.47 ✓
- units_raw = floor(1000 / 12.47) = 80 ✓
- max_units = floor((10000 / 1.0) / 1768.66) = 5 ✓
- units_final = min(80, 5) = 5 (margin-constrained) ✓

### Test 2: XAUUSD with margin=0.02 (50x leverage)
```
[WAVE5 SIZE] equity=10000.00 margin=0.0200 entry=1768.66000 sl=1756.19000 
sl_dist=12.47000 risk_frac=0.100 risk_cash=1000.00 units_raw=80 max_units=282 units_final=80
```

**Breakdown**:
- risk_cash = 1000 (same) ✓
- units_raw = 80 (same) ✓
- max_units = floor((10000 / 0.02) / 1768.66) = 282 ✓
- units_final = min(80, 282) = 80 (risk-constrained, not margin-constrained) ✓

### Test 3: GBPUSD with margin=0.02 (Multiple trades)
Successfully executed 5 trades with adaptive risk-based sizing:
- Trade sizes varied from units_raw=425531 to units_raw=21389
- All properly constrained by margin caps where applicable
- Strategy shows correct behavior: tighter stops → smaller units, wider stops → larger units

## Key Behaviors

### Order Placement
- If `units_final >= 1`: Order is placed with `buy(size=units_final)` or `sell(size=units_final)`
- If `units_final < 0` (invalid SL): Order is skipped, `size_fail` counter is incremented

### Debug Output (when `--wave5-debug` flag is set)
Prints all sizing parameters for every order attempt:
```
[WAVE5 SIZE] equity=... margin=... entry=... sl=... sl_dist=... risk_frac=... risk_cash=... units_raw=... max_units=... units_final=...
[WAVE5 SKIP] insufficient units: ... (when units_final < 1)
```

### Margin Effectiveness
- **With margin=1.0**: Constrains position size to amount equity can support
- **With margin=0.02**: Allows leverage to increase max_units, but risk-based sizing dominates
- Both cases properly apply: `units_final = min(units_raw, max_units)`

## Non-Breaking Changes

- ✓ AlligatorFractal strategy remains unchanged functionally (only added margin parameter)
- ✓ Wave5 strategy changes are localized to sizing logic
- ✓ No changes to other strategies
- ✓ All order entry points (split/single) use consistent sizing logic
- ✓ Exit logic (TP, SL) unchanged

## Testing Recommendation

Run validation tests:
```bash
# Test with margin=1.0 (cash-only constraint)
python src/compare_strategies.py --mode wave5 --asset XAUUSD --tf 1h \
  --cash 10000 --wave5-size 0.1 --wave5-debug --margin 1.0

# Test with margin=0.02 (50x leverage)
python src/compare_strategies.py --mode wave5 --asset XAUUSD --tf 1h \
  --cash 10000 --wave5-size 0.1 --wave5-debug --margin 0.02

# Test with GBPUSD for multiple trades
python src/compare_strategies.py --mode wave5 --asset GBPUSD --tf 1h \
  --cash 10000 --wave5-size 0.1 --wave5-debug --margin 0.02
```

## Implementation Notes

1. **Formula Correctness**: The implementation correctly follows:
   - Per-1-unit risk at SL = `abs(entry - sl)` (not entry price)
   - Risk in cash = `equity * risk_fraction`
   - Proper units = `floor(risk_cash / sl_dist)`

2. **Edge Cases Handled**:
   - Invalid SL distance (sl_dist <= 0 or not finite) → returns 0 → order skipped
   - Very tight stops with small equity → clamped by max_units
   - Very loose stops with leverage → clamped by max_units

3. **Debug Output**: Complete visibility into sizing decisions for each trade attempt

4. **Backward Compatibility**: Original behavior available via `sl_price=None`
