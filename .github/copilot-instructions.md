# BT3 Framework - AI Coding Agent Instructions

## Project Overview

BT3 is a specialized backtesting framework wrapper around [backtesting.py](https://github.com/kernc/backtesting.py) for forex trading strategies. It fetches historical data from [ejtraderLabs/historical-data](https://github.com/ejtraderLabs/historical-data) and provides forex-specific enhancements like automatic price scaling and spread modeling.

## Critical Architecture Patterns

### Data Pipeline & Auto-Scaling System

The `fetch_data()` function in [bt3.py](bt3.py) implements **symbol-aware automatic price normalization**:

- **JPY pairs** (GBPJPY, USDJPY, etc.): Target range [50, 300]
- **XAUUSD**: Target range [800, 4000]  
- **Other FX pairs**: Target range [0.5, 3.0]

The scaling algorithm evaluates divisors [1, 10, 100, 1000, 10000] and validates OHLC integrity with an 80% threshold check. **Never bypass or modify this logic** - it's essential for handling raw data that may be in different units (pips vs price). See lines 195-245 in [bt3.py](bt3.py).

### Spread Modeling vs Commission

BT3 supports two cost modeling approaches:

1. **Commission-based** (traditional): `run_backtest(data, Strategy, commission=0.0002)`
2. **Spread-based** (forex-realistic): `run_backtest(data, Strategy, spread_pips=1.5, symbol='GBPJPY')`

When `spread_pips` is provided:
- Automatically detects pip_size: JPY pairs=0.01, XAUUSD=0.1, others=0.0001
- Converts spread to price units: `spread_price = spread_pips * pip_size`
- Injects `spread_price` into strategy via `strategy_params`
- Sets `commission=0` unless explicitly overridden

Strategies must have a `spread_price = 0.0` class attribute to receive this value. See [alligator_fractal.py](alligator_fractal.py) line 160 for implementation.

## Strategy Development Patterns

### AlligatorFractal Strategy Architecture

Located in [alligator_fractal.py](alligator_fractal.py). Key components:

**Alligator State Machine** (`_alligator_state()` function):
- `sleeping`: Alligator lines too close (spread < min_spread_factor)
- `eating_up`: lips > teeth > jaw (bullish trend)
- `eating_down`: lips < teeth < jaw (bearish trend)
- `crossing`: Lines not aligned (no trade zone)

**SMMA with Forward Shifting**: `_smma_np()` computes smoothed moving averages, then `_shift_forward()` applies Bill Williams' future shifts (jaw_shift=8, teeth_shift=5, lips_shift=3). This is NOT lag compensation - it's the original indicator design.

**Fractal Detection**: 5-bar fractals (2 bars each side). The strategy uses **confirmed fractals** (shifted +2 bars) to ensure complete pattern formation before triggering entries.

**Stop Order Pattern**: The strategy places stop orders at fractal levels rather than market orders:
```python
entry_stop = last_bull + eps + self.spread_price / 2.0  # Long entry
entry_stop = last_bear - eps - self.spread_price / 2.0  # Short entry
```

Duplicate order prevention uses `_last_long_stop` and `_last_short_stop` tracking. The `_arm_brackets` pattern defers SL/TP application until after position opens to avoid same-bar contingent order errors.

### Indicator Caching Pattern

All custom strategies should cache indicators in `init()`:
```python
def init(self):
    # Compute once on full dataframe
    alligator = compute_alligator_ohlc(df, self._params)
    # Wrap in self.I() for backtesting.py integration
    self.jaw = self.I(lambda x=alligator['jaw'].to_numpy(): x)
```

Cache numpy arrays of price data in `init()` to avoid repeated conversions in `next()`:
```python
self._closes = np.asarray(self.data.Close)
```

## Testing & Validation

### Test Structure

- [test_bt3.py](test_bt3.py): Basic framework tests with synthetic data
- [test_alligator.py](test_alligator.py): Strategy tests with real forex data
- [test_improvements.py](test_improvements.py): Validates recent enhancements (auto-scaling, spread modeling)

Run tests directly: `python test_bt3.py` (no pytest required)

### Testing with Real Data

Always test strategies with **real forex data** from ejtraderLabs, not just synthetic data:
```python
data = fetch_data('GBPJPY', '4h')  # Fetches from GitHub repo
stats = run_backtest(data, AlligatorFractal, cash=10000)
```

Common test symbols: GBPJPY (high liquidity JPY pair), EURUSD (major pair), XAUUSD (gold, different price range)

## Data Format Requirements

DataFrames must have:
- DatetimeIndex (parsed from first column if no date column found)
- Columns: `Open`, `High`, `Low`, `Close`, `Volume` (exact capitalization)
- Sorted chronologically with no duplicate timestamps
- Float64 dtype for OHLC columns

The `fetch_data()` function handles all normalization automatically, including deduplication (`~df.index.duplicated(keep='first')`) and sorting (`df.sort_index(inplace=True)`).

## Common Development Workflows

### Adding a New Strategy

1. Inherit from `backtesting.Strategy`
2. Add `spread_price = 0.0` class attribute for spread modeling support
3. Compute indicators on full DataFrame in `init()`, wrap with `self.I()`
4. Cache numpy arrays of price data in `init()`
5. Implement `next()` with state management for pending orders if using stops

### Modifying Existing Strategies

When changing [alligator_fractal.py](alligator_fractal.py):
- **Never modify** the SMMA computation or forward-shifting logic (lines 45-80) - this matches Bill Williams' published algorithm
- Respect the state machine transitions in `_alligator_state()`
- Maintain the duplicate order prevention pattern with `_last_long_stop`/`_last_short_stop`
- Keep the `_arm_brackets` pattern for SL/TP application

### Debugging Strategies

Use the integrated test scripts in [test_alligator.py](test_alligator.py) which print detailed diagnostics. Check:
- OHLC integrity after auto-scaling (High >= max(Open,Close), Low <= min(Open,Close))
- Fractal detection (should see bullish/bearish fractal confirmations)
- Alligator state transitions (not stuck in 'sleeping' or 'unknown')

## External Dependencies & Integration

**backtesting.py Library**: Core dependency. We use `Backtest` class and inherit from `Strategy`. Key methods: `self.buy()`, `self.sell()`, `self.I()` (indicator wrapper), `self.position`.

**Data Source**: GitHub raw content from ejtraderLabs at `https://raw.githubusercontent.com/ejtraderLabs/historical-data/main/{SYMBOL}/{SYMBOL}{SUFFIX}.csv`. Only forex symbols supported (see `SUPPORTED_SYMBOLS` set in [bt3.py](bt3.py) lines 37-50).

**No Database or State Persistence**: All data fetched per-run from remote source. No caching layer.

## Project-Specific Conventions

- **Timeframe notation**: Accept both `"1d"`/`"d1"` formats, normalize with `_map_timeframe_suffix()` to ejtraderLabs convention
- **Price unit tolerance**: Use `eps = max(1e-4, abs(price) * 1e-6)` for floating-point comparisons in strategies
- **Stop order requirement**: Always set `exclusive_orders=True` when backtesting strategies with stop orders to prevent duplicate pending orders
- **Parameter classes**: Use `@dataclass` for strategy parameters (see `AlligatorParams`) to enable clean stats display

## Known Issues & Workarounds

**Same-bar contingent order error**: backtesting.py raises error if you set SL/TP when placing the entry order. Workaround: defer bracket application using `_arm_brackets` pattern (see [alligator_fractal.py](alligator_fractal.py) lines 225-230, 280-285).

**Duplicate pending orders**: Without tracking, strategy may place multiple stop orders at same level each bar. Workaround: track last stop levels and reset when position closes.

**OHLC scaling validation failures**: If auto-scaling produces invalid OHLC (High < Close or Low > Close), the function reverts to divisor=1. This indicates data quality issues at source.
