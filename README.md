# BT3 Backtesting Toolkit

Lightweight forex backtesting utilities on top of `backtesting.py`, plus a CLI runner for strategy comparisons and an FX cross-sectional momentum benchmark.

## Features
- Fetches forex OHLCV from the `ejtraderLabs/historical-data` repo or local CSV/Parquet.
- Alligator + Fractal strategies (strict/classic/pullback) and Wave5 AO divergence with extensive tuning flags.
- New FX 12-month cross-sectional momentum benchmark with optional carry, volatility targeting, and monthly rebalance.
- Reporting helpers to export trades, equity curves, metrics, and summary tables to `reports/`.

## Installation
```
pip install -r requirements.txt
```

## Quick Start
- Alligator vs classic compare:
  ```
  python src/compare_strategies.py --asset GBPJPY --tf 1h --spread 1.5 --exclusive_orders
  ```
- Wave5 AO divergence (explicit sizing example):
  ```
  python src/compare_strategies.py --mode wave5 --asset XAUUSD --tf 1h --spread 30 \
      --wave5-size 0.2 --wave5-entry-mode break --wave5-trigger-lag 24
  ```
- FX momentum benchmark (12m spot, monthly rebalance):
  ```
  python src/compare_strategies.py --mode fxmom \
      --fxmom-pairs EURUSD,GBPUSD,USDJPY,USDCHF,AUDUSD,NZDUSD,USDCAD \
      --fxmom-k 2 --fxmom-target-vol 0.10 --fxmom-use-carry 0
  ```

## Data
- Remote fetch pattern: `https://raw.githubusercontent.com/ejtraderLabs/historical-data/main/{SYMBOL}/{SYMBOL}{suffix}.csv`
- Timeframe suffix examples: `1d/d1`, `4h/h4`, `1h/h1`, `15m/m15`, `30m/m30`, `5m/m5`.
- Supported symbols: `AUDJPY`, `AUDUSD`, `EURCHF`, `EURGBP`, `EURJPY`, `EURUSD`, `GBPJPY`, `GBPUSD`, `USDCAD`, `USDCHF`, `USDJPY`, `XAUUSD`.
- You can also pass `--data <csv|parquet>` or `--fxmom-data-dir` for local files.

## FX Momentum (benchmark mode)
- Spot-only 12m log-momentum on USD crosses; longs top `k`, shorts bottom `k`, net 0; monthly rebalance.
- Optional carry: `--fxmom-use-carry 1 --fxmom-rates-csv data/rates_monthly.csv` (columns: `date,USD,EUR,JPY,GBP,CHF,AUD,NZD,CAD` in decimal).
- Vol targeting: `--fxmom-target-vol 0.10 --fxmom-vol-lookback 12 --fxmom-max-lev 3.0`.
- Reports saved under `reports/fxmom_*timestamp*/`: currency scores/weights, pair weights, returns, equity, metrics JSON, leverage, and equity PNG.

## Outputs
- Stats/trades/equity CSVs for strategy runs (under `reports/` with timestamped subfolders).
- Equity PNG for FX momentum; use `reporting.plot_equity_curve` for other strategies.

## Development
- Tests are in `tests/` and `src/test_*.py`; run with `pytest`.
