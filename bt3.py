"""
Simple Backtesting Framework using backtesting.py library

This module provides a simple wrapper around the backtesting.py library
to fetch historical data from ejtraderLabs historical data repository and
make it easy to backtest trading strategies.

Usage:
    from backtesting import Strategy, Backtest
    from bt3 import fetch_data, run_backtest
    
    # Fetch data from remote source (forex only)
    data = fetch_data(symbol="GBPJPY", timeframe="1d")
    
    # Define your strategy
    class MyStrategy(Strategy):
        def init(self):
            pass
        
        def next(self):
            if self.data.Close[-1] > self.data.Close[-2]:
                if not self.position:
                    self.buy()
            elif self.data.Close[-1] < self.data.Close[-2]:
                if self.position:
                    self.position.close()
    
    # Run backtest
    stats = run_backtest(data, MyStrategy)
    print(stats)
"""

import pandas as pd
import numpy as np
import urllib.request
from io import StringIO
from backtesting import Backtest, Strategy
from typing import Optional

# Allowed forex symbols (no crypto)
SUPPORTED_SYMBOLS = {
    "AUDJPY",
    "AUDUSD",
    "EURCHF",
    "EURGBP",
    "EURJPY",
    "EURUSD",
    "GBPJPY",
    "GBPUSD",
    "USDCAD",
    "USDCHF",
    "USDJPY",
    "XAUUSD",
}


def _map_timeframe_suffix(tf: str) -> str:
    """Map common timeframe inputs to ejtraderLabs filename suffixes.

    Examples:
    - "1d" or "d1" -> "d1"
    - "1h" or "h1" -> "h1"
    - "4h" or "h4" -> "h4"
    - "15m" or "m15" -> "m15"
    - "30m" or "m30" -> "m30"
    - "5m" or "m5" -> "m5"
    - "1w" or "w1" -> "w1"
    """
    tf = tf.strip().lower()
    mapping = {
        "1d": "d1",
        "d1": "d1",
        "1h": "h1",
        "h1": "h1",
        "4h": "h4",
        "h4": "h4",
        "15m": "m15",
        "m15": "m15",
        "30m": "m30",
        "m30": "m30",
        "5m": "m5",
        "m5": "m5",
        "1w": "w1",
        "w1": "w1",
    }
    if tf in mapping:
        return mapping[tf]
    raise ValueError(
        "Unsupported timeframe. Use one of: 1d/d1, 1h/h1, 4h/h4, 15m/m15, 30m/m30, 5m/m5, 1w/w1"
    )


def fetch_data(symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Fetch historical data from ejtraderLabs historical data repository.
    
    Parameters:
    -----------
    symbol : str
        Forex trading symbol (supported: AUDJPY, AUDUSD, EURCHF, EURGBP,
        EURJPY, EURUSD, GBPJPY, GBPUSD, USDCAD, USDCHF, USDJPY, XAUUSD)
    timeframe : str
        Timeframe for the data (e.g., "1d", "4h", "1h").
        Internally mapped to ejtraderLabs suffix (e.g., "1d" -> "d1").
    
    Returns:
    --------
    pd.DataFrame : DataFrame with OHLCV data indexed by datetime
    """
    symbol_upper = symbol.strip().upper()
    if symbol_upper not in SUPPORTED_SYMBOLS:
        raise ValueError(
            f"Unsupported symbol '{symbol}'. Supported forex symbols: {sorted(SUPPORTED_SYMBOLS)}"
        )

    suffix = _map_timeframe_suffix(timeframe)
    url = "https://raw.githubusercontent.com/ejtraderLabs/historical-data/main/"
    url += symbol_upper + "/" + symbol_upper + suffix + ".csv"
    
    try:
        print(f"Fetching data from: {url}")
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read().decode('utf-8')
        
        df = pd.read_csv(StringIO(data))
        
        # Standardize column names to match backtesting.py requirements
        # backtesting.py expects: Open, High, Low, Close, Volume (capitalized)
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ['open']:
                column_mapping[col] = 'Open'
            elif col_lower in ['high']:
                column_mapping[col] = 'High'
            elif col_lower in ['low']:
                column_mapping[col] = 'Low'
            elif col_lower in ['close']:
                column_mapping[col] = 'Close'
            elif col_lower in ['volume', 'vol']:
                column_mapping[col] = 'Volume'
        
        if column_mapping:
            df = df.rename(columns=column_mapping)
        
        # Try to parse date column (common names: Date, date, timestamp, Timestamp)
        date_cols = ['Date', 'date', 'timestamp', 'Timestamp', 'time', 'Time', 'datetime', 'Datetime']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                df.set_index(col, inplace=True)
                break
        
        # If no date column found, use the first column as index
        if not isinstance(df.index, pd.DatetimeIndex):
            if len(df.columns) > 0:
                df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
                df.set_index(df.columns[0], inplace=True)
        
        # Ensure required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Add Volume column if missing
        if 'Volume' not in df.columns:
            df['Volume'] = 0
        
        # Quality fixes: sort, deduplicate, ensure dtypes
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        
        # Auto-normalize forex prices with robust scaling selector
        # Convert all price columns to numeric floats first
        price_cols = [c for c in ['Open', 'High', 'Low', 'Close'] if c in df.columns]
        if price_cols:
            for c in price_cols:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            
            median_close = float(df['Close'].median())
            
            # Determine target range based on symbol
            if symbol_upper.endswith('JPY'):
                target_min, target_max = 50, 300
            elif symbol_upper == 'XAUUSD':
                target_min, target_max = 800, 4000
            else:
                target_min, target_max = 0.5, 3.0
            
            target_mid = (target_min + target_max) / 2.0
            
            # Evaluate candidate divisors
            candidates = [1, 10, 100, 1000, 10000]
            best_divisor = 1
            best_score = float('inf')
            
            for divisor in candidates:
                scaled = median_close / divisor
                # Check if within target range
                if target_min <= scaled <= target_max:
                    # Distance to midpoint
                    score = abs(scaled - target_mid)
                else:
                    # Distance to nearest boundary
                    if scaled < target_min:
                        score = target_min - scaled
                    else:
                        score = scaled - target_max
                
                if score < best_score:
                    best_score = score
                    best_divisor = divisor
            
            # Apply scaling
            if best_divisor != 1:
                original_median = median_close
                for c in price_cols:
                    df[c] = df[c] / best_divisor
                scaled_median = df['Close'].median()
                
                # Sanity check: ensure OHLC integrity after scaling
                high_valid = (df['High'] >= df[['Open', 'Close']].max(axis=1)).sum()
                low_valid = (df['Low'] <= df[['Open', 'Close']].min(axis=1)).sum()
                total_rows = len(df)
                
                if high_valid / total_rows < 0.8 or low_valid / total_rows < 0.8:
                    # Scaling broke OHLC, revert to divisor=1
                    for c in price_cols:
                        df[c] = df[c] * best_divisor
                    print(f"Warning: Auto-scaling by /{best_divisor} produced invalid OHLC, keeping original prices")
                else:
                    print(f"Auto-scaled prices for {symbol_upper} by /{best_divisor} (median {original_median:.2f} -> {scaled_median:.4f})")

        print(f"Successfully loaded {len(df)} rows of data")
        return df
        
    except urllib.error.HTTPError as e:
        raise Exception(f"HTTP Error {e.code}: Failed to fetch data from {url}. "
                       f"Please verify the symbol and timeframe are correct.")
    except urllib.error.URLError as e:
        raise Exception(f"URL Error: Failed to fetch data from {url}. Reason: {e.reason}")
    except Exception as e:
        raise Exception(f"Failed to fetch data from {url}: {str(e)}")


def run_backtest(
    data: pd.DataFrame,
    strategy: type,
    cash: float = 100000.0,
    commission: float = 0.0002,
    strategy_params: Optional[dict] = None,
    spread_pips: Optional[float] = None,
    pip_size: Optional[float] = None,
    symbol: Optional[str] = None,
    **kwargs
) -> dict:
    """
    Run a backtest using the backtesting.py library.
    
    Parameters:
    -----------
    data : pd.DataFrame
        OHLCV data with datetime index
    strategy : Strategy class
        Strategy class (not instance) that inherits from backtesting.Strategy
    cash : float
        Initial cash amount (default: 100000)
    commission : float
        Commission rate per trade (default: 0.0002 = 0.02%).
        If spread_pips is provided and commission is not explicitly overridden,
        commission will be set to 0 (spread modeling replaces it).
    strategy_params : dict, optional
        Parameters to pass to strategy.run()
    spread_pips : float, optional
        Spread cost in pips (e.g., 1.5 for 1.5 pip spread).
        If provided, will inject 'spread_price' into strategy via strategy_params.
    pip_size : float, optional
        Size of one pip in price units. If not provided, auto-detected:
        - JPY pairs: 0.01
        - XAUUSD: 0.1
        - Others: 0.0001
    symbol : str, optional
        Trading symbol for pip_size auto-detection. Can be inferred from data.attrs.
    **kwargs : dict
        Additional arguments to pass to Backtest
    
    Returns:
    --------
    dict : Backtest statistics
    """
    params = strategy_params or {}
    
    # Handle spread modeling
    if spread_pips is not None:
        # Infer symbol if not provided
        if symbol is None:
            symbol = data.attrs.get('symbol', '')
        
        # Determine pip_size
        if pip_size is None:
            symbol_upper = symbol.upper() if symbol else ''
            if symbol_upper.endswith('JPY'):
                pip_size = 0.01
            elif symbol_upper == 'XAUUSD':
                pip_size = 0.1
            else:
                pip_size = 0.0001
        
        # Convert spread to price units
        spread_price = spread_pips * pip_size
        params['spread_price'] = spread_price
        
        # Set commission=0 if not explicitly overridden in kwargs
        if 'commission' not in kwargs:
            commission = 0.0
    
    bt = Backtest(data, strategy, cash=cash, commission=commission, **kwargs)
    stats = bt.run(**params)
    return stats


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Simple Backtesting Framework Example")
    print("=" * 60)
    
    # Define a simple moving average crossover strategy
    class SMAStrategy(Strategy):
        # Strategy parameters
        fast_period = 10
        slow_period = 30
        
        def init(self):
            # Pre-calculate indicators
            from backtesting.lib import crossover
            from backtesting.test import SMA
            
            self.sma_fast = self.I(SMA, self.data.Close, self.fast_period)
            self.sma_slow = self.I(SMA, self.data.Close, self.slow_period)
        
        def next(self):
            # Trading logic
            if self.sma_fast[-1] > self.sma_slow[-1]:
                if not self.position:
                    self.buy()
            elif self.sma_fast[-1] < self.sma_slow[-1]:
                if self.position:
                    self.position.close()
    
    # Try to fetch real data, fall back to sample data if unavailable
    try:
        print("\nAttempting to fetch real data from remote source...")
        data = fetch_data(symbol="GBPJPY", timeframe="1d")
    except Exception as e:
        print(f"\nCould not fetch real data: {e}")
        print("\nUsing sample data for demonstration...")
        
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
        np.random.seed(42)
        
        # Generate realistic price data with proper OHLC constraints
        price = 100
        opens, highs, lows, closes = [], [], [], []
        
        for _ in range(200):
            open_price = price
            close_price = price + np.random.randn() * 2
            
            # Ensure High is the maximum and Low is the minimum
            high_price = max(open_price, close_price) + abs(np.random.randn())
            low_price = min(open_price, close_price) - abs(np.random.randn())
            low_price = max(low_price, 1)  # Ensure Low is positive
            
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
            'Volume': np.random.randint(1000, 10000, 200)
        }, index=dates)
        
        print(f"Created sample data with {len(data)} rows")
    
    print("\n" + "=" * 60)
    print("Running Backtest...")
    print("=" * 60)
    
    # Run the backtest
    stats = run_backtest(
        data=data,
        strategy=SMAStrategy,
        cash=10000.0,
        commission=0.001
    )
    
    print("\nBacktest Results:")
    print("=" * 60)
    print(stats)
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
