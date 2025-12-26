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
    cash: float = 10000.0,
    commission: float = 0.001,
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
        Initial cash amount (default: 10000)
    commission : float
        Commission rate per trade (default: 0.001 = 0.1%)
    **kwargs : dict
        Additional arguments to pass to Backtest
    
    Returns:
    --------
    dict : Backtest statistics
    """
    bt = Backtest(data, strategy, cash=cash, commission=commission, **kwargs)
    stats = bt.run()
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
