# reporting.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
import pandas as pd


def export_trades_csv(stats, out_path: Union[str, Path] = "trades.csv") -> Path:
    """
    Export backtesting.py trades table to CSV.
    stats is the result from bt.run() (pandas Series-like).
    """
    out_path = Path(out_path)
    trades = stats.get("_trades", None)
    if trades is None:
        raise ValueError("stats['_trades'] not found. Ensure you are using backtesting.py Stats output.")
    if not isinstance(trades, pd.DataFrame):
        # Some versions may return a different structure; try coercion
        trades = pd.DataFrame(trades)

    trades.to_csv(out_path, index=False)
    return out_path


def export_equity_curve_csv(stats, out_path: Union[str, Path] = "equity_curve.csv") -> Path:
    """
    Export equity curve to CSV.
    backtesting.py Stats usually contains '_equity_curve' DataFrame.
    """
    out_path = Path(out_path)
    eq = stats.get("_equity_curve", None)
    if eq is None:
        raise ValueError("stats['_equity_curve'] not found.")
    if not isinstance(eq, pd.DataFrame):
        eq = pd.DataFrame(eq)

    # Common columns include 'Equity' and 'DrawdownPct' depending on version
    eq.to_csv(out_path, index=True)
    return out_path


def plot_equity_curve(
    stats,
    title: str = "Equity Curve",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> None:
    """
    Plot equity curve from stats['_equity_curve'].
    """
    import matplotlib.pyplot as plt
    eq = stats.get("_equity_curve", None)
    if eq is None:
        raise ValueError("stats['_equity_curve'] not found.")
    if not isinstance(eq, pd.DataFrame):
        eq = pd.DataFrame(eq)

    # Try common column names
    col = None
    for candidate in ("Equity", "equity", "Value", "value"):
        if candidate in eq.columns:
            col = candidate
            break
    if col is None:
        # fallback: first numeric column
        num_cols = eq.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            raise ValueError("No numeric equity column found in stats['_equity_curve'].")
        col = num_cols[0]

    plt.figure()
    eq[col].plot()
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Equity")

    if save_path:
        save_path = Path(save_path)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
