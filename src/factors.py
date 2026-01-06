from __future__ import annotations

import pandas as pd


def compute_momentum(close: pd.Series, window: int) -> pd.Series:
    """
    Rolling momentum as percentage change over `window` bars.

    Parameters
    ----------
    close : pd.Series
        Close price series.
    window : int
        Lookback window length.

    Returns
    -------
    pd.Series
        Momentum series (Close / Close.shift(window) - 1) with the same index.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    close = pd.Series(close)
    return close / close.shift(window) - 1


def compute_carry_stub(df: pd.DataFrame, method: str = "rate_diff") -> pd.Series:
    """
    Placeholder for carry computation.

    This will require external inputs such as:
    - interest rate differentials per currency (for spot FX),
    - forward points or swap rates per pair,
    - contract-specific rollover data for commodities/indices.

    Parameters
    ----------
    df : pd.DataFrame
        Price DataFrame expected to include rate/forward data when implemented.
    method : str
        Carry methodology to apply, e.g., 'rate_diff' for cash rate spreads.

    Raises
    ------
    NotImplementedError
        Always, until carry inputs and implementation are added.
    """
    raise NotImplementedError(
        "Carry computation requires rate/forward inputs. Supply interest rate differentials, "
        "forward points, or swap data to implement this function."
    )


if __name__ == "__main__":
    # Lightweight self-check
    s = pd.Series([1, 2, 3, 5, 8], index=pd.date_range("2024-01-01", periods=5, freq="D"))
    print("Momentum sample:", compute_momentum(s, 2).dropna().iloc[-1])
    try:
        compute_carry_stub(pd.DataFrame({"Close": s}))
    except NotImplementedError as exc:
        print("Carry stub ok:", exc)
