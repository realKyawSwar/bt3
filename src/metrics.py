from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def sharpe_ratio(returns, periods_per_year: int = 12) -> float:
    r = pd.Series(returns).dropna()
    if r.empty:
        return float("nan")
    std = r.std(ddof=0)
    if std == 0 or not np.isfinite(std):
        return float("nan")
    return float(r.mean() / std * np.sqrt(periods_per_year))


def max_drawdown(equity_curve) -> float:
    eq = pd.Series(equity_curve).dropna()
    if eq.empty:
        return float("nan")
    running_max = eq.cummax()
    dd = (eq - running_max) / running_max
    return float(dd.min())


def turnover(weights: Optional[pd.DataFrame]) -> float:
    if weights is None:
        return float("nan")
    df = pd.DataFrame(weights).sort_index()
    if len(df) < 2:
        return 0.0
    changes = df.diff().iloc[1:]
    per_period = changes.abs().sum(axis=1) / 2.0
    return float(per_period.mean())


def exposure_summary(weights: Optional[pd.DataFrame]) -> dict:
    if weights is None or weights.empty:
        return {"gross_leverage": float("nan"), "net_exposure": float("nan")}
    abs_sum = weights.abs().sum(axis=1)
    net_sum = weights.sum(axis=1)
    return {
        "gross_leverage": float(abs_sum.mean()),
        "net_exposure": float(net_sum.mean()),
    }


def compute_metrics(returns: pd.Series, weights: Optional[pd.DataFrame], periods_per_year: int = 12) -> dict:
    ret = pd.Series(returns).dropna()
    equity = (1.0 + ret).cumprod()
    exposure = exposure_summary(weights)
    result = {
        "sharpe": sharpe_ratio(ret, periods_per_year=periods_per_year),
        "turnover": turnover(weights),
        "max_drawdown": max_drawdown(equity),
        "total_return": float(equity.iloc[-1] - 1.0) if not equity.empty else float("nan"),
        "vol": float(ret.std(ddof=0) * np.sqrt(periods_per_year)) if not ret.empty else float("nan"),
        "periods": int(len(ret)),
    }
    result.update(exposure)
    return result
