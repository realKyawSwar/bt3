from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from fx_momentum_12m import map_currency_weights_to_pairs  # noqa: E402
from metrics import turnover  # noqa: E402


def test_map_currency_weights_direction() -> None:
    idx = pd.to_datetime(["2020-01-31"])
    weights = pd.DataFrame([{"JPY": 1.0, "CAD": -0.5}], index=idx)
    pairs = ["USDJPY", "USDCAD"]

    pair_weights = map_currency_weights_to_pairs(weights, pairs)

    assert pair_weights.loc[idx[0], "USDJPY"] == -1.0  # long JPY -> short USDJPY
    assert pair_weights.loc[idx[0], "USDCAD"] == 0.5  # short CAD -> long USD/CAD


def test_turnover_formula() -> None:
    idx = pd.to_datetime(["2020-01-31", "2020-02-29"])
    weights = pd.DataFrame(
        [[0.5, -0.5], [-0.5, 0.5]],
        index=idx,
        columns=["EURUSD", "USDJPY"],
    )

    t = turnover(weights)

    # Change per leg: 1.0 each -> sum=2.0, turnover = 2/2 = 1.0
    assert t == 1.0
