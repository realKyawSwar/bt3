from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from custom_alligator import run_comparison


def _make_sample_data(rows: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range(start="2020-01-01", periods=rows, freq="h")

    price = 100 + rng.standard_normal(rows).cumsum()
    open_ = price + rng.standard_normal(rows) * 0.1
    close = price + rng.standard_normal(rows) * 0.1
    high = np.maximum(open_, close) + np.abs(rng.standard_normal(rows) * 0.2)
    low = np.minimum(open_, close) - np.abs(rng.standard_normal(rows) * 0.2)

    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": rng.integers(100, 1000, size=rows),
        },
        index=dates,
    )


def main() -> None:
    data = _make_sample_data()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_comparison(
            data,
            cash=10000,
            commission=0.0,
            spread_pips=None,
            outdir=Path(tmpdir),
            export=True,
            print_table=False,
        )

    required_keys = ["Return [%]", "# Trades", "Win Rate [%]"]
    for key in required_keys:
        if key not in result["strict"] or key not in result["classic"]:
            raise AssertionError(f"Missing required stat '{key}' in output")

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
