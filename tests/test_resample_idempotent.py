from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from bt3 import fetch_data  # noqa: E402
from compare_strategies import _resample_ohlcv  # noqa: E402


def test_resample_idempotent_fetch_data() -> None:
    try:
        df = fetch_data("GBPJPY", "1h")
    except Exception as exc:
        pytest.skip(f"fetch_data unavailable: {exc}")

    if df.empty:
        pytest.skip("fetch_data returned empty dataframe.")

    resampled = _resample_ohlcv(df, "1h")

    assert len(resampled) == len(df)
    assert resampled.index.equals(df.index)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        assert np.allclose(
            resampled[col].to_numpy(),
            df[col].to_numpy(),
            equal_nan=True,
        )
