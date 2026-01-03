"""Unit tests for Wave5 order disappearance classification."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from wave5_ao import Wave5AODivergenceStrategy  # noqa: E402


def _order_desc(
    order_id: int,
    *,
    side: str = "BUY",
    stop: float | None = None,
    limit: float | None = None,
    size: float = 1.0,
    parent_trade_id: int | None = None,
    is_contingent: bool = False,
) -> dict:
    return {
        "id": order_id,
        "side": side,
        "stop": stop,
        "limit": limit,
        "size": size,
        "parent_trade_id": parent_trade_id,
        "is_contingent": is_contingent,
    }


def _trade_desc(trade_id: int, *, size: float) -> dict:
    return {"id": trade_id, "size": size, "is_long": size > 0}


def _snapshot(
    orders: list[dict],
    *,
    trades: list[dict] | None = None,
    position: float = 0.0,
    removals: list[dict] | None = None,
) -> dict:
    trades = trades or []
    return {
        "orders": orders,
        "order_ids": {o["id"] for o in orders},
        "trades": trades,
        "trade_ids": {t["id"] for t in trades},
        "position_size": position,
        "recent_removals": removals or [],
    }


def test_filled_stop_entry_does_not_alert() -> None:
    """Missing parent order that became a trade should be classified as filled."""
    prev_orders = [_order_desc(1, side="SELL", stop=100.0, size=5.0)]
    prev_snapshot = _snapshot(prev_orders, position=0.0)

    # Trade opens; contingent SL/TP orders appear
    new_trade_id = 99
    curr_orders = [
        _order_desc(2, side="BUY", stop=110.0, size=5.0, parent_trade_id=new_trade_id, is_contingent=True),
        _order_desc(3, side="BUY", limit=90.0, size=5.0, parent_trade_id=new_trade_id, is_contingent=True),
    ]
    curr_snapshot = _snapshot(curr_orders, trades=[_trade_desc(new_trade_id, size=-5.0)], position=-5.0)

    result = Wave5AODivergenceStrategy._classify_missing_orders(
        missing_orders=prev_orders,
        prev_snapshot=prev_snapshot,
        curr_snapshot=curr_snapshot,
    )

    assert result["reason"] == "filled"
    assert result["alert"] is False


def test_explicit_cancel_is_not_alerted() -> None:
    """Order removed with an explicit reason should be treated as canceled."""
    prev_orders = [_order_desc(10, side="BUY", limit=50.5, size=3.0)]
    prev_snapshot = _snapshot(prev_orders, position=0.0)
    curr_snapshot = _snapshot([], removals=[{"order_id": 10, "reason": "EXPLICIT_REMOVE"}], position=0.0)

    result = Wave5AODivergenceStrategy._classify_missing_orders(
        missing_orders=prev_orders,
        prev_snapshot=prev_snapshot,
        curr_snapshot=curr_snapshot,
    )

    assert result["reason"] == "canceled"
    assert result["alert"] is False


def test_missing_without_reason_triggers_alert() -> None:
    """If no fill, no cancel reason, disappearance should raise an alert."""
    prev_orders = [_order_desc(20, side="BUY", stop=75.0, size=2.0)]
    prev_snapshot = _snapshot(prev_orders, position=0.0)
    curr_snapshot = _snapshot([], position=0.0)

    result = Wave5AODivergenceStrategy._classify_missing_orders(
        missing_orders=prev_orders,
        prev_snapshot=prev_snapshot,
        curr_snapshot=curr_snapshot,
    )

    assert result["reason"] == "unknown"
    assert result["alert"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
