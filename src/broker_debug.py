"""
Broker order lifecycle instrumentation for debugging order disappearance.

This module provides monkeypatching capabilities to add detailed logging
to the backtesting.py broker without modifying the installed library.

Key enhancements:
1. Wraps broker.orders list with TrackedOrderList to catch all removals
2. Patches _process_orders to log before/after state with reasons
3. Logs broker configuration at startup
4. Captures stack traces for silent clearing operations
"""

import functools
import traceback
from collections import deque
from typing import Any, Callable, List


class TrackedOrderList:
    """Wrapper around broker.orders list that logs all removals."""
    
    def __init__(self, original_list, broker, debug=True):
        self._list = original_list
        self._broker = broker
        self._debug = debug
    
    def _log_removal(self, reason: str, item=None):
        """Log an order removal with reason."""
        if not self._debug:
            return

        def _record_event(event: dict) -> None:
            try:
                events = getattr(self._broker, "_recent_order_removals", None)
                if not isinstance(events, deque):
                    events = deque(maxlen=500)
                events.append(event)
                setattr(self._broker, "_recent_order_removals", events)
            except Exception:
                pass
        
        stack = traceback.extract_stack(limit=8)
        caller = None
        for frame in reversed(stack[:-1]):
            # Find first frame outside this wrapper
            if 'TrackedOrderList' not in frame.filename and 'broker_debug.py' not in frame.filename:
                caller = f"{frame.filename.split('/')[-1]}:{frame.name}:{frame.lineno}"
                break
        
        ts = None
        try:
            data = getattr(self._broker, "_data", None)
            if data is not None:
                ts = data.index[-1] if hasattr(data, "index") else None
        except Exception:
            ts = None

        item_info = ""
        event = {
            "reason": reason,
            "order_id": id(item) if item else None,
            "caller": caller,
            "ts": ts,
        }
        if item and hasattr(item, '__dict__'):
            attrs = vars(item)
            side = "BUY" if attrs.get('_is_long', False) else "SELL"
            size = abs(attrs.get('_size', 0))
            item_info = f" side={side} size={size}"
            event.update({
                "side": side,
                "size": size,
                "stop": attrs.get('_stop') or attrs.get('stop'),
                "limit": attrs.get('_limit') or attrs.get('limit'),
            })

        _record_event(event)
        
        print(f"[BROKER ORDER] action=REMOVE reason={reason}{item_info} caller={caller}")
    
    def remove(self, item):
        """Remove item from list (called by broker)."""
        self._log_removal("EXPLICIT_REMOVE", item)
        self._list.remove(item)
    
    def clear(self):
        """Clear all items from list."""
        count = len(self._list)
        if count > 0:
            self._log_removal(f"CLEAR_ALL(count={count})")
        self._list.clear()
    
    def pop(self, index=-1):
        """Remove and return item at index."""
        item = self._list[index] if self._list else None
        self._log_removal("POP", item)
        return self._list.pop(index)
    
    def __len__(self):
        return len(self._list)
    
    def __iter__(self):
        return iter(self._list)
    
    def __getitem__(self, index):
        return self._list[index]
    
    def __bool__(self):
        return bool(self._list)
    
    def __repr__(self):
        return repr(self._list)
    
    def append(self, item):
        """Add item to list."""
        self._list.append(item)
    
    def extend(self, items):
        """Extend list with items."""
        self._list.extend(items)
    
    def insert(self, index, item):
        """Insert item at index."""
        self._list.insert(index, item)


def _patch_broker_orders_list(broker_instance, debug=True):
    """Replace broker.orders with TrackedOrderList wrapper."""
    if not debug or not hasattr(broker_instance, 'orders'):
        return
    
    if not isinstance(broker_instance.orders, TrackedOrderList):
        broker_instance.orders = TrackedOrderList(broker_instance.orders, broker_instance, debug)


def _log_broker_config(broker_instance):
    """Log broker configuration at startup."""
    attrs = {
        'cash': 'cash',
        'margin': '_margin',
        'commission': '_commission',
        'exclusive_orders': 'exclusive_orders',
        'trade_on_close': 'trade_on_close',
        'hedging': 'hedging',
    }
    
    config = {}
    for display_name, attr_name in attrs.items():
        if hasattr(broker_instance, attr_name):
            config[display_name] = getattr(broker_instance, attr_name)
    
    print(f"[BROKER CONFIG] {config}")


def install_broker_debug_hooks(broker_class: type, debug: bool = True) -> None:
    """
    Monkeypatch the broker class to add order lifecycle logging.
    
    This patches:
    1. _process_orders: logs before/after order counts with margin/equity state
    2. new_order: logs placement attempts
    3. orders list: wrapped to catch all removals
    4. _update_equity: logs margin/cash available changes
    
    Args:
        broker_class: The Broker class from backtesting._broker
        debug: Whether to enable debug logging (controlled by --wave5-debug)
    """
    if not debug:
        return  # Skip patching if debug is disabled
    
    # Save original methods
    original_process_orders = broker_class._process_orders
    original_update_equity = broker_class._update_equity if hasattr(broker_class, '_update_equity') else None
    
    @functools.wraps(original_process_orders)
    def _process_orders_with_logging(self):
        """Wrapped version of _process_orders that logs order lifecycle events."""
        
        # Wrap orders list if not already wrapped
        _patch_broker_orders_list(self, debug)
        
        # Log broker state at start of bar
        equity_val = self.equity if hasattr(self, 'equity') else 0
        margin_avail = self.margin_available if hasattr(self, 'margin_available') else 0
        ts = self._data.index[-1] if hasattr(self, '_data') and hasattr(self._data, 'index') else "?"
        print(f"[BROKER] ts={ts} equity={equity_val:.2f} margin_available={margin_avail:.2f}")
        
        # Log current orders before processing
        orders_before = len(self.orders) if hasattr(self, 'orders') else 0
        if orders_before > 0:
            print(f"[BROKER] orders_pending={orders_before}")
            for order in self.orders:
                side = "BUY" if order.is_long else "SELL"
                stop_val = getattr(order, '_stop', None) or getattr(order, 'stop', None)
                limit_val = getattr(order, '_limit', None) or getattr(order, 'limit', None)
                size = abs(getattr(order, '_size', getattr(order, 'size', 0)))
                price = getattr(order, 'created_price', None)
                print(f"[BROKER ORDER] action=PENDING side={side} size={size} stop={stop_val} limit={limit_val} created_at={price}")
        
        # Call original method
        try:
            result = original_process_orders(self)
        except Exception as e:
            print(f"[BROKER ERROR] _process_orders raised: {type(e).__name__}: {e}")
            raise
        
        # Log orders after processing (to detect cancellations)
        orders_after = len(self.orders) if hasattr(self, 'orders') else 0
        if orders_before > orders_after:
            print(f"[BROKER] orders_removed={orders_before - orders_after}")
        
        return result
    
    # Apply the patches
    broker_class._process_orders = _process_orders_with_logging
    
    # Patch __init__ to log config and wrap orders list
    original_init = broker_class.__init__
    
    @functools.wraps(original_init)
    def __init__with_logging(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _log_broker_config(self)
        _patch_broker_orders_list(self, debug)
    
    broker_class.__init__ = __init__with_logging
    
    print("[BROKER DEBUG] Installed order removal tracking and config logging")


def install_broker_order_hooks(broker_class: type, debug: bool = True) -> None:
    """
    Monkeypatch broker's new_order method to log order placement attempts.
    
    Args:
        broker_class: The Broker class from backtesting._broker
        debug: Whether to enable debug logging
    """
    if not debug:
        return
    
    original_new_order = broker_class.new_order
    
    @functools.wraps(original_new_order)
    def new_order_with_logging(self, *args, **kwargs):
        """Wrapped version of new_order that logs placement attempts."""
        side = "BUY" if (args and args[0] > 0) or kwargs.get('size', 0) > 0 else "SELL"
        size_arg = (args[0] if args else kwargs.get('size', 0))
        size = abs(size_arg) if size_arg else 0
        
        # Extract order parameters
        stop = kwargs.get('stop', None) if not args or len(args) < 2 else (args[1] if len(args) > 1 else None)
        limit = kwargs.get('limit', None) if not args or len(args) < 3 else (args[2] if len(args) > 2 else None)
        sl = kwargs.get('sl', None)
        tp = kwargs.get('tp', None)
        
        # Log placement attempt with broker state
        equity_val = self.equity if hasattr(self, 'equity') else 0
        margin_avail = self.margin_available if hasattr(self, 'margin_available') else 0
        ts = self._data.index[-1] if hasattr(self, '_data') and hasattr(self._data, 'index') else "?"
        
        print(f"[BROKER ORDER] action=NEW side={side} size={size} stop={stop} limit={limit} sl={sl} tp={tp}")
        print(f"[BROKER STATE] ts={ts} equity={equity_val:.2f} margin_available={margin_avail:.2f} for_order")
        
        try:
            order = original_new_order(self, *args, **kwargs)
            if order is None:
                print(f"[BROKER ORDER] action=REJECT reason=RETURNED_NONE side={side} size={size}")
            else:
                print(f"[BROKER ORDER] action=ACCEPTED side={side} size={size} order_id={id(order)}")
            return order
        except Exception as e:
            print(f"[BROKER ORDER] action=REJECT reason={type(e).__name__} msg={str(e)[:100]} side={side} size={size}")
            raise
    
    broker_class.new_order = new_order_with_logging
    print("[BROKER DEBUG] Installed new_order hooks")


def install_all_broker_hooks(debug: bool = True) -> None:
    """
    Install all broker debugging hooks.
    
    This function should be called before running a backtest when debug mode is enabled.
    
    Args:
        debug: Whether to enable debug logging (controlled by --wave5-debug)
    """
    if not debug:
        return
    
    try:
        # Import the broker class from backtesting.py
        from backtesting.backtesting import _Broker
        
        # Install hooks
        install_broker_order_hooks(_Broker, debug=debug)
        install_broker_debug_hooks(_Broker, debug=debug)
        
        print("[BROKER DEBUG] All hooks installed successfully")
        
    except ImportError as e:
        print(f"[BROKER DEBUG] Warning: Could not import Broker class: {e}")
    except Exception as e:
        print(f"[BROKER DEBUG] Warning: Failed to install hooks: {e}")
