import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run_case(margin: float) -> int:
    cmd = [
        sys.executable,
        "src/compare_strategies.py",
        "--mode",
        "wave5",
        "--asset",
        "XAUUSD",
        "--tf",
        "1h",
        "--spread",
        "30",
        "--wave5-size",
        "0.1",
        "--wave5-entry-mode",
        "break",
        "--wave5-trigger-lag",
        "24",
        "--wave5-zone-mode",
        "either",
        "--margin",
        str(margin),
    ]

    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    output = result.stdout + "\n" + result.stderr

    if result.returncode != 0:
        raise AssertionError(f"Backtest failed for margin={margin}: exit {result.returncode}\n{output}")

    match = re.search(r"# Trades\s*[:=]?\s*(\d+)", output)
    if not match:
        raise AssertionError(f"Could not parse # Trades for margin={margin}. Output:\n{output}")

    trades = int(match.group(1))
    if trades < 1:
        raise AssertionError(f"Expected at least 1 trade for margin={margin}, got {trades}\n{output}")

    return trades


def main() -> None:
    trades_one = _run_case(1.0)
    trades_levered = _run_case(0.02)
    print(
        f"Regression OK: margin=1.0 trades={trades_one}, margin=0.02 trades={trades_levered}"
    )


if __name__ == "__main__":
    main()
