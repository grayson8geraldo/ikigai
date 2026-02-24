"""Fibonacci retracement and extension calculations."""

from bot.models import FibLevel, SwingPoint
import config


def retracement_levels(
    swing_start: SwingPoint,
    swing_end: SwingPoint,
    ratios: list[float] = None,
) -> list[FibLevel]:
    """Calculate Fibonacci retracement levels between two swing points.

    For an upswing (start < end), retracement levels are below the end.
    For a downswing (start > end), retracement levels are above the end.
    """
    ratios = ratios or config.FIB_RETRACEMENT
    diff = swing_end.price - swing_start.price
    levels: list[FibLevel] = []

    for r in ratios:
        price = swing_end.price - diff * r
        levels.append(FibLevel(ratio=r, price=price, label=f"Fib {r:.3f}"))

    return levels


def extension_levels(
    swing_start: SwingPoint,
    swing_end: SwingPoint,
    ratios: list[float] = None,
) -> list[FibLevel]:
    """Calculate Fibonacci extension levels beyond the swing.

    Projects targets beyond swing_end in the direction of the move.
    """
    ratios = ratios or config.FIB_EXTENSION
    diff = swing_end.price - swing_start.price
    levels: list[FibLevel] = []

    for r in ratios:
        price = swing_start.price + diff * r
        levels.append(FibLevel(ratio=r, price=price, label=f"Ext {r:.3f}"))

    return levels


def trend_based_extension(
    point_a: SwingPoint,
    point_b: SwingPoint,
    point_c: SwingPoint,
    ratios: list[float] = None,
) -> list[FibLevel]:
    """Trend-based Fibonacci extension (3-point).

    Measures the move from A to B, then projects from C.
    Used for wave targets: e.g., wave 3 target from wave 1 length projected from wave 2 end.
    """
    ratios = ratios or config.FIB_EXTENSION
    move = point_b.price - point_a.price
    levels: list[FibLevel] = []

    for r in ratios:
        price = point_c.price + move * r
        levels.append(
            FibLevel(ratio=r, price=price, label=f"TrendExt {r:.3f}")
        )

    return levels


def wave_ratio(wave_a_length: float, wave_b_length: float) -> float:
    """Calculate the ratio between two wave lengths."""
    if wave_a_length == 0:
        return 0.0
    return wave_b_length / wave_a_length


def is_near_fib_level(
    price: float, levels: list[FibLevel], tolerance_pct: float = 0.015
) -> FibLevel | None:
    """Check if price is near any Fibonacci level (within tolerance).

    Returns the matching FibLevel or None.
    """
    for level in levels:
        if level.price == 0:
            continue
        diff_pct = abs(price - level.price) / level.price
        if diff_pct <= tolerance_pct:
            return level
    return None
