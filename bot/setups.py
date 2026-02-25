"""Setup detectors for the 3 trading setups from the IKIGAI strategy.

Setup A: Zigzag + Breakout  — trend continuation after ABC correction
Setup B: Ending Diagonal    — trend reversal at the end of a wedge
Setup C: Triangle + Breakout — impulse after triangle breakout
"""

import logging
from typing import Optional

import config
from bot.models import (
    Signal, SetupType, Direction, Trend, WaveStructure, WaveType,
    SwingPoint, FibLevel,
)
from bot import fibonacci

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Setup A: Zigzag + Breakout
# ---------------------------------------------------------------------------

def check_zigzag_setup(
    zigzag: WaveStructure,
    trend: Trend,
    current_price: float,
    symbol: str,
    timeframe: str,
) -> Optional[Signal]:
    """Check if a zigzag correction creates a valid Setup A signal.

    Conditions:
    - Must be counter-trend (correction against the main trend)
    - Current price should be near the end of wave C
    - R:R must be >= MIN_RR_RATIO
    """
    if zigzag.wave_type != WaveType.ZIGZAG:
        return None

    wa, wb, wc = zigzag.waves[0], zigzag.waves[1], zigzag.waves[2]

    # Determine trade direction based on main trend
    if trend == Trend.UP:
        direction = Direction.LONG
        # Zigzag should be a downward correction
        if wa.direction == Direction.LONG:
            return None
    elif trend == Trend.DOWN:
        direction = Direction.SHORT
        if wa.direction == Direction.SHORT:
            return None
    else:
        return None  # no trades in sideways market

    # Wave C reference level (structural level for SL)
    wave_c_level = wc.end.price

    # Check if current price is near wave C end (within max deviation)
    if current_price > 0:
        dist = abs(current_price - wave_c_level) / wave_c_level
        if dist > config.ENTRY_PRICE_MAX_DEVIATION:
            return None

    # Entry: use current market price (bot trades at market, not at historical wave level)
    entry_price = current_price if current_price > 0 else wave_c_level

    # Stop loss: based on wave structure (invalidation of the pattern)
    min_stop_pct = config.MIN_STOP_DISTANCE_PCT

    if direction == Direction.LONG:
        # SL below the lowest point of wave C
        stop_loss = min(wc.end.price, wc.start.price) * 0.995
        # Enforce minimum stop distance from actual entry
        if entry_price > 0 and (entry_price - stop_loss) / entry_price < min_stop_pct:
            stop_loss = entry_price * (1 - min_stop_pct)
    else:
        # SL above the highest point of wave C
        stop_loss = max(wc.end.price, wc.start.price) * 1.005
        # Enforce minimum stop distance from actual entry
        if entry_price > 0 and (stop_loss - entry_price) / entry_price < min_stop_pct:
            stop_loss = entry_price * (1 + min_stop_pct)

    # Target: trend-based extension from the wave before zigzag
    targets = fibonacci.trend_based_extension(
        point_a=SwingPoint(
            index=0, price=wa.start.price - wa.length * (1 if direction == Direction.LONG else -1),
            timestamp=0, is_high=not wa.start.is_high,
        ),
        point_b=wa.start,
        point_c=wc.end,
        ratios=[1.618, 2.618, 3.618],
    )

    # Calculate R:R
    risk = abs(entry_price - stop_loss)
    if risk == 0 or not targets:
        return None
    reward = abs(targets[0].price - entry_price)
    rr = reward / risk

    if rr < config.MIN_RR_RATIO:
        return None

    factors = [
        f"Trend: {trend.value}",
        f"Zigzag ABC correction (confidence: {zigzag.confidence:.0%})",
        f"Wave C level: {wave_c_level:.4f}, entry at market: {entry_price:.4f}",
        f"R:R = {rr:.1f}:1",
    ]

    # Check if price is at Fibonacci level
    fib_levels = fibonacci.retracement_levels(
        SwingPoint(index=0, price=wa.start.price - wa.length, timestamp=0, is_high=False),
        wa.start,
    )
    fib_match = fibonacci.is_near_fib_level(current_price, fib_levels)
    if fib_match:
        factors.append(f"Price at Fibonacci {fib_match.label}")

    return Signal(
        symbol=symbol,
        timeframe=timeframe,
        setup_type=SetupType.ZIGZAG_BREAKOUT,
        direction=direction,
        entry_price=entry_price,
        stop_loss=stop_loss,
        targets=targets,
        confidence=zigzag.confidence,
        factors=factors,
    )


# ---------------------------------------------------------------------------
# Setup B: Ending Diagonal (Wedge)
# ---------------------------------------------------------------------------

def check_diagonal_setup(
    diagonal: WaveStructure,
    current_price: float,
    symbol: str,
    timeframe: str,
) -> Optional[Signal]:
    """Check if an ending diagonal creates a valid Setup B signal.

    This is a reversal setup — trade against the diagonal's direction.
    """
    if diagonal.wave_type != WaveType.DIAGONAL:
        return None

    w1, w2, w3, w4, w5 = diagonal.waves

    # Direction: OPPOSITE to the diagonal (it's ending, so we trade the reversal)
    if w1.direction == Direction.LONG:
        direction = Direction.SHORT  # ascending diagonal → short reversal
    else:
        direction = Direction.LONG  # descending diagonal → long reversal

    # Expected completion zone of wave 5
    w3_length = w3.length
    min_stop_pct = config.MIN_STOP_DISTANCE_PCT

    if direction == Direction.SHORT:
        # Ascending diagonal: wave 5 end expected at w4.end + 0.618 * w3
        expected_w5_end = w4.end.price + w3_length * 0.618
        stop_loss = w4.end.price + w3_length * 1.05  # beyond wave 3 projection
    else:
        expected_w5_end = w4.end.price - w3_length * 0.618
        stop_loss = w4.end.price - w3_length * 1.05

    # Check if current price is near expected wave 5 end
    if current_price > 0:
        dist = abs(current_price - expected_w5_end) / expected_w5_end
        if dist > config.ENTRY_PRICE_MAX_DEVIATION:
            return None

    # Use current market price as entry
    entry_price = current_price if current_price > 0 else expected_w5_end

    # Enforce minimum stop distance from actual entry
    if direction == Direction.SHORT:
        if entry_price > 0 and (stop_loss - entry_price) / entry_price < min_stop_pct:
            stop_loss = entry_price * (1 + min_stop_pct)
    else:
        if entry_price > 0 and (entry_price - stop_loss) / entry_price < min_stop_pct:
            stop_loss = entry_price * (1 - min_stop_pct)

    # Target: base of the diagonal (start of wave 1) — gives 4-5:1 R:R
    target_price = w1.start.price
    risk = abs(entry_price - stop_loss)
    if risk == 0:
        return None
    reward = abs(target_price - entry_price)
    rr = reward / risk

    if rr < config.MIN_RR_RATIO:
        return None

    targets = [
        FibLevel(ratio=0.618, price=(entry_price + target_price) / 2 + (target_price - entry_price) * 0.118,
                 label="Target 0.618"),
        FibLevel(ratio=1.0, price=target_price, label="Full target (base)"),
    ]

    factors = [
        f"Ending diagonal (wedge) detected (confidence: {diagonal.confidence:.0%})",
        f"Wave 5 completing near {entry_price:.2f}",
        f"Reversal target: {target_price:.2f} (base of diagonal)",
        f"R:R = {rr:.1f}:1",
        f"W1>{w1.length:.2f} > W3>{w3.length:.2f} > W5>{w5.length:.2f} (valid)",
    ]

    return Signal(
        symbol=symbol,
        timeframe=timeframe,
        setup_type=SetupType.ENDING_DIAGONAL,
        direction=direction,
        entry_price=entry_price,
        stop_loss=stop_loss,
        targets=targets,
        confidence=diagonal.confidence,
        factors=factors,
    )


# ---------------------------------------------------------------------------
# Setup C: Triangle + Breakout
# ---------------------------------------------------------------------------

def check_triangle_setup(
    triangle: WaveStructure,
    trend: Trend,
    current_price: float,
    symbol: str,
    timeframe: str,
) -> Optional[Signal]:
    """Check if a triangle creates a valid Setup C signal.

    Triangle is the penultimate wave — after it, one more impulse follows.
    Trade in the direction of the trend (breakout).
    """
    if triangle.wave_type != WaveType.TRIANGLE:
        return None

    wa, wb, wc, wd, we = triangle.waves

    # Direction based on trend (triangle = continuation pattern in symmetric case)
    min_stop_pct = config.MIN_STOP_DISTANCE_PCT

    if trend == Trend.UP:
        direction = Direction.LONG
        # Breakout confirmation: price above wave B high
        breakout_level = max(wb.start.price, wb.end.price)
        if current_price > 0 and current_price < breakout_level:
            return None  # not yet broken out
        # Use current price as entry (already past breakout)
        entry_price = current_price if current_price > 0 else breakout_level
        stop_loss = min(we.start.price, we.end.price) * 0.995
        # Enforce minimum stop distance
        if entry_price > 0 and (entry_price - stop_loss) / entry_price < min_stop_pct:
            stop_loss = entry_price * (1 - min_stop_pct)
    elif trend == Trend.DOWN:
        direction = Direction.SHORT
        breakout_level = min(wb.start.price, wb.end.price)
        if current_price > 0 and current_price > breakout_level:
            return None
        entry_price = current_price if current_price > 0 else breakout_level
        stop_loss = max(we.start.price, we.end.price) * 1.005
        # Enforce minimum stop distance
        if entry_price > 0 and (stop_loss - entry_price) / entry_price < min_stop_pct:
            stop_loss = entry_price * (1 + min_stop_pct)
    else:
        return None

    # Potential = distance of wave A, projected from breakout point (structural target)
    wave_a_length = wa.length

    if direction == Direction.LONG:
        target_price = breakout_level + wave_a_length
    else:
        target_price = breakout_level - wave_a_length

    risk = abs(entry_price - stop_loss)
    if risk == 0:
        return None
    reward = abs(target_price - entry_price)
    rr = reward / risk

    if rr < config.MIN_RR_RATIO:
        return None

    # Also calculate trend-based Fibonacci target
    targets = [
        FibLevel(ratio=1.0, price=target_price, label="Triangle potential (normal)"),
    ]

    # Log-scale potential (more optimistic)
    if entry_price > 0 and wa.start.price > 0:
        log_ratio = wave_a_length / wa.start.price
        log_target = entry_price * (1 + log_ratio) if direction == Direction.LONG \
            else entry_price * (1 - log_ratio)
        targets.append(
            FibLevel(ratio=1.618, price=log_target, label="Triangle potential (log)")
        )

    factors = [
        f"Trend: {trend.value}",
        f"Triangle ABCDE completed (confidence: {triangle.confidence:.0%})",
        f"Breakout level: {breakout_level:.2f}",
        f"Target (wave A potential): {target_price:.2f}",
        f"R:R = {rr:.1f}:1",
        "Triangle = penultimate wave → expect final impulse",
    ]

    return Signal(
        symbol=symbol,
        timeframe=timeframe,
        setup_type=SetupType.TRIANGLE_BREAKOUT,
        direction=direction,
        entry_price=entry_price,
        stop_loss=stop_loss,
        targets=targets,
        confidence=triangle.confidence,
        factors=factors,
    )
