"""Elliott Wave pattern detection.

Detects swing points on the chart and labels them as potential
impulse (1-2-3-4-5), zigzag (A-B-C), triangle (A-B-C-D-E),
and diagonal structures.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional

import config
from bot.models import (
    SwingPoint, Wave, WaveStructure, WaveType, Trend, Direction, WaveContext,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Swing detection
# ---------------------------------------------------------------------------

def find_swings(
    df: pd.DataFrame,
    lookback: int = None,
    min_pct: float = None,
) -> list[SwingPoint]:
    """Find swing highs and lows in OHLCV data.

    A swing high is a bar whose high is the highest within `lookback` bars
    on each side.  A swing low is the mirror image.
    """
    lookback = lookback or config.SWING_LOOKBACK
    min_pct = min_pct or config.MIN_SWING_PCT

    highs = df["high"].values
    lows = df["low"].values
    timestamps = df["timestamp"].values

    swings: list[SwingPoint] = []

    for i in range(lookback, len(df) - lookback):
        window_high = highs[i - lookback : i + lookback + 1]
        window_low = lows[i - lookback : i + lookback + 1]

        is_high = highs[i] == window_high.max()
        is_low = lows[i] == window_low.min()

        if is_high:
            swings.append(
                SwingPoint(
                    index=i,
                    price=float(highs[i]),
                    timestamp=float(pd.Timestamp(timestamps[i]).timestamp()),
                    is_high=True,
                )
            )
        if is_low:
            swings.append(
                SwingPoint(
                    index=i,
                    price=float(lows[i]),
                    timestamp=float(pd.Timestamp(timestamps[i]).timestamp()),
                    is_high=False,
                )
            )

    # Sort by index
    swings.sort(key=lambda s: s.index)

    # Remove consecutive same-type swings — keep the most extreme
    filtered: list[SwingPoint] = []
    for s in swings:
        if not filtered:
            filtered.append(s)
            continue
        prev = filtered[-1]
        if prev.is_high == s.is_high:
            # Same type — keep the more extreme one
            if s.is_high:
                if s.price >= prev.price:
                    filtered[-1] = s
            else:
                if s.price <= prev.price:
                    filtered[-1] = s
        else:
            filtered.append(s)

    # Filter by minimum swing size
    result: list[SwingPoint] = []
    for i, s in enumerate(filtered):
        if i == 0:
            result.append(s)
            continue
        prev = result[-1]
        pct = abs(s.price - prev.price) / prev.price
        if pct >= min_pct:
            result.append(s)

    return result


def swings_to_waves(swings: list[SwingPoint]) -> list[Wave]:
    """Convert consecutive swing points into Wave segments."""
    waves: list[Wave] = []
    for i in range(len(swings) - 1):
        waves.append(Wave(start=swings[i], end=swings[i + 1]))
    return waves


# ---------------------------------------------------------------------------
# Trend detection (EMA stack + ADX + structure)
# ---------------------------------------------------------------------------

def _ema(values: np.ndarray, period: int) -> np.ndarray:
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False).mean().values


def _adx(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate the latest ADX value (Average Directional Index).

    ADX measures trend *strength* regardless of direction:
      > 25  = trending
      < 20  = ranging / sideways
    """
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values

    if len(df) < period * 2:
        return 0.0

    plus_dm = np.zeros(len(df))
    minus_dm = np.zeros(len(df))
    tr = np.zeros(len(df))

    for i in range(1, len(df)):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]

        plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0

        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    # Smoothed averages (Wilder's method)
    atr = pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean().values
    smooth_plus = pd.Series(plus_dm).ewm(alpha=1 / period, adjust=False).mean().values
    smooth_minus = pd.Series(minus_dm).ewm(alpha=1 / period, adjust=False).mean().values

    # Directional indicators
    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100 * smooth_plus / atr
        minus_di = 100 * smooth_minus / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)

    dx = np.nan_to_num(dx, nan=0.0)
    adx_values = pd.Series(dx).ewm(alpha=1 / period, adjust=False).mean().values

    return float(adx_values[-1])


def _detect_structure(swings: list, min_points: int = 4) -> str:
    """Detect higher-highs/higher-lows or lower-highs/lower-lows structure.

    Returns: 'hh_hl' (uptrend), 'lh_ll' (downtrend), or 'mixed'.
    """
    if len(swings) < min_points:
        return "mixed"

    recent = swings[-min_points:]
    highs = [s for s in recent if s.is_high]
    lows = [s for s in recent if not s.is_high]

    hh = all(highs[i].price > highs[i - 1].price for i in range(1, len(highs))) if len(highs) >= 2 else False
    hl = all(lows[i].price > lows[i - 1].price for i in range(1, len(lows))) if len(lows) >= 2 else False
    lh = all(highs[i].price < highs[i - 1].price for i in range(1, len(highs))) if len(highs) >= 2 else False
    ll = all(lows[i].price < lows[i - 1].price for i in range(1, len(lows))) if len(lows) >= 2 else False

    if hh and hl:
        return "hh_hl"
    if lh and ll:
        return "lh_ll"
    return "mixed"


def detect_trend(df: pd.DataFrame, swings: list = None) -> Trend:
    """Multi-factor trend detection.

    Combines three independent signals:
      1. EMA stack (20 > 50 > 200 = bullish, inverse = bearish)
      2. ADX strength (> 25 = trending, otherwise ignore)
      3. Market structure (HH/HL vs LH/LL from swing points)

    Each factor votes UP / DOWN / SIDEWAYS. Majority wins.
    """
    if len(df) < 50:
        return Trend.SIDEWAYS

    closes = df["close"].values
    votes: list[Trend] = []

    # --- Factor 1: EMA alignment ---
    ema20 = _ema(closes, 20)
    ema50 = _ema(closes, 50)

    if len(df) >= 200:
        ema200 = _ema(closes, 200)
        last_20, last_50, last_200 = ema20[-1], ema50[-1], ema200[-1]

        if last_20 > last_50 > last_200:
            votes.append(Trend.UP)
        elif last_20 < last_50 < last_200:
            votes.append(Trend.DOWN)
        else:
            # Partial alignment
            if last_20 > last_50:
                votes.append(Trend.UP)
            elif last_20 < last_50:
                votes.append(Trend.DOWN)
            else:
                votes.append(Trend.SIDEWAYS)
    else:
        last_20, last_50 = ema20[-1], ema50[-1]
        if last_20 > last_50:
            votes.append(Trend.UP)
        elif last_20 < last_50:
            votes.append(Trend.DOWN)
        else:
            votes.append(Trend.SIDEWAYS)

    # --- Factor 2: ADX + directional bias ---
    adx_val = _adx(df)
    if adx_val >= 25:
        # ADX is strong — use EMA slope to determine direction
        slope = (ema20[-1] - ema20[-5]) / ema20[-5] if len(ema20) >= 5 and ema20[-5] != 0 else 0
        if slope > 0.005:
            votes.append(Trend.UP)
        elif slope < -0.005:
            votes.append(Trend.DOWN)
        else:
            votes.append(Trend.SIDEWAYS)
    else:
        votes.append(Trend.SIDEWAYS)  # weak trend

    # --- Factor 3: Market structure (HH/HL vs LH/LL) ---
    if swings and len(swings) >= 4:
        structure = _detect_structure(swings)
    else:
        # Quick swing detection for trend purpose
        quick_swings = find_swings(df, lookback=3, min_pct=0.01)
        structure = _detect_structure(quick_swings)

    if structure == "hh_hl":
        votes.append(Trend.UP)
    elif structure == "lh_ll":
        votes.append(Trend.DOWN)
    else:
        votes.append(Trend.SIDEWAYS)

    # --- Majority vote ---
    up_count = votes.count(Trend.UP)
    down_count = votes.count(Trend.DOWN)

    if up_count >= 2:
        return Trend.UP
    if down_count >= 2:
        return Trend.DOWN
    return Trend.SIDEWAYS


# ---------------------------------------------------------------------------
# Impulse (5-wave) detection
# ---------------------------------------------------------------------------

def _validate_impulse_up(waves: list[Wave]) -> tuple[bool, float]:
    """Validate a 5-wave upward impulse structure. Returns (valid, confidence)."""
    if len(waves) < 5:
        return False, 0.0

    w1, w2, w3, w4, w5 = waves[0], waves[1], waves[2], waves[3], waves[4]
    confidence = 0.0

    # Rule 1: wave 2 never retraces beyond start of wave 1
    if w2.end.price <= w1.start.price:
        return False, 0.0
    confidence += 0.2

    # Rule 2: wave 3 is never the shortest
    len1, len3, len5 = w1.length, w3.length, w5.length
    if len3 <= len1 and len3 <= len5:
        return False, 0.0
    confidence += 0.2

    # Rule 3: wave 3 goes beyond wave 1 end
    if w3.end.price <= w1.end.price:
        return False, 0.0
    confidence += 0.2

    # Rule 4: wave 4 never enters wave 1 territory
    if w4.end.price <= w1.end.price:
        return False, 0.0
    confidence += 0.2

    # Norm: wave 2 and 4 alternate (one deep, one shallow)
    retr2 = w2.length / w1.length if w1.length > 0 else 0
    retr4 = w4.length / w3.length if w3.length > 0 else 0
    if abs(retr2 - retr4) > 0.1:
        confidence += 0.1

    # Norm: wave 3 often ~1.618 of wave 1
    if w1.length > 0:
        ratio = w3.length / w1.length
        if 1.4 < ratio < 2.0:
            confidence += 0.1

    return True, min(confidence, 1.0)


def _validate_impulse_down(waves: list[Wave]) -> tuple[bool, float]:
    """Validate a 5-wave downward impulse structure."""
    if len(waves) < 5:
        return False, 0.0

    w1, w2, w3, w4, w5 = waves[0], waves[1], waves[2], waves[3], waves[4]
    confidence = 0.0

    # Rule 1: wave 2 never retraces beyond start of wave 1
    if w2.end.price >= w1.start.price:
        return False, 0.0
    confidence += 0.2

    # Rule 2: wave 3 is never the shortest
    len1, len3, len5 = w1.length, w3.length, w5.length
    if len3 <= len1 and len3 <= len5:
        return False, 0.0
    confidence += 0.2

    # Rule 3: wave 3 goes beyond wave 1 end
    if w3.end.price >= w1.end.price:
        return False, 0.0
    confidence += 0.2

    # Rule 4: wave 4 never enters wave 1 territory
    if w4.end.price >= w1.end.price:
        return False, 0.0
    confidence += 0.2

    # Norms (same as up)
    retr2 = w2.length / w1.length if w1.length > 0 else 0
    retr4 = w4.length / w3.length if w3.length > 0 else 0
    if abs(retr2 - retr4) > 0.1:
        confidence += 0.1

    if w1.length > 0:
        ratio = w3.length / w1.length
        if 1.4 < ratio < 2.0:
            confidence += 0.1

    return True, min(confidence, 1.0)


def find_impulses(swings: list[SwingPoint]) -> list[WaveStructure]:
    """Find all valid 5-wave impulse structures in swing data."""
    structures: list[WaveStructure] = []
    if len(swings) < 6:
        return structures

    for i in range(len(swings) - 5):
        seg = swings[i : i + 6]
        waves = swings_to_waves(seg)

        # Try upward impulse (starts with low)
        if not seg[0].is_high:
            valid, conf = _validate_impulse_up(waves)
            if valid:
                for j, label in enumerate(["1", "2", "3", "4", "5"]):
                    waves[j].label = label
                structures.append(
                    WaveStructure(WaveType.IMPULSE, waves, confidence=conf)
                )

        # Try downward impulse (starts with high)
        if seg[0].is_high:
            valid, conf = _validate_impulse_down(waves)
            if valid:
                for j, label in enumerate(["1", "2", "3", "4", "5"]):
                    waves[j].label = label
                structures.append(
                    WaveStructure(WaveType.IMPULSE, waves, confidence=conf)
                )

    return structures


# ---------------------------------------------------------------------------
# Zigzag (A-B-C) detection
# ---------------------------------------------------------------------------

def find_zigzags(swings: list[SwingPoint]) -> list[WaveStructure]:
    """Find all valid A-B-C zigzag corrections."""
    structures: list[WaveStructure] = []
    if len(swings) < 4:
        return structures

    for i in range(len(swings) - 3):
        seg = swings[i : i + 4]
        waves = swings_to_waves(seg)
        wa, wb, wc = waves[0], waves[1], waves[2]

        # Downward zigzag (bearish correction in uptrend)
        if seg[0].is_high:
            # B does not retrace beyond start of A
            if wb.end.price >= wa.start.price:
                continue
            # C goes beyond end of A
            if wc.end.price >= wa.end.price:
                continue

            confidence = 0.5
            # Check Fibonacci ratios
            if wa.length > 0:
                b_ratio = wb.length / wa.length
                if 0.3 < b_ratio < 0.8:
                    confidence += 0.2
                c_ratio = wc.length / wa.length
                if 0.9 < c_ratio < 2.0:
                    confidence += 0.2

            for j, label in enumerate(["A", "B", "C"]):
                waves[j].label = label
            structures.append(
                WaveStructure(WaveType.ZIGZAG, waves, confidence=min(confidence, 1.0))
            )

        # Upward zigzag (bullish correction in downtrend)
        if not seg[0].is_high:
            if wb.end.price <= wa.start.price:
                continue
            if wc.end.price <= wa.end.price:
                continue

            confidence = 0.5
            if wa.length > 0:
                b_ratio = wb.length / wa.length
                if 0.3 < b_ratio < 0.8:
                    confidence += 0.2
                c_ratio = wc.length / wa.length
                if 0.9 < c_ratio < 2.0:
                    confidence += 0.2

            for j, label in enumerate(["A", "B", "C"]):
                waves[j].label = label
            structures.append(
                WaveStructure(WaveType.ZIGZAG, waves, confidence=min(confidence, 1.0))
            )

    return structures


# ---------------------------------------------------------------------------
# Triangle (A-B-C-D-E) detection
# ---------------------------------------------------------------------------

def find_triangles(swings: list[SwingPoint]) -> list[WaveStructure]:
    """Find contracting triangle patterns (5 waves: A-B-C-D-E)."""
    structures: list[WaveStructure] = []
    if len(swings) < 6:
        return structures

    for i in range(len(swings) - 5):
        seg = swings[i : i + 6]
        waves = swings_to_waves(seg)

        # Check contracting pattern: each wave shorter than the previous
        lengths = [w.length for w in waves]
        contracting = all(lengths[j] > lengths[j + 1] for j in range(len(lengths) - 1))

        if not contracting:
            # Allow one violation for flexibility
            violations = sum(
                1 for j in range(len(lengths) - 1) if lengths[j] <= lengths[j + 1]
            )
            if violations > 1:
                continue

        # Check that waves alternate direction
        directions_valid = True
        for j in range(len(waves) - 1):
            if waves[j].direction == waves[j + 1].direction:
                directions_valid = False
                break
        if not directions_valid:
            continue

        confidence = 0.4

        # Check Fibonacci relationship: each wave ~0.618 of the previous
        for j in range(1, len(lengths)):
            if lengths[j - 1] > 0:
                ratio = lengths[j] / lengths[j - 1]
                if 0.5 < ratio < 0.8:
                    confidence += 0.1

        # Check C doesn't go beyond A, D doesn't go beyond B
        if seg[0].is_high:
            # Starts high: A down, B up, C down, D up, E down
            if waves[2].end.price < waves[0].end.price:  # C beyond A - OK for triangle
                pass
            if seg[3].price > seg[1].price:  # D beyond B - invalid
                continue
        else:
            if waves[2].end.price > waves[0].end.price:
                pass
            if seg[3].price < seg[1].price:
                continue

        for j, label in enumerate(["A", "B", "C", "D", "E"]):
            waves[j].label = label

        structures.append(
            WaveStructure(WaveType.TRIANGLE, waves, confidence=min(confidence, 1.0))
        )

    return structures


# ---------------------------------------------------------------------------
# Ending diagonal detection
# ---------------------------------------------------------------------------

def find_ending_diagonals(swings: list[SwingPoint]) -> list[WaveStructure]:
    """Find ending diagonal (wedge) patterns — 5 waves with overlapping."""
    structures: list[WaveStructure] = []
    if len(swings) < 6:
        return structures

    for i in range(len(swings) - 5):
        seg = swings[i : i + 6]
        waves = swings_to_waves(seg)

        # Must be 5 alternating waves
        directions_valid = True
        for j in range(len(waves) - 1):
            if waves[j].direction == waves[j + 1].direction:
                directions_valid = False
                break
        if not directions_valid:
            continue

        w1, w2, w3, w4, w5 = waves

        # Ascending diagonal (bullish then reversal)
        if not seg[0].is_high:
            # Rule: wave 1 > wave 3 > wave 5
            if not (w1.length > w3.length > w5.length):
                continue
            # Rule: wave 2 > wave 4
            if not (w2.length > w4.length):
                continue
            # Diagonal special: wave 4 overlaps wave 1 territory
            if w4.end.price >= w1.end.price:
                continue  # no overlap — not a diagonal

            confidence = 0.5
            # Check 60% ratio (golden section)
            if w1.length > 0:
                r3 = w3.length / w1.length
                if 0.5 < r3 < 0.75:
                    confidence += 0.15
            if w3.length > 0:
                r5 = w5.length / w3.length
                if 0.5 < r5 < 0.75:
                    confidence += 0.15
            if w2.length > 0:
                r4 = w4.length / w2.length
                if 0.5 < r4 < 0.75:
                    confidence += 0.1

            for j, label in enumerate(["1", "2", "3", "4", "5"]):
                waves[j].label = label
            structures.append(
                WaveStructure(WaveType.DIAGONAL, waves, confidence=min(confidence, 1.0))
            )

        # Descending diagonal
        if seg[0].is_high:
            if not (w1.length > w3.length > w5.length):
                continue
            if not (w2.length > w4.length):
                continue
            if w4.end.price <= w1.end.price:
                continue

            confidence = 0.5
            if w1.length > 0:
                r3 = w3.length / w1.length
                if 0.5 < r3 < 0.75:
                    confidence += 0.15
            if w3.length > 0:
                r5 = w5.length / w3.length
                if 0.5 < r5 < 0.75:
                    confidence += 0.15
            if w2.length > 0:
                r4 = w4.length / w2.length
                if 0.5 < r4 < 0.75:
                    confidence += 0.1

            for j, label in enumerate(["1", "2", "3", "4", "5"]):
                waves[j].label = label
            structures.append(
                WaveStructure(WaveType.DIAGONAL, waves, confidence=min(confidence, 1.0))
            )

    return structures


# ---------------------------------------------------------------------------
# Public API: full analysis
# ---------------------------------------------------------------------------

def get_wave_context(analysis: dict, timeframe: str) -> WaveContext:
    """Determine the current wave context from analysis results.

    Examines the most recent wave structures to infer:
    - Where we are in the wave count
    - What direction the next move is expected to be

    This context is passed to lower-timeframe analysis so that
    entry signals can be validated against the larger wave structure.
    """
    trend = analysis["trend"]

    all_structures = (
        analysis["impulses"] + analysis["zigzags"]
        + analysis["triangles"] + analysis["diagonals"]
    )

    if not all_structures:
        return WaveContext(trend=trend, timeframe=timeframe)

    # Pick the most recent structure (by last wave end bar index)
    most_recent = max(all_structures, key=lambda s: s.waves[-1].end.index)
    last_wave = most_recent.waves[-1]
    expected: Direction | None = None
    label = ""

    if most_recent.wave_type == WaveType.IMPULSE:
        # After a completed impulse, expect a correction (opposite direction)
        if last_wave.direction == Direction.LONG:
            expected = Direction.SHORT
        else:
            expected = Direction.LONG
        label = "post_impulse"

    elif most_recent.wave_type == WaveType.ZIGZAG:
        # After a zigzag correction, expect trend continuation
        if trend == Trend.UP:
            expected = Direction.LONG
        elif trend == Trend.DOWN:
            expected = Direction.SHORT
        label = "post_zigzag"

    elif most_recent.wave_type == WaveType.TRIANGLE:
        # After a triangle, expect breakout in the trend direction
        if trend == Trend.UP:
            expected = Direction.LONG
        elif trend == Trend.DOWN:
            expected = Direction.SHORT
        label = "post_triangle"

    elif most_recent.wave_type == WaveType.DIAGONAL:
        # After an ending diagonal, expect reversal
        if last_wave.direction == Direction.LONG:
            expected = Direction.SHORT
        else:
            expected = Direction.LONG
        label = "post_diagonal"

    return WaveContext(
        trend=trend,
        timeframe=timeframe,
        expected_direction=expected,
        wave_label=label,
        structure_type=most_recent.wave_type.value,
        confidence=most_recent.confidence,
    )


def analyze(df: pd.DataFrame) -> dict:
    """Run full wave analysis on OHLCV data.

    Returns a dict with:
        trend: Trend
        swings: list[SwingPoint]
        impulses: list[WaveStructure]
        zigzags: list[WaveStructure]
        triangles: list[WaveStructure]
        diagonals: list[WaveStructure]
    """
    swings = find_swings(df)
    trend = detect_trend(df, swings=swings)

    result = {
        "trend": trend,
        "swings": swings,
        "impulses": find_impulses(swings),
        "zigzags": find_zigzags(swings),
        "triangles": find_triangles(swings),
        "diagonals": find_ending_diagonals(swings),
    }

    logger.info(
        "Analysis: trend=%s swings=%d impulses=%d zigzags=%d triangles=%d diagonals=%d",
        trend.value,
        len(swings),
        len(result["impulses"]),
        len(result["zigzags"]),
        len(result["triangles"]),
        len(result["diagonals"]),
    )

    return result
