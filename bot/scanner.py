"""Market scanner — scans all configured symbols for trading setups."""

import logging
import time

import config
from bot.exchange import Exchange
from bot.wave_analyzer import analyze
from bot.setups import check_zigzag_setup, check_diagonal_setup, check_triangle_setup
from bot.models import Signal

logger = logging.getLogger(__name__)

# Confluence bonus when a signal direction matches a higher-timeframe pattern
_MTF_CONFLUENCE_BONUS = 0.15


class Scanner:
    """Scans multiple symbols and timeframes for trading setups."""

    def __init__(self, exchange: Exchange):
        self.exchange = exchange

    def scan_symbol(self, symbol: str) -> list[Signal]:
        """Scan a single symbol across working timeframes for setups."""
        signals: list[Signal] = []

        # 1. Get global trend from daily chart
        df_daily = self.exchange.fetch_ohlcv(symbol, config.TIMEFRAMES["mid"], limit=200)
        if df_daily.empty:
            return signals

        global_analysis = analyze(df_daily)
        trend = global_analysis["trend"]

        # 2. Fetch current price ONCE per symbol
        current_price = self.exchange.get_current_price(symbol)

        # 3. Collect signals per timeframe (higher TF first for confluence)
        tf_signals: dict[str, list[Signal]] = {}

        for tf_name in ["work", "entry"]:
            tf = config.TIMEFRAMES[tf_name]
            df = self.exchange.fetch_ohlcv(symbol, tf, limit=200)
            if df.empty:
                continue

            analysis = analyze(df)
            tf_sigs: list[Signal] = []

            # Check Setup A: Zigzag + Breakout
            for zigzag in analysis["zigzags"]:
                signal = check_zigzag_setup(
                    zigzag, trend, current_price, symbol, tf,
                )
                if signal:
                    tf_sigs.append(signal)

            # Check Setup B: Ending Diagonal
            for diagonal in analysis["diagonals"]:
                signal = check_diagonal_setup(
                    diagonal, current_price, symbol, tf,
                )
                if signal:
                    tf_sigs.append(signal)

            # Check Setup C: Triangle + Breakout
            for triangle in analysis["triangles"]:
                signal = check_triangle_setup(
                    triangle, trend, current_price, symbol, tf,
                )
                if signal:
                    tf_sigs.append(signal)

            tf_signals[tf_name] = tf_sigs

        # 4. Multi-timeframe confluence: boost entry-TF signals confirmed by work-TF
        work_directions = set()
        for sig in tf_signals.get("work", []):
            work_directions.add(sig.direction)

        for sig in tf_signals.get("entry", []):
            if sig.direction in work_directions:
                sig.confidence = min(sig.confidence + _MTF_CONFLUENCE_BONUS, 1.0)
                sig.factors.append(
                    f"Multi-TF confluence: {sig.direction.value} confirmed on {config.TIMEFRAMES['work']}"
                )

        # Merge all timeframe signals
        for tf_sigs in tf_signals.values():
            signals.extend(tf_sigs)

        # 5. Deduplicate: keep only the best signal per symbol+direction
        signals = _deduplicate_signals(signals)

        return signals

    def scan_all(self) -> list[Signal]:
        """Scan all configured symbols and return all valid signals."""
        all_signals: list[Signal] = []

        # Clear price cache at the start of each full scan cycle
        self.exchange.clear_price_cache()

        for symbol in config.SYMBOLS:
            logger.info("Scanning %s ...", symbol)
            try:
                signals = self.scan_symbol(symbol)
                all_signals.extend(signals)
            except Exception as e:
                logger.error("Error scanning %s: %s", symbol, e)
            # Rate limiting
            time.sleep(0.5)

        # Sort by confidence (highest first)
        all_signals.sort(key=lambda s: s.confidence, reverse=True)

        logger.info("Scan complete: %d signals found", len(all_signals))
        return all_signals


def _deduplicate_signals(signals: list[Signal]) -> list[Signal]:
    """Keep only the highest-confidence signal per symbol+direction."""
    best: dict[str, Signal] = {}
    for sig in signals:
        key = f"{sig.symbol}:{sig.direction.value}"
        existing = best.get(key)
        if existing is None or sig.confidence > existing.confidence:
            best[key] = sig
    return list(best.values())
