"""Market scanner — scans all configured symbols for trading setups."""

import logging
import time

import config
from bot.exchange import Exchange
from bot.wave_analyzer import analyze
from bot.setups import check_zigzag_setup, check_diagonal_setup, check_triangle_setup
from bot.models import Signal, Direction, Trend

logger = logging.getLogger(__name__)

# Confluence bonus when a signal direction matches a higher-timeframe pattern
_MTF_CONFLUENCE_BONUS = 0.15

_BTC_SYMBOL = "BTC/USDT"


class Scanner:
    """Scans multiple symbols and timeframes for trading setups."""

    def __init__(self, exchange: Exchange):
        self.exchange = exchange
        self._btc_trend: Trend = Trend.SIDEWAYS

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

    def _detect_btc_trend(self) -> Trend:
        """Analyze BTC/USDT on multiple timeframes + price momentum.

        Uses daily trend, 4h trend, and 4h price momentum (last 12h).
        Price momentum overrides when trend analysis lags behind a reversal.
        """
        # Daily trend (long-term context)
        df_daily = self.exchange.fetch_ohlcv(_BTC_SYMBOL, config.TIMEFRAMES["mid"], limit=200)
        if df_daily.empty:
            logger.warning("Cannot fetch BTC daily data, defaulting to SIDEWAYS")
            return Trend.SIDEWAYS

        daily_trend = analyze(df_daily)["trend"]

        # 4h trend + price momentum
        df_4h = self.exchange.fetch_ohlcv(_BTC_SYMBOL, config.TIMEFRAMES["work"], limit=200)
        if df_4h.empty:
            logger.info("BTC trend: %s (daily only)", daily_trend.value)
            return daily_trend

        work_trend = analyze(df_4h)["trend"]

        # 4h price momentum: last 3 candles (~12 hours)
        momentum = Trend.SIDEWAYS
        if len(df_4h) >= 4:
            recent_close = float(df_4h["close"].iloc[-1])
            past_close = float(df_4h["close"].iloc[-4])
            change_pct = (recent_close - past_close) / past_close

            if change_pct > 0.01:       # BTC up >1% in 12h
                momentum = Trend.UP
            elif change_pct < -0.01:     # BTC down >1% in 12h
                momentum = Trend.DOWN

        # Decision logic (priority order):
        # 1. Daily + 4h agree → strong signal
        if daily_trend == work_trend and daily_trend != Trend.SIDEWAYS:
            result = daily_trend
        # 2. 4h trend matches momentum → recent consensus
        elif work_trend == momentum and work_trend != Trend.SIDEWAYS:
            result = work_trend
        # 3. Daily matches momentum → confirmed despite 4h lag
        elif daily_trend == momentum and daily_trend != Trend.SIDEWAYS:
            result = daily_trend
        # 4. Momentum is clear but trends are mixed → trust price action
        elif momentum != Trend.SIDEWAYS:
            result = momentum
        # 5. Single clear trend (no momentum) → use it
        elif work_trend != Trend.SIDEWAYS:
            result = work_trend
        elif daily_trend != Trend.SIDEWAYS:
            result = daily_trend
        else:
            result = Trend.SIDEWAYS

        logger.info(
            "BTC trend: %s (daily=%s, 4h=%s, momentum=%s)",
            result.value, daily_trend.value, work_trend.value, momentum.value,
        )
        return result

    def scan_all(self) -> list[Signal]:
        """Scan all configured symbols and return all valid signals."""
        all_signals: list[Signal] = []

        # Clear price cache at the start of each full scan cycle
        self.exchange.clear_price_cache()

        # Detect BTC trend BEFORE scanning altcoins
        if config.BTC_TREND_FILTER:
            self._btc_trend = self._detect_btc_trend()
        else:
            self._btc_trend = Trend.SIDEWAYS

        for symbol in config.SYMBOLS:
            logger.info("Scanning %s ...", symbol)
            try:
                signals = self.scan_symbol(symbol)
                all_signals.extend(signals)
            except Exception as e:
                logger.error("Error scanning %s: %s", symbol, e)
            # Rate limiting
            time.sleep(0.5)

        # Filter signals against BTC trend (altcoins only)
        if config.BTC_TREND_FILTER and self._btc_trend != Trend.SIDEWAYS:
            filtered = []
            for sig in all_signals:
                # Don't filter BTC itself
                if sig.symbol == _BTC_SYMBOL:
                    filtered.append(sig)
                    continue

                # BTC UP → reject SHORT on alts
                if self._btc_trend == Trend.UP and sig.direction == Direction.SHORT:
                    logger.info(
                        "Filtered %s %s: BTC trend is UP, rejecting altcoin shorts",
                        sig.symbol, sig.direction.value,
                    )
                    continue

                # BTC DOWN → reject LONG on alts
                if self._btc_trend == Trend.DOWN and sig.direction == Direction.LONG:
                    logger.info(
                        "Filtered %s %s: BTC trend is DOWN, rejecting altcoin longs",
                        sig.symbol, sig.direction.value,
                    )
                    continue

                filtered.append(sig)

            rejected = len(all_signals) - len(filtered)
            if rejected > 0:
                logger.info(
                    "BTC trend filter (%s): rejected %d/%d signals",
                    self._btc_trend.value, rejected, len(all_signals),
                )
            all_signals = filtered

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
