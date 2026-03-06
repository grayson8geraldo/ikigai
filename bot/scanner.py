"""Market scanner — hierarchical multi-timeframe wave analysis.

Scans from higher timeframes to lower:
  trend TF (4h) → work TF (1h) → entry TF (15m)

Each level builds a WaveContext that is passed down so that
lower-TF signals are validated against the larger wave structure.
"""

import logging
import time
from typing import Optional

import numpy as np
import config
from bot.exchange import Exchange
from bot.wave_analyzer import analyze, get_wave_context
from bot.setups import check_zigzag_setup, check_diagonal_setup, check_triangle_setup
from bot.models import Signal, Direction, Trend, WaveContext
from bot.learner import Learner

logger = logging.getLogger(__name__)

# Confidence bonuses for hierarchical alignment
_TREND_CONTEXT_BONUS = 0.15   # entry signal aligns with trend-TF context
_WORK_CONTEXT_BONUS = 0.10    # entry signal aligns with work-TF context
_MTF_DIRECTION_BONUS = 0.10   # entry signal direction confirmed on work TF

_BTC_SYMBOL = "BTC/USDT"
_VOLUME_AVG_PERIOD = 20  # bars for average volume calculation


class Scanner:
    """Scans multiple symbols with hierarchical wave analysis."""

    def __init__(self, exchange: Exchange, learner: Learner = None):
        self.exchange = exchange
        self.learner = learner
        self._btc_trend: Trend = Trend.SIDEWAYS

    # ------------------------------------------------------------------
    # Per-symbol hierarchical scan
    # ------------------------------------------------------------------

    def scan_symbol(self, symbol: str) -> list[Signal]:
        """Scan a single symbol using 3-level hierarchical analysis.

        Level 1 (trend TF, e.g. 4h):  determine trend + wave context
        Level 2 (work TF, e.g. 1h):   detect patterns, build work context
        Level 3 (entry TF, e.g. 15m):  find entry signals, validate vs context
        """
        signals: list[Signal] = []

        # ── Level 1: Trend timeframe ──────────────────────────────────
        df_trend = self.exchange.fetch_ohlcv(
            symbol, config.TIMEFRAMES["trend"], limit=200,
        )
        if df_trend.empty:
            return signals

        analysis_trend = analyze(df_trend)
        trend = analysis_trend["trend"]
        ctx_trend = get_wave_context(analysis_trend, config.TIMEFRAMES["trend"])

        if ctx_trend.wave_label:
            logger.info(
                "%s trend-TF context: %s → expects %s (conf %.0f%%)",
                symbol,
                ctx_trend.wave_label,
                ctx_trend.expected_direction.value if ctx_trend.expected_direction else "?",
                ctx_trend.confidence * 100,
            )

        # Fetch current price ONCE per symbol
        current_price = self.exchange.get_current_price(symbol)

        # ── Level 2: Work timeframe ───────────────────────────────────
        df_work = self.exchange.fetch_ohlcv(
            symbol, config.TIMEFRAMES["work"], limit=200,
        )
        if df_work.empty:
            return signals

        analysis_work = analyze(df_work)
        ctx_work = get_wave_context(analysis_work, config.TIMEFRAMES["work"])

        # Generate signals from work-TF patterns (with trend context)
        work_signals = self._generate_signals(
            analysis_work, trend, current_price, symbol,
            config.TIMEFRAMES["work"], ctx_trend,
        )

        # ── Level 3: Entry timeframe ──────────────────────────────────
        df_entry = self.exchange.fetch_ohlcv(
            symbol, config.TIMEFRAMES["entry"], limit=200,
        )
        entry_signals: list[Signal] = []
        if not df_entry.empty:
            analysis_entry = analyze(df_entry)
            entry_signals = self._generate_signals(
                analysis_entry, trend, current_price, symbol,
                config.TIMEFRAMES["entry"], ctx_work,
            )

            # Boost entry signals that align with trend-TF context
            for sig in entry_signals:
                if (ctx_trend.expected_direction
                        and sig.direction == ctx_trend.expected_direction):
                    sig.confidence = min(sig.confidence + _TREND_CONTEXT_BONUS, 1.0)
                    sig.factors.append(
                        f"Hierarchical: {config.TIMEFRAMES['trend']} "
                        f"{ctx_trend.wave_label} expects {sig.direction.value}"
                    )

        # Boost entry signals whose direction matches a work-TF signal
        work_directions = {s.direction for s in work_signals}
        for sig in entry_signals:
            if sig.direction in work_directions:
                sig.confidence = min(sig.confidence + _MTF_DIRECTION_BONUS, 1.0)
                sig.factors.append(
                    f"Multi-TF confluence: {sig.direction.value} "
                    f"confirmed on {config.TIMEFRAMES['work']}"
                )

        # Volume confirmation: reject signals when entry-TF volume is below average
        if not df_entry.empty and entry_signals:
            entry_signals = self._filter_by_volume(entry_signals, df_entry)

        # Merge and deduplicate
        all_sigs = work_signals + entry_signals
        return _deduplicate_signals(all_sigs)

    # ------------------------------------------------------------------
    # Signal generation helper
    # ------------------------------------------------------------------

    def _generate_signals(
        self,
        analysis: dict,
        trend: Trend,
        current_price: float,
        symbol: str,
        timeframe: str,
        context: Optional[WaveContext] = None,
    ) -> list[Signal]:
        """Generate signals from analysis, optionally boosted by higher-TF context."""
        sigs: list[Signal] = []

        for zigzag in analysis["zigzags"]:
            signal = check_zigzag_setup(
                zigzag, trend, current_price, symbol, timeframe,
            )
            if signal:
                self._apply_context_bonus(signal, context)
                sigs.append(signal)

        for diagonal in analysis["diagonals"]:
            signal = check_diagonal_setup(
                diagonal, current_price, symbol, timeframe,
            )
            if signal:
                self._apply_context_bonus(signal, context)
                sigs.append(signal)

        for triangle in analysis["triangles"]:
            signal = check_triangle_setup(
                triangle, trend, current_price, symbol, timeframe,
            )
            if signal:
                self._apply_context_bonus(signal, context)
                sigs.append(signal)

        return sigs

    @staticmethod
    def _apply_context_bonus(signal: Signal, context: Optional[WaveContext]):
        """Boost signal confidence if it aligns with higher-TF wave context."""
        if not context or not context.expected_direction:
            return
        if signal.direction == context.expected_direction:
            signal.confidence = min(signal.confidence + _WORK_CONTEXT_BONUS, 1.0)
            signal.factors.append(
                f"Aligned with {context.timeframe} {context.wave_label}"
            )

    # ------------------------------------------------------------------
    # Volume confirmation
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_by_volume(signals: list[Signal], df) -> list[Signal]:
        """Reject signals when current volume is below the rolling average.

        This filters out false breakouts and weak setups that lack
        market participation.
        """
        if "volume" not in df.columns or len(df) < _VOLUME_AVG_PERIOD + 1:
            return signals  # no volume data — pass through

        volumes = df["volume"].values
        avg_vol = np.mean(volumes[-_VOLUME_AVG_PERIOD - 1 : -1])  # exclude current bar
        current_vol = volumes[-1]

        if avg_vol <= 0:
            return signals

        vol_ratio = current_vol / avg_vol
        threshold = config.VOLUME_CONFIRM_MULTIPLIER

        if vol_ratio >= threshold:
            # Volume confirmed — boost confidence slightly
            for sig in signals:
                sig.confidence = min(sig.confidence + 0.05, 1.0)
                sig.factors.append(f"Volume confirmed: {vol_ratio:.1f}x avg")
            return signals
        else:
            logger.info(
                "Volume filter: rejected %d signal(s) — vol ratio %.2f < %.2f threshold",
                len(signals), vol_ratio, threshold,
            )
            return []

    # ------------------------------------------------------------------
    # BTC trend detection
    # ------------------------------------------------------------------

    def _detect_btc_trend(self) -> Trend:
        """Analyze BTC/USDT using trend + work timeframes + price momentum.

        Adapts automatically to the trading profile:
          intraday: 4h (trend) + 1h (work) + 1h momentum
          swing:    1d (trend) + 4h (work) + 4h momentum
        """
        # Trend-TF analysis
        df_trend = self.exchange.fetch_ohlcv(
            _BTC_SYMBOL, config.TIMEFRAMES["trend"], limit=200,
        )
        if df_trend.empty:
            logger.warning("Cannot fetch BTC trend data, defaulting to SIDEWAYS")
            return Trend.SIDEWAYS

        trend_result = analyze(df_trend)["trend"]

        # Work-TF analysis + price momentum
        df_work = self.exchange.fetch_ohlcv(
            _BTC_SYMBOL, config.TIMEFRAMES["work"], limit=200,
        )
        if df_work.empty:
            logger.info("BTC trend: %s (trend-TF only)", trend_result.value)
            return trend_result

        work_result = analyze(df_work)["trend"]

        # Price momentum: last 3 candles on work TF
        momentum = Trend.SIDEWAYS
        if len(df_work) >= 4:
            recent_close = float(df_work["close"].iloc[-1])
            past_close = float(df_work["close"].iloc[-4])
            change_pct = (recent_close - past_close) / past_close

            if change_pct > 0.01:
                momentum = Trend.UP
            elif change_pct < -0.01:
                momentum = Trend.DOWN

        # Decision logic (priority order)
        if trend_result == work_result and trend_result != Trend.SIDEWAYS:
            result = trend_result
        elif work_result == momentum and work_result != Trend.SIDEWAYS:
            result = work_result
        elif trend_result == momentum and trend_result != Trend.SIDEWAYS:
            result = trend_result
        elif momentum != Trend.SIDEWAYS:
            result = momentum
        elif work_result != Trend.SIDEWAYS:
            result = work_result
        elif trend_result != Trend.SIDEWAYS:
            result = trend_result
        else:
            result = Trend.SIDEWAYS

        logger.info(
            "BTC trend: %s (trend-TF=%s, work-TF=%s, momentum=%s)",
            result.value, trend_result.value, work_result.value, momentum.value,
        )
        return result

    # ------------------------------------------------------------------
    # Full scan
    # ------------------------------------------------------------------

    def scan_all(self) -> list[Signal]:
        """Scan all configured symbols and return all valid signals."""
        all_signals: list[Signal] = []

        self.exchange.clear_price_cache()

        # Detect BTC trend
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
            time.sleep(0.5)

        # BTC trend filter for altcoins
        if config.BTC_TREND_FILTER and self._btc_trend != Trend.SIDEWAYS:
            filtered = []
            for sig in all_signals:
                if sig.symbol == _BTC_SYMBOL:
                    filtered.append(sig)
                    continue
                if self._btc_trend == Trend.UP and sig.direction == Direction.SHORT:
                    logger.info(
                        "Filtered %s %s: BTC trend UP, rejecting altcoin shorts",
                        sig.symbol, sig.direction.value,
                    )
                    continue
                if self._btc_trend == Trend.DOWN and sig.direction == Direction.LONG:
                    logger.info(
                        "Filtered %s %s: BTC trend DOWN, rejecting altcoin longs",
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

        # Apply self-learning weights to adjust signal confidence
        if self.learner:
            for sig in all_signals:
                mult = self.learner.get_signal_multiplier(
                    setup_type=sig.setup_type.value,
                    symbol=sig.symbol,
                    direction=sig.direction.value,
                )
                if mult != 1.0:
                    old_conf = sig.confidence
                    sig.confidence = min(max(sig.confidence * mult, 0.05), 1.0)
                    sig.factors.append(f"Learned weight: {mult:.2f}x")
                    logger.info(
                        "Learning adjusted %s %s confidence: %.2f → %.2f (x%.2f)",
                        sig.symbol, sig.direction.value,
                        old_conf, sig.confidence, mult,
                    )

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
