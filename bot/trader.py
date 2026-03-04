"""Paper and live trader — manages orders and positions."""

import json
import os
import time
import uuid
import logging
from typing import Optional

import config
from bot.models import Signal, Position, Direction
from bot.exchange import Exchange
from bot.risk_manager import RiskManager
from bot.learner import Learner

logger = logging.getLogger(__name__)

POSITIONS_FILE = os.path.join(config.LOG_DIR, "positions.json")
HISTORY_FILE = os.path.join(config.LOG_DIR, "trade_history.json")


class Trader:
    """Manages opening, closing, and tracking positions."""

    def __init__(self, exchange: Exchange, risk_manager: RiskManager):
        self.exchange = exchange
        self.rm = risk_manager
        self.positions: list[Position] = []
        self.history: list[dict] = []
        # Cooldown: signal_key -> timestamp of last trade close
        self._signal_cooldowns: dict[str, float] = {}
        self.learner = Learner()
        self._load_state()

    @staticmethod
    def _signal_key(signal: Signal) -> str:
        """Generate a unique key for a signal to prevent duplicates."""
        return f"{signal.symbol}:{signal.direction.value}:{signal.setup_type.value}"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_state(self):
        """Load open positions and history from disk with validation."""
        if os.path.exists(POSITIONS_FILE):
            try:
                with open(POSITIONS_FILE) as f:
                    data = json.load(f)

                # Maximum sane margin: 5x initial deposit
                max_margin = config.PAPER_DEPOSIT * 5
                max_age_sec = config.MAX_POSITION_AGE_HOURS * 3600
                seen_symbols: set[str] = set()

                for p in data:
                    pos = Position(
                        id=p["id"],
                        symbol=p["symbol"],
                        direction=Direction(p["direction"]),
                        entry_price=p["entry_price"],
                        size=p["size"],
                        margin=p["margin"],
                        leverage=p["leverage"],
                        stop_loss=p["stop_loss"],
                        take_profit=p["take_profit"],
                        signal=None,  # signal not persisted
                        open_time=p.get("open_time", 0),
                        is_open=p.get("is_open", True),
                        setup_type=p.get("setup_type", ""),
                        signal_timeframe=p.get("signal_timeframe", ""),
                        signal_confidence=p.get("signal_confidence", 0.0),
                        signal_factors=p.get("signal_factors", []),
                    )
                    if not pos.is_open:
                        continue

                    # Validate: reject positions with corrupted/unreasonable margin
                    if pos.margin > max_margin:
                        logger.warning(
                            "Discarding corrupted position %s %s: "
                            "margin=$%.2f exceeds max $%.2f",
                            pos.symbol, pos.id, pos.margin, max_margin,
                        )
                        continue

                    # Validate: reject positions older than MAX_POSITION_AGE_HOURS
                    if pos.open_time > 0:
                        age = time.time() - pos.open_time
                        if age > max_age_sec:
                            logger.warning(
                                "Discarding stale position %s %s: "
                                "age=%.1fh exceeds max %dh",
                                pos.symbol, pos.id,
                                age / 3600, config.MAX_POSITION_AGE_HOURS,
                            )
                            continue

                    # Validate: reject positions with zero/negative entry price
                    if pos.entry_price <= 0:
                        logger.warning(
                            "Discarding invalid position %s %s: entry_price=%.4f",
                            pos.symbol, pos.id, pos.entry_price,
                        )
                        continue

                    # Deduplicate: keep only one position per symbol
                    if pos.symbol in seen_symbols:
                        logger.warning(
                            "Discarding duplicate position %s %s "
                            "(already have one for this symbol)",
                            pos.symbol, pos.id,
                        )
                        continue

                    seen_symbols.add(pos.symbol)
                    self.positions.append(pos)

                if self.positions:
                    logger.info(
                        "Loaded %d valid position(s) from disk", len(self.positions)
                    )
            except Exception as e:
                logger.error("Failed to load positions: %s", e)

        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE) as f:
                    self.history = json.load(f)
            except Exception as e:
                logger.error("Failed to load history: %s", e)

    def _save_state(self):
        """Save open positions and history to disk."""
        data = []
        for p in self.positions:
            data.append({
                "id": p.id,
                "symbol": p.symbol,
                "direction": p.direction.value,
                "entry_price": p.entry_price,
                "size": p.size,
                "margin": p.margin,
                "leverage": p.leverage,
                "stop_loss": p.stop_loss,
                "take_profit": p.take_profit,
                "open_time": p.open_time,
                "is_open": p.is_open,
                "setup_type": p.setup_type,
                "signal_timeframe": p.signal_timeframe,
                "signal_confidence": p.signal_confidence,
                "signal_factors": p.signal_factors,
            })
        with open(POSITIONS_FILE, "w") as f:
            json.dump(data, f, indent=2)

        with open(HISTORY_FILE, "w") as f:
            json.dump(self.history, f, indent=2)

    # ------------------------------------------------------------------
    # Open position
    # ------------------------------------------------------------------

    def open_position(self, signal: Signal) -> Optional[Position]:
        """Open a new position based on a signal."""
        # Risk checks
        can, reason = self.rm.can_trade()
        if not can:
            logger.warning("Cannot trade: %s", reason)
            return None

        valid, reason = self.rm.validate_signal(signal)
        if not valid:
            logger.warning("Signal rejected: %s", reason)
            return None

        # --- Safety: prevent duplicate positions on the same symbol ---
        for pos in self.positions:
            if pos.is_open and pos.symbol == signal.symbol:
                logger.warning(
                    "Skipping %s: already have an open position on this symbol",
                    signal.symbol,
                )
                return None

        # --- Safety: signal cooldown (prevent re-trading same setup) ---
        sig_key = self._signal_key(signal)
        cooldown_sec = config.SIGNAL_COOLDOWN_HOURS * 3600
        last_trade = self._signal_cooldowns.get(sig_key, 0)
        if time.time() - last_trade < cooldown_sec:
            remaining = cooldown_sec - (time.time() - last_trade)
            logger.warning(
                "Signal cooldown active for %s (%.0f min remaining)",
                sig_key, remaining / 60,
            )
            return None

        # --- Safety: validate entry price against current market price ---
        current_price = self.exchange.get_current_price(signal.symbol)
        if current_price == 0:
            logger.error("Cannot get current price for %s", signal.symbol)
            return None

        deviation = abs(current_price - signal.entry_price) / signal.entry_price
        if deviation > config.ENTRY_PRICE_MAX_DEVIATION:
            logger.warning(
                "Entry price stale for %s: signal=%.4f, market=%.4f, deviation=%.2f%% > %.2f%%",
                signal.symbol,
                signal.entry_price,
                current_price,
                deviation * 100,
                config.ENTRY_PRICE_MAX_DEVIATION * 100,
            )
            return None

        # --- Safety: verify TP hasn't already been reached ---
        take_profit = signal.targets[0].price if signal.targets else 0
        if take_profit > 0:
            if signal.direction == Direction.LONG and current_price >= take_profit:
                logger.warning(
                    "TP already reached for LONG %s: price=%.4f >= tp=%.4f",
                    signal.symbol, current_price, take_profit,
                )
                return None
            elif signal.direction == Direction.SHORT and current_price <= take_profit:
                logger.warning(
                    "TP already reached for SHORT %s: price=%.4f <= tp=%.4f",
                    signal.symbol, current_price, take_profit,
                )
                return None

        # Calculate position size
        sizing = self.rm.calculate_position_size(signal)
        if sizing["size"] == 0:
            logger.warning("Position size is zero")
            return None

        # Set leverage
        self.exchange.set_leverage(signal.symbol, sizing["leverage"])

        # Use current market price as actual entry in paper mode
        actual_entry = current_price if config.TRADING_MODE != "live" else signal.entry_price

        # --- Use wave-based SL/TP from signal directly ---
        # SL/TP are structural levels (wave invalidation points) and should NOT
        # be adjusted proportionally when entry price shifts — that destroys
        # the structural meaning of the stop level.
        actual_sl = signal.stop_loss
        actual_tp = take_profit

        # Validate: SL must be on the correct side of actual entry
        if signal.direction == Direction.LONG and actual_sl >= actual_entry:
            logger.warning(
                "Rejecting %s LONG: SL %.4f >= entry %.4f (SL above entry)",
                signal.symbol, actual_sl, actual_entry,
            )
            return None
        if signal.direction == Direction.SHORT and actual_sl <= actual_entry:
            logger.warning(
                "Rejecting %s SHORT: SL %.4f <= entry %.4f (SL below entry)",
                signal.symbol, actual_sl, actual_entry,
            )
            return None

        # --- Enforce minimum stop distance ---
        min_stop_pct = config.MIN_STOP_DISTANCE_PCT
        if actual_entry > 0:
            current_sl_pct = abs(actual_entry - actual_sl) / actual_entry
            if current_sl_pct < min_stop_pct:
                old_sl = actual_sl
                if signal.direction == Direction.LONG:
                    actual_sl = actual_entry * (1 - min_stop_pct)
                else:
                    actual_sl = actual_entry * (1 + min_stop_pct)
                logger.info(
                    "Enforced min stop distance for %s: SL %.4f->%.4f (%.2f%% -> %.2f%%)",
                    signal.symbol, old_sl, actual_sl,
                    current_sl_pct * 100, min_stop_pct * 100,
                )

        # --- Validate actual R:R from market entry ---
        actual_risk = abs(actual_entry - actual_sl)
        actual_reward = abs(actual_tp - actual_entry) if actual_tp > 0 else 0
        if actual_risk > 0 and actual_reward > 0:
            actual_rr = actual_reward / actual_risk
            if actual_rr < config.MIN_RR_RATIO:
                logger.warning(
                    "Actual R:R too low for %s: %.1f:1 < %.1f:1 "
                    "(entry=%.4f, sl=%.4f, tp=%.4f)",
                    signal.symbol, actual_rr, config.MIN_RR_RATIO,
                    actual_entry, actual_sl, actual_tp,
                )
                return None

        if actual_entry != signal.entry_price:
            logger.info(
                "Trade at market for %s: entry %.4f (signal %.4f), SL %.4f, TP %.4f",
                signal.symbol, actual_entry, signal.entry_price, actual_sl, actual_tp,
            )

        # Place order
        side = "buy" if signal.direction == Direction.LONG else "sell"

        order_params = {}
        if config.TRADING_MODE == "live":
            order_params = {
                "stopLoss": {"triggerPrice": str(actual_sl)},
                "takeProfit": {"triggerPrice": str(actual_tp)},
            }

        order = self.exchange.create_market_order(
            signal.symbol,
            side,
            sizing["size_base"],
            params=order_params,
        )

        if not order:
            logger.error("Failed to place order for %s", signal.symbol)
            return None

        # Create position record (use actual market price for realistic paper trading)
        position = Position(
            id=str(uuid.uuid4())[:8],
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=actual_entry,
            size=sizing["size_base"],
            margin=sizing["margin"],
            leverage=sizing["leverage"],
            stop_loss=actual_sl,
            take_profit=actual_tp,
            signal=signal,
            setup_type=signal.setup_type.value,
            signal_timeframe=signal.timeframe,
            signal_confidence=signal.confidence,
            signal_factors=list(signal.factors),
        )

        self.positions.append(position)
        self.rm.open_positions += 1
        self._save_state()

        logger.info(
            "OPENED %s %s | entry=%.4f (signal=%.4f) | sl=%.4f | tp=%.4f | margin=$%.2f | x%d",
            signal.direction.value.upper(),
            signal.symbol,
            actual_entry,
            signal.entry_price,
            actual_sl,
            actual_tp,
            sizing["margin"],
            sizing["leverage"],
        )

        return position

    # ------------------------------------------------------------------
    # Check positions (paper mode — simulate stop/take)
    # ------------------------------------------------------------------

    def check_positions(self):
        """Check open positions against current prices (paper trading)."""
        for pos in self.positions:
            if not pos.is_open:
                continue

            price = self.exchange.get_current_price(pos.symbol)
            if price == 0:
                continue

            pos.close_price = price  # for unrealized PnL tracking

            # Check max hold time — auto-close stale positions
            hold_hours = (time.time() - pos.open_time) / 3600
            if hold_hours > config.MAX_POSITION_AGE_HOURS:
                logger.info(
                    "Max hold time (%.1fh > %dh) for %s — closing at market",
                    hold_hours, config.MAX_POSITION_AGE_HOURS, pos.symbol,
                )
                self._close_position(pos, price, "MAX HOLD TIME")
                continue

            # Check stop loss
            hit_stop = False
            hit_target = False

            if pos.direction == Direction.LONG:
                hit_stop = price <= pos.stop_loss
                hit_target = pos.take_profit > 0 and price >= pos.take_profit
            else:
                hit_stop = price >= pos.stop_loss
                hit_target = pos.take_profit > 0 and price <= pos.take_profit

            if hit_stop:
                self._close_position(pos, price, "STOP LOSS")
            elif hit_target:
                self._close_position(pos, price, "TAKE PROFIT")

    def _close_position(self, pos: Position, close_price: float, reason: str):
        """Close a position and record PnL."""
        pos.is_open = False
        pos.close_price = close_price
        pos.close_time = time.time()

        if pos.direction == Direction.LONG:
            pnl_pct = (close_price - pos.entry_price) / pos.entry_price
        else:
            pnl_pct = (pos.entry_price - close_price) / pos.entry_price

        pos.pnl = pnl_pct * pos.margin * pos.leverage

        # Update risk manager
        self.rm.open_positions = max(0, self.rm.open_positions - 1)
        if pos.pnl >= 0:
            self.rm.record_win(pos.pnl)
        else:
            self.rm.record_loss(pos.pnl)

        self.rm.update_balance(self.rm.balance + pos.pnl)

        # Record signal cooldown to prevent re-trading the same setup
        if pos.signal:
            sig_key = self._signal_key(pos.signal)
            self._signal_cooldowns[sig_key] = time.time()
            logger.info("Signal cooldown set for %s (%dh)", sig_key, config.SIGNAL_COOLDOWN_HOURS)

        # Record to history (including metadata for self-learning)
        self.history.append({
            "id": pos.id,
            "symbol": pos.symbol,
            "direction": pos.direction.value,
            "entry_price": pos.entry_price,
            "close_price": close_price,
            "pnl": round(pos.pnl, 2),
            "pnl_pct": round(pnl_pct * 100, 2),
            "reason": reason,
            "open_time": pos.open_time,
            "close_time": pos.close_time,
            "setup_type": pos.setup_type,
            "timeframe": pos.signal_timeframe,
            "confidence": pos.signal_confidence,
            "factors": pos.signal_factors,
        })

        self._save_state()

        # Trigger self-learning update
        self.learner.update(self.history)

        emoji = "+" if pos.pnl >= 0 else ""
        logger.info(
            "CLOSED %s %s | %s | PnL: %s$%.2f (%.1f%%) | Balance: $%.2f",
            pos.direction.value.upper(),
            pos.symbol,
            reason,
            emoji,
            pos.pnl,
            pnl_pct * 100,
            self.rm.balance,
        )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_open_positions(self) -> list[Position]:
        return [p for p in self.positions if p.is_open]

    def get_balance(self) -> float:
        return self.rm.balance

    def get_stats(self) -> dict:
        """Get trading statistics."""
        if not self.history:
            return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0, "total_pnl": 0}

        wins = [t for t in self.history if t["pnl"] > 0]
        losses = [t for t in self.history if t["pnl"] <= 0]
        total_pnl = sum(t["pnl"] for t in self.history)

        return {
            "total": len(self.history),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(self.history) * 100 if self.history else 0,
            "total_pnl": round(total_pnl, 2),
            "avg_win": round(sum(t["pnl"] for t in wins) / len(wins), 2) if wins else 0,
            "avg_loss": round(sum(t["pnl"] for t in losses) / len(losses), 2) if losses else 0,
        }
