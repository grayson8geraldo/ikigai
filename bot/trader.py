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
        self._load_state()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_state(self):
        """Load open positions and history from disk."""
        if os.path.exists(POSITIONS_FILE):
            try:
                with open(POSITIONS_FILE) as f:
                    data = json.load(f)
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
                    )
                    if pos.is_open:
                        self.positions.append(pos)
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

        # Calculate position size
        sizing = self.rm.calculate_position_size(signal)
        if sizing["size"] == 0:
            logger.warning("Position size is zero")
            return None

        # Set leverage
        self.exchange.set_leverage(signal.symbol, sizing["leverage"])

        # Determine take profit (first target)
        take_profit = signal.targets[0].price if signal.targets else 0

        # Place order
        side = "buy" if signal.direction == Direction.LONG else "sell"

        order_params = {}
        if config.TRADING_MODE == "live":
            order_params = {
                "stopLoss": {"triggerPrice": str(signal.stop_loss)},
                "takeProfit": {"triggerPrice": str(take_profit)},
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

        # Create position record
        position = Position(
            id=str(uuid.uuid4())[:8],
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=signal.entry_price,
            size=sizing["size_base"],
            margin=sizing["margin"],
            leverage=sizing["leverage"],
            stop_loss=signal.stop_loss,
            take_profit=take_profit,
            signal=signal,
        )

        self.positions.append(position)
        self.rm.open_positions += 1
        self._save_state()

        logger.info(
            "OPENED %s %s | entry=%.2f | sl=%.2f | tp=%.2f | margin=$%.2f | x%d",
            signal.direction.value.upper(),
            signal.symbol,
            signal.entry_price,
            signal.stop_loss,
            take_profit,
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

        # Record to history
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
        })

        self._save_state()

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
