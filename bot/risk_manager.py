"""Risk manager — position sizing, limits, and trade validation."""

import logging
from bot.models import Signal, Direction

import config

logger = logging.getLogger(__name__)


class RiskManager:
    """Manages risk for each trade and overall portfolio."""

    def __init__(self, balance: float):
        self.balance = balance
        self.open_positions = 0
        self.daily_losses = 0
        self.daily_loss_limit = 0.20  # 20% of balance
        self.consecutive_losses = 0
        self.max_consecutive_losses = 3
        self.trades_today = 0

    def update_balance(self, balance: float):
        self.balance = balance

    def reset_daily(self):
        """Call at the start of each trading day."""
        self.daily_losses = 0
        self.trades_today = 0

    def can_trade(self) -> tuple[bool, str]:
        """Check if we are allowed to open a new trade."""
        if self.open_positions >= config.MAX_POSITIONS:
            return False, f"Max positions reached ({config.MAX_POSITIONS})"

        if self.consecutive_losses >= self.max_consecutive_losses:
            return False, f"Max consecutive losses reached ({self.max_consecutive_losses})"

        if self.balance > 0 and self.daily_losses / self.balance >= self.daily_loss_limit:
            return False, f"Daily loss limit reached ({self.daily_loss_limit:.0%})"

        return True, "OK"

    def validate_signal(self, signal: Signal) -> tuple[bool, str]:
        """Validate a signal against risk rules."""
        # Check R:R ratio
        if signal.rr_ratio < config.MIN_RR_RATIO:
            return False, f"R:R too low: {signal.rr_ratio:.1f} < {config.MIN_RR_RATIO}"

        # Check risk percentage
        risk_pct = signal.risk_pct
        if risk_pct > 0.15:  # stop more than 15% away is suspicious
            return False, f"Stop too far: {risk_pct:.1%}"

        if risk_pct < 0.001:  # stop too tight
            return False, f"Stop too tight: {risk_pct:.1%}"

        return True, "OK"

    def calculate_position_size(self, signal: Signal) -> dict:
        """Calculate position size based on risk parameters.

        Returns dict with:
            margin: how much USDT to use as margin
            size: position size in USDT (margin * leverage)
            size_base: position size in base currency
            risk_amount: dollar amount at risk
        """
        # Determine risk percentage based on balance phase
        if self.balance < 300:
            risk_pct = min(config.RISK_PER_TRADE, 0.10)  # aggressive: up to 10%
            leverage = min(config.LEVERAGE, 10)
        elif self.balance < 600:
            risk_pct = min(config.RISK_PER_TRADE, 0.07)
            leverage = min(config.LEVERAGE, 7)
        else:
            risk_pct = min(config.RISK_PER_TRADE, 0.05)  # conservative
            leverage = min(config.LEVERAGE, 5)

        risk_amount = self.balance * risk_pct

        # Position size = risk_amount / distance_to_stop_pct
        stop_distance_pct = signal.risk_pct
        if stop_distance_pct == 0:
            return {"margin": 0, "size": 0, "size_base": 0, "risk_amount": 0, "leverage": leverage}

        position_size = risk_amount / stop_distance_pct  # in USDT
        margin = position_size / leverage

        # Cap margin at 50% of balance
        if margin > self.balance * 0.5:
            margin = self.balance * 0.5
            position_size = margin * leverage

        size_base = position_size / signal.entry_price if signal.entry_price > 0 else 0

        result = {
            "margin": round(margin, 2),
            "size": round(position_size, 2),
            "size_base": size_base,
            "risk_amount": round(risk_amount, 2),
            "leverage": leverage,
        }

        logger.info(
            "Position size: margin=$%.2f, size=$%.2f (x%d), risk=$%.2f",
            result["margin"], result["size"], leverage, result["risk_amount"],
        )

        return result

    def record_win(self, pnl: float):
        """Record a winning trade."""
        self.consecutive_losses = 0
        self.trades_today += 1

    def record_loss(self, pnl: float):
        """Record a losing trade."""
        self.consecutive_losses += 1
        self.daily_losses += abs(pnl)
        self.trades_today += 1
