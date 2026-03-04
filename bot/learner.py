"""Self-learning engine — analyzes trade history and adapts strategy weights.

After each trade closes the Learner recalculates:
  1. Win-rate per setup type  → confidence multiplier
  2. Win-rate per symbol      → confidence multiplier
  3. Win-rate per direction   → directional bias
  4. Avg profitable hold time → suggests optimal hold window
  5. Avg winning gain %       → suggests optimal target distance

Weights are saved to disk and loaded on restart so that learning
persists across bot restarts.
"""

import json
import logging
import os
import time
from typing import Optional

import config

logger = logging.getLogger(__name__)

WEIGHTS_FILE = os.path.join(config.LOG_DIR, "learned_weights.json")

# Need at least this many trades before applying learned adjustments
MIN_TRADES_TOTAL = 5
# Need at least this many trades per category before adjusting its weight
MIN_TRADES_PER_CATEGORY = 3

# Multiplier range: weights are clamped to [MIN, MAX]
# 1.0 = neutral, >1.0 = boost, <1.0 = penalty
_MIN_MULT = 0.5
_MAX_MULT = 1.5

# How much recent trades matter more than older ones (exponential decay)
# 0.0 = all trades equal, 1.0 = only last trade matters
_RECENCY_DECAY = 0.02


def _default_weights() -> dict:
    """Return a fresh weights dictionary with neutral values."""
    return {
        "setup_type": {},       # setup_type_str -> multiplier
        "symbol": {},           # symbol -> multiplier
        "direction": {          # "long"/"short" -> multiplier
            "long": 1.0,
            "short": 1.0,
        },
        "stats": {
            "avg_profitable_hold_hours": 0.0,
            "avg_winning_pct": 0.0,
            "avg_losing_pct": 0.0,
            "total_trades_analyzed": 0,
        },
        "updated_at": 0.0,
    }


class Learner:
    """Analyzes trade history and produces adaptive weights for the scanner."""

    def __init__(self):
        self.weights: dict = self._load_weights()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_weights(self) -> dict:
        if os.path.exists(WEIGHTS_FILE):
            try:
                with open(WEIGHTS_FILE) as f:
                    data = json.load(f)
                logger.info(
                    "Loaded learned weights (%d trades analyzed)",
                    data.get("stats", {}).get("total_trades_analyzed", 0),
                )
                return data
            except Exception as e:
                logger.error("Failed to load weights: %s", e)
        return _default_weights()

    def _save_weights(self):
        self.weights["updated_at"] = time.time()
        try:
            with open(WEIGHTS_FILE, "w") as f:
                json.dump(self.weights, f, indent=2)
        except Exception as e:
            logger.error("Failed to save weights: %s", e)

    # ------------------------------------------------------------------
    # Core learning
    # ------------------------------------------------------------------

    def update(self, history: list[dict]):
        """Recalculate all adaptive weights from the full trade history.

        Called after each trade close.  The history list contains dicts
        with keys: symbol, direction, pnl, pnl_pct, reason,
        open_time, close_time, setup_type, timeframe, confidence, factors.
        """
        if len(history) < MIN_TRADES_TOTAL:
            logger.info(
                "Not enough trades for learning (%d/%d)",
                len(history), MIN_TRADES_TOTAL,
            )
            return

        weights = _default_weights()

        # Compute recency-weighted win stats per category
        weights["setup_type"] = self._learn_category(
            history, key="setup_type",
        )
        weights["symbol"] = self._learn_category(
            history, key="symbol",
        )
        weights["direction"] = self._learn_category(
            history, key="direction",
        )

        # Aggregate stats
        winners = [t for t in history if t["pnl"] > 0]
        losers = [t for t in history if t["pnl"] <= 0]

        if winners:
            hold_times = []
            for t in winners:
                ot = t.get("open_time", 0)
                ct = t.get("close_time", 0)
                if ot > 0 and ct > ot:
                    hold_times.append((ct - ot) / 3600)
            if hold_times:
                weights["stats"]["avg_profitable_hold_hours"] = round(
                    sum(hold_times) / len(hold_times), 2,
                )
            weights["stats"]["avg_winning_pct"] = round(
                sum(t["pnl_pct"] for t in winners) / len(winners), 2,
            )

        if losers:
            weights["stats"]["avg_losing_pct"] = round(
                sum(t["pnl_pct"] for t in losers) / len(losers), 2,
            )

        weights["stats"]["total_trades_analyzed"] = len(history)

        self.weights = weights
        self._save_weights()

        logger.info(
            "Learning update: %d trades → setup_weights=%s, symbol_count=%d",
            len(history),
            {k: round(v, 2) for k, v in weights["setup_type"].items()},
            len(weights["symbol"]),
        )

    def _learn_category(
        self,
        history: list[dict],
        key: str,
    ) -> dict[str, float]:
        """Compute a confidence multiplier per unique value of `key`.

        Uses recency-weighted win rate:
          multiplier = 0.5 + win_rate  (range: 0.5 – 1.5)
        """
        # Group trades by category
        groups: dict[str, list[dict]] = {}
        for t in history:
            val = t.get(key, "")
            if not val:
                continue
            groups.setdefault(val, []).append(t)

        result: dict[str, float] = {}
        now = time.time()

        for cat, trades in groups.items():
            if len(trades) < MIN_TRADES_PER_CATEGORY:
                result[cat] = 1.0  # neutral until enough data
                continue

            # Recency-weighted win rate
            weighted_wins = 0.0
            total_weight = 0.0
            for t in trades:
                age_hours = (now - t.get("close_time", now)) / 3600
                weight = max(0.01, 1.0 - _RECENCY_DECAY * age_hours)
                total_weight += weight
                if t["pnl"] > 0:
                    weighted_wins += weight

            if total_weight > 0:
                win_rate = weighted_wins / total_weight
            else:
                win_rate = 0.5

            # Map win_rate to multiplier: 0% → 0.5x, 50% → 1.0x, 100% → 1.5x
            mult = _MIN_MULT + win_rate * (_MAX_MULT - _MIN_MULT)
            result[cat] = round(max(_MIN_MULT, min(_MAX_MULT, mult)), 3)

        return result

    # ------------------------------------------------------------------
    # Public API: query weights
    # ------------------------------------------------------------------

    def get_signal_multiplier(
        self,
        setup_type: str,
        symbol: str,
        direction: str,
    ) -> float:
        """Return a combined confidence multiplier for a signal.

        Result is the product of per-category multipliers, clamped to [0.5, 1.5].
        Values > 1.0 boost the signal, < 1.0 penalize it.
        """
        if self.weights["stats"].get("total_trades_analyzed", 0) < MIN_TRADES_TOTAL:
            return 1.0  # not enough data yet

        setup_mult = self.weights["setup_type"].get(setup_type, 1.0)
        symbol_mult = self.weights["symbol"].get(symbol, 1.0)
        dir_mult = self.weights["direction"].get(direction, 1.0)

        combined = setup_mult * symbol_mult * dir_mult
        return max(_MIN_MULT, min(_MAX_MULT, combined))

    def get_stats_summary(self) -> dict:
        """Return a summary of learned statistics for display."""
        stats = self.weights.get("stats", {})
        n = stats.get("total_trades_analyzed", 0)
        if n == 0:
            return {"status": "waiting", "trades_analyzed": 0}

        # Find best/worst setup types
        setup_w = self.weights.get("setup_type", {})
        best_setup = max(setup_w, key=setup_w.get) if setup_w else "-"
        worst_setup = min(setup_w, key=setup_w.get) if setup_w else "-"

        # Find best/worst symbols
        sym_w = self.weights.get("symbol", {})
        best_sym = max(sym_w, key=sym_w.get) if sym_w else "-"
        worst_sym = min(sym_w, key=sym_w.get) if sym_w else "-"

        return {
            "status": "active",
            "trades_analyzed": n,
            "avg_profitable_hold_h": stats.get("avg_profitable_hold_hours", 0),
            "avg_winning_pct": stats.get("avg_winning_pct", 0),
            "avg_losing_pct": stats.get("avg_losing_pct", 0),
            "best_setup": f"{best_setup} ({setup_w.get(best_setup, 1.0):.2f}x)",
            "worst_setup": f"{worst_setup} ({setup_w.get(worst_setup, 1.0):.2f}x)",
            "best_symbol": f"{best_sym} ({sym_w.get(best_sym, 1.0):.2f}x)",
            "worst_symbol": f"{worst_sym} ({sym_w.get(worst_sym, 1.0):.2f}x)",
        }
