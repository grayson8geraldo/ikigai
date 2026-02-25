#!/usr/bin/env python3
"""Backtesting engine — replay strategy on historical OHLCV data.

Usage:
    python -m bot.backtester                        # backtest all symbols, last 6 months
    python -m bot.backtester --symbol BTC/USDT      # single symbol
    python -m bot.backtester --months 12            # last 12 months
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

import config
from bot.exchange import Exchange
from bot.wave_analyzer import analyze
from bot.setups import check_zigzag_setup, check_diagonal_setup, check_triangle_setup
from bot.risk_manager import RiskManager
from bot.models import Signal, Direction

logger = logging.getLogger(__name__)

MAX_CANDLES_PER_REQUEST = 200


@dataclass
class BacktestTrade:
    """Record of a single backtest trade."""
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_bar: int
    exit_bar: int = 0
    exit_price: float = 0.0
    pnl_pct: float = 0.0
    result: str = ""  # "win", "loss", "timeout"


@dataclass
class BacktestResult:
    """Aggregated backtest results."""
    symbol: str
    timeframe: str
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    timeouts: int = 0
    total_pnl_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    trades: list = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return (self.wins / self.total_trades * 100) if self.total_trades > 0 else 0.0

    @property
    def avg_pnl_pct(self) -> float:
        return (self.total_pnl_pct / self.total_trades) if self.total_trades > 0 else 0.0

    @property
    def profit_factor(self) -> float:
        gross_wins = sum(t.pnl_pct for t in self.trades if t.pnl_pct > 0)
        gross_losses = abs(sum(t.pnl_pct for t in self.trades if t.pnl_pct < 0))
        return (gross_wins / gross_losses) if gross_losses > 0 else float("inf")


def fetch_historical_data(
    exchange: Exchange,
    symbol: str,
    timeframe: str,
    months: int = 6,
) -> pd.DataFrame:
    """Fetch historical OHLCV data by paginating backwards."""
    all_data = []
    now = int(time.time() * 1000)
    start_ms = int((datetime.utcnow() - timedelta(days=months * 30)).timestamp() * 1000)

    # Estimate milliseconds per candle
    tf_minutes = {
        "1m": 1, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "2h": 120, "4h": 240,
        "1d": 1440, "1w": 10080,
    }
    minutes = tf_minutes.get(timeframe, 240)
    ms_per_candle = minutes * 60 * 1000

    since = start_ms
    while since < now:
        data = exchange._retry(
            lambda s=since: exchange.client.fetch_ohlcv(
                symbol, timeframe, since=s, limit=MAX_CANDLES_PER_REQUEST
            ),
            f"fetch_historical({symbol} {timeframe})",
        )
        if not data:
            break
        all_data.extend(data)
        # Move forward past last candle
        since = data[-1][0] + ms_per_candle
        if len(data) < MAX_CANDLES_PER_REQUEST:
            break
        time.sleep(0.3)  # rate limit

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(
        all_data, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def _simulate_trade(
    df: pd.DataFrame,
    signal: Signal,
    entry_bar: int,
    max_bars: int = 50,
) -> BacktestTrade:
    """Simulate a trade forward from entry_bar using OHLCV candles."""
    trade = BacktestTrade(
        symbol=signal.symbol,
        direction=signal.direction.value,
        entry_price=signal.entry_price,
        stop_loss=signal.stop_loss,
        take_profit=signal.targets[0].price if signal.targets else 0.0,
        entry_bar=entry_bar,
    )

    for i in range(entry_bar + 1, min(entry_bar + max_bars + 1, len(df))):
        high = df.iloc[i]["high"]
        low = df.iloc[i]["low"]

        if signal.direction == Direction.LONG:
            # Check stop first (conservative: assume stop hit before TP on same bar)
            if low <= signal.stop_loss:
                trade.exit_bar = i
                trade.exit_price = signal.stop_loss
                trade.pnl_pct = (trade.exit_price - trade.entry_price) / trade.entry_price * 100
                trade.result = "loss"
                return trade
            if trade.take_profit > 0 and high >= trade.take_profit:
                trade.exit_bar = i
                trade.exit_price = trade.take_profit
                trade.pnl_pct = (trade.exit_price - trade.entry_price) / trade.entry_price * 100
                trade.result = "win"
                return trade
        else:  # SHORT
            if high >= signal.stop_loss:
                trade.exit_bar = i
                trade.exit_price = signal.stop_loss
                trade.pnl_pct = (trade.entry_price - trade.exit_price) / trade.entry_price * 100
                trade.result = "loss"
                return trade
            if trade.take_profit > 0 and low <= trade.take_profit:
                trade.exit_bar = i
                trade.exit_price = trade.take_profit
                trade.pnl_pct = (trade.entry_price - trade.exit_price) / trade.entry_price * 100
                trade.result = "win"
                return trade

    # Timeout — close at last bar's close
    last_idx = min(entry_bar + max_bars, len(df) - 1)
    trade.exit_bar = last_idx
    trade.exit_price = df.iloc[last_idx]["close"]
    if signal.direction == Direction.LONG:
        trade.pnl_pct = (trade.exit_price - trade.entry_price) / trade.entry_price * 100
    else:
        trade.pnl_pct = (trade.entry_price - trade.exit_price) / trade.entry_price * 100
    trade.result = "timeout"
    return trade


def backtest_symbol(
    exchange: Exchange,
    symbol: str,
    timeframe: str = "4h",
    months: int = 6,
    window: int = 200,
) -> BacktestResult:
    """Run backtest for a single symbol by sliding a window over historical data."""
    result = BacktestResult(symbol=symbol, timeframe=timeframe)

    logger.info("Fetching historical data for %s %s (%d months)...", symbol, timeframe, months)
    df = fetch_historical_data(exchange, symbol, timeframe, months)
    if len(df) < window + 50:
        logger.warning("Not enough data for %s: %d bars", symbol, len(df))
        return result

    logger.info("Got %d bars for %s. Running backtest...", len(df), symbol)

    # Also fetch daily for trend detection
    df_daily = fetch_historical_data(exchange, symbol, "1d", months)

    last_trade_bar = -50  # avoid overlapping trades

    for start in range(0, len(df) - window, 10):  # slide by 10 bars
        end = start + window
        if end >= len(df):
            break
        if start < last_trade_bar + 20:  # minimum gap between trades
            continue

        chunk = df.iloc[start:end].reset_index(drop=True)
        analysis = analyze(chunk)

        # Get trend from daily data at this point in time
        chunk_time = chunk.iloc[-1]["timestamp"]
        daily_mask = df_daily["timestamp"] <= chunk_time
        if daily_mask.sum() < 50:
            trend = analysis["trend"]
        else:
            daily_chunk = df_daily[daily_mask].tail(200).reset_index(drop=True)
            daily_analysis = analyze(daily_chunk)
            trend = daily_analysis["trend"]

        current_price = float(chunk.iloc[-1]["close"])
        signals: list[Signal] = []

        for zigzag in analysis["zigzags"]:
            sig = check_zigzag_setup(zigzag, trend, current_price, symbol, timeframe)
            if sig:
                signals.append(sig)

        for diagonal in analysis["diagonals"]:
            sig = check_diagonal_setup(diagonal, current_price, symbol, timeframe)
            if sig:
                signals.append(sig)

        for triangle in analysis["triangles"]:
            sig = check_triangle_setup(triangle, trend, current_price, symbol, timeframe)
            if sig:
                signals.append(sig)

        if not signals:
            continue

        # Take the best signal
        best = max(signals, key=lambda s: s.confidence)

        # Simulate the trade on bars AFTER the analysis window
        trade = _simulate_trade(df, best, entry_bar=end - 1, max_bars=50)
        result.trades.append(trade)
        result.total_trades += 1

        if trade.result == "win":
            result.wins += 1
        elif trade.result == "loss":
            result.losses += 1
        else:
            result.timeouts += 1

        result.total_pnl_pct += trade.pnl_pct
        last_trade_bar = end

    # Calculate max drawdown
    if result.trades:
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for t in result.trades:
            cumulative += t.pnl_pct
            peak = max(peak, cumulative)
            dd = peak - cumulative
            max_dd = max(max_dd, dd)
        result.max_drawdown_pct = max_dd

    return result


def print_backtest_report(results: list[BacktestResult]):
    """Print a formatted backtest report."""
    print(f"\n{'='*80}")
    print(f"  BACKTEST REPORT")
    print(f"{'='*80}")

    total_trades = 0
    total_wins = 0
    total_losses = 0
    total_pnl = 0.0

    for r in results:
        if r.total_trades == 0:
            continue

        total_trades += r.total_trades
        total_wins += r.wins
        total_losses += r.losses
        total_pnl += r.total_pnl_pct

        print(f"\n  {r.symbol} ({r.timeframe})")
        print(f"    Trades: {r.total_trades}  |  Wins: {r.wins}  |  Losses: {r.losses}  |  Timeouts: {r.timeouts}")
        print(f"    Win rate: {r.win_rate:.1f}%  |  Avg PnL: {r.avg_pnl_pct:+.2f}%")
        print(f"    Total PnL: {r.total_pnl_pct:+.2f}%  |  Max DD: {r.max_drawdown_pct:.2f}%")
        print(f"    Profit factor: {r.profit_factor:.2f}")

    print(f"\n{'='*80}")
    print(f"  TOTALS")
    print(f"{'='*80}")
    print(f"    Total trades:   {total_trades}")
    print(f"    Total wins:     {total_wins}")
    print(f"    Total losses:   {total_losses}")
    win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
    print(f"    Win rate:       {win_rate:.1f}%")
    print(f"    Total PnL:      {total_pnl:+.2f}%")

    # Simulate with leverage and compounding
    if total_trades > 0:
        balance = config.PAPER_DEPOSIT
        rm = RiskManager(balance)
        for r in results:
            for t in r.trades:
                risk_pct = rm.calculate_position_size(
                    Signal(
                        symbol=t.symbol,
                        timeframe="4h",
                        setup_type=config,  # placeholder
                        direction=Direction(t.direction),
                        entry_price=t.entry_price,
                        stop_loss=t.stop_loss,
                        targets=[],
                        confidence=0.5,
                    )
                ) if False else None  # skip detailed sim for summary

        print(f"\n    Starting balance: ${config.PAPER_DEPOSIT:.2f}")
        # Simple estimate: apply avg trade PnL * leverage
        avg_leverage = 7  # approximate mid-range
        estimated_balance = config.PAPER_DEPOSIT
        for r in results:
            for t in r.trades:
                trade_impact = t.pnl_pct / 100 * avg_leverage * 0.08  # ~8% risk
                estimated_balance *= (1 + trade_impact)
        print(f"    Estimated final: ${estimated_balance:.2f} (with ~x{avg_leverage} leverage, 8% risk)")

    print()


def main():
    parser = argparse.ArgumentParser(description="IKIGAI Strategy Backtester")
    parser.add_argument("--symbol", type=str, default=None,
                        help="Single symbol to backtest (e.g. BTC/USDT)")
    parser.add_argument("--months", type=int, default=6,
                        help="How many months of history to test (default: 6)")
    parser.add_argument("--timeframe", type=str, default="4h",
                        help="Working timeframe (default: 4h)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    exchange = Exchange()
    symbols = [args.symbol] if args.symbol else config.SYMBOLS[:5]  # top-5 by default

    results = []
    for symbol in symbols:
        try:
            r = backtest_symbol(exchange, symbol, args.timeframe, args.months)
            results.append(r)
        except Exception as e:
            logger.error("Backtest failed for %s: %s", symbol, e, exc_info=True)
        time.sleep(1)

    print_backtest_report(results)


if __name__ == "__main__":
    main()
