#!/usr/bin/env python3
"""IKIGAI Trading Bot — Elliott Wave + Fibonacci strategy.

Usage:
    python main.py              # Run one scan cycle
    python main.py --loop       # Run continuously (scan every N minutes)
    python main.py --status     # Show current positions and stats
"""

import argparse
import logging
import sys
import time
from datetime import datetime, date

import config
from bot.exchange import Exchange
from bot.scanner import Scanner
from bot.trader import Trader
from bot.risk_manager import RiskManager

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            f"{config.LOG_DIR}/bot_{datetime.now():%Y%m%d}.log"
        ),
    ],
)
logger = logging.getLogger("ikigai")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_banner():
    profile = config.TRADING_PROFILE.upper()
    tfs = "/".join(config.TIMEFRAMES.values())
    print("""
╔══════════════════════════════════════════════════╗
║          IKIGAI Trading Bot v2.0                 ║
║    Elliott Wave + Fibonacci Strategy             ║
║                                                  ║
║    Mode: {mode:<10s}  Deposit: ${deposit:<10.2f}  ║
║    Profile: {profile:<8s}  TFs: {tfs:<16s}    ║
╚══════════════════════════════════════════════════╝
""".format(
        mode=config.TRADING_MODE.upper(),
        deposit=config.PAPER_DEPOSIT,
        profile=profile,
        tfs=tfs,
    ))


def print_signals(signals):
    if not signals:
        print("\n  No signals found.\n")
        return

    print(f"\n{'='*70}")
    print(f"  SIGNALS FOUND: {len(signals)}")
    print(f"{'='*70}")

    for i, sig in enumerate(signals, 1):
        dir_icon = "LONG" if sig.direction.value == "long" else "SHORT"
        print(f"\n  [{i}] {sig.symbol} | {dir_icon} | {sig.setup_type.value}")
        print(f"      Timeframe: {sig.timeframe}")
        print(f"      Entry:     {sig.entry_price:.4f}")
        print(f"      Stop:      {sig.stop_loss:.4f} ({sig.risk_pct:.2%} risk)")
        print(f"      R:R:       {sig.rr_ratio:.1f}:1")
        print(f"      Confidence: {sig.confidence:.0%}")
        print(f"      Factors:")
        for f in sig.factors:
            print(f"        - {f}")
        if sig.targets:
            print(f"      Targets:")
            for t in sig.targets:
                print(f"        - {t.label}: {t.price:.4f}")


def print_status(trader: Trader):
    positions = trader.get_open_positions()
    stats = trader.get_stats()

    print(f"\n{'='*70}")
    print(f"  BALANCE: ${trader.get_balance():.2f}")
    print(f"{'='*70}")

    print(f"\n  Open Positions: {len(positions)}")
    for p in positions:
        dir_icon = "LONG" if p.direction.value == "long" else "SHORT"
        print(f"    [{p.id}] {p.symbol} {dir_icon} @ {p.entry_price:.4f}"
              f" | SL: {p.stop_loss:.4f} | TP: {p.take_profit:.4f}"
              f" | Margin: ${p.margin:.2f} x{p.leverage}")

    print(f"\n  Statistics:")
    print(f"    Total trades:  {stats['total']}")
    print(f"    Wins:          {stats['wins']}")
    print(f"    Losses:        {stats['losses']}")
    print(f"    Win rate:      {stats['win_rate']:.1f}%")
    print(f"    Total PnL:     ${stats['total_pnl']:.2f}")
    if stats['total'] > 0:
        print(f"    Avg win:       ${stats['avg_win']:.2f}")
        print(f"    Avg loss:      ${stats['avg_loss']:.2f}")
    print()


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def run_scan_cycle(scanner: Scanner, trader: Trader, auto_trade: bool = False):
    """Run one full scan cycle: scan -> signal -> optionally trade."""
    logger.info("Starting scan cycle at %s", datetime.now().strftime("%Y-%m-%d %H:%M"))

    # Check open positions first
    trader.check_positions()

    # Scan for new signals
    signals = scanner.scan_all()

    # Show BTC trend if filter is active
    if config.BTC_TREND_FILTER:
        btc_trend = scanner._btc_trend.value.upper()
        print(f"\n  BTC trend: {btc_trend}  (altcoin filter: {'active' if btc_trend != 'SIDEWAYS' else 'off'})")

    print_signals(signals)

    if not signals:
        return

    if auto_trade:
        # Auto-execute the best signal(s)
        for signal in signals:
            can, reason = trader.rm.can_trade()
            if not can:
                logger.info("Cannot open more trades: %s", reason)
                break

            logger.info(
                "Auto-trading: %s %s %s",
                signal.direction.value,
                signal.symbol,
                signal.setup_type.value,
            )
            position = trader.open_position(signal)
            if position:
                logger.info("Position opened: %s", position.id)
    else:
        # Manual mode: show signals and let user decide
        print("\n  Auto-trade is OFF. Use --trade to enable auto-execution.")
        print("  Showing signals only.\n")

    print_status(trader)


def main():
    parser = argparse.ArgumentParser(description="IKIGAI Trading Bot")
    parser.add_argument("--loop", action="store_true",
                        help="Run continuously (scan every 15 minutes)")
    parser.add_argument("--interval", type=int, default=config.DEFAULT_SCAN_INTERVAL,
                        help=f"Scan interval in minutes (default: {config.DEFAULT_SCAN_INTERVAL})")
    parser.add_argument("--status", action="store_true",
                        help="Show current positions and statistics")
    parser.add_argument("--trade", action="store_true",
                        help="Enable auto-trading (execute signals automatically)")
    args = parser.parse_args()

    print_banner()

    # Initialize components
    exchange = Exchange()
    risk_manager = RiskManager(balance=config.PAPER_DEPOSIT)
    trader = Trader(exchange, risk_manager)
    scanner = Scanner(exchange)

    if args.status:
        print_status(trader)
        return

    if args.loop:
        logger.info("Starting continuous scanning (every %d min)...", args.interval)
        last_reset_date = date.today()
        while True:
            try:
                # Daily reset of risk manager counters
                today = date.today()
                if today != last_reset_date:
                    risk_manager.reset_daily()
                    last_reset_date = today
                    logger.info("Daily reset: losses and trade counters cleared")

                run_scan_cycle(scanner, trader, auto_trade=args.trade)
                logger.info("Next scan in %d minutes...", args.interval)
                time.sleep(args.interval * 60)
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                print_status(trader)
                break
            except Exception as e:
                logger.error("Error in scan cycle: %s", e, exc_info=True)
                time.sleep(60)  # wait 1 min on error
    else:
        run_scan_cycle(scanner, trader, auto_trade=args.trade)


if __name__ == "__main__":
    main()
