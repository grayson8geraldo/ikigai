import os
from dotenv import load_dotenv

load_dotenv()

# --- Exchange ---
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")

# --- Trading mode ---
TRADING_MODE = os.getenv("TRADING_MODE", "paper")  # "paper" or "live"
PAPER_DEPOSIT = float(os.getenv("PAPER_DEPOSIT", "150"))

# --- Trading profile ---
# "intraday" = 4h/1h/15m with tight parameters (trades close within hours)
# "swing"    = 1d/4h/1h with wider parameters (trades last days)
TRADING_PROFILE = os.getenv("TRADING_PROFILE", "intraday")

# --- Risk management (base values, overridden by profile below) ---
LEVERAGE = int(os.getenv("LEVERAGE", "5"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.08"))
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "3"))

# --- BTC correlation filter ---
BTC_TREND_FILTER = os.getenv("BTC_TREND_FILTER", "true").lower() in ("true", "1", "yes")

# --- Top-20 coins to scan ---
SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "ADA/USDT", "AVAX/USDT", "DOGE/USDT", "LINK/USDT", "DOT/USDT",
    "NEAR/USDT", "SUI/USDT", "TON/USDT", "POL/USDT", "UNI/USDT",
    "ATOM/USDT", "FIL/USDT", "APT/USDT", "ARB/USDT", "OP/USDT",
]

# --- Fibonacci levels ---
FIB_RETRACEMENT = [0.236, 0.382, 0.5, 0.618, 0.786]
FIB_EXTENSION = [1.0, 1.618, 2.618, 3.618]

# ---------------------------------------------------------------------------
# Profile-specific parameters
# ---------------------------------------------------------------------------

if TRADING_PROFILE == "intraday":
    # Timeframes: 4h for trend, 1h for patterns, 15m for entry
    TIMEFRAMES = {
        "trend": "4h",
        "work": "1h",
        "entry": "15m",
    }
    SWING_LOOKBACK = int(os.getenv("SWING_LOOKBACK", "3"))
    MIN_SWING_PCT = float(os.getenv("MIN_SWING_PCT", "0.005"))       # 0.5%
    MIN_STOP_DISTANCE_PCT = float(os.getenv("MIN_STOP_DISTANCE_PCT", "0.005"))  # 0.5%
    SIGNAL_COOLDOWN_HOURS = int(os.getenv("SIGNAL_COOLDOWN_HOURS", "1"))
    MIN_RR_RATIO = float(os.getenv("MIN_RR_RATIO", "2.0"))
    MAX_POSITION_AGE_HOURS = int(os.getenv("MAX_POSITION_AGE_HOURS", "12"))
    DEFAULT_SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "5"))
    ENTRY_PRICE_MAX_DEVIATION = float(os.getenv("ENTRY_PRICE_MAX_DEVIATION", "0.03"))  # 3%
    # Fibonacci target ratios for intraday (closer targets)
    TARGET_RATIOS_ZIGZAG = [0.618, 1.0, 1.618]
    TARGET_RATIOS_TRIANGLE = [0.618, 1.0]
else:
    # Swing profile (original)
    TIMEFRAMES = {
        "trend": "1d",
        "work": "4h",
        "entry": "1h",
    }
    SWING_LOOKBACK = int(os.getenv("SWING_LOOKBACK", "5"))
    MIN_SWING_PCT = float(os.getenv("MIN_SWING_PCT", "0.02"))        # 2%
    MIN_STOP_DISTANCE_PCT = float(os.getenv("MIN_STOP_DISTANCE_PCT", "0.015"))  # 1.5%
    SIGNAL_COOLDOWN_HOURS = int(os.getenv("SIGNAL_COOLDOWN_HOURS", "4"))
    MIN_RR_RATIO = float(os.getenv("MIN_RR_RATIO", "3.0"))
    MAX_POSITION_AGE_HOURS = int(os.getenv("MAX_POSITION_AGE_HOURS", "48"))
    DEFAULT_SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "15"))
    ENTRY_PRICE_MAX_DEVIATION = float(os.getenv("ENTRY_PRICE_MAX_DEVIATION", "0.05"))  # 5%
    TARGET_RATIOS_ZIGZAG = [1.618, 2.618, 3.618]
    TARGET_RATIOS_TRIANGLE = [1.0, 1.618]

# --- Logging ---
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
