import os
from dotenv import load_dotenv

load_dotenv()

# --- Exchange ---
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")

# --- Trading mode ---
TRADING_MODE = os.getenv("TRADING_MODE", "paper")  # "paper" or "live"
PAPER_DEPOSIT = float(os.getenv("PAPER_DEPOSIT", "150"))

# --- Risk management ---
LEVERAGE = int(os.getenv("LEVERAGE", "5"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.08"))
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "3"))
MIN_RR_RATIO = float(os.getenv("MIN_RR_RATIO", "3.0"))

# --- Top-20 coins to scan ---
SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "ADA/USDT", "AVAX/USDT", "DOGE/USDT", "LINK/USDT", "DOT/USDT",
    "NEAR/USDT", "SUI/USDT", "TON/USDT", "MATIC/USDT", "UNI/USDT",
    "ATOM/USDT", "FIL/USDT", "APT/USDT", "ARB/USDT", "OP/USDT",
]

# --- Timeframes (from global to entry) ---
TIMEFRAMES = {
    "global": "1w",   # weekly - global trend
    "mid": "1d",      # daily  - local trend
    "work": "4h",     # 4-hour - working timeframe
    "entry": "1h",    # 1-hour - entry timeframe
}

# --- Wave analysis parameters ---
SWING_LOOKBACK = 5         # bars each side to confirm swing
MIN_SWING_PCT = 0.02       # minimum swing size as fraction of price (2%)

# --- Fibonacci levels ---
FIB_RETRACEMENT = [0.236, 0.382, 0.5, 0.618, 0.786]
FIB_EXTENSION = [1.0, 1.618, 2.618, 3.618]

# --- Logging ---
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
