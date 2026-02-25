"""Exchange connector for Bybit via ccxt."""

import ccxt
import pandas as pd
import logging
import time
from typing import Optional

import config

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
BACKOFF_BASE = 2  # seconds


class Exchange:
    """Bybit exchange connector."""

    def __init__(self):
        params = {
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},  # perpetual futures
        }
        if config.BYBIT_API_KEY:
            params["apiKey"] = config.BYBIT_API_KEY
            params["secret"] = config.BYBIT_API_SECRET

        self.client = ccxt.bybit(params)
        # Price cache: symbol -> (price, timestamp)
        self._price_cache: dict[str, tuple[float, float]] = {}
        self._price_cache_ttl = 10.0  # seconds

    def _retry(self, func, description: str):
        """Execute a function with retry and exponential backoff on network errors."""
        for attempt in range(MAX_RETRIES):
            try:
                return func()
            except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable) as e:
                wait = BACKOFF_BASE ** (attempt + 1)
                logger.warning(
                    "%s failed (attempt %d/%d): %s — retrying in %ds",
                    description, attempt + 1, MAX_RETRIES, e, wait,
                )
                time.sleep(wait)
            except ccxt.BadSymbol as e:
                logger.error("%s: invalid symbol — %s", description, e)
                return None
            except Exception as e:
                logger.error("%s failed: %s", description, e)
                return None
        logger.error("%s failed after %d retries", description, MAX_RETRIES)
        return None

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "4h",
        limit: int = 200,
    ) -> pd.DataFrame:
        """Fetch OHLCV candles and return as DataFrame."""
        data = self._retry(
            lambda: self.client.fetch_ohlcv(symbol, timeframe, limit=limit),
            f"fetch_ohlcv({symbol} {timeframe})",
        )
        if data is None:
            return pd.DataFrame()

        df = pd.DataFrame(
            data, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

    def fetch_ticker(self, symbol: str) -> Optional[dict]:
        """Fetch current ticker for a symbol."""
        return self._retry(
            lambda: self.client.fetch_ticker(symbol),
            f"fetch_ticker({symbol})",
        )

    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol (cached with TTL)."""
        cached = self._price_cache.get(symbol)
        if cached and time.time() - cached[1] < self._price_cache_ttl:
            return cached[0]

        ticker = self.fetch_ticker(symbol)
        if ticker and ticker.get("last"):
            price = float(ticker["last"])
            self._price_cache[symbol] = (price, time.time())
            return price
        return 0.0

    def clear_price_cache(self):
        """Clear the price cache (call at the start of each scan cycle)."""
        self._price_cache.clear()

    # --- Order methods (only used in live mode) ---

    def create_limit_order(
        self, symbol: str, side: str, amount: float, price: float, params: dict = None
    ) -> Optional[dict]:
        """Place a limit order. side: 'buy' or 'sell'."""
        if config.TRADING_MODE != "live":
            logger.info("[PAPER] Limit %s %s %.6f @ %.2f", side, symbol, amount, price)
            return {"id": "paper", "status": "open"}
        try:
            return self.client.create_limit_order(
                symbol, side, amount, price, params=params or {}
            )
        except Exception as e:
            logger.error("Order failed: %s", e)
            return None

    def create_market_order(
        self, symbol: str, side: str, amount: float, params: dict = None
    ) -> Optional[dict]:
        """Place a market order."""
        if config.TRADING_MODE != "live":
            logger.info("[PAPER] Market %s %s %.6f", side, symbol, amount)
            return {"id": "paper", "status": "filled"}
        try:
            return self.client.create_market_order(
                symbol, side, amount, params=params or {}
            )
        except Exception as e:
            logger.error("Order failed: %s", e)
            return None

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol."""
        if config.TRADING_MODE != "live":
            return True
        try:
            self.client.set_leverage(leverage, symbol)
            return True
        except Exception as e:
            logger.error("Set leverage failed: %s", e)
            return False
