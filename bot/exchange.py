"""Exchange connector for Bybit via ccxt."""

import ccxt
import pandas as pd
import logging
from typing import Optional

import config

logger = logging.getLogger(__name__)


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

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "4h",
        limit: int = 200,
    ) -> pd.DataFrame:
        """Fetch OHLCV candles and return as DataFrame."""
        try:
            data = self.client.fetch_ohlcv(symbol, timeframe, limit=limit)
        except Exception as e:
            logger.error("Failed to fetch %s %s: %s", symbol, timeframe, e)
            return pd.DataFrame()

        df = pd.DataFrame(
            data, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

    def fetch_ticker(self, symbol: str) -> Optional[dict]:
        """Fetch current ticker for a symbol."""
        try:
            return self.client.fetch_ticker(symbol)
        except Exception as e:
            logger.error("Failed to fetch ticker %s: %s", symbol, e)
            return None

    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        ticker = self.fetch_ticker(symbol)
        if ticker and ticker.get("last"):
            return float(ticker["last"])
        return 0.0

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
