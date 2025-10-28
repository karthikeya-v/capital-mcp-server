#!/usr/bin/env python3
"""
Real-Time Market Data Streaming
Supports multiple data providers with WebSocket connections
"""

import asyncio
import json
import logging
from typing import Callable, Dict, List, Optional
from datetime import datetime
import websockets

logger = logging.getLogger(__name__)


class MarketDataStream:
    """
    Unified interface for real-time market data streaming
    Supports: Capital.com, Binance, Polygon, Alpaca, TwelveData
    """

    def __init__(self, provider: str, api_key: Optional[str] = None):
        self.provider = provider.lower()
        self.api_key = api_key
        self.subscriptions = {}
        self.callbacks = {}
        self.ws = None
        self.running = False

    async def connect(self):
        """Connect to WebSocket based on provider"""
        if self.provider == "binance":
            await self._connect_binance()
        elif self.provider == "polygon":
            await self._connect_polygon()
        elif self.provider == "alpaca":
            await self._connect_alpaca()
        elif self.provider == "twelvedata":
            await self._connect_twelvedata()
        elif self.provider == "capital":
            await self._connect_capital()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        self.running = True
        logger.info(f"✅ Connected to {self.provider} WebSocket")

    # ========== BINANCE (CRYPTO - FREE) ==========

    async def _connect_binance(self):
        """Binance WebSocket - Free real-time crypto data"""
        # Binance uses multiple streams, one per symbol
        # We'll use the aggregate trade stream for real-time trades
        pass  # Will be implemented when subscribing

    async def subscribe_binance(self, symbol: str, callback: Callable):
        """Subscribe to Binance stream"""
        # Convert symbol format: BTCUSDT, ETHUSDT, etc.
        symbol = symbol.upper().replace("/", "")

        uri = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@aggTrade"

        async def stream_handler():
            async with websockets.connect(uri) as ws:
                self.ws = ws
                logger.info(f"📊 Subscribed to Binance {symbol}")

                while self.running:
                    try:
                        message = await ws.recv()
                        data = json.loads(message)

                        # Parse Binance tick
                        tick = {
                            "symbol": symbol,
                            "price": float(data['p']),
                            "quantity": float(data['q']),
                            "timestamp": data['T'],
                            "is_buyer_maker": data['m'],
                            "provider": "binance"
                        }

                        await callback(tick)

                    except Exception as e:
                        logger.error(f"Binance stream error: {e}")
                        await asyncio.sleep(1)

        asyncio.create_task(stream_handler())

    # ========== POLYGON.IO (STOCKS - PAID) ==========

    async def _connect_polygon(self):
        """Polygon WebSocket - Real-time stock data"""
        if not self.api_key:
            raise ValueError("Polygon requires API key")

        uri = "wss://socket.polygon.io/stocks"

        async with websockets.connect(uri) as ws:
            self.ws = ws

            # Authenticate
            await ws.send(json.dumps({
                "action": "auth",
                "params": self.api_key
            }))

            auth_response = await ws.recv()
            logger.info(f"Polygon auth: {auth_response}")

    async def subscribe_polygon(self, symbol: str, callback: Callable):
        """Subscribe to Polygon stream"""
        if not self.ws:
            await self._connect_polygon()

        # Subscribe to trades
        await self.ws.send(json.dumps({
            "action": "subscribe",
            "params": f"T.{symbol}"  # T = Trades
        }))

        logger.info(f"📊 Subscribed to Polygon {symbol}")

        # Handle messages
        async def message_handler():
            while self.running:
                try:
                    message = await self.ws.recv()
                    data = json.loads(message)

                    if data[0].get('ev') == 'T':  # Trade event
                        for trade in data:
                            tick = {
                                "symbol": trade['sym'],
                                "price": trade['p'],
                                "quantity": trade['s'],
                                "timestamp": trade['t'],
                                "exchange": trade['x'],
                                "provider": "polygon"
                            }
                            await callback(tick)

                except Exception as e:
                    logger.error(f"Polygon stream error: {e}")
                    await asyncio.sleep(1)

        asyncio.create_task(message_handler())

    # ========== ALPACA (STOCKS - FREE) ==========

    async def _connect_alpaca(self):
        """Alpaca WebSocket - Free IEX real-time data"""
        if not self.api_key:
            raise ValueError("Alpaca requires API key")

        # Split key (format: "KEY:SECRET")
        key_id, secret = self.api_key.split(":")

        uri = "wss://stream.data.alpaca.markets/v2/iex"

        async with websockets.connect(uri) as ws:
            self.ws = ws

            # Authenticate
            await ws.send(json.dumps({
                "action": "auth",
                "key": key_id,
                "secret": secret
            }))

            auth_response = await ws.recv()
            logger.info(f"Alpaca auth: {auth_response}")

    async def subscribe_alpaca(self, symbol: str, callback: Callable):
        """Subscribe to Alpaca IEX stream"""
        if not self.ws:
            await self._connect_alpaca()

        # Subscribe to trades
        await self.ws.send(json.dumps({
            "action": "subscribe",
            "trades": [symbol]
        }))

        logger.info(f"📊 Subscribed to Alpaca {symbol}")

        # Handle messages
        async def message_handler():
            while self.running:
                try:
                    message = await self.ws.recv()
                    data = json.loads(message)

                    if data[0].get('T') == 't':  # Trade
                        for trade in data:
                            tick = {
                                "symbol": trade['S'],
                                "price": trade['p'],
                                "quantity": trade['s'],
                                "timestamp": trade['t'],
                                "provider": "alpaca"
                            }
                            await callback(tick)

                except Exception as e:
                    logger.error(f"Alpaca stream error: {e}")
                    await asyncio.sleep(1)

        asyncio.create_task(message_handler())

    # ========== TWELVEDATA (MULTI-ASSET - PAID) ==========

    async def _connect_twelvedata(self):
        """TwelveData WebSocket - Forex, stocks, crypto"""
        if not self.api_key:
            raise ValueError("TwelveData requires API key")

        uri = f"wss://ws.twelvedata.com/v1/quotes/price?apikey={self.api_key}"

        self.ws = await websockets.connect(uri)
        logger.info("Connected to TwelveData")

    async def subscribe_twelvedata(self, symbol: str, callback: Callable):
        """Subscribe to TwelveData stream"""
        if not self.ws:
            await self._connect_twelvedata()

        # Subscribe
        await self.ws.send(json.dumps({
            "action": "subscribe",
            "params": {"symbols": symbol}
        }))

        logger.info(f"📊 Subscribed to TwelveData {symbol}")

        # Handle messages
        async def message_handler():
            while self.running:
                try:
                    message = await self.ws.recv()
                    data = json.loads(message)

                    if data.get('event') == 'price':
                        tick = {
                            "symbol": data['symbol'],
                            "price": float(data['price']),
                            "timestamp": data['timestamp'],
                            "provider": "twelvedata"
                        }
                        await callback(tick)

                except Exception as e:
                    logger.error(f"TwelveData stream error: {e}")
                    await asyncio.sleep(1)

        asyncio.create_task(message_handler())

    # ========== CAPITAL.COM (YOUR BROKER) ==========

    async def _connect_capital(self):
        """Capital.com WebSocket - Check their API docs"""
        # NOTE: Check Capital.com API documentation for WebSocket endpoint
        # This is a placeholder implementation

        if not self.api_key:
            raise ValueError("Capital.com requires API credentials")

        # Example (verify with their docs):
        # uri = "wss://api-streaming-capital.backend-capital.com/..."

        logger.warning("Capital.com WebSocket not yet implemented - check API docs")
        raise NotImplementedError("Capital.com WebSocket endpoint needed")

    # ========== UNIFIED INTERFACE ==========

    async def subscribe(self, symbols: List[str], callback: Callable):
        """
        Subscribe to multiple symbols with unified callback

        Args:
            symbols: List of trading symbols
            callback: async function(tick_data) called on each tick
        """
        for symbol in symbols:
            if self.provider == "binance":
                await self.subscribe_binance(symbol, callback)
            elif self.provider == "polygon":
                await self.subscribe_polygon(symbol, callback)
            elif self.provider == "alpaca":
                await self.subscribe_alpaca(symbol, callback)
            elif self.provider == "twelvedata":
                await self.subscribe_twelvedata(symbol, callback)

    async def close(self):
        """Close WebSocket connection"""
        self.running = False
        if self.ws:
            await self.ws.close()
        logger.info(f"Closed {self.provider} WebSocket")


# ========== EXAMPLE USAGE ==========

async def on_tick(tick):
    """Callback for each price tick"""
    print(f"🔔 {tick['symbol']}: ${tick['price']:.2f} @ {tick.get('quantity', 0)}")


async def example_binance():
    """Example: Stream Bitcoin price from Binance (FREE)"""
    stream = MarketDataStream("binance")
    await stream.subscribe(["BTCUSDT", "ETHUSDT"], on_tick)

    # Keep running
    while True:
        await asyncio.sleep(1)


async def example_alpaca():
    """Example: Stream stocks from Alpaca (FREE)"""
    stream = MarketDataStream(
        "alpaca",
        api_key="YOUR_KEY:YOUR_SECRET"
    )
    await stream.subscribe(["AAPL", "TSLA", "NVDA"], on_tick)

    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    # Test Binance (FREE)
    print("Testing Binance WebSocket (FREE crypto data)...")
    asyncio.run(example_binance())

    # Uncomment to test others:
    # asyncio.run(example_alpaca())
