#!/usr/bin/env python3
"""
Quick test of real-time streaming data
Run this to see millisecond-level price updates!
"""

import asyncio
import websockets
import json
from datetime import datetime


async def test_binance_stream():
    """
    Test Binance WebSocket - Completely FREE!
    No API key needed!
    """
    print("=" * 70)
    print("🚀 TESTING BINANCE REAL-TIME STREAM (FREE)")
    print("=" * 70)
    print("Connecting to Bitcoin price stream...")
    print("Updates will show every ~100-500ms")
    print("Press Ctrl+C to stop")
    print("=" * 70)
    print()

    # Binance WebSocket for Bitcoin trades
    uri = "wss://stream.binance.com:9443/ws/btcusdt@aggTrade"

    tick_count = 0
    start_time = datetime.now()

    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected! Streaming Bitcoin trades...\n")

            while True:
                message = await websocket.recv()
                data = json.loads(message)

                tick_count += 1
                elapsed = (datetime.now() - start_time).total_seconds()

                # Parse trade data
                price = float(data['p'])
                quantity = float(data['q'])
                timestamp = data['T']
                is_buyer = not data['m']  # m = is buyer maker

                # Color code buys vs sells
                direction = "🟢 BUY " if is_buyer else "🔴 SELL"

                print(f"{direction} | ${price:,.2f} | {quantity:.6f} BTC | "
                      f"Tick #{tick_count} | {elapsed:.1f}s")

                # Show stats every 50 ticks
                if tick_count % 50 == 0:
                    ticks_per_sec = tick_count / elapsed
                    print(f"\n📊 STATS: {tick_count} ticks in {elapsed:.1f}s "
                          f"({ticks_per_sec:.1f} ticks/second)\n")

    except KeyboardInterrupt:
        print(f"\n\n{'=' * 70}")
        print("⏹️  Stopped stream")
        print(f"Total ticks received: {tick_count}")
        print(f"Time elapsed: {elapsed:.1f} seconds")
        print(f"Average: {tick_count/elapsed:.1f} ticks/second")
        print("=" * 70)


async def test_multiple_symbols():
    """
    Test multiple crypto symbols at once
    """
    print("=" * 70)
    print("🚀 TESTING MULTIPLE SYMBOLS")
    print("=" * 70)
    print("Streaming: Bitcoin, Ethereum, Solana")
    print("Press Ctrl+C to stop")
    print("=" * 70)
    print()

    symbols = {
        "btcusdt": "Bitcoin",
        "ethusdt": "Ethereum",
        "solusdt": "Solana"
    }

    prices = {}

    async def stream_symbol(symbol, name):
        """Stream one symbol"""
        uri = f"wss://stream.binance.com:9443/ws/{symbol}@aggTrade"

        async with websockets.connect(uri) as websocket:
            while True:
                message = await websocket.recv()
                data = json.loads(message)
                price = float(data['p'])

                # Update price
                prices[name] = price

                # Print all prices in one line
                line = " | ".join([
                    f"{name}: ${p:,.2f}"
                    for name, p in prices.items()
                ])
                print(f"\r{line}", end="", flush=True)

    # Stream all symbols concurrently
    tasks = [stream_symbol(symbol, name) for symbol, name in symbols.items()]
    await asyncio.gather(*tasks)


async def test_yahoo_fallback():
    """
    Fallback to Yahoo Finance if you don't want WebSocket
    Slower but still works
    """
    print("=" * 70)
    print("📊 TESTING YAHOO FINANCE (NO WEBSOCKET)")
    print("=" * 70)
    print("Polling stocks every 2 seconds...")
    print("Press Ctrl+C to stop")
    print("=" * 70)
    print()

    try:
        import yfinance as yf

        tickers = ["AAPL", "TSLA", "NVDA"]

        while True:
            prices = {}

            for ticker in tickers:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    prices[ticker] = info.get('regularMarketPrice', 0)
                except:
                    prices[ticker] = 0

            line = " | ".join([
                f"{ticker}: ${price:.2f}"
                for ticker, price in prices.items()
            ])

            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {line}")

            await asyncio.sleep(2)

    except ImportError:
        print("❌ yfinance not installed")
        print("Install: pip install yfinance")
    except KeyboardInterrupt:
        print("\n⏹️  Stopped")


def main():
    """Main menu"""
    print("\n" + "=" * 70)
    print("🎯 REAL-TIME DATA STREAMING TEST")
    print("=" * 70)
    print()
    print("Choose a test:")
    print()
    print("1. Binance Bitcoin Stream (FREE, Millisecond updates)")
    print("2. Multiple Crypto Symbols (FREE)")
    print("3. Yahoo Finance Stocks (FREE, 2-second polling)")
    print()

    choice = input("Enter choice (1-3): ").strip()

    if choice == "1":
        asyncio.run(test_binance_stream())
    elif choice == "2":
        asyncio.run(test_multiple_symbols())
    elif choice == "3":
        asyncio.run(test_yahoo_fallback())
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()
