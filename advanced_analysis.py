#!/usr/bin/env python3
"""
Advanced Technical Analysis for US Tech 100
Generates actionable trade recommendations
"""

import os
import asyncio
import json
from dotenv import load_dotenv
import httpx
from datetime import datetime

load_dotenv()

API_KEY = os.getenv("CAPITAL_API_KEY")
IDENTIFIER = os.getenv("CAPITAL_IDENTIFIER")
API_PASSWORD = os.getenv("CAPITAL_API_PASSWORD")
USE_DEMO = os.getenv("CAPITAL_USE_DEMO", "true").lower() == "true"
API_URL = "https://demo-api-capital.backend-capital.com" if USE_DEMO else "https://api-capital.backend-capital.com"

EPIC = "US100"  # US Tech 100


async def get_session():
    """Authenticate and get session tokens"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_URL}/api/v1/session",
            headers={"X-CAP-API-KEY": API_KEY, "Content-Type": "application/json"},
            json={"identifier": IDENTIFIER, "password": API_PASSWORD, "encryptedPassword": False}
        )
        if response.status_code == 200:
            return response.headers.get("X-SECURITY-TOKEN"), response.headers.get("CST")
        return None, None


async def get_price_history(security_token, cst_token, resolution, max_points):
    """Get historical price data"""
    headers = {"X-SECURITY-TOKEN": security_token, "CST": cst_token}
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{API_URL}/api/v1/prices/{EPIC}",
            headers=headers,
            params={"resolution": resolution, "max": max_points}
        )
        if response.status_code == 200:
            return response.json()
        return None


async def get_market_details(security_token, cst_token):
    """Get current market details"""
    headers = {"X-SECURITY-TOKEN": security_token, "CST": cst_token}
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/api/v1/markets/{EPIC}", headers=headers)
        if response.status_code == 200:
            return response.json()
        return None


def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    if len(prices) < period + 1:
        return None

    gains = []
    losses = []

    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))

    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period

    if avg_loss == 0:
        return 100

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices):
    """Calculate MACD (12, 26, 9)"""
    if len(prices) < 26:
        return None, None, None

    # Simple EMA calculation
    def ema(data, period):
        multiplier = 2 / (period + 1)
        ema_values = [sum(data[:period]) / period]
        for price in data[period:]:
            ema_values.append((price - ema_values[-1]) * multiplier + ema_values[-1])
        return ema_values[-1]

    ema_12 = ema(prices, 12)
    ema_26 = ema(prices, 26)
    macd_line = ema_12 - ema_26

    # Signal line (9-period EMA of MACD)
    # Simplified: using last 9 values
    signal_line = macd_line  # Simplified

    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    if len(prices) < period:
        return None, None, None

    recent_prices = prices[-period:]
    sma = sum(recent_prices) / period

    variance = sum((x - sma) ** 2 for x in recent_prices) / period
    std = variance ** 0.5

    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)

    return upper_band, sma, lower_band


def find_support_resistance(prices, window=20):
    """Find support and resistance levels"""
    highs = []
    lows = []

    for i in range(window, len(prices) - window):
        # Check if it's a local high
        if all(prices[i] >= prices[i-j] for j in range(1, window)) and \
           all(prices[i] >= prices[i+j] for j in range(1, window)):
            highs.append(prices[i])

        # Check if it's a local low
        if all(prices[i] <= prices[i-j] for j in range(1, window)) and \
           all(prices[i] <= prices[i+j] for j in range(1, window)):
            lows.append(prices[i])

    # Get recent support/resistance
    resistance = max(highs[-3:]) if highs else None
    support = min(lows[-3:]) if lows else None

    return support, resistance


async def main():
    """Main analysis function"""
    print("=" * 70)
    print("📊 ADVANCED US TECH 100 ANALYSIS & TRADE RECOMMENDATION")
    print("=" * 70)
    print(f"Mode: {'DEMO' if USE_DEMO else 'LIVE'} Account")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Authenticate
    print("🔐 Authenticating...")
    security_token, cst_token = await get_session()
    if not security_token:
        print("❌ Authentication failed")
        return

    print("✅ Connected to Capital.com")
    print()

    # Get current market data
    print("📈 Fetching current market data...")
    market_data = await get_market_details(security_token, cst_token)

    if not market_data:
        print("❌ Failed to fetch market data")
        return

    snapshot = market_data.get('snapshot', {})
    current_bid = snapshot.get('bid')
    current_offer = snapshot.get('offer')
    current_price = (current_bid + current_offer) / 2

    print(f"Current Price: {current_price:.2f}")
    print(f"Bid/Offer: {current_bid:.2f} / {current_offer:.2f}")
    print(f"Daily Change: {snapshot.get('netChange')} ({snapshot.get('percentageChange')}%)")
    print()

    # Fetch multiple timeframes
    print("📊 Fetching multi-timeframe data...")

    # 4-hour for trend
    data_4h = await get_price_history(security_token, cst_token, "HOUR_4", 100)
    # 1-hour for entry timing
    data_1h = await get_price_history(security_token, cst_token, "HOUR", 200)
    # 15-min for precise entry
    data_15m = await get_price_history(security_token, cst_token, "MINUTE_15", 200)

    if not all([data_4h, data_1h, data_15m]):
        print("❌ Failed to fetch historical data")
        return

    # Extract close prices
    closes_4h = [p['closePrice']['ask'] for p in data_4h['prices'] if 'closePrice' in p]
    closes_1h = [p['closePrice']['ask'] for p in data_1h['prices'] if 'closePrice' in p]
    closes_15m = [p['closePrice']['ask'] for p in data_15m['prices'] if 'closePrice' in p]

    print(f"✅ Retrieved {len(closes_4h)} 4H candles, {len(closes_1h)} 1H candles, {len(closes_15m)} 15M candles")
    print()

    # Calculate technical indicators
    print("🔬 TECHNICAL ANALYSIS")
    print("=" * 70)

    # Moving averages (1H timeframe)
    ma_20 = sum(closes_1h[-20:]) / 20
    ma_50 = sum(closes_1h[-50:]) / 50
    ma_100 = sum(closes_1h[-100:]) / 100

    print(f"Moving Averages (1H):")
    print(f"  MA(20):  {ma_20:.2f}")
    print(f"  MA(50):  {ma_50:.2f}")
    print(f"  MA(100): {ma_100:.2f}")
    print()

    # RSI
    rsi_1h = calculate_rsi(closes_1h, 14)
    rsi_4h = calculate_rsi(closes_4h, 14)

    print(f"RSI:")
    print(f"  1H RSI: {rsi_1h:.2f}")
    print(f"  4H RSI: {rsi_4h:.2f}")

    if rsi_1h > 70:
        print("  ⚠️  Overbought territory")
    elif rsi_1h < 30:
        print("  ✅ Oversold territory")
    else:
        print("  ℹ️  Neutral zone")
    print()

    # MACD
    macd, signal, histogram = calculate_macd(closes_1h)
    print(f"MACD (1H):")
    print(f"  MACD Line: {macd:.2f}")
    print(f"  Signal: {signal:.2f}")
    print(f"  Histogram: {histogram:.2f}")
    print()

    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(closes_1h, 20, 2)
    print(f"Bollinger Bands (1H, 20, 2):")
    print(f"  Upper: {bb_upper:.2f}")
    print(f"  Middle: {bb_middle:.2f}")
    print(f"  Lower: {bb_lower:.2f}")

    bb_position = ((current_price - bb_lower) / (bb_upper - bb_lower)) * 100
    print(f"  Current Position: {bb_position:.1f}%")
    print()

    # Support/Resistance
    support, resistance = find_support_resistance(closes_1h, 10)
    print(f"Support/Resistance (1H):")
    print(f"  Resistance: {resistance:.2f}" if resistance else "  Resistance: Not detected")
    print(f"  Support: {support:.2f}" if support else "  Support: Not detected")
    print()

    # Trend Analysis
    print("📈 TREND ANALYSIS")
    print("=" * 70)

    trend_4h = "BULLISH" if ma_20 > ma_50 > ma_100 else "BEARISH" if ma_20 < ma_50 < ma_100 else "MIXED"
    momentum = "STRONG" if abs(closes_1h[-1] - closes_1h[-10]) > 100 else "MODERATE" if abs(closes_1h[-1] - closes_1h[-10]) > 50 else "WEAK"

    print(f"4H Trend: {trend_4h}")
    print(f"Momentum: {momentum}")
    print(f"Price vs MA(20): {'+' if current_price > ma_20 else '-'}{abs(current_price - ma_20):.2f} ({((current_price - ma_20) / ma_20 * 100):.2f}%)")
    print()

    # Generate Trade Recommendation
    print("💡 TRADE RECOMMENDATION")
    print("=" * 70)

    # Decision logic
    signals_bullish = 0
    signals_bearish = 0

    # Check each indicator
    if ma_20 > ma_50:
        signals_bullish += 1
    else:
        signals_bearish += 1

    if rsi_1h < 40:
        signals_bullish += 1
    elif rsi_1h > 60:
        signals_bearish += 1

    if current_price > bb_middle:
        signals_bullish += 0.5
    else:
        signals_bearish += 0.5

    if macd > 0:
        signals_bullish += 1
    else:
        signals_bearish += 1

    # Determine trade direction
    if signals_bullish > signals_bearish and rsi_1h < 70:
        direction = "BUY"
        color = "🟢"
    elif signals_bearish > signals_bullish and rsi_1h > 30:
        direction = "SELL"
        color = "🔴"
    else:
        direction = "NO TRADE"
        color = "⚪"

    print(f"{color} DIRECTION: {direction}")
    print()

    if direction != "NO TRADE":
        # Calculate entry, stop loss, and take profit
        if direction == "BUY":
            entry = current_offer
            stop_loss = support if support else entry - 100
            take_profit_1 = entry + (entry - stop_loss) * 1.5  # 1.5:1 RR
            take_profit_2 = entry + (entry - stop_loss) * 3.0  # 3:1 RR

            print(f"📍 Entry Price: {entry:.2f}")
            print(f"🛑 Stop Loss: {stop_loss:.2f} ({entry - stop_loss:.2f} points)")
            print(f"🎯 Take Profit 1: {take_profit_1:.2f} (1.5:1 RR)")
            print(f"🎯 Take Profit 2: {take_profit_2:.2f} (3:1 RR)")
            print()
            print(f"💰 Risk: {entry - stop_loss:.2f} points")
            print(f"💰 Reward (TP1): {take_profit_1 - entry:.2f} points")
            print(f"💰 Reward (TP2): {take_profit_2 - entry:.2f} points")

        else:  # SELL
            entry = current_bid
            stop_loss = resistance if resistance else entry + 100
            take_profit_1 = entry - (stop_loss - entry) * 1.5
            take_profit_2 = entry - (stop_loss - entry) * 3.0

            print(f"📍 Entry Price: {entry:.2f}")
            print(f"🛑 Stop Loss: {stop_loss:.2f} ({stop_loss - entry:.2f} points)")
            print(f"🎯 Take Profit 1: {take_profit_1:.2f} (1.5:1 RR)")
            print(f"🎯 Take Profit 2: {take_profit_2:.2f} (3:1 RR)")
            print()
            print(f"💰 Risk: {stop_loss - entry:.2f} points")
            print(f"💰 Reward (TP1): {entry - take_profit_1:.2f} points")
            print(f"💰 Reward (TP2): {entry - take_profit_2:.2f} points")

        print()
        print("📋 REASONING:")
        print(f"  • Bullish Signals: {signals_bullish}")
        print(f"  • Bearish Signals: {signals_bearish}")
        print(f"  • RSI: {rsi_1h:.2f} ({'Overbought' if rsi_1h > 70 else 'Oversold' if rsi_1h < 30 else 'Neutral'})")
        print(f"  • Trend: {trend_4h}")
        print(f"  • MACD: {'Bullish' if macd > 0 else 'Bearish'}")
        print(f"  • BB Position: {bb_position:.1f}%")

        print()
        print("⚠️  RISK MANAGEMENT:")
        print(f"  • Suggested position size: 0.1 - 0.5 lots (depending on account size)")
        print(f"  • Risk 1-2% of account per trade")
        print(f"  • Move stop to breakeven after TP1 is hit")
        print(f"  • Consider trailing stop after price moves in your favor")

    else:
        print("⚠️  No clear trading signal at this time.")
        print("   Wait for better setup or clearer trend direction.")
        print()
        print("Reasons:")
        print(f"  • Conflicting signals (Bullish: {signals_bullish}, Bearish: {signals_bearish})")
        print(f"  • RSI in extreme zone: {rsi_1h:.2f}")
        print("  • Recommend waiting for pullback or breakout")

    print()
    print("=" * 70)
    print("⚠️  DISCLAIMER: This is for educational purposes only.")
    print("   Always do your own analysis and manage risk appropriately.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
