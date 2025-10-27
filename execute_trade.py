#!/usr/bin/env python3
"""
Analyze US Tech 100 and execute a trade on demo account
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

EPIC = "US100"
RISK_AMOUNT = 500  # $500 risk


async def get_session():
    """Authenticate and get session tokens"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_URL}/api/v1/session",
            headers={"X-CAP-API-KEY": API_KEY, "Content-Type": "application/json"},
            json={"identifier": IDENTIFIER, "password": API_PASSWORD, "encryptedPassword": False}
        )
        if response.status_code == 200:
            data = response.json()
            return response.headers.get("X-SECURITY-TOKEN"), response.headers.get("CST"), data
        return None, None, None


async def get_market_details(security_token, cst_token):
    """Get current market details"""
    headers = {"X-SECURITY-TOKEN": security_token, "CST": cst_token}
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/api/v1/markets/{EPIC}", headers=headers)
        if response.status_code == 200:
            return response.json()
        return None


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


def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    if len(prices) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        gains.append(max(change, 0))
        losses.append(max(-change, 0))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


async def place_trade(security_token, cst_token, direction, size, stop_loss, take_profit):
    """Place a trade order"""
    headers = {
        "X-SECURITY-TOKEN": security_token,
        "CST": cst_token,
        "Content-Type": "application/json"
    }

    payload = {
        "epic": EPIC,
        "direction": direction,
        "size": size,
        "guaranteedStop": False
    }

    if stop_loss:
        payload["stopLevel"] = stop_loss
    if take_profit:
        payload["profitLevel"] = take_profit

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_URL}/api/v1/positions",
            headers=headers,
            json=payload
        )
        return response.status_code, response.json() if response.status_code in [200, 201] else response.text


async def main():
    """Main execution function"""
    print("=" * 70)
    print("🎯 US TECH 100 TRADE EXECUTION")
    print("=" * 70)
    print(f"Mode: {'DEMO' if USE_DEMO else 'LIVE'} Account")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Risk Amount: ${RISK_AMOUNT}")
    print()

    # Authenticate
    print("🔐 Authenticating...")
    security_token, cst_token, account_data = await get_session()

    if not security_token:
        print("❌ Authentication failed")
        return

    print("✅ Connected to Capital.com")

    # Show account info
    account_info = account_data.get('accountInfo', {})
    print(f"💰 Account Balance: {account_data.get('currencySymbol', '$')}{account_info.get('balance', 0):.2f}")
    print(f"💵 Available Funds: {account_data.get('currencySymbol', '$')}{account_info.get('available', 0):.2f}")
    print()

    # Get market data
    print("📊 Fetching market data...")
    market_data = await get_market_details(security_token, cst_token)

    if not market_data:
        print("❌ Failed to fetch market data")
        return

    snapshot = market_data.get('snapshot', {})
    instrument = market_data.get('instrument', {})
    dealing_rules = market_data.get('dealingRules', {})

    current_bid = snapshot.get('bid')
    current_offer = snapshot.get('offer')
    current_price = (current_bid + current_offer) / 2

    print(f"📈 US Tech 100 (US100)")
    print(f"   Current Bid: {current_bid:.2f}")
    print(f"   Current Offer: {current_offer:.2f}")
    print(f"   Mid Price: {current_price:.2f}")
    print(f"   Daily Change: {snapshot.get('netChange')} ({snapshot.get('percentageChange')}%)")
    print()

    # Get historical data
    print("📉 Analyzing technical indicators...")
    price_data = await get_price_history(security_token, cst_token, "HOUR", 200)

    if not price_data:
        print("❌ Failed to fetch price history")
        return

    closes = [p['closePrice']['ask'] for p in price_data['prices'] if 'closePrice' in p]

    # Calculate indicators
    rsi = calculate_rsi(closes, 14)
    ma_20 = sum(closes[-20:]) / 20
    ma_50 = sum(closes[-50:]) / 50

    print(f"   RSI: {rsi:.2f}")
    print(f"   MA(20): {ma_20:.2f}")
    print(f"   MA(50): {ma_50:.2f}")
    print()

    # Analyze and decide
    print("🔬 MARKET ANALYSIS")
    print("=" * 70)

    bullish_signals = 0
    bearish_signals = 0
    reasons = []

    # Moving averages
    if ma_20 > ma_50:
        bullish_signals += 1
        reasons.append("✅ MA(20) > MA(50) - Bullish trend")
    else:
        bearish_signals += 1
        reasons.append("❌ MA(20) < MA(50) - Bearish trend")

    # RSI
    if rsi < 30:
        bullish_signals += 2
        reasons.append(f"✅ RSI {rsi:.1f} - Oversold (strong buy signal)")
    elif rsi > 70:
        bearish_signals += 2
        reasons.append(f"❌ RSI {rsi:.1f} - Overbought (caution)")
    elif 40 < rsi < 60:
        reasons.append(f"ℹ️  RSI {rsi:.1f} - Neutral zone")
    else:
        reasons.append(f"ℹ️  RSI {rsi:.1f}")

    # Price vs MA
    if current_price > ma_20:
        bullish_signals += 0.5
        reasons.append(f"✅ Price above MA(20) by {current_price - ma_20:.2f} points")
    else:
        bearish_signals += 0.5
        reasons.append(f"⚠️  Price below MA(20) by {ma_20 - current_price:.2f} points")

    for reason in reasons:
        print(f"  {reason}")

    print()
    print(f"📊 Signal Score:")
    print(f"   Bullish: {bullish_signals}")
    print(f"   Bearish: {bearish_signals}")
    print()

    # Determine direction
    if bullish_signals > bearish_signals and rsi < 75:
        direction = "BUY"
        entry_price = current_offer
        stop_distance = 100  # 100 points
        stop_loss = entry_price - stop_distance
        take_profit = entry_price + (stop_distance * 2)  # 2:1 RR
        print(f"💡 RECOMMENDATION: {direction} (Bullish signals dominant)")
    elif bearish_signals > bullish_signals and rsi > 25:
        direction = "SELL"
        entry_price = current_bid
        stop_distance = 100
        stop_loss = entry_price + stop_distance
        take_profit = entry_price - (stop_distance * 2)
        print(f"💡 RECOMMENDATION: {direction} (Bearish signals dominant)")
    else:
        print("⚠️  CONFLICTING SIGNALS - Market unclear")
        print("   Proceeding with conservative position based on trend...")
        # Default to trend direction with tight stops
        if ma_20 > ma_50:
            direction = "BUY"
            entry_price = current_offer
            stop_distance = 80
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + (stop_distance * 1.5)
        else:
            direction = "SELL"
            entry_price = current_bid
            stop_distance = 80
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - (stop_distance * 1.5)

    print()

    # Calculate position size
    # Capital.com typically uses 1 point = $1 for indices at minimum size
    # For US100, 1 lot typically = $1 per point
    # Risk = Position Size * Stop Distance
    # Position Size = Risk / Stop Distance

    min_size = dealing_rules.get('minDealSize', {}).get('value', 0.1)
    max_size = dealing_rules.get('maxDealSize', {}).get('value', 100)

    position_size = RISK_AMOUNT / stop_distance

    # Round to acceptable increments (usually 0.1)
    position_size = round(position_size * 10) / 10

    # Ensure within limits
    position_size = max(min_size, min(position_size, max_size))

    # Calculate actual risk with rounded position
    actual_risk = position_size * stop_distance

    print("📍 TRADE SETUP")
    print("=" * 70)
    print(f"Direction: {direction}")
    print(f"Entry Price: {entry_price:.2f}")
    print(f"Stop Loss: {stop_loss:.2f} ({stop_distance} points)")
    print(f"Take Profit: {take_profit:.2f}")
    print(f"Position Size: {position_size} lots")
    print(f"Risk per Point: ${position_size:.2f}")
    print(f"Total Risk: ${actual_risk:.2f}")
    print(f"Potential Reward: ${abs(take_profit - entry_price) * position_size:.2f}")
    print(f"Risk/Reward: 1:{abs((take_profit - entry_price) / (entry_price - stop_loss)) if direction == 'BUY' else abs((take_profit - entry_price) / (stop_loss - entry_price)):.1f}")
    print()

    # Confirm trade
    print("⚠️  IMPORTANT: This is a DEMO account trade")
    print()

    input("Press Enter to execute trade (or Ctrl+C to cancel)...")
    print()

    # Place the trade
    print(f"🚀 Placing {direction} order...")
    status_code, result = await place_trade(
        security_token, cst_token, direction, position_size, stop_loss, take_profit
    )

    print()
    if status_code in [200, 201]:
        print("✅ TRADE EXECUTED SUCCESSFULLY!")
        print("=" * 70)
        print(json.dumps(result, indent=2))
        print()

        deal_reference = result.get('dealReference')
        print(f"📋 Deal Reference: {deal_reference}")
        print()
        print(f"💰 Trade Summary:")
        print(f"   Direction: {direction}")
        print(f"   Size: {position_size} lots")
        print(f"   Entry: {entry_price:.2f}")
        print(f"   Stop: {stop_loss:.2f}")
        print(f"   Target: {take_profit:.2f}")
        print(f"   Risk: ${actual_risk:.2f}")
        print()
        print("✅ Position is now OPEN on your demo account!")
        print("   Monitor it in your Capital.com platform")

    else:
        print("❌ TRADE FAILED")
        print(f"Status Code: {status_code}")
        print(f"Error: {result}")

    print()
    print("=" * 70)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n❌ Trade cancelled by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
