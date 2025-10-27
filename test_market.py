#!/usr/bin/env python3
"""
Quick test script to fetch and analyze US Tech 100
"""

import os
import asyncio
import json
from dotenv import load_dotenv
import httpx

# Load credentials
load_dotenv()

API_KEY = os.getenv("CAPITAL_API_KEY")
IDENTIFIER = os.getenv("CAPITAL_IDENTIFIER")
API_PASSWORD = os.getenv("CAPITAL_API_PASSWORD")
USE_DEMO = os.getenv("CAPITAL_USE_DEMO", "true").lower() == "true"
API_URL = "https://demo-api-capital.backend-capital.com" if USE_DEMO else "https://api-capital.backend-capital.com"


async def get_session():
    """Authenticate and get session tokens"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_URL}/api/v1/session",
            headers={
                "X-CAP-API-KEY": API_KEY,
                "Content-Type": "application/json"
            },
            json={
                "identifier": IDENTIFIER,
                "password": API_PASSWORD,
                "encryptedPassword": False
            }
        )

        if response.status_code == 200:
            security_token = response.headers.get("X-SECURITY-TOKEN")
            cst_token = response.headers.get("CST")
            return security_token, cst_token
        else:
            print(f"❌ Authentication failed: {response.status_code}")
            print(response.text)
            return None, None


async def search_market(security_token, cst_token, search_term):
    """Search for a market"""
    headers = {
        "X-SECURITY-TOKEN": security_token,
        "CST": cst_token
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{API_URL}/api/v1/markets",
            headers=headers,
            params={"searchTerm": search_term}
        )

        if response.status_code == 200:
            return response.json()
        return None


async def get_market_details(security_token, cst_token, epic):
    """Get detailed market information"""
    headers = {
        "X-SECURITY-TOKEN": security_token,
        "CST": cst_token
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{API_URL}/api/v1/markets/{epic}",
            headers=headers
        )

        if response.status_code == 200:
            return response.json()
        return None


async def get_price_history(security_token, cst_token, epic, resolution="HOUR", max_points=100):
    """Get historical price data"""
    headers = {
        "X-SECURITY-TOKEN": security_token,
        "CST": cst_token
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{API_URL}/api/v1/prices/{epic}",
            headers=headers,
            params={
                "resolution": resolution,
                "max": max_points
            }
        )

        if response.status_code == 200:
            return response.json()
        return None


async def main():
    """Main analysis function"""
    print("=" * 60)
    print("🚀 Capital.com Market Analysis Tool")
    print("=" * 60)
    print(f"Mode: {'DEMO' if USE_DEMO else 'LIVE'}")
    print(f"API URL: {API_URL}")
    print()

    # Authenticate
    print("🔐 Authenticating...")
    security_token, cst_token = await get_session()

    if not security_token:
        print("❌ Failed to authenticate. Check your credentials in .env file")
        return

    print("✅ Authentication successful!")
    print()

    # Search for US Tech 100
    print("🔍 Searching for US Tech 100...")
    search_results = await search_market(security_token, cst_token, "US Tech 100")

    if not search_results or not search_results.get('markets'):
        print("❌ Market not found")
        return

    # Find the right market
    markets = search_results.get('markets', [])
    us_tech = None

    for market in markets:
        if "US Tech 100" in market.get('instrumentName', ''):
            us_tech = market
            break

    if not us_tech:
        print("❌ US Tech 100 not found")
        print("Available markets:")
        for m in markets[:5]:
            print(f"  - {m.get('instrumentName')} ({m.get('epic')})")
        return

    epic = us_tech.get('epic')
    print(f"✅ Found: {us_tech.get('instrumentName')} (Epic: {epic})")
    print()

    # Get current price
    print("📊 Fetching current market data...")
    market_details = await get_market_details(security_token, cst_token, epic)

    if market_details:
        snapshot = market_details.get('snapshot', {})
        print(f"")
        print(f"💰 Current Price:")
        print(f"   Bid: {snapshot.get('bid')}")
        print(f"   Offer: {snapshot.get('offer')}")
        print(f"   Spread: {snapshot.get('offer', 0) - snapshot.get('bid', 0):.2f}")
        print(f"")
        print(f"📈 Daily Stats:")
        print(f"   High: {snapshot.get('high')}")
        print(f"   Low: {snapshot.get('low')}")
        print(f"   Change: {snapshot.get('netChange')} ({snapshot.get('percentageChange')}%)")
        print()

    # Get historical data
    print("📉 Fetching price history (last 50 hours)...")
    price_history = await get_price_history(security_token, cst_token, epic, "HOUR", 50)

    if price_history and 'prices' in price_history:
        prices = price_history['prices']
        print(f"✅ Retrieved {len(prices)} candles")
        print()

        # Simple technical analysis
        closes = [p['closePrice']['ask'] for p in prices if 'closePrice' in p]

        if len(closes) >= 20:
            print("📊 TECHNICAL ANALYSIS:")
            print("-" * 60)

            # Current price
            current = closes[-1]
            print(f"Current Price: {current:.2f}")

            # Moving averages
            ma_10 = sum(closes[-10:]) / 10
            ma_20 = sum(closes[-20:]) / 20
            print(f"10-period MA: {ma_10:.2f}")
            print(f"20-period MA: {ma_20:.2f}")

            # Trend analysis
            if ma_10 > ma_20:
                print("📈 Trend: BULLISH (10 MA > 20 MA)")
            else:
                print("📉 Trend: BEARISH (10 MA < 20 MA)")

            # Price position
            if current > ma_10:
                print(f"✅ Price is {((current - ma_10) / ma_10 * 100):.2f}% above 10 MA")
            else:
                print(f"⚠️  Price is {((ma_10 - current) / ma_10 * 100):.2f}% below 10 MA")

            # Recent change
            hour_ago = closes[-2]
            hour_change = ((current - hour_ago) / hour_ago) * 100
            print(f"1-hour change: {hour_change:+.2f}%")

            # Range
            recent_high = max(closes[-20:])
            recent_low = min(closes[-20:])
            position_in_range = ((current - recent_low) / (recent_high - recent_low)) * 100
            print(f"20-hour range: {recent_low:.2f} - {recent_high:.2f}")
            print(f"Position in range: {position_in_range:.1f}%")

            print()
            print("💡 ANALYSIS SUMMARY:")
            print("-" * 60)

            if ma_10 > ma_20 and current > ma_10:
                print("Strong bullish momentum. Price above both MAs.")
                if position_in_range > 80:
                    print("⚠️  Near resistance - consider waiting for pullback")
                else:
                    print("✅ Good entry zone for long positions")
            elif ma_10 < ma_20 and current < ma_10:
                print("Strong bearish momentum. Price below both MAs.")
                if position_in_range < 20:
                    print("⚠️  Near support - oversold conditions")
                else:
                    print("⚠️  Bearish trend - caution advised")
            else:
                print("Mixed signals. Market in consolidation.")
                print("💡 Wait for clearer trend before entering")

    print()
    print("=" * 60)
    print("✅ Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
