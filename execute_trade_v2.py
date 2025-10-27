#!/usr/bin/env python3
"""
Execute trade with proper risk management for demo account
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


async def get_session():
    """Authenticate"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_URL}/api/v1/session",
            headers={"X-CAP-API-KEY": API_KEY, "Content-Type": "application/json"},
            json={"identifier": IDENTIFIER, "password": API_PASSWORD, "encryptedPassword": False}
        )
        if response.status_code == 200:
            return response.headers.get("X-SECURITY-TOKEN"), response.headers.get("CST")
        return None, None


async def get_market_details(security_token, cst_token):
    """Get market details"""
    headers = {"X-SECURITY-TOKEN": security_token, "CST": cst_token}
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/api/v1/markets/{EPIC}", headers=headers)
        if response.status_code == 200:
            return response.json()
        return None


async def place_trade(security_token, cst_token, direction, size, stop_loss, take_profit):
    """Place trade"""
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


async def check_deal(security_token, cst_token, deal_ref):
    """Check deal confirmation"""
    headers = {"X-SECURITY-TOKEN": security_token, "CST": cst_token}
    async with httpx.AsyncClient() as client:
        # Wait a moment for processing
        await asyncio.sleep(1)
        response = await client.get(
            f"{API_URL}/api/v1/confirms/{deal_ref}",
            headers=headers
        )
        return response.status_code, response.json() if response.status_code == 200 else None


async def main():
    print("=" * 70)
    print("🎯 US TECH 100 TRADE EXECUTION (Adjusted)")
    print("=" * 70)
    print(f"Mode: DEMO Account")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Authenticate
    print("🔐 Authenticating...")
    security_token, cst_token = await get_session()

    if not security_token:
        print("❌ Authentication failed")
        return

    print("✅ Connected")
    print()

    # Get market data
    print("📊 Fetching market data...")
    market_data = await get_market_details(security_token, cst_token)

    if not market_data:
        print("❌ Failed to fetch market data")
        return

    snapshot = market_data.get('snapshot', {})
    current_bid = snapshot.get('bid')
    current_offer = snapshot.get('offer')
    current_price = (current_bid + current_offer) / 2

    print(f"📈 US Tech 100")
    print(f"   Current Price: {current_price:.2f}")
    print(f"   Bid: {current_bid:.2f}")
    print(f"   Offer: {current_offer:.2f}")
    print()

    # TRADE SETUP - Conservative sizing for demo account
    # Based on previous rejection, using much smaller size
    direction = "SELL"  # Market is overbought (RSI 86.7)
    entry_price = current_bid
    stop_distance = 50  # Tighter stop
    stop_loss = entry_price + stop_distance
    take_profit = entry_price - (stop_distance * 2)  # 2:1 RR
    position_size = 0.5  # Small size to pass risk checks

    print("📍 TRADE SETUP (Conservative)")
    print("=" * 70)
    print(f"Direction: {direction}")
    print(f"Entry: {entry_price:.2f}")
    print(f"Stop Loss: {stop_loss:.2f} (+{stop_distance} points)")
    print(f"Take Profit: {take_profit:.2f} (-{stop_distance * 2} points)")
    print(f"Position Size: {position_size} lots")
    print(f"Risk: ${position_size * stop_distance:.2f}")
    print(f"Potential Reward: ${position_size * stop_distance * 2:.2f}")
    print(f"Risk/Reward: 1:2")
    print()

    print("⚠️  This is a DEMO account trade")
    print()

    input("Press Enter to execute (Ctrl+C to cancel)...")
    print()

    # Place trade
    print(f"🚀 Placing {direction} order...")
    status_code, result = await place_trade(
        security_token, cst_token, direction, position_size, stop_loss, take_profit
    )

    print()
    if status_code in [200, 201]:
        deal_reference = result.get('dealReference')
        print(f"✅ Order submitted!")
        print(f"📋 Deal Reference: {deal_reference}")
        print()

        # Check confirmation
        print("⏳ Checking deal confirmation...")
        conf_status, confirmation = await check_deal(security_token, cst_token, deal_reference)

        if confirmation:
            deal_status = confirmation.get('dealStatus')
            reject_reason = confirmation.get('rejectReason')

            if deal_status == "ACCEPTED":
                print()
                print("✅ TRADE EXECUTED SUCCESSFULLY!")
                print("=" * 70)
                print(json.dumps(confirmation, indent=2))
                print()
                print(f"💰 Position Summary:")
                print(f"   Direction: {direction}")
                print(f"   Size: {position_size} lots")
                print(f"   Entry: {entry_price:.2f}")
                print(f"   Stop: {stop_loss:.2f}")
                print(f"   Target: {take_profit:.2f}")
                print(f"   Deal ID: {confirmation.get('dealId')}")
                print()
                print("✅ Position is OPEN on your demo account!")

            else:
                print()
                print(f"❌ TRADE REJECTED")
                print(f"Status: {deal_status}")
                print(f"Reason: {reject_reason}")
                print()
                print("Possible reasons:")
                print("  • Demo account limitations")
                print("  • Market closed or suspended")
                print("  • Position size still too large")
                print("  • Insufficient margin")
        else:
            print("⚠️  Could not retrieve confirmation")

    else:
        print("❌ ORDER FAILED")
        print(f"Status: {status_code}")
        print(f"Error: {result}")

    print()
    print("=" * 70)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
