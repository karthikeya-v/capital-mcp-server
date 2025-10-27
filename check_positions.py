#!/usr/bin/env python3
"""Check open positions"""

import os
import asyncio
import json
from dotenv import load_dotenv
import httpx

load_dotenv()

API_KEY = os.getenv("CAPITAL_API_KEY")
IDENTIFIER = os.getenv("CAPITAL_IDENTIFIER")
API_PASSWORD = os.getenv("CAPITAL_API_PASSWORD")
USE_DEMO = os.getenv("CAPITAL_USE_DEMO", "true").lower() == "true"
API_URL = "https://demo-api-capital.backend-capital.com" if USE_DEMO else "https://api-capital.backend-capital.com"


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


async def get_positions(security_token, cst_token):
    """Get open positions"""
    headers = {"X-SECURITY-TOKEN": security_token, "CST": cst_token}
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/api/v1/positions", headers=headers)
        if response.status_code == 200:
            return response.json()
        return None


async def get_account_info(security_token, cst_token):
    """Get account info"""
    headers = {"X-SECURITY-TOKEN": security_token, "CST": cst_token}
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/api/v1/accounts", headers=headers)
        if response.status_code == 200:
            return response.json()
        return None


async def main():
    print("=" * 70)
    print("📊 CHECKING OPEN POSITIONS")
    print("=" * 70)
    print()

    # Authenticate
    security_token, cst_token = await get_session()
    if not security_token:
        print("❌ Authentication failed")
        return

    print("✅ Authenticated")
    print()

    # Get account info
    print("💰 Account Information:")
    account = await get_account_info(security_token, cst_token)
    if account:
        accounts = account.get('accounts', [])
        for acc in accounts:
            if acc.get('preferred'):
                balance_info = acc.get('balance', {})
                print(f"   Account: {acc.get('accountName')}")
                print(f"   Balance: {acc.get('symbol')}{balance_info.get('balance', 0):.2f}")
                print(f"   Available: {acc.get('symbol')}{balance_info.get('available', 0):.2f}")
                print(f"   P&L: {acc.get('symbol')}{balance_info.get('profitLoss', 0):.2f}")
    print()

    # Get positions
    print("📊 Open Positions:")
    positions = await get_positions(security_token, cst_token)

    if not positions or not positions.get('positions'):
        print("   No open positions")
    else:
        for pos in positions.get('positions', []):
            position = pos.get('position', {})
            market = pos.get('market', {})

            print(f"\n{'='*70}")
            print(f"   Instrument: {market.get('instrumentName')} ({market.get('epic')})")
            print(f"   Direction: {position.get('direction')}")
            print(f"   Size: {position.get('dealSize')} lots")
            print(f"   Open Level: {position.get('openLevel')}")
            print(f"   Current Level: {position.get('level')}")
            print(f"   Stop Loss: {position.get('stopLevel')}")
            print(f"   Take Profit: {position.get('limitLevel')}")
            print(f"   P&L: {market.get('symbol')}{position.get('profit', 0):.2f}")
            print(f"   Deal ID: {position.get('dealId')}")
            print(f"   Created: {position.get('createdDateUTC')}")

    print()
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
