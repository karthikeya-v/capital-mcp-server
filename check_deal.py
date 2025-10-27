#!/usr/bin/env python3
"""Check deal confirmation"""

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

DEAL_REFERENCE = "o_7de2e630-bcc2-4a48-84b0-f711b78f8f05"


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


async def check_deal(security_token, cst_token, deal_ref):
    """Check deal confirmation"""
    headers = {"X-SECURITY-TOKEN": security_token, "CST": cst_token}
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{API_URL}/api/v1/confirms/{deal_ref}",
            headers=headers
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        return response.status_code, response.json() if response.status_code == 200 else response.text


async def main():
    print("=" * 70)
    print("🔍 CHECKING DEAL CONFIRMATION")
    print("=" * 70)
    print(f"Deal Reference: {DEAL_REFERENCE}")
    print()

    # Authenticate
    security_token, cst_token = await get_session()
    if not security_token:
        print("❌ Authentication failed")
        return

    print("✅ Authenticated")
    print()

    # Check deal
    status, result = await check_deal(security_token, cst_token, DEAL_REFERENCE)

    if status == 200:
        print("✅ Deal Confirmation:")
        print(json.dumps(result, indent=2))
    else:
        print(f"❌ Could not retrieve deal confirmation")
        print(f"This might be normal - the deal may still be processing")

    print()
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
