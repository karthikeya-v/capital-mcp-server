#!/usr/bin/env python3
"""Test authentication with correct format"""

import os
import asyncio
from dotenv import load_dotenv
import httpx

load_dotenv()

API_KEY = os.getenv("CAPITAL_API_KEY")
IDENTIFIER = os.getenv("CAPITAL_IDENTIFIER")
API_PASSWORD = os.getenv("CAPITAL_API_PASSWORD")
USE_DEMO = os.getenv("CAPITAL_USE_DEMO", "true").lower() == "true"

API_URL = "https://demo-api-capital.backend-capital.com" if USE_DEMO else "https://api-capital.backend-capital.com"

print("=" * 60)
print("Capital.com Authentication Test")
print("=" * 60)
print(f"API URL: {API_URL}")
print(f"API Key: {API_KEY}")
print(f"Identifier: {IDENTIFIER}")
print(f"Password: {'*' * len(API_PASSWORD) if API_PASSWORD else 'NOT SET'}")
print()

async def test_auth():
    """Test authentication with correct format"""

    if IDENTIFIER == "your_email@example.com":
        print("⚠️  ERROR: Please update CAPITAL_IDENTIFIER in .env file")
        print("   Set it to your Capital.com login email/username")
        print()
        return

    print("🔐 Attempting authentication...")
    print()

    async with httpx.AsyncClient() as client:
        try:
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

            print(f"Status Code: {response.status_code}")
            print()

            if response.status_code == 200:
                print("✅ Authentication SUCCESSFUL!")
                print()

                # Extract tokens
                cst = response.headers.get("CST")
                security_token = response.headers.get("X-SECURITY-TOKEN")

                print(f"CST Token: {cst[:20]}..." if cst else "CST Token: NOT FOUND")
                print(f"Security Token: {security_token[:20]}..." if security_token else "Security Token: NOT FOUND")
                print()

                print("Response body:")
                print(response.text)

            else:
                print("❌ Authentication FAILED")
                print()
                print("Response:")
                print(response.text)
                print()
                print("Possible issues:")
                print("  - CAPITAL_IDENTIFIER should be your Capital.com email")
                print("  - Check your API password is correct")
                print("  - Verify API access is enabled in your account")

        except Exception as e:
            print(f"❌ Error: {e}")

asyncio.run(test_auth())
