#!/usr/bin/env python3
"""Debug authentication with Capital.com"""

import os
import asyncio
import json
from dotenv import load_dotenv
import httpx

load_dotenv()

API_KEY = os.getenv("CAPITAL_API_KEY")
API_PASSWORD = os.getenv("CAPITAL_API_PASSWORD")
USE_DEMO = os.getenv("CAPITAL_USE_DEMO", "true").lower() == "true"

print(f"API Key: {API_KEY}")
print(f"Password: {'*' * len(API_PASSWORD) if API_PASSWORD else 'NOT SET'}")
print(f"Use Demo: {USE_DEMO}")
print()

API_URL = "https://demo-api-capital.backend-capital.com" if USE_DEMO else "https://api-capital.backend-capital.com"

print(f"Attempting to connect to: {API_URL}")
print()

async def test_auth():
    """Test different authentication methods"""

    # Method 1: Standard authentication
    print("Method 1: Standard POST /api/v1/session")
    print("-" * 60)

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{API_URL}/api/v1/session",
                headers={
                    "X-CAP-API-KEY": API_KEY,
                    "Content-Type": "application/json"
                },
                json={
                    "identifier": API_KEY,
                    "password": API_PASSWORD
                }
            )

            print(f"Status Code: {response.status_code}")
            print(f"Headers: {dict(response.headers)}")
            print(f"Body: {response.text}")
            print()

        except Exception as e:
            print(f"Error: {e}")
            print()

asyncio.run(test_auth())
