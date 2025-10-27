#!/usr/bin/env python3
"""Try alternative authentication methods"""

import os
import asyncio
import base64
from dotenv import load_dotenv
import httpx

load_dotenv()

API_KEY = os.getenv("CAPITAL_API_KEY")
API_PASSWORD = os.getenv("CAPITAL_API_PASSWORD")
API_URL = "https://api-capital.backend-capital.com"

async def test_methods():
    """Test different auth methods"""

    # Method 1: encryptedPassword (some brokers use this)
    print("Method 1: Using 'encryptedPassword' field")
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
                    "encryptedPassword": "false",
                    "password": API_PASSWORD
                }
            )
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text}\n")
        except Exception as e:
            print(f"Error: {e}\n")

    # Method 2: Just identifier and password
    print("Method 2: Minimal payload")
    print("-" * 60)
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{API_URL}/api/v1/session",
                headers={
                    "X-CAP-API-KEY": API_KEY,
                },
                json={
                    "identifier": API_KEY,
                    "password": API_PASSWORD
                }
            )
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text}\n")
        except Exception as e:
            print(f"Error: {e}\n")

    # Method 3: Check if endpoint is accessible
    print("Method 3: Test API availability")
    print("-" * 60)
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{API_URL}/api/v1/ping")
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text}\n")
        except Exception as e:
            print(f"Error: {e}\n")

asyncio.run(test_methods())
